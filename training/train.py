import argparse
import contextlib
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from data import dataset as D
from model.main_model import MainModel
from model.sampling import ensemble_metrics, sample

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = Path.cwd()


def build_parser(description=__doc__):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--fasta", type=Path, default=ROOT / "data/alphafold_embeddings/train_data.fasta")
    a("--embedding-path", type=Path, default=ROOT / "data/alphafold_embeddings/train_embeddings")
    a("--data-dirs", type=Path, nargs="+",
      default=[ROOT.parent / "temp_sim_dataset/filtered_train_data"],
      help="searched in order; a domain's snapshot file is preferred over its full file")
    a("--val-fasta", type=Path, default=ROOT / "data/alphafold_embeddings/val_data.fasta",
      help="validation is skipped while this file does not exist")
    a("--val-embedding-path", type=Path, default=ROOT / "data/alphafold_embeddings/val_embeddings")
    a("--val-data-dirs", type=Path, nargs="+", help="default: --data-dirs")
    a("--save-path", type=Path, default=ROOT / "weights/tempformer_5200_af.pth")
    a("--resume", action=argparse.BooleanOptionalAction, default=True)
    # data
    a("--frames-per-temp", type=int, default=10)
    a("--crop-size", type=int, default=256)
    a("--limit-domains", type=int, help="first N FASTA domains only (smoke tests)")
    a("--num-workers", type=int, default=4)
    # optimisation
    a("--batch-size", type=int, default=6)
    a("--accum-steps", type=int, default=8, help="effective batch = batch-size x accum-steps")
    a("--epochs", type=int, default=300)
    a("--lr", type=float, default=1e-4)
    a("--min-lr-ratio", type=float, default=0.01)
    a("--warmup-steps", type=int, default=2000, help="optimiser steps")
    a("--weight-decay", type=float, default=1e-5)
    a("--ema-decay", type=float, default=0.999)
    a("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    a("--seed", type=int, default=42)
    # model
    a("--dropout", type=float, default=0.0)
    a("--aux-max-sigma", type=float, default=2.0, help="FAPE/torsion/bond losses where sigma_tr < this (A)")
    a("--w-fape", type=float, default=1.0)
    a("--w-torsion", type=float, default=1.0)
    a("--w-bond", type=float, default=1.0)
    a("--w-chi", type=float, default=1.0, help="weight of the side-chain chi score loss")
    a("--temp-smooth-weight", type=float, default=0.0, help="curvature penalty on the temperature embedding")
    # validation
    a("--val-seed", type=int, default=0)
    a("--sample-val-every", type=int, default=5, help="epochs; 0 disables sampling-based validation")
    a("--sample-val-domains", type=int, default=10)
    a("--sample-val-n", type=int, default=16, help="samples per domain and temperature")
    a("--sample-val-steps", type=int, default=100)
    a("--sample-val-max-len", type=int, default=320)
    a("--sc-weight", type=float, default=1.0, help="weight of the chi1 rotamer error in the sampling score")
    a("--check", action="store_true")
    return p


def parse_args(parser=None):
    args = (parser or build_parser()).parse_args()
    for name in ("fasta", "embedding_path", "val_fasta", "val_embedding_path", "save_path"):
        setattr(args, name, LAUNCH_DIR / getattr(args, name))
    args.data_dirs = [LAUNCH_DIR / d for d in args.data_dirs]
    args.val_data_dirs = [LAUNCH_DIR / d for d in args.val_data_dirs] if args.val_data_dirs else args.data_dirs
    return args


class EMA:
    """Exponential moving average of all floating-point state, with the usual warm-up of the decay."""

    def __init__(self, model, decay):
        self.decay, self.updates = decay, 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def weights(self, like):
        return {k: v.to(like[k].dtype) for k, v in self.shadow.items()}

    @contextlib.contextmanager
    def applied(self, model):
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.weights(raw))
        try:
            yield
        finally:
            model.load_state_dict(raw)


def save_atomic(obj, path):
    """torch.save to a temporary file, then rename: readers (evaluate.py) never see a half-written checkpoint,
    and a preempted save leaves the previous checkpoint intact."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        v = v.to(device, non_blocking=True)
        if k in ("single", "pair"):
            v = v.to(dtype)
        out[k] = v
    return out


@torch.no_grad()
def validation_loss(model, loader, device, amp, embed_dtype, seed, max_batches=None):
    """Mean loss with dropout off and the same timesteps/noise every call (RNG states restored afterwards)."""
    cuda = torch.cuda.is_available()
    saved = torch.get_rng_state(), np.random.get_state(), torch.cuda.get_rng_state_all() if cuda else None
    torch.manual_seed(seed)
    np.random.seed(seed)
    was = model.training
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        with amp():
            total += float(model(to_device(batch, device, embed_dtype))["loss"])
        n += 1
        if max_batches and n >= max_batches:
            break
    model.train(was)
    torch.set_rng_state(saved[0])
    np.random.set_state(saved[1])
    if cuda:
        torch.cuda.set_rng_state_all(saved[2])
    return total / max(n, 1)


def sampling_validation(model, pool, args, device, n=None, steps=None, domains=None, temps=None):
    """Ensembles vs MD for the first validation domains: per-temperature metrics and the overall score."""
    doms = [d for d in pool.domains if len(d["aatype"]) <= args.sample_val_max_len][: domains or args.sample_val_domains]
    rows = []
    with tempfile.TemporaryDirectory(dir=args.save_path.parent) as tmp:        # DSSP scratch files, removed after
        for d in doms:
            feats = D.protein_batch(d)
            for temp in temps or D.TEMPS:
                if temp not in d["md"]["rmsf"]:
                    continue
                atoms = sample(model, feats, float(temp), n or args.sample_val_n, n_steps=steps or args.sample_val_steps,
                               seed=args.val_seed, device=device)
                rows.append({"domain": d["domain_id"], "T": temp,
                             **ensemble_metrics(atoms, d["aatype"], d["md"], temp, Path(tmp))})
    if not rows:
        return None, []
    keys = ("rmsf_r", "q_err", "ss_err", "chi1_err", "chi1_err_floor", "chi2_err", "chi2_err_floor")
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    chi1 = 0.0 if np.isnan(mean["chi1_err"]) else mean["chi1_err"]          # e.g. domains without chis
    score = mean["rmsf_r"] - mean["q_err"] - mean["ss_err"] - args.sc_weight * chi1
    return score, [{"mean": mean}] + rows


def main(args=None, index_fn=D.index_embeddings, model_extra=None):
    """index_fn(embedding dir) finds the embedding files (data.build_pool); model_extra: further MainModel
    arguments. train_dca.py and train_esm.py pass their own; the embedding widths come from the files."""
    args = args or parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    (ROOT / "logs").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    amp = lambda: torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    embed_dtype = torch.float16 if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else torch.float32)
    print(f"device {device.type}, precision {precision}", flush=True)

    train_pool = D.build_pool("train", args.fasta, args.embedding_path, args.data_dirs, args.crop_size,
                              limit=args.limit_domains, index_fn=index_fn)
    val_pool, val_loader = None, None
    if args.val_fasta.exists():
        val_pool = D.build_pool("val", args.val_fasta, args.val_embedding_path, args.val_data_dirs, args.crop_size,
                                limit=args.limit_domains, keep_md_reference=args.sample_val_every > 0,
                                index_fn=index_fn)
        val_frames = val_pool.draw(args.frames_per_temp, [args.val_seed, 1], random_rank=False, random_crop=False)
        if len(val_frames) >= args.batch_size:
            val_loader = D.make_loader(val_frames, args.batch_size, args.num_workers, shuffle=False)
            print(f"val: {len(val_frames)} frames, {len(val_loader)} batches", flush=True)
        else:
            print(f"WARNING: only {len(val_frames)} validation frames: validation off", flush=True)
            val_pool = None
    else:
        print(f"no validation set ({args.val_fasta} not found)", flush=True)
    sample_val = val_pool is not None and args.sample_val_every > 0
    metric = "sample_score" if sample_val else ("val_loss" if val_loader else "train_loss")
    print(f"best checkpoint chosen by {metric}", flush=True)

    single_dim, pair_dim = D.embedding_widths(train_pool)
    print(f"embeddings: single {single_dim}, pair {pair_dim} channels", flush=True)
    model_kwargs = dict(input_shape=single_dim, pair_shape=pair_dim, dropout=args.dropout,
                        aux_max_sigma=args.aux_max_sigma, w_fape=args.w_fape, w_torsion=args.w_torsion,
                        w_bond=args.w_bond, w_chi=args.w_chi, temp_smooth_weight=args.temp_smooth_weight,
                        **(model_extra or {}))
    model = MainModel(**model_kwargs).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimiser steps happen after every full group of accum-steps batches (a partial last group is dropped)
    steps_per_epoch = len(train_pool.draw(args.frames_per_temp, [args.seed, 0])) // args.batch_size // args.accum_steps
    total_steps = max(1, steps_per_epoch * args.epochs)
    def lr_factor(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(enabled=precision == "fp16")
    ema = EMA(model, args.ema_decay)
    print(f"{steps_per_epoch} optimiser steps per epoch (effective batch {args.batch_size * args.accum_steps}), "
          f"{total_steps} in total", flush=True)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.save_path.with_name(args.save_path.stem + "_last.pth")
    start_epoch, best = 0, -float("inf")          # everything is maximised: losses are stored negated
    if args.resume and last_path.exists() and not args.check:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["raw_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        ema.shadow, ema.updates = ckpt["ema_state_dict"], ckpt["ema_updates"]
        start_epoch, best = ckpt["epoch"], ckpt["best"]
        if ckpt.get("best_metric") != metric:
            print(f"best checkpoint was chosen by {ckpt.get('best_metric')}, now by {metric}: best reset", flush=True)
            best = -float("inf")
        print(f"resumed from {last_path} after epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        frames = train_pool.draw(args.frames_per_temp, [args.seed, epoch])
        loader = D.make_loader(frames, args.batch_size, args.num_workers, shuffle=True)
        model.train()
        sums, count, start = {}, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            with amp():
                out = model(to_device(batch, device, embed_dtype))
            scaler.scale(out["loss"] / args.accum_steps).backward()
            for k, v in out.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            count += 1
            if (i + 1) % args.accum_steps == 0 or args.check:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                step = scheduler.last_epoch
                if step % 200 == 0:
                    print(f"  step {step} | " + " | ".join(f"{k} {v / count:.4f}" for k, v in sums.items())
                          + f" | lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
            if args.check:
                print("check ok: training step | " + " | ".join(f"{k} {float(v):.4f}" for k, v in out.items()), flush=True)
                break
        train_loss = sums["loss"] / max(count, 1)

        val_loss, score, val_rows = None, None, []
        with ema.applied(model):
            if val_loader:
                val_loss = validation_loss(model, val_loader, device, amp, embed_dtype, args.val_seed,
                                           max_batches=1 if args.check else None)
            if sample_val and (args.check or (epoch + 1) % args.sample_val_every == 0):
                kw = dict(n=2, steps=5, domains=1, temps=["320"]) if args.check else {}
                score, val_rows = sampling_validation(model, val_pool, args, device, **kw)
        if args.check:
            print(f"check ok: validation loss {val_loss} | sampling score {score} | {val_rows[:1]}", flush=True)
            return

        value = {"sample_score": score, "val_loss": -val_loss if val_loss is not None else None,
                 "train_loss": -train_loss}[metric]
        state = {"epoch": epoch + 1, "model_state_dict": ema.weights(model.state_dict()),
                 "raw_model_state_dict": model.state_dict(), "config": model.config, "loss": train_loss,
                 "val_loss": val_loss, "sample_score": score, "sample_metrics": val_rows}
        if value is not None and value > best:
            best = value
            save_atomic(state, args.save_path)
        save_atomic({**state, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                     "scaler_state_dict": scaler.state_dict(), "ema_state_dict": ema.shadow, "ema_updates": ema.updates,
                     "best": best, "best_metric": metric}, last_path)
        if val_rows:
            with open(args.save_path.with_name(args.save_path.stem + "_sample_validation.jsonl"), "a") as fp:
                fp.write(json.dumps({"epoch": epoch + 1, "score": score, "rows": val_rows}) + "\n")
        parts = [f"train loss {train_loss:.4f}"] + ([f"val loss {val_loss:.4f}"] if val_loss is not None else []) \
            + ([f"sampling score {score:.4f}"] if score is not None else []) \
            + ([f"chi1 err {val_rows[0]['mean']['chi1_err']:.3f} (floor {val_rows[0]['mean']['chi1_err_floor']:.3f}) "
                f"chi2 err {val_rows[0]['mean']['chi2_err']:.3f} (floor {val_rows[0]['mean']['chi2_err_floor']:.3f})"]
               if val_rows else [])
        shown = best if metric == "sample_score" else -best
        print(f"Epoch {epoch + 1} | " + " | ".join(parts) + f" | best {metric} {shown:.4f} | "
              f"{len(frames)} frames | {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
