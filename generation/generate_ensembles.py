import argparse
import contextlib
import math
import os
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from data import dataset as D
from evaluation.evaluate import finished, load_checkpoint, minimize_ensemble, write_ensemble
from model import residue_geometry as G
from model.main_model import MainModel
from model.sampling import sample

ROOT = Path(__file__).resolve().parents[1]
STALE_LOCK = 3600          # seconds: a join lock this old was left by a task killed while joining


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--checkpoint", type=Path, default=ROOT / "weights/tempformer500_af.pth")
    a("--fasta", type=Path, required=True)
    a("--embedding-path", type=Path, required=True, help="ColabFold output directory of the FASTA's proteins")
    a("--temperatures", type=float, nargs="+", default=[320, 348, 379, 413, 450])
    a("--systems", nargs="+", metavar="NAME:T", help="run these (protein, temperature) pairs instead of every "
      "FASTA protein at every --temperatures")
    a("--num-samples", type=int, default=50)
    a("--chunk-size", type=int, help="write each ensemble in chunks of this many structures, then join them")
    a("--steps", type=int, default=100)
    a("--sampler", choices=["sde", "ode"], default="sde")
    a("--chi-source", choices=["diffused", "predicted"], default="diffused")
    a("--batch", type=int, default=10, help="samples generated together (GPU memory)")
    a("--seed", type=int, default=0)
    a("--minimize", action="store_true", help="also write ensembles minimized as aSAMt's are, in <run dir>_min")
    a("--keep-raw", action=argparse.BooleanOptionalAction, default=True,
      help="--no-keep-raw (with --minimize): write only the minimized ensembles")
    a("--stable-for", type=int, default=120, help="seconds the checkpoint must be unchanged before loading")
    a("--task-id", type=int, default=0, help="this task takes work items task-id, task-id + num-tasks, ...")
    a("--num-tasks", type=int, default=1)
    a("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if not args.keep_raw and not args.minimize:
        p.error("--no-keep-raw needs --minimize")
    return args


def join_chunks(folder, name, n_chunks):
    """<name>.partKKKK ensembles in folder -> <name>.{top.pdb,traj.dcd} once every chunk is finished: the first
    chunk's topology, every frame superposed on it (CA); the chunks are deleted afterwards. A lock file keeps two
    tasks from joining the same ensemble. True if the joined ensemble exists (now or before)."""
    import mdtraj as md
    from mdtraj.formats import DCDTrajectoryFile
    top, dcd = folder / f"{name}.top.pdb", folder / f"{name}.traj.dcd"
    if finished(top, dcd):
        return True
    parts = [(folder / f"{name}.part{k:04d}.top.pdb", folder / f"{name}.part{k:04d}.traj.dcd") for k in range(n_chunks)]
    if not all(finished(*p) for p in parts):
        return False
    lock = folder / f"{name}.join.lock"
    with contextlib.suppress(FileNotFoundError):
        if time.time() - lock.stat().st_mtime > STALE_LOCK:
            lock.unlink()
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        print(f"{folder / name}: another task is joining the chunks", flush=True)
        return False
    try:
        ref = md.load_pdb(str(parts[0][0]))
        ca = ref.topology.select("name CA")
        tmp_dcd, tmp_top = dcd.with_name(dcd.name + ".tmp"), top.with_name(top.name + ".tmp")
        with DCDTrajectoryFile(str(tmp_dcd), "w", force_overwrite=True) as f:
            for _, part_dcd in parts:
                traj = md.load_dcd(str(part_dcd), top=ref.topology)
                traj.superpose(ref, 0, atom_indices=ca)
                f.write(traj.xyz * 10.0)                                   # DCD files are in Angstrom
        tmp_dcd.replace(dcd)
        tmp_top.write_bytes(parts[0][0].read_bytes())                     # top after dcd: finished() is true
        tmp_top.replace(top)
        for part in parts:
            for path in part:
                path.unlink()
    finally:
        lock.unlink(missing_ok=True)
    return True


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, args.stable_for)
    model = MainModel(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])                       # EMA weights
    model.to(device).eval()
    run_dir = args.out_dir / f"{args.checkpoint.stem}_epoch{ckpt['epoch']:03d}"
    min_dir = run_dir.with_name(run_dir.name + "_min")

    seqs = D.read_fasta(args.fasta)
    emb = D.index_embeddings(args.embedding_path)
    if args.systems:
        systems = [(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1])) for s in args.systems]
        missing = sorted({dom for dom, _ in systems if dom not in seqs})
        if missing:
            raise SystemExit(f"--systems: not in {args.fasta}: {', '.join(missing)}")
    else:
        systems = [(dom, T) for dom in seqs for T in args.temperatures]
    chunk = min(args.chunk_size or args.num_samples, args.num_samples)
    n_chunks = math.ceil(args.num_samples / chunk)
    chunked = n_chunks > 1
    work = [(dom, T, k) for dom, T in systems for k in range(n_chunks)][args.task_id::args.num_tasks]
    outs = ([run_dir] if args.keep_raw else []) + ([min_dir] if args.minimize else [])
    print(f"{args.checkpoint.name}: epoch {ckpt['epoch']} ({device.type}) | task {args.task_id}/{args.num_tasks}: "
          + (f"{len(work)} chunks of {chunk} structures" if chunked else f"{len(work)} systems")
          + f" -> {' and '.join(map(str, outs))}", flush=True)

    feats_of = {}
    for dom, T, k in work:
        if dom not in feats_of:
            seq, entry = seqs[dom], emb.get(dom)
            aatype = np.array([G.RESTYPES.index(c) if c in G.RESTYPES else G.UNK for c in seq], dtype=np.int64)
            if not entry or not entry["ranks"]:
                print(f"{dom}: no embeddings, skipped", flush=True)
                feats_of[dom] = None
            elif (aatype == G.UNK).any():
                print(f"{dom}: non-standard residues in the FASTA sequence, skipped", flush=True)
                feats_of[dom] = None
            elif np.load(entry["ranks"][min(entry["ranks"])]["single"], mmap_mode="r").shape[0] < len(seq):
                print(f"{dom}: embedding shorter than the sequence, skipped", flush=True)
                feats_of[dom] = None
            else:
                tmpl = D.template_ca(entry["template"], len(seq)) if "template" in entry else None
                if tmpl is None:
                    print(f"{dom}: no rank_001 template of the sequence length, template features masked", flush=True)
                feats_of[dom] = aatype, D.protein_batch({"aatype": aatype, "ranks": entry["ranks"], "template_ca": tmpl})
        if feats_of[dom] is None:
            continue
        aatype, feats = feats_of[dom]
        name = f"{dom}.{T:g}"
        label = f"{name}" + (f" chunk {k + 1}/{n_chunks}" if chunked else "") + f" (L={len(aatype)})"
        if chunked and all(finished(d / dom / f"{name}.top.pdb", d / dom / f"{name}.traj.dcd") for d in outs):
            print(f"{label}: already joined", flush=True)
            continue
        stem = f"{name}.part{k:04d}" if chunked else name
        lo, hi = k * chunk, min((k + 1) * chunk, args.num_samples)          # this chunk's structures
        min_top, min_dcd = min_dir / dom / f"{stem}.top.pdb", min_dir / dom / f"{stem}.traj.dcd"
        start = time.time()
        did = []
        if args.keep_raw or not finished(min_top, min_dcd):
            # --no-keep-raw: the raw ensemble is only the minimizer's input, in a folder deleted afterwards
            with (contextlib.nullcontext(run_dir / dom) if args.keep_raw
                  else tempfile.TemporaryDirectory(prefix=f"{stem}_")) as raw_dir:
                top_path, dcd_path = Path(raw_dir) / f"{stem}.top.pdb", Path(raw_dir) / f"{stem}.traj.dcd"
                if not finished(top_path, dcd_path):
                    top_path.parent.mkdir(parents=True, exist_ok=True)
                    atoms = []
                    for c in range(lo, hi, args.batch):
                        n = min(args.batch, hi - c)
                        seed = zlib.crc32(f"{args.seed}:{dom}:{T:g}:{c}".encode())
                        atoms.append(sample(model, feats, float(T), n, n_steps=args.steps, sampler=args.sampler,
                                            seed=seed, device=device, chi_source=args.chi_source).cpu().numpy())
                    write_ensemble(np.concatenate(atoms), aatype, top_path, dcd_path)
                    did.append("sampled")
                if args.minimize and not finished(min_top, min_dcd):
                    min_top.parent.mkdir(parents=True, exist_ok=True)
                    write_ensemble(minimize_ensemble(aatype, top_path, dcd_path, device.type), aatype, min_top, min_dcd)
                    did.append("minimized")
        if chunked and all([join_chunks(d / dom, name, n_chunks) for d in outs]):
            did.append(f"all {n_chunks} chunks joined")
        print(f"{label}: {' + '.join(did) or 'already done'} in {time.time() - start:.0f}s", flush=True)
    print("all systems of this task done", flush=True)


if __name__ == "__main__":
    main()
