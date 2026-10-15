import argparse
import csv
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from data import dataset as D
from model import residue_geometry as G
from model.main_model import MainModel
from model.sampling import ensemble_metrics, sample

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "temp_sim_dataset"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--checkpoint", type=Path, default=ROOT / "weights/tempformer500_af.pth")
    a("--fasta", type=Path, default=ROOT / "data/splits/tempformer500_val.fasta")
    a("--embedding-path", type=Path, default=DATASET / "alphafold_embeddings")
    a("--data-dirs", type=Path, nargs="+", default=[DATASET / "filtered_train_data"])
    a("--domains", nargs="+", help="default: the first --num-domains FASTA domains of at most --max-len residues")
    a("--num-domains", type=int, default=10)
    a("--max-len", type=int, default=320)
    a("--temperatures", type=int, nargs="+", default=[320, 348, 379, 413, 450])
    a("--num-samples", type=int, default=50)
    a("--steps", type=int, default=100)
    a("--sampler", choices=["sde", "ode"], default="sde")
    a("--chi-source", choices=["diffused", "predicted"], default="diffused")
    a("--batch", type=int, default=10, help="samples generated together (GPU memory)")
    a("--seed", type=int, default=0)
    a("--sc-weight", type=float, default=1.0, help="chi1 weight in the summary score, as in train.py")
    a("--stable-for", type=int, default=120, help="seconds the checkpoint must be unchanged before loading")
    a("--out-dir", type=Path, default=ROOT / "outputs/evaluation")
    a("--minimize", action="store_true",
      help="also minimize each ensemble as aSAMt's are; written to <checkpoint>_epoch<E>_min and scored there")
    return p.parse_args()


def load_checkpoint(path, stable_for, retries=10):
    """Load once the file has not been modified for stable_for seconds; retry if it changes during the load."""
    for _ in range(retries):
        mtime = path.stat().st_mtime
        wait = stable_for - (time.time() - mtime)
        if wait > 0:
            print(f"checkpoint modified {stable_for - wait:.0f}s ago, waiting {wait:.0f}s", flush=True)
            time.sleep(wait + 1)
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:                      # e.g. replaced while reading
            print(f"load failed ({type(e).__name__}), retrying", flush=True)
            time.sleep(30)
            continue
        if path.stat().st_mtime == mtime:
            return ckpt
    raise RuntimeError(f"could not load a stable {path}")


def pick_domains(args):
    if args.domains:
        return args.domains
    seqs = D.read_fasta(args.fasta)
    return [d for d, s in seqs.items() if len(s) <= args.max_len][: args.num_domains]


def write_ensemble(atoms, aatype, top_path, dcd_path):
    """atom14 (n, L, 14, 3) in Angstrom -> topology PDB (first structure) + DCD, superposed on the first (CA)."""
    import mdtraj as md
    mask = G._C["atom14_mask"][aatype] > 0.5
    top_path.write_text("\n".join(G.atom14_to_pdb_lines(atoms[0], aatype) + ["END"]))
    top = md.load_pdb(str(top_path)).topology
    traj = md.Trajectory(atoms[:, mask] / 10.0, top)                     # same atom order as the PDB writer
    traj.superpose(traj, 0, atom_indices=top.select("name CA"))
    traj.save_dcd(str(dcd_path))
    traj[0].save_pdb(str(top_path))


def read_ensemble(aatype, top_path, dcd_path):
    import mdtraj as md
    mask = G._C["atom14_mask"][aatype] > 0.5
    traj = md.load_dcd(str(dcd_path), top=str(top_path))
    atoms = np.zeros((traj.n_frames, len(aatype), 14, 3), np.float32)
    atoms[:, mask] = traj.xyz * 10.0
    return atoms


def finished(top_path, dcd_path):
    """write_ensemble writes the top, the dcd, then the top again: a finished ensemble's top is no older than its dcd."""
    return top_path.exists() and dcd_path.exists() and top_path.stat().st_mtime >= dcd_path.stat().st_mtime


def minimize_ensemble(aatype, top_path, dcd_path, device):
    """Written ensemble -> atom14 (n, L, 14, 3) in Angstrom after aSAMt's minimizer (minimizer/), with the mdcath
    protocol that aSAMt's ensembles are minimized with."""
    from minimizer.runner import Minimizer
    traj = Minimizer(name=top_path.stem, top_fp=str(top_path), ens_fp=str(dcd_path), protocol="mdcath").run(
        device=device, verbose=False)
    mask = G._C["atom14_mask"][aatype] > 0.5
    if [a.name for a in traj.topology.atoms] != [str(n) for n in G.ATOM14_NAMES[aatype][mask]]:
        raise RuntimeError(f"{top_path.name}: minimizer output atoms are not in atom14 order")
    atoms = np.zeros((traj.n_frames, len(aatype), 14, 3), np.float32)
    atoms[:, mask] = traj.xyz * 10.0
    return atoms


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, args.stable_for)
    epoch = ckpt["epoch"]
    model = MainModel(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])                       # EMA weights
    model.to(device).eval()
    s16 = ckpt.get("sample_score")
    print(f"{args.checkpoint.name}: epoch {epoch}, training-validation score {s16 if s16 is None else round(s16, 4)} "
          f"({device.type})", flush=True)

    names = pick_domains(args)
    pool = D.build_pool("eval", args.fasta, args.embedding_path, args.data_dirs, crop_size=10 ** 6,
                        keep_md_reference=True, only=names)
    run_dir = args.out_dir / f"{args.checkpoint.stem}_epoch{epoch:03d}"
    min_dir = run_dir.with_name(run_dir.name + "_min")                   # --minimize: same layout as run_dir
    scored_dir = min_dir if args.minimize else run_dir
    history = args.out_dir / ("metrics_history_min.csv" if args.minimize else "metrics_history.csv")
    fields = ["time", "checkpoint", "epoch", "train_val_score", "n_samples", "steps", "sampler", "chi_source",
              "domain", "T", "rmsf_r", "q_err", "ss_err", "chi1_err", "chi1_err_floor", "chi2_err", "chi2_err_floor",
              "q_gen", "q_md"]
    done = set()
    if history.exists():
        with open(history) as fp:
            done = {(r["checkpoint"], r["epoch"], r["n_samples"], r["steps"], r["sampler"], r["chi_source"],
                     r["domain"], r["T"]) for r in csv.DictReader(fp)}

    rows = []
    for dom in pool.domains:
        dom_id, aatype = dom["domain_id"], dom["aatype"]
        folder = run_dir / dom_id
        folder.mkdir(parents=True, exist_ok=True)
        if args.minimize:
            (min_dir / dom_id).mkdir(parents=True, exist_ok=True)
        feats = D.protein_batch(dom)
        for T in args.temperatures:
            top_path, dcd_path = folder / f"{dom_id}.{T}.top.pdb", folder / f"{dom_id}.{T}.traj.dcd"
            min_top, min_dcd = min_dir / dom_id / top_path.name, min_dir / dom_id / dcd_path.name
            start = time.time()
            if args.minimize and finished(min_top, min_dcd):
                atoms = read_ensemble(aatype, min_top, min_dcd)
            else:
                if finished(top_path, dcd_path):
                    atoms = read_ensemble(aatype, top_path, dcd_path)
                else:
                    atoms = []
                    for c in range(0, args.num_samples, args.batch):
                        n = min(args.batch, args.num_samples - c)
                        seed = zlib.crc32(f"{args.seed}:{dom_id}:{T}:{c}".encode())
                        atoms.append(sample(model, feats, float(T), n, n_steps=args.steps, sampler=args.sampler,
                                            seed=seed, device=device, chi_source=args.chi_source).cpu().numpy())
                    atoms = np.concatenate(atoms)
                    write_ensemble(atoms, aatype, top_path, dcd_path)
                if args.minimize:
                    atoms = minimize_ensemble(aatype, top_path, dcd_path, device.type)
                    write_ensemble(atoms, aatype, min_top, min_dcd)
            if str(T) not in dom["md"]["rmsf"]:
                print(f"{dom_id} {T} K: {len(atoms)} structures written (no MD at this temperature, not scored)", flush=True)
                continue
            m = ensemble_metrics(atoms, aatype, dom["md"], str(T), folder)
            row = {"time": time.strftime("%Y-%m-%d %H:%M"), "checkpoint": str(args.checkpoint), "epoch": epoch,
                   "train_val_score": s16, "n_samples": len(atoms), "steps": args.steps, "sampler": args.sampler,
                   "chi_source": args.chi_source, "domain": dom_id, "T": T, **m}
            rows.append(row)
            print(f"{dom_id} {T} K: {len(atoms)} {'minimized ' if args.minimize else ''}structures in "
                  f"{time.time() - start:.0f}s | RMSF r {m['rmsf_r']:.3f} | "
                  f"|dQ| {m['q_err']:.3f} | SS err {m['ss_err']:.3f} | chi1 {m['chi1_err']:.3f} "
                  f"(floor {m['chi1_err_floor']:.3f})", flush=True)

    new = [r for r in rows if (str(r["checkpoint"]), str(r["epoch"]), str(r["n_samples"]), str(r["steps"]),
                               r["sampler"], r["chi_source"], r["domain"], str(r["T"])) not in done]
    if new:
        write_header = not history.exists()
        with open(history, "a", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(new)
    if rows:
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in
                ("rmsf_r", "q_err", "ss_err", "chi1_err", "chi1_err_floor", "chi2_err", "chi2_err_floor")}
        score = mean["rmsf_r"] - mean["q_err"] - mean["ss_err"] - args.sc_weight * mean["chi1_err"]
        print(f"\nepoch {epoch} | {len(rows)} ensembles of {rows[0]['n_samples']} | score {score:.4f} "
              f"(training validation, 16 samples: {s16 if s16 is None else round(s16, 4)})")
        print("  " + " | ".join(f"{k} {v:.3f}" for k, v in mean.items()))
        for T in args.temperatures:
            rs = [r for r in rows if r["T"] == T]
            if rs:
                print(f"  {T} K: RMSF r {np.nanmean([r['rmsf_r'] for r in rs]):.3f} | |dQ| "
                      f"{np.nanmean([r['q_err'] for r in rs]):.3f} | SS err {np.nanmean([r['ss_err'] for r in rs]):.3f} | "
                      f"chi1 {np.nanmean([r['chi1_err'] for r in rs]):.3f}")
        print(f"ensembles: {scored_dir} | history: {history}")


if __name__ == "__main__":
    main()
