import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from evaluation.evaluate import finished

DATASET = Path(__file__).resolve().parents[2] / "temp_sim_dataset"
BENCH = DATASET / "03_NMR_ensemble_benchmark"
TF = DATASET / "all_atom_tempformer/NMR_benchmark/tempformer1000_af_epoch175_min"
METRICS = ["rmsf_mae", "div_err", "div_bias", "rmsf_r"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--bench", type=Path, default=BENCH, help="holds fasta/<PDB>.fasta and native_ensembles/<PDB>/*.pdb")
    a("--ensembles", nargs="+", default=[f"tempformer1000_min={TF}", f"predictions={BENCH / 'predictions'}"],
      metavar="NAME=DIR")
    a("--proteins", nargs="+", help="default: every fasta/<PDB>.fasta")
    a("--workers", type=int, default=1)
    a("--out-csv", type=Path, help="also write the per-protein rows here")
    return p.parse_args()


def load_ca(paths, top=None):
    """PDB files (one conformation each), or a DCD with its topology -> (sequence, mdtraj CA trajectory)."""
    topo = md.load_topology(str(top or paths[0]))
    ca = topo.select("protein and name CA")
    traj = md.load_dcd(str(paths[0]), top=str(top), atom_indices=ca) if top else \
        md.load([str(p) for p in paths], atom_indices=ca)
    return "".join(r.code or "X" for r in traj.topology.residues), traj


def fasta_index(seq, fasta):
    """Position in the FASTA sequence of every residue of seq (-1 if unmatched)."""
    idx = np.full(len(seq), -1)
    for a, b, n in SequenceMatcher(None, seq, fasta, autojunk=False).get_matching_blocks():
        idx[a:a + n] = np.arange(b, b + n)
    return idx


def rmsf(traj, n_iter=3):
    """Per-residue RMSF (Angstrom) after iteratively superposing every frame on the ensemble mean."""
    t = traj[:]
    ref = t[0]
    for _ in range(n_iter):
        t.superpose(ref)
        ref = md.Trajectory(t.xyz.mean(0, keepdims=True), t.topology)
    t.superpose(ref)
    return 10 * np.sqrt(((t.xyz - t.xyz.mean(0)) ** 2).sum(-1).mean(0))


def mean_pairwise_rmsd(traj):
    """Mean optimally superposed RMSD (Angstrom) over all pairs of frames."""
    M = traj.n_frames
    d = np.stack([md.rmsd(traj, traj, i) for i in range(M)])
    return float(10 * d[np.triu_indices(M, 1)].mean())


def ensembles_of(name, root, pdb):
    """[(label, T, paths, top)] of one method for one protein."""
    d = root / pdb
    dcds = sorted(d.glob(f"{pdb}.*.traj.dcd"))
    if dcds:
        out = []
        for dcd in dcds:
            T = float(dcd.name[len(pdb) + 1:-len(".traj.dcd")])
            top = dcd.with_name(f"{pdb}.{T:g}.top.pdb")
            if finished(top, dcd):
                out.append((name, T, [dcd], top))
        return out
    pdbs = sorted(p for p in d.glob("*.pdb") if not p.name.endswith(".top.pdb"))
    return [(name, np.nan, pdbs, None)] if pdbs else []


def evaluate_protein(job):
    bench, methods, pdb = job
    fasta = "".join(l.strip() for l in open(bench / f"fasta/{pdb}.fasta") if not l.startswith(">"))
    nat_seq, nat = load_ca(sorted(glob.glob(str(bench / f"native_ensembles/{pdb}/*.pdb"))))
    nat_idx = fasta_index(nat_seq, fasta)
    rows = []
    for name, root in methods:
        for label, T, paths, top in ensembles_of(name, root, pdb):
            seq, ens = load_ca(paths, top)
            ens_idx = fasta_index(seq, fasta)
            common = np.intersect1d(nat_idx[nat_idx >= 0], ens_idx[ens_idx >= 0])
            if ens.n_frames < 2 or len(common) < 0.9 * len(fasta):
                print(f"{pdb} {label} {T}: {ens.n_frames} conformations, {len(common)}/{len(fasta)} residues matched, "
                      "skipped", flush=True)
                continue
            n_sel = nat.atom_slice([int(np.where(nat_idx == i)[0][0]) for i in common])
            e_sel = ens.atom_slice([int(np.where(ens_idx == i)[0][0]) for i in common])
            r_nat, r_ens = rmsf(n_sel), rmsf(e_sel)
            d_nat, d_ens = mean_pairwise_rmsd(n_sel), mean_pairwise_rmsd(e_sel)
            rows.append({"protein": pdb, "method": label, "T": T, "M_pred": ens.n_frames, "M_nmr": nat.n_frames,
                         "n_res": len(common), "L_fasta": len(fasta),
                         "rmsf_mae": float(np.abs(r_ens - r_nat).mean()), "rmsf_r": float(np.corrcoef(r_ens, r_nat)[0, 1]),
                         "D_pred": d_ens, "D_nmr": d_nat, "div_err": abs(d_ens - d_nat), "div_bias": d_ens - d_nat})
    return rows


def summarize(rows):
    g = rows.groupby(["method", "T"], dropna=False)
    s = g[METRICS].agg(["mean", "median"])
    s.columns = [f"{m}_{f}" for m, f in s.columns]
    return pd.concat([g.size().rename("n_proteins"), s], axis=1).reset_index()


def main():
    args = parse_args()
    methods = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.ensembles]
    proteins = args.proteins or sorted(p.stem for p in (args.bench / "fasta").glob("*.fasta"))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = pd.DataFrame([r for rs in ex.map(evaluate_protein, [(args.bench, methods, p) for p in proteins]) for r in rs])
    if rows.empty:
        raise SystemExit("no generated ensembles found")
    pd.set_option("display.width", 200)
    print(f"\n{len(proteins)} proteins; per method (and temperature), over the proteins it has:")
    print(summarize(rows).round(3).to_string(index=False))
    groups = rows.groupby(["method", "T"], dropna=False).protein.apply(set)
    common = set.intersection(*groups)
    if len(common) < max(map(len, groups)):
        print(f"\nover the {len(common)} proteins every method (and temperature) has:")
        print(summarize(rows[rows.protein.isin(common)]).round(3).to_string(index=False))
    if args.out_csv:
        rows.to_csv(args.out_csv, index=False)
        print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
