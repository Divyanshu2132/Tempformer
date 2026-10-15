import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from data.dataset import read_fasta
from evaluation.evaluate import finished

DATASET = Path(__file__).resolve().parents[2] / "temp_sim_dataset"
MELT = DATASET / "case_study/melting_temperature"
ENS = DATASET / "all_atom_tempformer"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--ens-dirs", type=Path, nargs="+", default=[ENS / "melting_500_af/tempformer500_af_epoch175_min",
                                                   ENS / "melting_1000_af/tempformer1000_af_epoch175_min"])
    a("--fasta", type=Path, default=MELT / "melting_pdbs.fasta")
    a("--ref-dir", type=Path, default=MELT / "alphafold_structures", help="<pdb>_unrelaxed_rank_001_*.pdb models")
    a("--threshold", type=float, default=0.5, help="Tm: where the native SS kept crosses this value")
    a("--workers", type=int, default=1)
    a("--plot", action="store_true")
    return p.parse_args()


def ss_curve(job):
    """One protein in one ensemble directory -> rows (T, native SS kept, helix, strand)."""
    ens_dir, ref_dir, pdb = job
    ref = md.compute_dssp(md.load(glob.glob(str(ref_dir / f"{pdb}_unrelaxed_rank_001_*.pdb"))[0]))[0]
    native = np.isin(ref, ["H", "E"])
    rows = []
    for dcd in (ens_dir / pdb).glob(f"{pdb}.*.traj.dcd"):
        T = float(dcd.name[len(pdb) + 1:-len(".traj.dcd")])
        top = dcd.with_name(f"{pdb}.{T:g}.top.pdb")
        if not finished(top, dcd):
            continue
        ss = md.compute_dssp(md.load_dcd(str(dcd), top=str(top)))
        if ss.shape[1] != len(ref):
            raise ValueError(f"{dcd}: {ss.shape[1]} residues, AlphaFold model has {len(ref)}")
        rows.append({"protein": pdb, "T": T, "n_frames": len(ss), "native_ss": float((ss[:, native] == ref[native]).mean()),
                     "helix": float((ss == "H").mean()), "strand": float((ss == "E").mean()),
                     "L": len(ref), "n_native_ss": int(native.sum())})
    return sorted(rows, key=lambda r: r["T"])


def threshold_tm(T, y, threshold):
    """First temperature where y drops below threshold, linearly interpolated; (Tm, status)."""
    if y[0] < threshold:
        return np.nan, "below_range"
    below = np.nonzero(y < threshold)[0]
    if not len(below):
        return np.nan, "above_range"
    i = below[0]
    return T[i - 1] + (y[i - 1] - threshold) * (T[i] - T[i - 1]) / (y[i - 1] - y[i]), "ok"


def plot(curves, tms, threshold, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 8
    nrow = int(np.ceil(len(tms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 1.9 * nrow), sharex=True, sharey=True)
    for ax, (_, t) in zip(axes.flat, tms.iterrows()):
        c = curves[curves.protein == t.protein]
        ax.plot(c["T"], c.native_ss, "o-", ms=3, lw=1.2, color="#2a78d6")
        ax.axhline(threshold, color="gray", ls=":", lw=0.8)
        if t.status == "ok":
            ax.axvline(t.Tm, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{t.protein}  " + (f"Tm {t.Tm:.0f}" if t.status == "ok" else t.status.replace("_", " ")), fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(tms):]:
        ax.axis("off")
    fig.supxlabel("Temperature (K)")
    fig.supylabel("Native secondary structure kept")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    prots = list(read_fasta(args.fasta))
    for ens_dir in args.ens_dirs:
        jobs = [(ens_dir, args.ref_dir, p) for p in prots if (ens_dir / p).is_dir()]
        with ProcessPoolExecutor(args.workers) as ex:
            curves = pd.DataFrame([r for rows in ex.map(ss_curve, jobs) for r in rows])
        tag = ens_dir.name
        curves.to_csv(ens_dir.parent / f"ss_melting_{tag}.csv", index=False)
        tms = []
        for pdb, c in curves.groupby("protein", sort=False):
            Tm, status = threshold_tm(c["T"].to_numpy(), c.native_ss.to_numpy(), args.threshold)
            tms.append({"protein": pdb, "L": c.L.iloc[0], "n_native_ss": c.n_native_ss.iloc[0], "n_T": len(c),
                        "Tm": Tm, "status": status, "native_ss_Tmin": c.native_ss.iloc[0],
                        "native_ss_Tmax": c.native_ss.iloc[-1]})
        tms = pd.DataFrame(tms)
        tms.to_csv(ens_dir.parent / f"ss_tm_{tag}.csv", index=False)
        n = tms.status.value_counts().to_dict()
        print(f"{ens_dir}: {len(tms)} proteins | Tm at native SS kept = {args.threshold}: {n.get('ok', 0)} inside the range "
              f"(median {tms.Tm.median():.0f} K), {n.get('above_range', 0)} above it, {n.get('below_range', 0)} below it")
        print(f"  -> {ens_dir.parent / f'ss_melting_{tag}.csv'}\n  -> {ens_dir.parent / f'ss_tm_{tag}.csv'}")
        if args.plot:
            plot(curves, tms, args.threshold, ens_dir.parent / f"ss_tm_{tag}.png", f"{tag}: secondary-structure melting")
            print(f"  -> {ens_dir.parent / f'ss_tm_{tag}.png'}")


if __name__ == "__main__":
    main()
