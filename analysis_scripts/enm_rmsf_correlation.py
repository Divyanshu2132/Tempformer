#!/usr/bin/env python
"""
Inputs
------
- Domain set: filenames in alphafold_rmsfs/{domain}_{temp}.npy
- Reference structure + ground-truth RMSF: data/mdcath_dataset_{domain}.h5
    {domain}/pdbProteinAtoms      all-atom PDB text of the reference structure
    {domain}/{T}/{replica}/rmsf   per-residue RMSF (Angstrom), one per replica
  Ground-truth RMSF at each temperature is the mean over the 5 replicas,
  matching davit_scripts/save_md_rmsfs.py.

Output
------
DATA_DIR/enm_correlations.csv with columns:
    Temperature, Correlation, Spearman, ID, Comparison, N_residues
matching the layout of correlations.csv / spectral_correlations.csv so it can
be concatenated with those for plotting.

Usage
-----
    python enm_rmsf_correlation.py [--model {anm,gnm}] [--cutoff 15.0] [--n-jobs -1]
"""

import argparse
import io
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import prody
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
prody.confProDy(verbosity="none")

DATA_DIR = Path("/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset")
AF_RMSF_DIR = DATA_DIR / "alphafold_rmsfs"   # domain population to evaluate
MDCATH_DIR = DATA_DIR / "data"               # mdcath_dataset_{domain}.h5 files
OUTPUT_CSV = DATA_DIR / "enm_correlations.csv"

TEMPERATURES = ["320", "348", "379", "413", "450"]
N_REPLICAS = 5


def discover_domains(af_dir: Path = AF_RMSF_DIR, temps=TEMPERATURES) -> list:
    """Domains cached in alphafold_rmsfs/ that have every temperature."""
    wanted = set(temps)
    by_domain = {}
    for f in os.listdir(af_dir):
        domain, temp = f[:-4].rsplit("_", 1)
        by_domain.setdefault(domain, set()).add(temp)
    return sorted(d for d, seen in by_domain.items() if wanted.issubset(seen))


def load_reference_ca(h5_group) -> "prody.AtomGroup":
    """Parse the reference all-atom PDB stored in the h5 group and return CA atoms."""
    pdb_text = h5_group["pdbProteinAtoms"][()]
    if isinstance(pdb_text, bytes):
        pdb_text = pdb_text.decode()
    structure = prody.parsePDBStream(io.StringIO(pdb_text))
    return structure.select("name CA and protein")


def load_mean_md_rmsf(h5_group, temperature: str, n_replicas: int = N_REPLICAS) -> np.ndarray:
    """Mean per-residue RMSF (Angstrom) over replicas at one temperature."""
    rmsfs = [
        np.asarray(h5_group[temperature][str(r)]["rmsf"][:], dtype=float)
        for r in range(n_replicas)
    ]
    return np.mean(rmsfs, axis=0)


def compute_enm_rmsf(ca_atoms, model: str = "anm", cutoff: float = 15.0, gamma: float = 1.0):
    """Theoretical per-residue RMSF from an elastic network model (arbitrary units).

    Uses all non-trivial modes (exact solution for a system this small), so the
    result matches the classical closed-form ENM fluctuation profile rather
    than a low-mode truncation.
    """
    if model == "gnm":
        enm = prody.GNM()
        enm.buildKirchhoff(ca_atoms, cutoff=cutoff, gamma=gamma)
    else:
        enm = prody.ANM()
        enm.buildHessian(ca_atoms, cutoff=cutoff, gamma=gamma)
    enm.calcModes(n_modes=None)
    msf = prody.calcSqFlucts(enm)
    return np.sqrt(msf)


def process_domain(domain: str, model: str, cutoff: float, gamma: float):
    """Return one row per temperature: correlation between ENM and mdCATH RMSF."""
    h5_path = MDCATH_DIR / f"mdcath_dataset_{domain}.h5"
    if not h5_path.exists():
        return []

    try:
        with h5py.File(h5_path, "r") as f:
            g = f[domain]
            ca_atoms = load_reference_ca(g)
            if ca_atoms is None:
                return []

            rmsf_md_by_temp = {}
            for T in TEMPERATURES:
                if T not in g:
                    return []
                rmsf_md_by_temp[T] = load_mean_md_rmsf(g, T)

            n_res = ca_atoms.numAtoms()
            if any(len(v) != n_res for v in rmsf_md_by_temp.values()):
                return []  # topology / residue-count mismatch, skip

            rmsf_enm = compute_enm_rmsf(ca_atoms, model=model, cutoff=cutoff, gamma=gamma)
    except Exception as e:
        print(f"skip {domain}: {e}", flush=True)
        return []

    rows = []
    for T in TEMPERATURES:
        rmsf_md = rmsf_md_by_temp[T]
        if np.std(rmsf_enm) == 0 or np.std(rmsf_md) == 0:
            continue
        pearson_r, _ = pearsonr(rmsf_enm, rmsf_md)
        spearman_r, _ = spearmanr(rmsf_enm, rmsf_md)
        rows.append({
            "Temperature": int(T),
            "Correlation": pearson_r,
            "Spearman": spearman_r,
            "ID": domain,
            "Comparison": model.upper(),
            "N_residues": n_res,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["anm", "gnm"], default="anm",
                     help="Elastic network model flavor (default: anm)")
    ap.add_argument("--cutoff", type=float, default=15.0,
                     help="Interaction cutoff in Angstrom (default: 15.0 for ANM; "
                          "use ~10.0 for GNM)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Spring constant (default: 1.0)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="joblib parallel workers")
    args = ap.parse_args()

    domains = discover_domains()
    print(f"Found {len(domains)} domains cached in {AF_RMSF_DIR} "
          f"with all {len(TEMPERATURES)} temperatures")

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_domain)(d, args.model, args.cutoff, args.gamma) for d in domains
    )

    rows = [row for domain_rows in results for row in domain_rows]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No domains produced results - check MDCATH_DIR and reference structures.")
        return

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows ({df['ID'].nunique()} domains) to {OUTPUT_CSV}")

    print("\n=== Correlation by temperature (Pearson) ===")
    print(df.groupby("Temperature")["Correlation"].agg(["mean", "median", "std", "count"]).round(4))

    print("\n=== Correlation by temperature (Spearman) ===")
    print(df.groupby("Temperature")["Spearman"].agg(["mean", "median", "std", "count"]).round(4))


if __name__ == "__main__":
    main()
