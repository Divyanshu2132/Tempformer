"""

Usage
-----
python compute_Tm_from_native_contacts.py \
    --generated-dir /lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/spectral_temperature_generated/AF_generated_data \
    --mdcath-dir    /lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/data \
    --out           /lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/contact_vs_T/Tm_correlation_native_contacts_AF.h5 \
    --csv-out       /lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/contact_vs_T/Tm_correlation_native_contacts_AF.csv \
    [--domain 1a0hA01] [--workers N]

Outputs
-------
--csv-out : one row per domain with Tm_gen, Tm_md, folded-state Q, and edge flags.
--out     : HDF5 with the same per-domain arrays plus summary attrs (n_pairs,
            pearson_r, spearman_rho, rmse_K, mae_K).
figures/Tm_correlation_native_contacts_AF.png : Tm_gen vs Tm_md scatter, y=x line.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_contacts_vs_T import list_domains
from compute_native_contacts_mdref import (
    TEMPERATURES, CONTACT_CUTOFF_NM, SEQ_SEP_MIN, MD_STRIDE,
    ca_distance_matrix_from_coords, compute_Q,
    load_generated_ensemble, load_mdcath_ca_coords_by_replica,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
GENERATED_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__), '..', 'spectral_temperature_generated', 'AF_generated_data')
MDCATH_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_H5_DEFAULT = os.path.join(
    os.path.dirname(__file__), '..', 'contact_vs_T', 'Tm_correlation_native_contacts_AF.h5')
OUT_CSV_DEFAULT = os.path.join(
    os.path.dirname(__file__), '..', 'contact_vs_T', 'Tm_correlation_native_contacts_AF.csv')

Q_THRESHOLD = 0.4
print(f"Threshold={Q_THRESHOLD}, Cutoff={CONTACT_CUTOFF_NM} nm, SeqSepMin={SEQ_SEP_MIN}")

def load_native_ca_coords(mdcath_path):
    """Parse the single-frame native reference structure mdCATH stores per domain.

    pdbProteinAtoms is one MODEL of protein-only atoms (the structure the MD run was
    seeded from) -- the actual "native pdb", as opposed to an ensemble average.
    Returns CA coordinates in nm, in the same residue order as pdbProteinAtoms (which
    is also the order load_mdcath_ca_coords_by_replica uses to slice trajectory coords).
    """
    with h5py.File(mdcath_path, 'r') as f:
        domain_id = list(f.keys())[0]
        pdb_txt = f[domain_id]['pdbProteinAtoms'][()]
    if isinstance(pdb_txt, bytes):
        pdb_txt = pdb_txt.decode('utf-8', errors='replace')

    coords = []
    for line in pdb_txt.splitlines():
        if not line.startswith('ATOM'):
            continue
        if line[12:16].strip() != 'CA':
            continue
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    if not coords:
        raise ValueError(f'No CA atoms found in native structure of {mdcath_path}')
    return np.asarray(coords, dtype=np.float32) * 0.1  # Angstrom -> nm


def native_contacts_from_static_structure(mdcath_path, cutoff_nm=CONTACT_CUTOFF_NM, sep_min=SEQ_SEP_MIN):
    """Native contact (i,j) pairs and their native distance r0_ij from the static
    reference structure -- the standard Go-model native-contact definition."""
    native_ca = load_native_ca_coords(mdcath_path)
    dists = ca_distance_matrix_from_coords(native_ca[None, :, :])[0]
    ii, jj = np.triu_indices(dists.shape[0], k=sep_min)
    keep = dists[ii, jj] < cutoff_nm
    pairs = np.stack([ii[keep], jj[keep]], axis=1).astype(np.int32)
    r0 = dists[ii[keep], jj[keep]].astype(np.float32)
    return pairs, r0, dists.shape[0]


def find_Tm_from_Q(T, Q, threshold=Q_THRESHOLD):
    """T at which Q(T) crosses threshold (linear interpolation between bracketing
    samples). Returns (Tm, is_edge); is_edge=True means no crossing was found in the
    sampled range (curve stayed on one side of the threshold throughout)."""
    T = np.asarray(T, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if len(T) < 2:
        return np.nan, True
    order = np.argsort(T)
    T_s, Q_s = T[order], Q[order]
    for i in range(len(T_s) - 1):
        if not (np.isfinite(Q_s[i]) and np.isfinite(Q_s[i + 1])):
            continue
        if Q_s[i] == Q_s[i + 1]:
            continue
        if (Q_s[i] - threshold) * (Q_s[i + 1] - threshold) <= 0:
            frac = (threshold - Q_s[i]) / (Q_s[i + 1] - Q_s[i])
            return float(T_s[i] + frac * (T_s[i + 1] - T_s[i])), False
    return np.nan, True


def Q_of_T(coords_by_T, native_pairs, native_r0):
    """coords_by_T: {T_str: [F, n_res, 3] nm}. Returns (T_arr, Q_mean_arr)."""
    T_list, Q_list = [], []
    for T_str in TEMPERATURES:
        coords = coords_by_T.get(T_str)
        if coords is None:
            continue
        dists = ca_distance_matrix_from_coords(coords)
        q = compute_Q(dists, native_pairs, native_r0)
        T_list.append(float(T_str))
        Q_list.append(float(q.mean()))
    return np.array(T_list), np.array(Q_list)


def process_domain(domain, generated_dir, mdcath_dir):
    mdcath_path = os.path.join(mdcath_dir, f'mdcath_dataset_{domain}.h5')
    if not os.path.exists(mdcath_path):
        raise RuntimeError(f'Missing mdCATH file for {domain}: {mdcath_path}')

    native_pairs, native_r0, n_res_native = native_contacts_from_static_structure(mdcath_path)
    if native_pairs.size == 0:
        raise RuntimeError('no native contacts found (check cutoff / sequence separation)')

    gen_coords_by_T = {}
    for T_str in TEMPERATURES:
        coords = load_generated_ensemble(generated_dir, domain, T_str)
        if coords is not None and coords.shape[1] == n_res_native:
            gen_coords_by_T[T_str] = coords
    T_gen, Q_gen = Q_of_T(gen_coords_by_T, native_pairs, native_r0)

    md_coords_by_T = {}
    for T_str in TEMPERATURES:
        replica_coords = load_mdcath_ca_coords_by_replica(mdcath_path, T_str, stride=MD_STRIDE)
        if not replica_coords:
            continue
        pooled = np.concatenate(list(replica_coords.values()), axis=0)
        if pooled.shape[1] == n_res_native:
            md_coords_by_T[T_str] = pooled
    T_md, Q_md = Q_of_T(md_coords_by_T, native_pairs, native_r0)

    Tm_gen, edge_gen = find_Tm_from_Q(T_gen, Q_gen)
    Tm_md, edge_md = find_Tm_from_Q(T_md, Q_md)

    return dict(
        domain=domain, Tm_gen=Tm_gen, Tm_md=Tm_md, edge_gen=edge_gen, edge_md=edge_md,
        Q_folded_gen=float(Q_gen[0]) if len(Q_gen) else np.nan,
        Q_folded_md=float(Q_md[0]) if len(Q_md) else np.nan,
        n_native_contacts=int(len(native_pairs)),
    )


def _process_domain_worker(domain, generated_dir, mdcath_dir):
    try:
        return process_domain(domain, generated_dir, mdcath_dir), None
    except Exception as e:
        return None, (domain, str(e))


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ['domain', 'Tm_gen', 'Tm_md', 'Q_folded_gen', 'Q_folded_md',
              'edge_gen', 'edge_md', 'n_native_contacts']
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_h5(rows, valid_mask, stats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    domains = np.array([r['domain'] for r in rows], dtype=h5py.string_dtype())
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('domain', data=domains)
        for key in ('Tm_gen', 'Tm_md', 'Q_folded_gen', 'Q_folded_md'):
            h5.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.float32))
        for key in ('edge_gen', 'edge_md'):
            h5.create_dataset(key, data=np.array([r[key] for r in rows], dtype=bool))
        h5.create_dataset('n_native_contacts', data=np.array([r['n_native_contacts'] for r in rows], dtype=np.int32))
        h5.create_dataset('valid_for_correlation', data=valid_mask)
        for k, v in stats.items():
            h5.attrs[k] = v


def plot_scatter(Tm_gen, Tm_md, stats, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(Tm_md, Tm_gen, alpha=0.5, s=20)
    lo = min(Tm_md.min(), Tm_gen.min()) - 5 if len(Tm_md) else 315
    hi = max(Tm_md.max(), Tm_gen.max()) + 5 if len(Tm_md) else 455
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.text(0.05, 0.95,
            f"N = {stats['n_pairs']}\nPearson r = {stats['pearson_r']:.2f}\n"
            f"Spearman rho = {stats['spearman_rho']:.2f}\nRMSE = {stats['rmse_K']:.1f} K",
            transform=ax.transAxes, va='top')
    ax.set_xlabel(r'$T_m^\mathrm{MD}$ (K, mdCATH)')
    ax.set_ylabel(r'$T_m^\mathrm{AF-generated}$ (K, from native-contact loss $Q(T){=}0.5$)')
    ax.set_title('Melting temperature agreement (native contacts)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated-dir', default=GENERATED_DIR_DEFAULT,
                     help='AF_generated_data layout: {domain}_{T}/{domain}_{T}_{i}_minimized.pdb')
    ap.add_argument('--mdcath-dir', default=MDCATH_DIR_DEFAULT,
                     help='Directory of mdcath_dataset_{domain}.h5 reference trajectories '
                          '(native structure is read from pdbProteinAtoms in this same file)')
    ap.add_argument('--out', default=OUT_H5_DEFAULT)
    ap.add_argument('--csv-out', default=OUT_CSV_DEFAULT)
    ap.add_argument('--domain', default=None, help='If set, process only this domain')
    ap.add_argument('--include-edge', action='store_true',
                     help='Include domains whose Q(T) never crosses 0.5 in the correlation')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                     help='Parallel worker processes (each domain is independent); default is all available cores')
    args = ap.parse_args()

    domains = [args.domain] if args.domain else list_domains(args.generated_dir)
    print(f'Processing {len(domains)} domains with {args.workers} worker(s)')

    rows = []
    if args.workers > 1 and len(domains) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_domain_worker, d, args.generated_dir, args.mdcath_dir): d
                       for d in domains}
            for i, fut in enumerate(as_completed(futures)):
                result, failure = fut.result()
                if failure is not None:
                    print(f'  {failure[0]} FAILED: {failure[1]}')
                else:
                    rows.append(result)
                if (i + 1) % 25 == 0:
                    print(f'  {i + 1}/{len(domains)}')
    else:
        for i, domain in enumerate(domains):
            try:
                rows.append(process_domain(domain, args.generated_dir, args.mdcath_dir))
                if (i + 1) % 25 == 0:
                    print(f'  {i + 1}/{len(domains)}')
            except Exception as e:
                print(f'  {domain} FAILED: {e}')

    Tm_gen = np.array([r['Tm_gen'] for r in rows])
    Tm_md = np.array([r['Tm_md'] for r in rows])
    edge_gen = np.array([r['edge_gen'] for r in rows])
    edge_md = np.array([r['edge_md'] for r in rows])

    finite = np.isfinite(Tm_gen) & np.isfinite(Tm_md)
    valid_mask = finite if args.include_edge else (finite & ~edge_gen & ~edge_md)

    n_pairs = int(valid_mask.sum())
    if n_pairs >= 3:
        r_p, _ = pearsonr(Tm_md[valid_mask], Tm_gen[valid_mask])
        r_s, _ = spearmanr(Tm_md[valid_mask], Tm_gen[valid_mask])
        diff = Tm_gen[valid_mask] - Tm_md[valid_mask]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
    else:
        r_p = r_s = rmse = mae = float('nan')

    stats = dict(n_pairs=n_pairs, pearson_r=float(r_p), spearman_rho=float(r_s),
                 rmse_K=rmse, mae_K=mae, n_domains_total=len(rows),
                 n_excluded_edge=int((finite & (edge_gen | edge_md)).sum()))

    print(f"Tm correlation over {n_pairs} domains: Pearson r = {r_p:.3f}, "
          f"Spearman rho = {r_s:.3f}, RMSE = {rmse:.1f} K, MAE = {mae:.1f} K")
    print(f"Excluded {stats['n_excluded_edge']} domain(s) whose Q(T) never crossed "
          f"{Q_THRESHOLD} in range (pass --include-edge to keep them)")

    write_csv(rows, args.csv_out)
    write_h5(rows, valid_mask, stats, args.out)
    plot_scatter(Tm_gen[valid_mask], Tm_md[valid_mask], stats,
                 os.path.join(FIG_DIR, 'Tm_correlation_native_contacts_AF.png'))
    print(f'Wrote {args.csv_out}, {args.out}, and figures/Tm_correlation_native_contacts_AF.png')


if __name__ == '__main__':
    main()
