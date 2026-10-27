import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from evaluation.evaluate import finished

DATASET = Path(__file__).resolve().parents[2] / "temp_sim_dataset"
PM = DATASET / "all_atom_tempformer/point_mutation"
K_B = 0.0019872041                                                       # kcal/(mol K)
CONTACT_CUTOFF, CONTACT_MIN_SEP, CONTACT_TOL = 8.0, 3, 1.2
# change -> (observable, expected sign of its correlation with ddg_exp, axis label)
CHANGES = {"ddg_ens": ("dg_ens", +1, "ΔΔG$_{ens}$ (kcal/mol)"),
           "dq": ("q", -1, "ΔQ"),
           "drmsd": ("rmsd", +1, "ΔRMSD (Å)"),
           "dnative_ss": ("native_ss", -1, "Δ native SS kept")}
# dataviz reference palette: categorical slots in fixed order, text tokens, surface (light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--temperature", type=float, default=320.0, help="the ensembles generated at this temperature (K)")
    a("--root", type=Path, default=PM, help="holds system_info.csv, alphafold_embeddings/ and the model folders")
    a("--models", nargs="+", default=["tempformer1000_af_epoch175_min", "asamt"],
      help="ensemble folders, in --root or as paths; the first two are compared")
    a("--q-fold", type=float, default=0.7, help="a conformation with Q at least this counts as folded")
    a("--no-common", action="store_true", help="correlate each model over all its mutants, not only those every "
      "model has (the difference and agreement rows always use the common set)")
    a("--n-boot", type=int, default=2000)
    a("--seed", type=int, default=0)
    a("--workers", type=int, default=1)
    a("--out-dir", type=Path, help="default: <root>/stability_correlation")
    return p.parse_args()


def ready(model_dir, name, T, time_marker):
    stem = model_dir / name / f"{name}.{T:g}"
    if time_marker:                                     # aSAMt minimizes in place; .time.txt is written last
        return Path(f"{stem}.time.txt").exists()
    return finished(Path(f"{stem}.top.pdb"), Path(f"{stem}.traj.dcd"))


def family_rows(job):
    """One wild type and its mutants in one model folder -> a row of observables per ensemble."""
    label, model_dir, wt, names, T, af_dir, q_fold = job
    refs = sorted(af_dir.glob(f"{wt}_unrelaxed_rank_001_*.pdb"))
    if not refs:
        print(f"{wt}: no rank_001 model in {af_dir}, family skipped", flush=True)
        return []
    ref = md.load(str(refs[0]))
    ref_ca = ref.topology.select("name CA")
    ca0 = ref.xyz[0, ref_ca] * 10
    i, j = np.triu_indices(len(ref_ca), k=CONTACT_MIN_SEP)
    d0 = np.linalg.norm(ca0[i] - ca0[j], axis=-1)
    keep = d0 < CONTACT_CUTOFF
    i, j, d0 = i[keep], j[keep], d0[keep]
    ref_ss = md.compute_dssp(ref)[0]
    native = np.isin(ref_ss, ["H", "E"])
    rows = []
    for name in names:
        stem = model_dir / name / f"{name}.{T:g}"
        traj = md.load_dcd(f"{stem}.traj.dcd", top=f"{stem}.top.pdb")
        ca = traj.topology.select("name CA")
        if len(ca) != len(ref_ca):
            print(f"{label}/{name}: {len(ca)} CA atoms, {refs[0].name} has {len(ref_ca)}, skipped", flush=True)
            continue
        xyz = traj.xyz[:, ca] * 10
        q = (np.linalg.norm(xyz[:, i] - xyz[:, j], axis=-1) < CONTACT_TOL * d0).mean(1)
        n, n_fold = len(q), int((q >= q_fold).sum())
        p = (n_fold + 0.5) / (n + 1)
        ss = md.compute_dssp(traj)
        rows.append({"model": label, "name": name, "name_wt": wt, "T": T, "n_frames": n, "q": float(q.mean()),
                     "p_fold": n_fold / n, "dg_ens": float(-K_B * T * np.log(p / (1 - p))),
                     "rmsd": float(10 * md.rmsd(traj.atom_slice(ca), ref.atom_slice(ref_ca)).mean()),
                     "native_ss": float((ss[:, native] == ref_ss[native]).mean()) if native.any() else np.nan})
    return rows


def boot_spearman(x, y, idx):
    """Spearman rho of every bootstrap resample (a row of idx) of the pairs (x, y); NaN for a constant resample."""
    rx, ry = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def ci(values):
    v = values[np.isfinite(values)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def correlate(x, y, idx):
    """n, Pearson and Spearman of y against x, with the bootstrap CI of rho; NaN if y is constant or n < 3."""
    out = {"n": len(x)}
    if len(x) < 3 or np.ptp(y) == 0:
        return {**out, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
                "rho_ci_low": np.nan, "rho_ci_high": np.nan}
    r, pr = pearsonr(x, y)
    rho, prho = spearmanr(x, y)
    lo, hi = ci(boot_spearman(x, y, idx))
    return {**out, "pearson_r": r, "pearson_p": pr, "spearman_rho": rho, "spearman_p": prho,
            "rho_ci_low": lo, "rho_ci_high": hi}


def plot(muts, summary, labels, T, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summary[summary.kind == "vs_ddg_exp"].set_index(["model", "change"])
    fig, axes = plt.subplots(len(CHANGES), len(labels), figsize=(4.2 * len(labels), 3.1 * len(CHANGES)),
                             sharex=True, sharey="row", squeeze=False, facecolor=SURFACE)
    for c, label in enumerate(labels):
        m = muts[muts.model == label]
        for r, (chg, (_, sign, ylabel)) in enumerate(CHANGES.items()):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.axhline(0, color=GRID, lw=1, zorder=1)
            ax.axvline(0, color=GRID, lw=1, zorder=1)
            ax.scatter(m.ddg_exp, m[chg], s=22, color=SERIES[c % len(SERIES)], alpha=0.8,
                       edgecolors=SURFACE, linewidths=0.6, zorder=2)
            s = rows.loc[(label, chg)]
            stats = (f"r {s.pearson_r:.2f} · ρ {s.spearman_rho:.2f} [{s.rho_ci_low:.2f}, {s.rho_ci_high:.2f}]"
                     f" · n {s.n:.0f}")
            ax.set_title(f"{label}\n{stats}" if r == 0 else stats, fontsize=8.5, color=TEXT_2)
            ax.tick_params(colors=TEXT_2, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            if c == 0:
                ax.set_ylabel(f"{ylabel}\nexpected sign of r: {'+' if sign > 0 else '−'}", color=TEXT)
            if r == len(CHANGES) - 1:
                ax.set_xlabel("ΔΔG$_{exp}$ (kcal/mol), > 0 destabilizing", color=TEXT)
    fig.suptitle(f"Point-mutation stability at {T:g} K: change in the ensemble (mutant − wild type) vs experiment",
                 color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    T = args.temperature
    out_dir = args.out_dir or args.root / "stability_correlation"
    af_dir = args.root / "alphafold_embeddings"
    info = pd.read_csv(args.root / "system_info.csv")
    info["mutant"] = info.mutant.astype(str).str.lower() == "true"
    wts = set(info.name[~info.mutant])
    mutants = info[info.mutant]
    orphans = mutants[~mutants.name_wt.isin(wts)]
    if len(orphans):
        print(f"no wild type in system_info.csv, left out: {' '.join(orphans.name)}")
    mutants = mutants[mutants.name_wt.isin(wts)][["name", "name_wt", "ddg_exp", "dg_exp"]]
    families = {wt: [wt] + mutants.name[mutants.name_wt == wt].tolist() for wt in sorted(wts)}

    model_dirs = {Path(m).name: (Path(m) if Path(m).is_dir() else args.root / m) for m in args.models}
    labels = list(model_dirs)
    jobs = []
    for label, d in model_dirs.items():
        time_marker = next(d.glob("*/*.time.txt"), None) is not None
        n_ready = 0
        for wt, names in families.items():
            names = [n for n in names if ready(d, n, T, time_marker)]
            n_ready += len(names)
            if names:
                jobs.append((label, d, wt, names, T, af_dir, args.q_fold))
        print(f"{label}: {n_ready} of {len(info)} systems have a finished ensemble at {T:g} K ({d})", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        ens = pd.DataFrame([r for rows in ex.map(family_rows, jobs) for r in rows])
    if ens.empty:
        raise SystemExit(f"no finished ensembles at {T:g} K")

    # mutant - wild type, per model
    muts = []
    for label in labels:
        e = ens[ens.model == label].set_index("name")
        m = mutants[mutants.name.isin(e.index) & mutants.name_wt.isin(e.index)].copy()
        for chg, (obs, _, _) in CHANGES.items():
            m[chg] = e.loc[m.name, obs].to_numpy() - e.loc[m.name_wt, obs].to_numpy()
        m.insert(0, "model", label)
        muts.append(m)
    muts = pd.concat(muts, ignore_index=True)
    common = set.intersection(*(set(muts.name[muts.model == lb]) for lb in labels))
    print(f"mutants with wild type: {', '.join(f'{lb} {(muts.model == lb).sum()}' for lb in labels)}; "
          f"common to all models: {len(common)}" + ("" if args.no_common else " (used for the correlations)"))
    if len(common) < 3 and not args.no_common:
        print("  fewer than 3 common mutants: correlations are NaN (generation still running? --no-common uses "
              "each model's own mutants)")

    rng = np.random.default_rng(args.seed)
    summary = []
    for chg, (_, sign, _) in CHANGES.items():
        wide = muts.pivot(index="name", columns="model", values=chg).reindex(columns=labels)
        ddg = mutants.set_index("name").ddg_exp
        both = wide.loc[sorted(common)].dropna()
        idx_common = rng.integers(0, len(both), (args.n_boot, len(both))) if len(both) else None
        rho_boot = {}
        for label in labels:
            col = both[label] if not args.no_common else wide[label].dropna()
            x, y = ddg.loc[col.index].to_numpy(), col.to_numpy()
            idx = idx_common if not args.no_common else rng.integers(0, max(len(x), 1), (args.n_boot, len(x)))
            s = correlate(x, y, idx)
            summary.append({"kind": "vs_ddg_exp", "model": label, "change": chg, "expected_sign": sign, **s,
                            "rho_oriented": sign * s["spearman_rho"]})
            if len(both) >= 3:
                rho_boot[label] = boot_spearman(ddg.loc[both.index].to_numpy(), both[label].to_numpy(), idx_common)
        if len(labels) >= 2 and len(both) >= 3:
            a, b = labels[:2]
            xa = ddg.loc[both.index].to_numpy()
            ra, rb = spearmanr(xa, both[a])[0], spearmanr(xa, both[b])[0]
            diff = sign * (rho_boot[a] - rho_boot[b])
            d = diff[np.isfinite(diff)]
            p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
            lo, hi = ci(diff)
            summary.append({"kind": "difference", "model": f"{a} - {b}", "change": chg, "expected_sign": sign,
                            "n": len(both), "rho_oriented": sign * (ra - rb), "rho_ci_low": lo, "rho_ci_high": hi,
                            "p_boot": p_boot})
            agree = boot_spearman(both[a].to_numpy(), both[b].to_numpy(), idx_common)
            lo, hi = ci(agree)
            summary.append({"kind": "agreement", "model": f"{a} vs {b}", "change": chg, "n": len(both),
                            "spearman_rho": spearmanr(both[a], both[b])[0], "rho_ci_low": lo, "rho_ci_high": hi})
    summary = pd.DataFrame(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"stability_{T:g}K"
    ens.to_csv(f"{stem}_ensembles.csv", index=False)
    muts.to_csv(f"{stem}_mutants.csv", index=False)
    summary.to_csv(f"{stem}_summary.csv", index=False)

    print(f"\nCorrelation with ddg_exp at {T:g} K (rho_oriented = expected sign x rho, higher = better):")
    for chg in CHANGES:
        print(f"  {chg}")
        for _, s in summary[summary.change == chg].iterrows():
            if s.kind == "vs_ddg_exp":
                print(f"    {s.model:34s} n {s.n:4.0f} | r {s.pearson_r:+.2f} | rho {s.spearman_rho:+.2f} "
                      f"[{s.rho_ci_low:+.2f}, {s.rho_ci_high:+.2f}] p {s.spearman_p:.1e}")
            elif s.kind == "difference":
                print(f"    {'oriented rho ' + s.model:34s} {s.rho_oriented:+.2f} [{s.rho_ci_low:+.2f}, "
                      f"{s.rho_ci_high:+.2f}] p_boot {s.p_boot:.3f}")
            else:
                print(f"    {'agreement ' + s.model:34s} rho {s.spearman_rho:+.2f} [{s.rho_ci_low:+.2f}, "
                      f"{s.rho_ci_high:+.2f}]")
    plot(muts if args.no_common else muts[muts.name.isin(common)], summary, labels, T, f"{stem}.png")
    print("\n" + "\n".join(f"-> {stem}{suffix}" for suffix in
                           ("_ensembles.csv", "_mutants.csv", "_summary.csv", ".png")))


if __name__ == "__main__":
    main()
