import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dual_conformation_sampling import BENCH, OUT_ROOT, load_target

OTHER = {"fold1": "fold2", "fold2": "fold1"}
LABELS = {"fold1": "Fold1 prior", "fold2": "Fold2 prior", "af2": "AF2 prior"}
# categorical slots 1-3 of the reference palette: the three that stay distinguishable in a scatter, also under CVD
SERIES = {"fold1": "#2a78d6", "fold2": "#eb6834", "af2": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--root", type=Path, default=OUT_ROOT, help="output root of dual_conformation_sampling.py")
    a("--runs", nargs="+", type=Path, help="model folders to score (default: every folder in --root but analysis/)")
    a("--bench", type=Path, default=BENCH)
    a("--thresholds", type=float, nargs="+", default=[2.0, 3.0], help="Angstrom, for frac_lt_<x>A")
    a("--out-dir", type=Path, help="CSV output folder (default: <root>/analysis)")
    a("--plot-dir", type=Path, help="also write the plots here (PNG)")
    return p.parse_args()


def rmsd_to(xyz, ref):
    """CA RMSD (Angstrom) of every frame of xyz (n, m, 3) to ref (m, 3), each optimally superposed."""
    x = xyz - xyz.mean(1, keepdims=True)
    r = ref - ref.mean(0)
    h = np.einsum("nmi,mj->nij", x, r)
    s = np.linalg.svd(h, compute_uv=False)
    s[:, -1] *= np.sign(np.linalg.det(h))                               # no reflections
    return np.sqrt(np.maximum((x ** 2).sum((1, 2)) + (r ** 2).sum() - 2 * s.sum(1), 0.0) / len(ref))


def load_ca(meta_path):
    """CA coordinates (n, L, 3) in Angstrom and sequence of the ensemble a run json describes."""
    import mdtraj as md
    stem = meta_path.name[:-len(".json")]
    top = md.load_topology(str(meta_path.with_name(f"{stem}.top.pdb")))
    traj = md.load_dcd(str(meta_path.with_name(f"{stem}.traj.dcd")), top=top, atom_indices=top.select("name CA"))
    return traj.xyz * 10.0, "".join(r.code for r in traj.topology.residues)


def score_run(xyz, meta, states):
    """RMSD of every conformation to both states over the residues resolved in both and modeled, and the
    RMSD between the states over the same residues."""
    s0, s1 = meta["span"]
    ca = {k: dict(zip(s["fasta_index"].tolist(), s["ca"])) for k, s in states.items()}
    common = [i for i in range(s0, s1) if i in ca["fold1"] and i in ca["fold2"]]
    ref = {k: np.array([ca[k][i] for i in common]) for k in ca}
    x = xyz[:, [i - s0 for i in common]]
    return {k: rmsd_to(x, ref[k]) for k in ref}, float(rmsd_to(ref["fold1"][None], ref["fold2"])[0]), len(common)


def metrics(r, d12, prior, thresholds):
    m = {"n_samples": len(r["fold1"]), "d_fold1_fold2": d12, "frac_closer_fold1": np.mean(r["fold1"] < r["fold2"])}
    for k in ("fold1", "fold2"):
        m[f"min_rmsd_{k}"], m[f"mean_rmsd_{k}"] = r[k].min(), r[k].mean()
    if prior in OTHER:
        rt, rp = r[OTHER[prior]], r[prior]
        m.update({"target": OTHER[prior], "d_prior": d12, "min_rmsd_target": rt.min(), "mean_rmsd_target": rt.mean(),
                  "delta_prior": d12 - rt.min(), "frac_toward": np.mean(rt < d12), "frac_target": np.mean(rt < rp),
                  **{f"frac_lt_{x:g}A": np.mean(rt < x) for x in thresholds}, "min_rmsd_prior": rp.min()})
    return m


def write_csv(rows, path):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows({k: fmt(v) for k, v in r.items()} for r in rows)


def summarize(runs, thresholds):
    """One row per (tag, T, direction): experimental prior against the AlphaFold prior."""
    by = {(r["tag"], r["T"], r["target_id"], r["prior"]): r for r in runs}
    rows = []
    for tag, T in sorted({(r["tag"], r["T"]) for r in runs}, key=lambda k: (k[0], str(k[1]))):
        for prior, tgt in (("fold2", "fold1"), ("fold1", "fold2")):
            exp = [r for r in runs if r["tag"] == tag and r["T"] == T and r["prior"] == prior]
            if not exp:
                continue
            paired = [(r, by[(tag, T, r["target_id"], "af2")]) for r in exp if (tag, T, r["target_id"], "af2") in by]
            row = {"tag": tag, "T": T, "direction": f"{prior}->{tgt}", "n_targets": len(exp),
                   "mean_min_rmsd_target": np.mean([r["min_rmsd_target"] for r in exp]),
                   "n_delta_prior_positive": sum(r["delta_prior"] > 0 for r in exp),
                   "mean_delta_prior": np.mean([r["delta_prior"] for r in exp]),
                   **{f"mean_{k}": np.mean([r[k] for r in exp]) for k in
                      ["frac_target", "frac_toward", *(f"frac_lt_{x:g}A" for x in thresholds), "min_rmsd_prior"]}}
            if paired:
                e, b = np.mean([r["min_rmsd_target"] for r, _ in paired]), np.mean([a[f"min_rmsd_{tgt}"] for _, a in paired])
                row.update({"n_paired_af2": len(paired), "paired_min_rmsd_target": e, "paired_af2_min_rmsd_target": b,
                            "change_vs_af2_pct": 100.0 * (e - b) / b})
            rows.append(row)
    return rows


def print_summary(rows, thresholds):
    for tag, T in dict.fromkeys((r["tag"], r["T"]) for r in rows):
        print(f"\n{tag}, {T if T == 'pooled' else f'{T:g} K'}:")
        for r in (r for r in rows if r["tag"] == tag and r["T"] == T):
            tgt = r["direction"].split("->")[1]
            vs = (f" (AF2 prior {r['paired_af2_min_rmsd_target']:.2f} A -> {r['paired_min_rmsd_target']:.2f} A, "
                  f"{r['change_vs_af2_pct']:+.1f}% over {r['n_paired_af2']})") if "n_paired_af2" in r else ""
            print(f"  {r['direction']}: min RMSD to {tgt} {r['mean_min_rmsd_target']:.2f} A{vs} | Delta_prior > 0: "
                  f"{r['n_delta_prior_positive']}/{r['n_targets']} (mean {r['mean_delta_prior']:+.2f} A) | closer to "
                  f"{tgt} than to the prior state: {r['mean_frac_target']:.3f} | toward {tgt}: {r['mean_frac_toward']:.3f} | "
                  + " | ".join(f"< {x:g} A: {r[f'mean_frac_lt_{x:g}A']:.3f}" for x in thresholds))


# ---------------------------------------------------------------- plots

def _style(ax, lim):
    ax.set_facecolor(SURFACE)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_rmsd_space(tag, target, rmsds, d12, path):
    """Conformations of every prior in (RMSD to Fold1, RMSD to Fold2), one panel per temperature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    temps = sorted({T for _, T in rmsds}, key=lambda T: (T == "pooled", T if T != "pooled" else 0))
    temps = [T for T in temps if T != "pooled"] or temps
    lim = 1.05 * max(d12, max(max(r["fold1"].max(), r["fold2"].max()) for r in rmsds.values()))
    fig, axes = plt.subplots(1, len(temps), figsize=(3.1 * len(temps) + 0.4, 3.7), squeeze=False, facecolor=SURFACE)
    for ax, T in zip(axes[0], temps):
        _style(ax, lim)
        for prior in ("af2", "fold1", "fold2"):
            r = rmsds.get((prior, T))
            if r is not None:
                ax.scatter(r["fold1"], r["fold2"], s=22, color=SERIES[prior], edgecolors=SURFACE, linewidths=0.8,
                           alpha=0.85, zorder=2, label=LABELS[prior])
        for name, xy, offset, ha, va in (("Fold1", (0.0, d12), (7, 0), "left", "center"),
                                         ("Fold2", (d12, 0.0), (0, 7), "center", "bottom")):
            ax.scatter(*xy, s=80, marker="*", color=INK, edgecolors=SURFACE, linewidths=1, zorder=3, clip_on=False)
            ax.annotate(name, xy, xytext=offset, textcoords="offset points", ha=ha, va=va, color=INK, fontsize=8)
        ax.set_title(f"{T:g} K", color=INK, fontsize=10)
        ax.set_xlabel("RMSD to Fold1 (Å)", color=MUTED, fontsize=9)
    axes[0][0].set_ylabel("RMSD to Fold2 (Å)", color=MUTED, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(f"{target} · {tag}", color=INK, fontsize=11, x=0.01, y=0.99, ha="left", va="top")
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.0, 0.94), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_vs_af2(tag, T, runs, path):
    """Min RMSD to the target state, AlphaFold prior (x) against experimental prior (y), one dot per target:
    below the diagonal, the experimental prior got closer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(r["target_id"], r["prior"]): r for r in runs if r["tag"] == tag and r["T"] == T}
    panels = []
    for prior, tgt in (("fold2", "fold1"), ("fold1", "fold2")):
        pts = [(by[(t, "af2")][f"min_rmsd_{tgt}"], r["min_rmsd_target"]) for (t, p), r in by.items()
               if p == prior and (t, "af2") in by]
        if pts:
            panels.append((prior, tgt, np.array(pts)))
    if not panels:
        return
    lim = 1.05 * max(p.max() for *_, p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8), squeeze=False, facecolor=SURFACE)
    for ax, (prior, tgt, pts) in zip(axes[0], panels):
        _style(ax, lim)
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color=SERIES[prior], edgecolors=SURFACE, linewidths=0.8, zorder=2)
        better = int((pts[:, 1] < pts[:, 0]).sum())
        ax.set_title(f"{prior.title()} → {tgt.title()}: {better}/{len(pts)} closer", color=INK, fontsize=10)
        ax.set_xlabel(f"AF2 prior: min RMSD to {tgt.title()} (Å)", color=MUTED, fontsize=9)
        ax.set_ylabel(f"{prior.title()} prior: min RMSD to {tgt.title()} (Å)", color=MUTED, fontsize=9)
    fig.suptitle(f"{tag} · {T if T == 'pooled' else f'{T:g} K'}", color=INK, fontsize=11, x=0.01, y=0.99, ha="left",
                 va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------- driver

def main():
    args = parse_args()
    folders = args.runs or sorted(d for d in args.root.iterdir() if d.is_dir() and d.name != "analysis")
    out_dir = args.out_dir or args.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    natives, samples, runs = {}, [], []
    rmsds = defaultdict(dict)                       # (tag, target) -> {(prior, T): {"fold1": (n,), "fold2": (n,)}}
    d12s = {}
    for folder in folders:
        tag = folder.name
        for meta_path in sorted(folder.glob("*/*.json")):
            meta = json.loads(meta_path.read_text())
            if not {"target", "prior", "span", "temperature"} <= meta.keys():
                continue                                                # not a dual_conformation_sampling.py run
            target, prior, T = meta["target"], meta["prior"], float(meta["temperature"])
            if target not in natives:
                natives[target] = load_target(target, args.bench)
            fasta, states = natives[target]
            xyz, seq = load_ca(meta_path)
            s0, s1 = meta["span"]
            if seq != fasta[s0:s1]:
                print(f"{meta_path}: ensemble sequence differs from the FASTA span, skipped", flush=True)
                continue
            r, d12, n_common = score_run(xyz, meta, states)
            rmsds[(tag, target)][(prior, T)] = r
            d12s[target] = d12
            samples += [{"tag": tag, "target_id": target, "prior": prior, "T": T, "sample": i,
                         "rmsd_fold1": a, "rmsd_fold2": b} for i, (a, b) in enumerate(zip(r["fold1"], r["fold2"]))]
            runs.append({"tag": tag, "target_id": target, "prior": prior, "T": T, "n_common": n_common,
                         **metrics(r, d12, prior, args.thresholds)})
    if not runs:
        raise SystemExit(f"no finished runs in {', '.join(map(str, folders))}")
    for (tag, target), by_run in rmsds.items():                         # every prior over all its temperatures
        for prior in {p for p, _ in by_run}:
            rs = [r for (p, _), r in by_run.items() if p == prior]
            if len(rs) > 1:
                pooled = {k: np.concatenate([r[k] for r in rs]) for k in ("fold1", "fold2")}
                runs.append({"tag": tag, "target_id": target, "prior": prior, "T": "pooled",
                             "n_common": next(r["n_common"] for r in runs if r["tag"] == tag and r["target_id"] == target),
                             **metrics(pooled, d12s[target], prior, args.thresholds)})
    summary = summarize(runs, args.thresholds)
    write_csv(samples, out_dir / "samples.csv")
    write_csv(runs, out_dir / "runs.csv")
    write_csv(summary, out_dir / "summary.csv")
    print_summary(summary, args.thresholds)
    print(f"\n{len(runs)} ensembles of {len(natives)} targets -> {out_dir}/{{samples,runs,summary}}.csv")
    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        for (tag, target), by_run in rmsds.items():
            plot_rmsd_space(tag, target, by_run, d12s[target], args.plot_dir / f"{tag}.{target}.rmsd_space.png")
        for tag, T in {(r["tag"], r["T"]) for r in runs}:
            plot_vs_af2(tag, T, runs, args.plot_dir / f"{tag}.{T if T == 'pooled' else f'{T:g}'}.vs_af2.png")
        print(f"plots -> {args.plot_dir}")


if __name__ == "__main__":
    main()
