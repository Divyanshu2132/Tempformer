#!/usr/bin/env python

import argparse

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from tica_fast_folding import (
    GEN_DIR,
    ca_distance_features,
    fit_tica,
    generated_features,
    load_ca_traj,
    project,
)


def make_edges(Y, bins=50, padding=0.05):
    """Bin edges spanning Y's range with symmetric padding on each axis."""
    lo = Y.min(axis=0)
    hi = Y.max(axis=0)
    pad = padding * np.maximum(hi - lo, 1e-6)

    edges_x = np.linspace(lo[0] - pad[0], hi[0] + pad[0], bins + 1)
    edges_y = np.linspace(lo[1] - pad[1], hi[1] + pad[1], bins + 1)

    return edges_x, edges_y


def shared_edges(md_Y, gen_Y, bins=18, padding=0.05):
    """Bin edges spanning the union of both point clouds.

    MD-only edges would clip generated points that land outside the MD
    basin, silently dropping them from Q's mass and understating the
    divergence exactly where it matters most (off-manifold structures).
    """
    lo = np.minimum(md_Y.min(axis=0), gen_Y.min(axis=0))
    hi = np.maximum(md_Y.max(axis=0), gen_Y.max(axis=0))
    pad = padding * np.maximum(hi - lo, 1e-6)

    edges_x = np.linspace(lo[0] - pad[0], hi[0] + pad[0], bins + 1)
    edges_y = np.linspace(lo[1] - pad[1], hi[1] + pad[1], bins + 1)

    return edges_x, edges_y


def js_divergence_2d(md_Y, gen_Y, bins=18, padding=0.05, eps=1e-12):
    """Jensen-Shannon divergence (base-2, range [0, 1]) between the MD and
    generated distributions in tIC1/tIC2 space.

    Deliberately COARSE (~15-20 bins/axis) compared to the fine grid used
    for the free-energy plots: with far fewer generated structures than MD
    frames, a fine shared grid leaves many bins with 0-1 generated counts,
    which inflates JSD as a sampling artifact rather than a real
    distributional difference. Coarser bins trade spatial resolution for a
    statistically reliable divergence estimate.
    """
    edges_x, edges_y = shared_edges(md_Y, gen_Y, bins=bins, padding=padding)

    P, _, _ = np.histogram2d(md_Y[:, 0], md_Y[:, 1], bins=[edges_x, edges_y])
    Q, _, _ = np.histogram2d(gen_Y[:, 0], gen_Y[:, 1], bins=[edges_x, edges_y])

    P = P.ravel() / P.sum()
    Q = Q.ravel() / Q.sum()
    P = P + eps
    Q = Q + eps
    P /= P.sum()
    Q /= Q.sum()

    return float(jensenshannon(P, Q, base=2) ** 2)


def js_divergence_kde_2d(md_Y, gen_Y, grid_size=100, bw_method="scott", padding=0.05, eps=1e-12):
    """KDE-based alternative to js_divergence_2d.

    Fits a separate Gaussian KDE to the MD and generated point clouds, then
    evaluates both on the same fixed grid and computes JSD on the resulting
    (normalized) densities. This smooths out the zero-bin sparsity that a
    histogram suffers when the generated sample is small, at the cost of
    introducing a bandwidth choice (`bw_method`) that should be reported and
    justified the same way the bin count is for the histogram version.

    `bw_method` is passed straight to `scipy.stats.gaussian_kde` for both
    distributions: 'scott' / 'silverman' are automatic rules, a float scales
    the automatic bandwidth (e.g. 0.5 for a tighter kernel).
    """
    edges_x, edges_y = shared_edges(md_Y, gen_Y, bins=grid_size, padding=padding)
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    grid_x, grid_y = np.meshgrid(centers_x, centers_y, indexing="ij")
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])

    md_kde = gaussian_kde(md_Y.T, bw_method=bw_method)
    gen_kde = gaussian_kde(gen_Y.T, bw_method=bw_method)

    P = md_kde(grid_points)
    Q = gen_kde(grid_points)

    P = P / P.sum()
    Q = Q / Q.sum()
    P = P + eps
    Q = Q + eps
    P /= P.sum()
    Q /= Q.sum()

    return float(jensenshannon(P, Q, base=2) ** 2)


def precision_fraction(md_Y, gen_Y, bins=50, min_count=1, padding=0.05):
    """Fraction of generated structures landing in an MD-occupied tIC bin.

    This is the "precision" companion to js_divergence_2d: JSD measures how
    well the overall SHAPE of the two distributions matches, while this
    measures whether generated points land where the reference simulation
    actually has density at all. A protein can score high precision (points
    land in real basins) alongside high JSD (the generated ensemble is
    narrower/more concentrated than MD) — that combination is a specific,
    informative failure mode worth reporting rather than averaging away.

    Uses an MD-only occupancy grid (unlike js_divergence_2d's shared-range
    grid): a generated point outside the MD range should count against
    precision, not be padded into a shared bin.
    """
    edges_x, edges_y = make_edges(md_Y, bins=bins, padding=padding)

    md_H, _, _ = np.histogram2d(md_Y[:, 0], md_Y[:, 1], bins=[edges_x, edges_y])
    occupied = md_H >= min_count

    xi = np.searchsorted(edges_x, gen_Y[:, 0], side="right") - 1
    yi = np.searchsorted(edges_y, gen_Y[:, 1], side="right") - 1

    inside = (
        (xi >= 0)
        & (xi < len(edges_x) - 1)
        & (yi >= 0)
        & (yi < len(edges_y) - 1)
    )

    hit = np.zeros(len(gen_Y), dtype=bool)
    hit[inside] = occupied[xi[inside], yi[inside]]

    total = len(gen_Y)

    return {
        "precision_fraction": float(hit.sum() / total) if total else 0.0,
        "generated_inside_range": int(inside.sum()),
        "generated_total": total,
    }


def compute_metrics(
    md_Y,
    gen_Y,
    js_bins=18,
    precision_bins=50,
    min_count=1,
    use_kde=False,
    kde_grid_size=100,
    kde_bw_method="scott",
):
    """JSD (shape) and precision fraction (location), reported together."""
    if use_kde:
        jsd = js_divergence_kde_2d(md_Y, gen_Y, grid_size=kde_grid_size, bw_method=kde_bw_method)
        js_method = f"kde (bw_method={kde_bw_method}, grid={kde_grid_size})"
    else:
        jsd = js_divergence_2d(md_Y, gen_Y, bins=js_bins)
        js_method = f"histogram ({js_bins}x{js_bins} bins)"

    precision = precision_fraction(md_Y, gen_Y, bins=precision_bins, min_count=min_count)

    return {
        "js_divergence": jsd,
        "js_method": js_method,
        **precision,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", default="Protein_B")
    parser.add_argument("--temperature", default="348")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--lag", type=int, default=20)
    parser.add_argument("--n-pca", type=int, default=100)
    parser.add_argument("--min-seq-sep", type=int, default=3)
    parser.add_argument("--js-bins", type=int, default=18, help="Coarse grid for JSD (default: 18x18)")
    parser.add_argument("--precision-bins", type=int, default=50, help="MD occupancy grid for precision")
    parser.add_argument("--min-count", type=int, default=1, help="Min MD frames for a bin to count as occupied")
    parser.add_argument("--use-kde", action="store_true", help="Use KDE instead of histograms for JSD")
    parser.add_argument("--kde-grid-size", type=int, default=100)
    parser.add_argument("--kde-bw-method", default="scott")
    args = parser.parse_args()

    xyz = load_ca_traj(args.protein, args.stride)
    md_X, pairs = ca_distance_features(xyz, min_seq_sep=args.min_seq_sep)

    # fit TICA ONCE, on MD only
    scaler, pca, tica_vecs, evals, md_Y = fit_tica(
        md_X,
        lag=args.lag,
        n_pca=args.n_pca,
    )

    gen_path = GEN_DIR / f"{args.protein}_{args.temperature}"
    gen_X, gen_files = generated_features(gen_path, pairs)

    # project generated structures through the same fixed transformation
    gen_Y = project(gen_X, scaler, pca, tica_vecs)

    metrics = compute_metrics(
        md_Y,
        gen_Y,
        js_bins=args.js_bins,
        precision_bins=args.precision_bins,
        min_count=args.min_count,
        use_kde=args.use_kde,
        kde_grid_size=args.kde_grid_size,
        kde_bw_method=args.kde_bw_method,
    )

    print(f"{args.protein} {args.temperature} K")
    print(f"MD frames: {len(md_Y)}, generated structures: {len(gen_Y)}")
    print(f"JS divergence [{metrics['js_method']}]: {metrics['js_divergence']:.4f}")
    print(
        f"Precision (generated in MD-occupied bins): "
        f"{metrics['precision_fraction']:.4f} "
        f"({metrics['generated_inside_range']}/{metrics['generated_total']} inside range)"
    )


if __name__ == "__main__":
    main()
