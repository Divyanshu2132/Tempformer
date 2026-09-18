#!/usr/bin/env python

from pathlib import Path
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams.update({'font.size': 18})
BASE = Path("/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/case_study/fast_folding_proteins")
GEN_DIR = Path("/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/spectral_temperature_generated/fast_folding_exact")

PROTEIN_TO_NPY = {
    "Protein_B": "proteinb_ca_coords.npy",
    "BBA": "bba_ca_coords.npy",
    "BBL": "bbl_ca_coords.npy",
    "a3D": "a3d_ca_coords.npy",
    "Trp-cage": "trpcage_ca_coords.npy",
    "Homeodomain": "homeodomain_ca_coords.npy",
    "NTL9": "ntl9_ca_coords.npy",
    "Protein_G": "proteing_ca_coords.npy",
    "WW-Domain": "wwdomain_ca_coords.npy",
    "Chignolin": "chignolin_ca_coords.npy",
    "Lambda-repressor": "lambda_ca_coords.npy",
    "Villin": "villin_ca_coords.npy",
}


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def load_ca_traj(protein, stride):
    arr = np.load(BASE / PROTEIN_TO_NPY[protein], mmap_mode="r")

    # Most files are frames,residues,xyz. Villin appears residues,xyz,frames.
    if arr.shape[-1] == 3:
        xyz = arr[::stride]
    elif arr.shape[1] == 3:
        xyz = np.moveaxis(arr, -1, 0)[::stride]
    else:
        raise ValueError(f"Unrecognized CA coord shape: {arr.shape}")

    return np.asarray(xyz, dtype=np.float32)


def ca_distance_features(xyz, min_seq_sep=3):
    n_ca = xyz.shape[1]
    pairs = [(i, j) for i in range(n_ca) for j in range(i + min_seq_sep, n_ca)]
    X = np.empty((xyz.shape[0], len(pairs)), dtype=np.float32)

    for k, (i, j) in enumerate(pairs):
        X[:, k] = np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1)

    return X, pairs


def read_ca_xyz_from_pdb(pdb_path):
    xyz = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(xyz, dtype=np.float32)


def generated_features(gen_path, pairs):
    pdbs = sorted(gen_path.glob("*_minimized.pdb"), key=natural_key)
    feats, kept = [], []

    for pdb in pdbs:
        xyz = read_ca_xyz_from_pdb(pdb)
        if len(xyz) == 0:
            continue

        feat = np.empty(len(pairs), dtype=np.float32)
        for k, (i, j) in enumerate(pairs):
            if max(i, j) >= len(xyz):
                raise ValueError(f"{pdb} has {len(xyz)} CA atoms, expected at least {max(i, j) + 1}")
            feat[k] = np.linalg.norm(xyz[i] - xyz[j])

        feats.append(feat)
        kept.append(pdb)

    if not feats:
        raise RuntimeError(f"No PDB files found in {gen_path}")

    return np.vstack(feats), kept


def fit_tica(X, lag=20, n_pca=100, n_tics=2, reg=1e-6):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_pca = min(n_pca, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_pca, random_state=0)
    Y = pca.fit_transform(Xs)
    Y -= Y.mean(axis=0, keepdims=True)

    X0 = Y[:-lag]
    Xt = Y[lag:]

    C00 = X0.T @ X0 / (len(X0) - 1)
    Ctt = Xt.T @ Xt / (len(Xt) - 1)
    C0t = X0.T @ Xt / (len(X0) - 1)

    Cauto = 0.5 * (C00 + Ctt) + reg * np.eye(C00.shape[0])
    Ctau = 0.5 * (C0t + C0t.T)

    evals, evecs = eigh(Ctau, Cauto)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order[:n_tics]]
    evals = evals[order[:n_tics]]

    return scaler, pca, evecs, evals, Y @ evecs


def project(X, scaler, pca, tica_vecs):
    return pca.transform(scaler.transform(X)) @ tica_vecs


def plot_tica(md_Y, gen_Y, out_png, title, bins=120):
    H, xedges, yedges = np.histogram2d(md_Y[:, 0], md_Y[:, 1], bins=bins)
    H = gaussian_filter(H.T, sigma=1.4)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=220)
    ax.imshow(
        H,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="Spectral_r",
        aspect="auto",
    )
    ax.scatter(gen_Y[:, 0], gen_Y[:, 1], s=18, c="black", alpha=0.75, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("tIC 1")
    ax.set_ylabel("tIC 2")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def free_energy_grid(Y, temperature, bins=50, k_b=0.0019872041):
    H, xedges, yedges = np.histogram2d(Y[:, 0], Y[:, 1], bins=bins)
    P = H / np.sum(H)

    with np.errstate(divide="ignore", invalid="ignore"):
        G = -k_b * float(temperature) * np.log(P)

    G[~np.isfinite(G)] = np.nan
    G -= np.nanmin(G)

    return G.T, xedges, yedges


def plot_free_energy_with_generated(md_Y, gen_Y, out_png, title, temperature, bins=50):
    G, xedges, yedges = free_energy_grid(md_Y, temperature=temperature, bins=bins)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=220)

    im = ax.imshow(
        G,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="Spectral_r",
        aspect="auto",
    )

    ax.scatter(
        gen_Y[:, 0],
        gen_Y[:, 1],
        s=12,
        c="black",
        alpha=0.65,
        linewidths=0,
        label="Generated",
    )

    ax.set_title(title)
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.legend()
    fig.colorbar(im, ax=ax, label="Free energy (kcal/mol)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", default="Protein_B")
    parser.add_argument("--temperature", default="348")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--lag", type=int, default=20)
    parser.add_argument("--n-pca", type=int, default=100)
    parser.add_argument("--min-seq-sep", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    xyz = load_ca_traj(args.protein, args.stride)
    md_X, pairs = ca_distance_features(xyz, min_seq_sep=args.min_seq_sep)

    scaler, pca, tica_vecs, evals, md_Y = fit_tica(
        md_X,
        lag=args.lag,
        n_pca=args.n_pca,
    )

    gen_path = GEN_DIR / f"{args.protein}_{args.temperature}"
    gen_X, gen_files = generated_features(gen_path, pairs)
    gen_Y = project(gen_X, scaler, pca, tica_vecs)

    out_png = args.out or f"tica_{args.protein}_{args.temperature}.png"
    plot_free_energy_with_generated(
        md_Y,
        gen_Y,
        out_png,
        title=f"{args.protein} {args.temperature} K",
        temperature=args.temperature,
        bins=50,
    )

    # np.savez(
    #     out_png.replace(".png", ".npz"),
    #     md_tica=md_Y,
    #     gen_tica=gen_Y,
    #     evals=evals,
    #     generated_files=np.array([str(p) for p in gen_files]),
    # )

    print(f"Saved: {out_png}")
    print(f"Saved: {out_png.replace('.png', '.npz')}")
    print(f"TICA eigenvalues: {evals}")


if __name__ == "__main__":
    main()