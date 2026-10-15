import os
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import residue_geometry as G

TEMPS = ["320", "348", "379", "413", "450"]


def read_fasta(path):
    seqs, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].split()[0]
            seqs[cur] = ""
        elif cur:
            seqs[cur] += line
    return seqs


def index_embeddings(emb_dir):
    """{domain: {"ranks": {rank: {"single": path, "pair": path}}, "template": rank_001 pdb path}}. Domain names may
    contain underscores (e.g. the point mutants 2JWS_I24D, HHH_rd1_I7P)."""
    npy = re.compile(r"^_?(.+?)_(single|pair)_repr_rank_(\d{3})_.*\.npy$")
    pdb = re.compile(r"^_?(.+?)_unrelaxed_rank_001_.*\.pdb$")
    index = {}
    for name in os.listdir(emb_dir):
        m = npy.match(name)
        if m:
            index.setdefault(m.group(1), {"ranks": {}}).setdefault("ranks", {}).setdefault(int(m.group(3)), {})[m.group(2)] = emb_dir / name
            continue
        m = pdb.match(name)
        if m:
            index.setdefault(m.group(1), {"ranks": {}})["template"] = emb_dir / name
    for entry in index.values():
        entry["ranks"] = {r: p for r, p in entry["ranks"].items() if "single" in p and "pair" in p}
    return index


def find_h5(dom, data_dirs):
    for prefix in ("mdcath_snapshots_", "mdcath_dataset_"):
        for d in data_dirs:
            path = d / f"{prefix}{dom}.h5"
            if path.exists():
                return path
    return None


def parse_topology(pdb_protein_atoms):
    """mdCATH protein topology -> aatype (L,), atom14 -> coords index (L, 14) (-1 = absent), residue numbers."""
    aatype, idx, resid, keys = [], [], [], {}
    for a, line in enumerate(l for l in pdb_protein_atoms.split("\n") if l.startswith("ATOM")):
        name, resname, rnum = line[12:16].strip(), line[17:21].strip(), line[22:27].strip()
        key = (rnum, resname)
        if key not in keys:
            keys[key] = len(aatype)
            aatype.append(G.resname_to_index(resname))
            idx.append([-1] * 14)
            resid.append(int(re.match(r"-?\d+", rnum).group()))
        if name.startswith("H"):
            continue
        slot = G.atom14_slot(aatype[keys[key]], name)
        if slot is not None:
            idx[keys[key]][slot] = a
    return np.array(aatype, dtype=np.int64), np.array(idx, dtype=np.int64), np.array(resid)


def template_ca(pdb_path, L):
    """CA coordinates of the AlphaFold model; None if its length does not match."""
    ca = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pdb_path)
          if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return np.array(ca, dtype=np.float32) if len(ca) == L else None


class FrameSet(Dataset):
    """Samples from a table of (domain, temperature, replica, frame, rank, crop start)."""

    def __init__(self, domains, rows, crop_size):
        self.domains, self.crop_size = domains, crop_size
        self.rows = np.asarray(rows, dtype=np.int64).reshape(-1, 6)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d, t, r, frame, rank, start = (int(v) for v in self.rows[i])
        dom = self.domains[d]
        L = len(dom["aatype"])
        s = slice(start, min(start + self.crop_size, L))
        with h5py.File(dom["h5_path"], "r") as f:
            coords = f[dom["domain_id"]][TEMPS[t]][str(r)]["coords"][frame]              # (atoms, 3), Angstrom
        idx = dom["atom14_index"][s]
        mask = (idx >= 0).astype(np.float32)
        atom14 = np.where(mask[..., None] > 0, coords[idx.clip(0)], 0.0).astype(np.float32)
        center = atom14[:, 1][mask[:, 1] > 0].mean(0)
        atom14 = (atom14 - center) * mask[..., None]
        paths = dom["ranks"][rank]
        single = np.array(np.load(paths["single"], mmap_mode="r")[s])
        pair = np.array(np.load(paths["pair"], mmap_mode="r")[s, s])
        tmpl = dom["template_ca"]
        bonded = dom["bonded"][s].copy()
        bonded[-1] = False                                   # the crop's last residue has no next residue
        return {
            "aatype": torch.from_numpy(dom["aatype"][s]),
            "atom14": torch.from_numpy(atom14),
            "atom14_mask": torch.from_numpy(mask),
            "bonded": torch.from_numpy(bonded),
            "single": torch.from_numpy(single).to(torch.float16),
            "pair": torch.from_numpy(pair).to(torch.float16),
            "template_ca": torch.from_numpy(tmpl[s] if tmpl is not None else np.zeros((len(idx), 3), np.float32)),
            "template_mask": torch.full((len(idx),), tmpl is not None, dtype=torch.bool),
            "temperature": torch.tensor(float(TEMPS[t])),
        }


def collate(batch):
    L = max(len(x["aatype"]) for x in batch)
    out = {"temperature": torch.stack([x["temperature"] for x in batch])}
    for key in batch[0]:
        if key == "temperature":
            continue
        v = batch[0][key]
        shape = (L, L) + v.shape[2:] if key == "pair" else (L,) + v.shape[1:]
        padded = torch.zeros((len(batch),) + shape, dtype=v.dtype)
        for b, x in enumerate(batch):
            n = len(x["aatype"])
            if key == "pair":
                padded[b, :n, :n] = x[key]
            else:
                padded[b, :n] = x[key]
        out[key] = padded
    out["res_mask"] = torch.zeros((len(batch), L), dtype=torch.bool)
    for b, x in enumerate(batch):
        out["res_mask"][b, :len(x["aatype"])] = True
    out["bonded"] = out["bonded"] & out["res_mask"]
    out["bonded"][:, :-1] &= out["res_mask"][:, 1:]
    return out


class FramePool:
    """Every (domain, temperature) with the number of saved frames per replica."""

    def __init__(self, domains, pools, crop_size):
        self.domains, self.pools, self.crop_size = domains, pools, crop_size

    def draw(self, per_temp, seed, random_rank=True, random_crop=True):
        """per_temp frames for every (domain, temperature), split evenly over the replicas (random replicas
        take the remainder), random distinct frames within each replica; each with an embedding rank and a
        crop start (random, or rank 1 and a centred crop when disabled)."""
        rng = np.random.default_rng(seed)
        rows = []
        for d, t, reps in self.pools:
            dom = self.domains[d]
            L, ranks = len(dom["aatype"]), sorted(dom["ranks"])
            k = np.full(len(reps), per_temp // len(reps))
            k[rng.choice(len(reps), per_temp % len(reps), replace=False)] += 1
            for (r, n), kr in zip(reps, k):
                for frame in rng.choice(n, min(kr, n), replace=False):
                    rank = int(rng.choice(ranks)) if random_rank else ranks[0]
                    extra = max(0, L - self.crop_size)
                    start = int(rng.integers(0, extra + 1)) if random_crop else extra // 2
                    rows.append((d, t, r, frame, rank, start))
        return FrameSet(self.domains, rows, self.crop_size)


def build_pool(label, fasta, emb_dir, data_dirs, crop_size, max_len=None, limit=None, keep_md_reference=False,
               only=None, index_fn=index_embeddings):
    """Index usable domains (all FASTA entries, or those in `only`). keep_md_reference also stores MD targets
    for sampling-based validation. index_fn(emb_dir) finds the embedding files, in the form of index_embeddings
    (train_dca.py and train_esm.py pass their own)."""
    seqs = read_fasta(fasta)
    names = [n for n in seqs if only is None or n in set(only)][:limit]
    emb = index_fn(emb_dir) if emb_dir.is_dir() else {}
    domains, pools = [], []
    skipped = {"no embeddings": 0, "no mdCATH file": 0, "sequence differs from FASTA": 0, "unknown residues": 0,
               "missing backbone atoms": 0, "embedding shorter than sequence": 0, "too long": 0, "no template": 0}
    start = time.time()
    for n, dom in enumerate(names, 1):
        entry = emb.get(dom)
        if not entry or not entry["ranks"]:
            skipped["no embeddings"] += 1
            continue
        h5_path = find_h5(dom, data_dirs)
        if h5_path is None:
            skipped["no mdCATH file"] += 1
            continue
        with h5py.File(h5_path, "r") as f:
            g = f[dom]
            aatype, idx, resid = parse_topology(g["pdbProteinAtoms"][()].decode("utf-8"))
            L = len(aatype)
            seq = "".join(G.RESTYPES[a] for a in aatype)
            if (aatype == G.UNK).any():
                skipped["unknown residues"] += 1
                continue
            if seq != seqs[dom]:
                skipped["sequence differs from FASTA"] += 1
                continue
            if (idx[:, :3] < 0).any():
                skipped["missing backbone atoms"] += 1
                continue
            if max_len and L > max_len:
                skipped["too long"] += 1
                continue
            emb_len = np.load(entry["ranks"][sorted(entry["ranks"])[0]]["single"], mmap_mode="r").shape[0]
            if emb_len < L:
                skipped["embedding shorter than sequence"] += 1
                continue
            tmpl = template_ca(entry["template"], L) if "template" in entry else None
            if tmpl is None:
                skipped["no template"] += 1       # kept: template features are masked out
            info = dict(h5_path=str(h5_path), domain_id=dom, aatype=aatype, atom14_index=idx,
                        bonded=np.r_[resid[1:] == resid[:-1] + 1, False], ranks=entry["ranks"], template_ca=tmpl)
            if keep_md_reference:
                info["md"] = md_reference(g, idx, aatype)
            d = len(domains)
            domains.append(info)
            for t, temp in enumerate(TEMPS):
                if temp in g:
                    reps = [(int(r), g[temp][r]["coords"].shape[0]) for r in g[temp].keys()]
                    reps = [(r, k) for r, k in reps if k > 0]
                    if reps:
                        pools.append((d, t, reps))
        if n % 500 == 0:
            print(f"  {label}: indexed {n}/{len(names)} domains ({time.time() - start:.0f}s)", flush=True)
    print(f"{label}: {len(domains)} of {len(names)} domains used, {len(pools)} domain-temperature pairs; skipped: "
          f"{', '.join(f'{v} {k}' for k, v in skipped.items() if v) or 'none'}", flush=True)
    return FramePool(domains, pools, crop_size)


def embedding_widths(pool):
    """Channels of the single and pair embeddings (MainModel input_shape, pair_shape), from the first domain."""
    ranks = pool.domains[0]["ranks"]
    return tuple(int(np.load(ranks[min(ranks)][k], mmap_mode="r").shape[-1]) for k in ("single", "pair"))


def md_reference(g, atom14_index, aatype, frames_per_replica=50):
    """MD targets per temperature: RMSF (mean over replicas), 3-state DSSP fractions per residue, and from
    up to frames_per_replica frames of every replica: CA coordinates (native contacts) and chi1/chi2 rotamer
    bins (side chains). Also the native CA structure (topology coordinates)."""
    ca = atom14_index[:, 1]
    lines = [l for l in g["pdbProteinAtoms"][()].decode("utf-8").split("\n") if l.startswith("ATOM")]
    native = np.array([[float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])] for i in ca], np.float32)
    ref = {"native_ca": native, "rmsf": {}, "ss": {}, "ca_frames": {}, "chi_bins": {}}
    atom_mask = (atom14_index >= 0).astype(np.float32)
    for temp in TEMPS:
        if temp not in g:
            continue
        reps = [g[temp][r] for r in g[temp].keys()]
        ref["rmsf"][temp] = 10.0 * np.mean([r["rmsf"][:] for r in reps], axis=0)          # nm -> Angstrom
        dssp = np.concatenate([(r["dssp_all_frames"] if "dssp_all_frames" in r else r["dssp"])[:] for r in reps])
        ref["ss"][temp] = ss_fractions(dssp.astype("S1"))
        coords = np.concatenate([r["coords"][:: max(1, r["coords"].shape[0] // frames_per_replica)] for r in reps])
        ref["ca_frames"][temp] = coords[:, ca].astype(np.float32)
        atom14 = coords[:, atom14_index.clip(0)] * atom_mask[None, ..., None]                 # (frames, L, 14, 3)
        ref["chi_bins"][temp] = G.atom14_rotamer_bins(atom14, torch.from_numpy(aatype), atom_mask[None]).numpy().astype(np.int8)
    return ref


def ss_fractions(dssp8):
    """(frames, L) 8-state DSSP codes -> (3, L) fraction of frames in helix (H, G, I), strand (E, B), coil."""
    h = np.isin(dssp8, [b"H", b"G", b"I"]).mean(0)
    e = np.isin(dssp8, [b"E", b"B"]).mean(0)
    return np.stack([h, e, 1.0 - h - e])


def protein_batch(dom, rank=None):
    """Uncropped conditioning features of one protein, batch dimension 1 (for sampling)."""
    rank = rank if rank is not None else sorted(dom["ranks"])[0]
    L, paths, tmpl = len(dom["aatype"]), dom["ranks"][rank], dom["template_ca"]
    return {
        "aatype": torch.from_numpy(dom["aatype"])[None],
        "single": torch.from_numpy(np.array(np.load(paths["single"], mmap_mode="r")[:L])).float()[None],
        "pair": torch.from_numpy(np.array(np.load(paths["pair"], mmap_mode="r")[:L, :L])).float()[None],
        "template_ca": torch.from_numpy(tmpl if tmpl is not None else np.zeros((L, 3), np.float32))[None],
        "template_mask": torch.full((1, L), tmpl is not None, dtype=torch.bool),
        "res_mask": torch.ones((1, L), dtype=torch.bool),
    }


def make_loader(frames, batch_size, num_workers, shuffle):
    return DataLoader(frames, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers,
                      pin_memory=True, persistent_workers=False,
                      prefetch_factor=2 if num_workers > 0 else None, collate_fn=collate)
