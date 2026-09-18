# === your preprocessing util ===
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

def pseudo_CB(seq, CB, CA):
    """
    use CA instead CB for GLY
    """
    seq = np.array(list(seq))
    new_CB = np.zeros([len(seq), 3])
    s = np.array(list(seq))
    pCB = np.where(np.expand_dims(s == "G", -1), CA, new_CB)
    counter = 0
    for i, j in enumerate(pCB):
        if (0.0 == np.sum(j)):
            pCB[i] = CB[counter]
            counter += 1
    return pCB

def pairwise_distance(x, y=None):
    if y is None:
        y = x
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)

def calc_rotate_imat(N, CA, C):
    assert N.shape == CA.shape == C.shape
    p1 = N - CA
    x = p1 / np.linalg.norm(p1, axis=-1, keepdims=True)
    p2 = C - N
    inner_1 = np.matmul(np.expand_dims(p1, axis=1), np.expand_dims(p1, axis=2))[:, :, 0]
    inner_2 = np.matmul(np.expand_dims(-p1, axis=1), np.expand_dims(p2, axis=2))[:, :, 0]
    alpha = inner_1 / inner_2
    y = alpha * p2 + p1
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    mat = np.concatenate([x, y, z], axis=-1)
    mat = mat.reshape(*mat.shape[:-1], 3, 3)
    return mat


# === LAZY dataset ===
class LazyProteinDataset(Dataset):
    """
    Each item must contain:
      h5_path, domain_id, temp, repl, frame,
      single_path, pair_path,
      ca_idx, n_idx, c_idx
    """
    def __init__(self, data_list, embed_dtype=torch.float16, geom_dtype=torch.float32):
        self.data = data_list
        self.embed_dtype = embed_dtype
        self.geom_dtype = geom_dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- read ONLY the requested frame from HDF5 ---
        with h5py.File(item["h5_path"], "r") as f:
            coords = f[item["domain_id"]][item["temp"]][item["repl"]]["coords"][item["frame"]]  # (num_atoms, 3)

        # coords is numpy; slice out backbone atoms
        N  = coords[item["n_idx"]]
        C  = coords[item["c_idx"]]
        CA = coords[item["ca_idx"]]

        # center
        center = np.nanmean(CA, axis=0)
        N, C, CA = N - center, C - center, CA - center

        # model inputs
        T  = CA
        IR = calc_rotate_imat(N, CA, C)

        # --- memmap embeddings from disk (no upfront load) ---
        # np.load(..., mmap_mode="r") -> backed by file, paged in as accessed
        single_repr = np.load(item["single_path"], mmap_mode="r")[:len(CA),:]
        pair_repr   = np.load(item["pair_path"],   mmap_mode="r")[:len(CA),:len(CA),:]

        # Convert to torch (still CPU). Use fp16 for embeddings to reduce memory traffic.
        # np.asarray keeps it as a view; torch.from_numpy shares memory where possible.
        single_t = torch.from_numpy(np.asarray(single_repr)).to(self.embed_dtype)
        pair_t   = torch.from_numpy(np.asarray(pair_repr)).to(self.embed_dtype)

        return {
            "single_repr": single_t,                                   # (L, Cs)
            "pair_repr": pair_t,                                       # (L, L, Cp)
            "T": torch.from_numpy(np.asarray(T)).to(self.geom_dtype),  # (L, 3)
            "IR": torch.from_numpy(np.asarray(IR)).to(self.geom_dtype),# (L, 3, 3)
            "temperature": torch.tensor(float(item["temp"]), dtype=torch.float32),
        }


def collate_protein_batch(batch, pad_values=None):
    pad_values = pad_values or {
        "single_repr": 0,
        "pair_repr": 0,
        "T": np.nan,
        "IR": np.nan,
        "cbcb": np.nan,
    }

    # --- determine max sequence length ---
    Lmax = max(x["single_repr"].shape[0] for x in batch)
    B = len(batch)

    out = {}

    for key in batch[0].keys():
        if key == "temperature":
            out[key] = torch.stack([x[key] for x in batch])
            continue

        pad_val = pad_values.get(key, 0.0)
        tensors = []

        for x in batch:
            arr = x[key]
            L = arr.shape[0]

            # (L, F)  — per-residue
            if arr.ndim == 2:
                padded = torch.full((Lmax, arr.shape[1]), pad_val, dtype=arr.dtype)
                padded[:L] = arr

            # (L, L, F) or (L, L)  — pairwise
            elif arr.ndim == 3 and arr.shape[0] == arr.shape[1]:
                padded = torch.full((Lmax, Lmax) + arr.shape[2:], pad_val, dtype=arr.dtype)
                padded[:L, :L] = arr

            # (L, 3, 3)  — rotation matrices
            elif arr.ndim == 3 and arr.shape[1:] == (3, 3):
                padded = torch.full((Lmax, 3, 3), pad_val, dtype=arr.dtype)
                padded[:L] = arr

            else:
                raise ValueError(f"Unexpected tensor shape {arr.shape} for key '{key}'")

            tensors.append(padded)

        out[key] = torch.stack(tensors)

    # --- build mask ---
    mask = torch.zeros((B, Lmax), dtype=torch.bool)
    for i, x in enumerate(batch):
        mask[i, :x["T"].shape[0]] = True
    out["mask"] = mask

    return out


def read_pdb_sequence(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ppb = PPBuilder()
    seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
    return "".join(seqs)

