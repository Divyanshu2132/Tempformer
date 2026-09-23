import numpy as np
from Bio.PDB import PDBParser, PPBuilder
import torch
from torch.utils.data import Dataset, DataLoader
import MDAnalysis as mda
from utils import LazyProteinDataset, read_pdb_sequence, collate_protein_batch
import torch.optim as optim
from model.main_model import MainModel
from run_inference import load_model
from tqdm import tqdm, trange
import os
import h5py
import random
import argparse

random.seed(42)
aa_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V","HSD":"H","HSE":"H","HSP":"H"}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tempformer on AlphaFold single embeddings + DCA pairwise embeddings")
    parser.add_argument("--data-dir", default="./data", help="Directory of mdcath HDF5 dataset files")
    parser.add_argument("--embedding-path", default="./embeddings/alphafold", help="Directory of AlphaFold single embedding .npy files")
    parser.add_argument("--pairwise-embedding-path", default="./embeddings/dca", help="Directory of DCA pairwise embedding .npy files")
    parser.add_argument("--save-path", default="./weights/checkpoint.pth", help="Path to save the best model checkpoint")
    return parser.parse_args()

args = parse_args()
save_path = args.save_path
embidding_path = args.embedding_path
pairwise_embidding_path = args.pairwise_embedding_path
data_dir = args.data_dir
os.makedirs(os.path.dirname(save_path), exist_ok=True)

all_files = random.sample(os.listdir(data_dir), 500)
val_files = all_files[:100]
train_files = all_files[100:]
temperature = ['320', '348', '379', '413', '450']
repl_list = ['0', '1', '2', '3', '4']

def find_embed_paths(domain_id: str):
    files = os.listdir(embidding_path)
    pair_files = os.listdir(pairwise_embidding_path)
    single = [x for x in files if domain_id in x and "single" in x and "rank_001" in x and x.endswith(".npy")]
    pair   = [x for x in pair_files if domain_id in x and "pair"   in x and x.endswith(".npy")]
    if len(single) == 0 or len(pair) == 0:
        return None, None
    return os.path.join(embidding_path, single[0]), os.path.join(pairwise_embidding_path, pair[0])

def build_samples(file_list):
    samples = []
    for fname in file_list:
        domain_id = fname.split('_')[-1].split('.')[0]
        h5_path = os.path.join(data_dir, fname)

        single_path, pair_path = find_embed_paths(domain_id)
        if single_path is None:
            continue

        with h5py.File(h5_path, "r") as f:
            pdbProteinAtoms = f[domain_id]['pdbProteinAtoms'][()].decode('utf-8').split('\n')[1:-3]
            atomtypes = [line.split()[2] for line in pdbProteinAtoms]
            ca_indices = np.where(np.array(atomtypes) == 'CA')[0]
            n_indices  = np.where(np.array(atomtypes) == 'N')[0]
            c_indices  = np.where(np.array(atomtypes) == 'C')[0]
            if len(ca_indices) != len(n_indices) or len(n_indices) != len(c_indices):
                continue

            seq = "".join([aa_3to1[pdbProteinAtoms[idx].split()[3]] for idx in ca_indices])
            for repl in repl_list:
                for tempera in temperature:
                    coords_ds = f[domain_id][tempera][str(repl)]["coords"]
                    # sample every 20 frames
                    for frame in range(0, coords_ds.shape[0], 60):
                        samples.append({
                            "h5_path": h5_path,
                            "domain_id": domain_id,
                            "repl": str(repl),
                            "temp": str(tempera),
                            "frame": int(frame),
                            "single_path": single_path,
                            "pair_path": pair_path,
                            "ca_idx": ca_indices.astype(np.int32),
                            "n_idx": n_indices.astype(np.int32),
                            "c_idx": c_indices.astype(np.int32),
                            "seq": seq,
                        })
    return samples

data = build_samples(train_files)
val_data = build_samples(val_files)

batch_size = 6
data = data[:-(len(data)%batch_size)] if len(data)%batch_size!=0 else data
val_data = val_data[:-(len(val_data)%batch_size)] if len(val_data)%batch_size!=0 else val_data
print("num train samples:", len(data))
print("num val samples:", len(val_data))
dataset = LazyProteinDataset(data)
val_dataset = LazyProteinDataset(val_data)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,             # tune (2-8) depending on node
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=collate_protein_batch,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_protein_batch,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MainModel().to(device)
# model = load_model(save_path).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("Training on device:", device.type)
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler(enabled=use_amp)

best_loss = float("inf")
epochs = 300
def gb(x): return x / 1024**3
# Only move these to GPU (adjust keys to match what MainModel actually uses)
GPU_KEYS = {"single_repr", "pair_repr","T", "IR", "temperature"}
c=0
for epoch in trange(epochs):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move only selected tensors to GPU
        pair_shape = tuple(batch["pair_repr"].shape)
        single_shape = tuple(batch["single_repr"].shape)
        for k in list(batch.keys()):
            v = batch[k]
            if k in GPU_KEYS and torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # try:
        with torch.amp.autocast(device_type="cuda",enabled=use_amp):
            output = model(batch)
            loss = output["loss"]

        # except:
        # torch.cuda.synchronize()
        # peak = torch.cuda.max_memory_allocated()
        # curr = torch.cuda.memory_allocated()
        # resv = torch.cuda.memory_reserved()

        # print(f"[step {epoch}] peak_alloc={gb(peak):.2f}GB curr={gb(curr):.2f}GB reserved={gb(resv):.2f}GB "
        #     f"pair={pair_shape} single={single_shape},T={batch['T'].shape} IR={batch['IR'].shape} temp={batch['temperature'].shape}")
        #     raise
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
    avg_loss = total_loss / len(loader)

    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            for k in list(batch.keys()):
                v = batch[k]
                if k in GPU_KEYS and torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                output = model(batch)
                val_loss_total += float(output["loss"].item())
    val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else float("nan")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_loss": val_loss,
            },
            save_path,
        )
    else:
        c+=1
        # if c==50:
        #     break
    print(f"Epoch {epoch+1} | train loss = {avg_loss:.6f} | val loss = {val_loss:.6f} | best = {best_loss:.6f}")