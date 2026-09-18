import numpy as np
from Bio.PDB import PDBParser, PPBuilder
import torch
from torch.utils.data import Dataset, DataLoader
import MDAnalysis as mda
from utils import LazyProteinDataset, read_pdb_sequence, collate_protein_batch
import torch.optim as optim
from model.main_model import MainModel
from run_inference import load_model,xyz2pdb,write_to_pdb
import time
from tqdm import tqdm
import os
import h5py
from scipy.spatial.transform import Rotation as R
import random
import argparse
from model import geometry, so3  # assumes same module structure

random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aa_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V","HSD":"H","HSE":"H","HSP":"H"}


def find_embed_paths(domain_id: str):
    files = os.listdir(embidding_path)
    single = [x for x in files if domain_id in x and "single" in x and "rank_001" in x and x.endswith(".npy")]
    pair   = [x for x in files if domain_id in x and "pair"   in x and "rank_001" in x and x.endswith(".npy")]
    if len(single) == 0 or len(pair) == 0:
        return None, None
    return os.path.join(embidding_path, single[0]), os.path.join(embidding_path, pair[0])


def convert_to_CANC_scaling(tr, rot_mat ,N_ref,C_ref,CA_dis,scale_factor):
    tr, rot_mat = tr.to(device), rot_mat.to(device)
    CA = tr
    length = len(CA)
    sum_tensor = torch.empty((length-1), device=CA.device)
    if scale_factor == 0 : 
        for i in range(length-1):
            sum_tensor[i] = torch.sum((CA[i] - CA[i+1]) ** 2)
        scale_factor = torch.sqrt(torch.sum(CA_dis)/torch.sum(sum_tensor))
    CA*=scale_factor
    print(scale_factor.device,rot_mat.device,N_ref.device,C_ref.device,CA.device)
    N = torch.matmul(rot_mat.transpose(-1, -2), N_ref)*scale_factor + CA
    C = torch.matmul(rot_mat.transpose(-1, -2), C_ref)*scale_factor + CA
    return CA, N, C, scale_factor

def write_to_pdb_scaling(seq, tr, rot_mat, file,N_ref,C_ref,CA_dis,scale_factor):
    CA, N, C, scale_factor = convert_to_CANC_scaling(tr, rot_mat,N_ref,C_ref,CA_dis,scale_factor=scale_factor)
    with open(file, "w") as fp:
        lines = xyz2pdb(seq, CA, N, C)
        fp.write("\n".join(lines))
    # print("sf:", scale_factor.item() if torch.is_tensor(scale_factor) else scale_factor)
    # print("mean |N-CA|:", torch.norm(N-CA, dim=1).mean().item())
    # print("mean |C-CA|:", torch.norm(C-CA, dim=1).mean().item())
    # print("mean |CA-CA|:", torch.norm(CA[1:]-CA[:-1], dim=1).mean().item())
    return scale_factor

def inference_fn(
    model,
    single_repr,
    pair_repr,
    tr_init=None,
    rot_mat_init=None,
    temperature=None,
    save_full_state=False,
    use_tqdm=True,
):
    """
    Diffusion reverse process: generate 3D conformations conditioned on temperature.

    Parameters
    ----------
    model : trained MainModel (temperature-aware)
    single_repr : torch.Tensor (L, F1)
    pair_repr   : torch.Tensor (L, L, F2)
    tr_init, rot_mat_init : optional initial translations and rotations
    temperature : torch.Tensor (1,) or (B,) in Kelvin (conditioning input)
    """
    device = single_repr.device
    inference_steps = 500

    # --- schedules ---
    t_schedule = np.linspace(1, 0, inference_steps + 1)[:-1]
    tr_schedule, rot_schedule = t_schedule, t_schedule
    tr_sigma_min, tr_sigma_max = model.tr_sigma_min, model.tr_sigma_max
    rot_sigma_min, rot_sigma_max = model.rot_sigma_min, model.rot_sigma_max

    def t_to_sigma(t_tr, t_rot):
        T_sigma = (tr_sigma_min ** (1 - t_tr)) * (tr_sigma_max ** t_tr)
        IR_sigma = (rot_sigma_min ** (1 - t_rot)) * (rot_sigma_max ** t_rot)
        return T_sigma, IR_sigma

    # --- initialize coordinates ---
    def init_conformer(feature):
        L = feature.shape[0]
        random_tr = torch.randn(L, 3) * tr_sigma_max
        random_rot = torch.from_numpy(R.random(num=L).as_matrix()).float()
        return random_tr, random_rot

    if tr_init is None or rot_mat_init is None:
        tr, rot_mat = init_conformer(single_repr)
        tr_init, rot_mat_init = tr.clone(), rot_mat.clone()
    else:
        tr, rot_mat = tr_init.clone(), rot_mat_init.clone()

    tr, rot_mat = tr.to(device), rot_mat.to(device)

    if save_full_state:
        tr_list, rot_mat_list = [tr_init.clone().to(device)], [rot_mat_init.clone().to(device)]

    start_time = time.time()

    # --- reverse diffusion ---
    for t_idx in tqdm(range(inference_steps), disable=not use_tqdm):
        t_tr, t_rot = tr_schedule[t_idx], rot_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]

        tr_sigma, rot_sigma = t_to_sigma(t_tr, t_rot)

        # predict score
        with torch.no_grad():
            tr_score, rot_score = model.forward_step(
                (tr[None], rot_mat[None]),
                torch.zeros((1, tr.shape[0]), dtype=bool, device=device),
                torch.tensor([t_idx], device=device),
                single_repr[None],
                pair_repr[None],
                temperature=temperature,        # <---- temperature conditioning
            )
            tr_score, rot_score = tr_score[0], rot_score[0]
            tr_score /= tr_sigma
            rot_score *= so3.score_norm(torch.tensor([rot_sigma], device=torch.device('cpu')))[0]

        # diffusion step updates
        tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(tr_sigma_max / tr_sigma_min)))
        rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(rot_sigma_max / rot_sigma_min)))

        tr_perturb = (tr_g**2) * dt_tr * tr_score
        rot_perturb = (rot_g**2) * dt_rot * rot_score

        rot_mat_perturb = geometry.axis_angle_to_matrix(rot_perturb)
        tr = tr + tr_perturb
        rot_mat = torch.matmul(rot_mat_perturb, rot_mat)

        #saving score norms for monitoring
        score_trace = {
            "step": [],
            "tr_score_norm": [],
            "rot_score_norm": [],
            "tr_perturb_norm": [],
            "rot_perturb_norm": [],
        }

        
        if t_idx % 10 == 0 or t_idx == inference_steps - 1:
            score_trace["step"].append(t_idx)
            score_trace["tr_score_norm"].append(torch.norm(tr_score, dim=-1).detach().cpu())
            score_trace["rot_score_norm"].append(torch.norm(rot_score, dim=-1).detach().cpu())
            score_trace["tr_perturb_norm"].append(torch.norm(tr_perturb, dim=-1).detach().cpu())
            score_trace["rot_perturb_norm"].append(torch.norm(rot_perturb, dim=-1).detach().cpu())

        
        if save_full_state:
            tr_list.append(tr.clone().to(device))
            rot_mat_list.append(rot_mat.clone().to(device))

    # --- summary ---
    x = torch.norm(tr[1:] - tr[:-1], dim=-1)
    print(
        f"CA-CA distance: {x.mean():.3f} ± {x.std():.3f} | "
        f"len={tr.shape[0]} | time={time.time() - start_time:.2f}s"
    )

    if save_full_state:
        return tr_list, rot_mat_list
    return tr_init, rot_mat_init, tr, rot_mat,score_trace

def parse_args():
    parser = argparse.ArgumentParser(description="Generate conformations at target temperatures with a trained Tempformer model")
    parser.add_argument("--checkpoint-path", default="./weights/checkpoint.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--data-dir", default="./data", help="Directory of mdcath HDF5 dataset files")
    parser.add_argument("--embedding-path", default="./embeddings/alphafold", help="Directory of AlphaFold single/pair embedding .npy files")
    parser.add_argument("--output-dir", default="./output", help="Directory to write generated PDB structures to")
    return parser.parse_args()

args = parse_args()
checkpoint_path = args.checkpoint_path
model=load_model(checkpoint_path).to(device)

data_dir = args.data_dir
temperature = ['320', '348', '379', '413', '450']
train_files = random.sample(os.listdir(data_dir),750)[500:]
# print(len(train_files))
embidding_path = args.embedding_path
output_dir = args.output_dir
for fname in train_files:
    # domain_id = fname.split('_')[-1].split('.')[0]
    domain_id = fname
    single_path, _ = find_embed_paths(domain_id)
    if single_path is None:
        continue
    # if np.load(single_path).shape[0] > 280:
    #     continue
    print(domain_id)
    seq=""
    repl=0
    ca_indices =[]
    n_indices = []
    c_indices = []
    for tem in temperature:
        with h5py.File(os.path.join(data_dir, f"mdcath_dataset_{domain_id}.h5"), 'r') as f:
            pdbProteinAtoms = f[domain_id]['pdbProteinAtoms'][()].decode('utf-8').split('\n')[1:-3] # remove header and footer
            atomtypes = [line.split()[2] for line in pdbProteinAtoms]
            ca_indices = np.where(np.array(atomtypes) == 'CA')[0]
            n_indices = np.where(np.array(atomtypes) == 'N')[0]
            c_indices = np.where(np.array(atomtypes) == 'C')[0]
            seq="".join([aa_3to1[pdbProteinAtoms[index].split()[3]] for index in ca_indices])
            for frame in [0]:
                temp=f[domain_id][tem][str(repl)]["coords"]
                ca_indices =temp[frame][ca_indices]
                n_indices = temp[frame][n_indices]
                c_indices = temp[frame][c_indices]
        single_repr=np.load(embidding_path+[i for i in os.listdir(embidding_path) if domain_id in i and 'single' in i and "rank_001" in i and ".npy" in i][0])[:(len(seq))]
        pair_repr=np.load(embidding_path+[i for i in os.listdir(embidding_path) if domain_id in i and 'pair' in i and "rank_001" in i and ".npy" in i][0])[:(len(seq)),:(len(seq))]
        print(n_indices.shape,ca_indices.shape,c_indices.shape)
        if ca_indices.shape!=n_indices.shape or c_indices.shape!=n_indices.shape:
            continue
        n_ref_vectors=n_indices - ca_indices
        c_ref_vectors=c_indices - ca_indices
        N_ref = np.array([1.45597958, 0.0, 0.0],dtype=np.float32)
        C_ref = np.array([-0.533655602, 1.42752619, 0.0],dtype=np.float32)
        ca_dis=np.empty([len(ca_indices)])
        for i,_ in enumerate(ca_indices-1):
            if i==len(ca_indices)-1:
                continue
            ca_dis[i]=np.sum(np.square(ca_indices[i]-ca_indices[i+1]))
        scale_factor=0
        file=os.path.join(output_dir, f"{domain_id}_{tem}/")
        if not os.path.exists(file):
            os.makedirs(file)
        print(tem)
        for i in range(50):
            if os.path.exists(f"{file}{domain_id}_{tem}_{i}.pdb"):
                continue
            tr, rot_mat=inference_fn(
                model,
                torch.tensor(single_repr).float().to(device),
                torch.tensor(pair_repr).float().to(device),
                temperature=torch.tensor([float(tem)]).to(device),
                save_full_state=True,
                use_tqdm=True,
            )
            print(n_indices.shape,ca_indices.shape,c_indices.shape)
            print(len(tr),len(rot_mat[-1].size()))
            print(tr[-1].size(),rot_mat[-1].size())
            write_to_pdb_scaling(seq, tr[-1], rot_mat[-1], f"{file}{domain_id}_{tem}_{i}.pdb",torch.from_numpy(N_ref).to(device),torch.from_numpy(C_ref).to(device),torch.from_numpy(ca_dis),scale_factor)
        # write_to_pdb(seq, tr[-1], rot_mat[-1], f"{file}{domain_id}_{temperature[2]}_{i}.pdb")