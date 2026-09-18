import os
import numpy as np
import mdtraj as md
from Bio.PDB import PDBParser
from tqdm import tqdm
import pickle

aa_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V","HSD":"H","HSE":"H","HSP":"H"}
def _drop_overlapping_atoms(traj, decimals=6):
    xyz = np.round(traj.xyz[0], decimals=decimals)
    _, keep_idx = np.unique(xyz, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    if len(keep_idx) != traj.n_atoms:
        traj = traj.atom_slice(keep_idx)
        return False
    return True

def build_aligned_structure_data(pdb_path, cutoff=0.2):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    traj = md.load(pdb_path)
    if _drop_overlapping_atoms(traj):
        protein_atom_idx = traj.topology.select("protein")
        traj_protein = traj.atom_slice(protein_atom_idx)

        sasa = md.shrake_rupley(traj_protein, mode="residue")[0]

        residues = []
        seq = []
        bfactor = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):
                        residues.append((chain.id, residue.id))
                        seq.append(aa_3to1[residue.get_resname()])
                        bfactor.append(residue["CA"].get_bfactor())

        seq = "".join(seq)
        bfactor = np.asarray(bfactor, dtype=float)

        # print(len(seq), len(bfactor), len(sasa))
        n = min(len(seq), len(bfactor), len(sasa))
        return {
            "seq": seq[:n],
            "bfactor": bfactor[:n],
            "sasa": sasa[:n],
            "core_mask": (sasa[:n] <= cutoff),
            "surface_mask": (sasa[:n] > cutoff),
            "residue_ids": residues[:n],
        }
    else:
        return {}

b_factor_paths = "/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/Bfactor/pdbs/"
aligned_data = {}

for fname in tqdm(os.listdir(b_factor_paths)):
    pdb_id = fname[:4]
    pdb_path = os.path.join(b_factor_paths, fname)
    try:
        temp_val=build_aligned_structure_data(pdb_path, cutoff=0.2)
        if len(temp_val)==0:
            print(pdb_id)
            continue
        else:
            aligned_data[pdb_id] = temp_val
    except Exception as e:
        print(f"Error processing {fname}: {e}")
pickle.dump(aligned_data, open("/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/Bfactor/sasa_vars.pkl", "wb"))
