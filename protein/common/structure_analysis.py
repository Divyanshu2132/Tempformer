"""
Analyze different types of features of mdtraj trajectories.
"""

ofo_restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "ND2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "ASP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "CD1",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LEU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LYS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
        "",
        "",
        "",
        "",
        "",
    ],
    "MET": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

ofo_restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

import os
import glob
import h5py
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

# ----------------------------
# paths
# ----------------------------
DATA_DIR = "/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/case_study/melting_temperature/"
H5_DIR = os.path.join(DATA_DIR, "data")
GEN_DIR = os.path.join(DATA_DIR, "generated_data")


def calc_mdtraj_rmsf(
        traj: md.Trajectory,
        ref_traj: md.Trajectory,
        ref_index: int = 0
    ) -> np.ndarray:
    traj_c = md.Trajectory(traj.xyz, topology=traj.topology)
    ref_traj = md.Trajectory(ref_traj.xyz, topology=traj.topology)
    rmsf = md.rmsf(traj_c, ref_traj, ref_index)
    return rmsf


std_atoms = set()
for k in ofo_restype_name_to_atom14_names:
    for a in ofo_restype_name_to_atom14_names[k]:
        if a:
            std_atoms.add(a)


def calc_q_values(
        traj,
        native_traj,
        beta=50.0,
        lambda_=1.2,
        delta=0.0,
        threshold=1.0  # in nanometers.
    ):

    if len(native_traj) != 1:
        raise NotImplementedError()
    
    dist = []
    top_atoms_dict = [[a for a in r.atoms if a.name in std_atoms] for r in native_traj.topology.residues]
    
    top_ids = []
    traj_ids = []
    top_residues = list(native_traj.topology.residues)  #
    top_atoms = list(native_traj.topology.atoms)        #
    traj_residues = list(traj.topology.residues)
    traj_atoms = list(traj.topology.atoms)           #
    for i, r_i in enumerate(native_traj.topology.residues):
        for j, r_j in enumerate(native_traj.topology.residues):
            if r_j.index - r_i.index > 3:
                a_k_ids = [a_k.index for a_k in top_atoms_dict[i]]
                a_k_atoms = top_atoms_dict[i]
                a_l_ids = [a_l.index for a_l in top_atoms_dict[j]]
                a_l_atoms = top_atoms_dict[j]
                dist_ij = np.sqrt(
                    np.sum(
                        np.square(
                            native_traj.xyz[:,a_k_ids,None,:] - native_traj.xyz[:,None,a_l_ids,:]
                        ),
                    axis=-1
                    )
                )[0]
                a_k_pos, a_l_pos = np.unravel_index(dist_ij.argmax(), dist_ij.shape)
                if threshold is not None:
                    if dist_ij[a_k_pos, a_l_pos] > threshold:
                        continue
                top_ids.append((a_k_ids[a_k_pos], a_l_ids[a_l_pos]))
                traj_ids.append(
                    (traj_residues[i].atom(a_k_atoms[a_k_pos].name).index,
                     traj_residues[j].atom(a_l_atoms[a_l_pos].name).index)
                )
                
    if not traj_ids:
        raise ValueError()
    ref_dist = md.compute_distances(native_traj, top_ids)
    traj_dist = md.compute_distances(traj, traj_ids)
    n_contacts = ref_dist.shape[1]
    q_x = np.sum(1/(1+np.exp(beta*(traj_dist - lambda_*(ref_dist + delta)))), axis=1)/n_contacts
    return q_x


def calc_rmsd(hat_traj, ref_traj, ref_idx=0, prealigned=True, get_tm=False):
    ref_traj = ref_traj[ref_idx:ref_idx+1]
    if not prealigned:
        raise NotImplementedError()
    ref_xyz = ref_traj.xyz * 10.0
    hat_xyz = hat_traj.xyz * 10.0
    sq_dev = np.sum(np.square(ref_xyz - hat_xyz), axis=-1)
    rsq_dev = np.sqrt(sq_dev)
    n_res = ref_xyz.shape[1]
    rmsd = np.sqrt(sq_dev.sum(axis=1)/n_res)
    tm_score = (1 / n_res) * np.sum(1 / (1 + (rsq_dev/d_0(n_res))**2), axis=1)
    if not get_tm:
        return rmsd*0.1
    else:
        return tm_score, rmsd*0.1

def d_0(n_res):
    return 1.24*(n_res - 15)**(1/3) - 1.8
    

def calc_ssep(traj, native_traj):
    if len(native_traj) > 1:
        raise ValueError()
    ref_dssp = md.compute_dssp(native_traj)
    traj_dssp = md.compute_dssp(traj)
    match = ref_dssp == traj_dssp
    match = match.astype(int)
    score = match.mean(axis=1)
    he_mask = np.isin(ref_dssp, ['H', 'E']).astype(int)
    match_mask = match * he_mask
    score_mask = match_mask.sum(axis=1)/he_mask.sum()
    return score_mask

def _calc_distances(xyz, eps=1e-9):
    d = np.sqrt(
        np.sum(np.square(xyz[:,:,None,:] - xyz[:,None,:,:]), axis=3)+eps
    )
    return d

# ----------------------------
# helpers
# ----------------------------
def to_str_array(arr):
    """Convert bytes/object/string arrays to plain string numpy array."""
    arr = np.asarray(arr)
    if arr.dtype.kind in {"S", "O"}:
        return arr.astype("U")
    return arr

def alpha_helix_frequency_from_dssp(dssp_array):
    """
    dssp_array: shape (n_frames, L)
    Returns: shape (L,), frequency of alpha helix state H at each residue.
    """
    dssp_array = to_str_array(dssp_array)
    return (dssp_array == "H").mean(axis=0)

def load_mdcath_alpha_profile(domain_id, temp):
    """
    Average alpha-helix frequency over all replicas and all frames
    from the stored HDF5 DSSP arrays.
    """
    h5_path = os.path.join(H5_DIR, f"mdcath_dataset_{domain_id}.h5")
    with h5py.File(h5_path, "r") as f:
        grp = f[domain_id][str(temp)]
        replica_profiles = []

        for repl in grp.keys():
            if repl.isdigit():
                dssp = grp[repl]["dssp"][:]          # (n_frames, L)
                replica_profiles.append(alpha_helix_frequency_from_dssp(dssp))

    return np.mean(replica_profiles, axis=0)

def load_generated_alpha_profile(domain_id, temp):
    """
    Compute DSSP from generated all-atom PDBs and average over samples.
    Important: use *_all_atom.pdb, not the backbone-only .pdb files.
    """
    folder = os.path.join(GEN_DIR, f"{domain_id}_{temp}")
    pdb_files = sorted(glob.glob(os.path.join(folder, "*_minimized.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No all-atom generated PDBs found in {folder}")

    sample_profiles = []
    for pdb_path in pdb_files:
        traj = md.load(pdb_path)
        dssp = md.compute_dssp(traj, simplified=True)   # shape (1, L), values H/E/C
        sample_profiles.append(alpha_helix_frequency_from_dssp(dssp))

    return np.mean(sample_profiles, axis=0)

def plot_generated_alpha_profiles(domain_id, temps=("320", "348", "379", "413", "450"), compare_temp="450", compare_temp1="320",colors=[]):
    temps = [str(t) for t in temps]
    compare_temp = str(compare_temp)
    compare_temp1 = str(compare_temp1)

    generated_profiles = {}

    for temp in temps:
        generated_profiles[temp] = load_generated_alpha_profile(domain_id, temp)

    L = len(generated_profiles[temps[0]])
    x = np.arange(L)

    plt.figure(figsize=(8, 6))

    # left: generated profiles across temperatures
    for temp in temps:
        if len(colors) > 0:
            plt.plot(x, generated_profiles[temp], label=temp, lw=1.5, color=colors.pop(0))
        else:
            plt.plot(x, generated_profiles[temp], label=temp, lw=1.5)
    plt.title(r"$\alpha$ helix profile (Generated)"+f"{domain_id}")
    plt.ylabel(r"$\alpha$ helix frequency")
    plt.xlabel("Residue index")
    plt.legend()
# ----------------------------
# plotting
# ----------------------------
def plot_alpha_profiles(domain_id, temps=("320", "348", "379", "413", "450"), compare_temp="450", compare_temp1="320",generated=False):
    temps = [str(t) for t in temps]
    compare_temp = str(compare_temp)
    compare_temp1 = str(compare_temp1)

    native_profiles = {}
    generated_profiles = {}

    for temp in temps:
        native_profiles[temp] = load_mdcath_alpha_profile(domain_id, temp)

    L = len(native_profiles[temps[0]])
    x = np.arange(L)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)

    # top-left: MDCATH native profiles across temperatures
    ax = axes[0, 0]
    for temp in temps:
        ax.plot(x, native_profiles[temp], label=temp, lw=1.5)
    ax.set_title(r"$\alpha$ helix profile (MDCATH)")
    ax.set_ylabel(r"$\alpha$ helix frequency")
    ax.set_xlabel("Residue index")
    ax.legend(fontsize=8)

    # bottom-left: generated profiles across temperatures
    if generated:
        for temp in temps:
            generated_profiles[temp] = load_generated_alpha_profile(domain_id, temp)
        ax = axes[1, 0]
        for temp in temps:
            ax.plot(x, generated_profiles[temp], label=temp, lw=1.5)
        ax.set_title(r"$\alpha$ helix profile (Generated)")
        ax.set_ylabel(r"$\alpha$ helix frequency")
        ax.set_xlabel("Residue index")
        ax.legend(fontsize=8)

        # top-right: comparison at one temperature
        ax = axes[0, 1]
        ax.plot(x, native_profiles[compare_temp1], label=f"MDCATH({compare_temp1} K)", lw=1.5)
        ax.plot(x, generated_profiles[compare_temp1], label=f"Generated({compare_temp1} K)", lw=1.5)
        ax.set_title(r"$\alpha$ helix profile")
        ax.set_ylabel(r"$\alpha$ helix frequency")
        ax.set_xlabel("Residue index")
        ax.legend(fontsize=8)

        # bottom-right: zoom on low-frequency region if desired
        ax = axes[1, 1]
        ax.plot(x, native_profiles[compare_temp], label=f"MDCATH({compare_temp} K)", lw=1.5)
        ax.plot(x, generated_profiles[compare_temp], label=f"Generated({compare_temp} K)", lw=1.5)
        ax.set_title(r"$\alpha$ helix profile")
        ax.set_ylabel(r"$\alpha$ helix frequency")
        ax.set_xlabel("Residue index")
        ax.set_ylim(0, 0.6)   # remove this line if you want full 0-1 scale
        ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_ylim(0, 1.1)

    # keep the zoom panel actually zoomed
    axes[1, 1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()
