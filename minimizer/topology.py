# ref: https://github.com/giacomo-janson/sam2 (Apache-2.0, see LICENSE): tables from sam/data/sequences.py; helpers
# from sam/data/aa_topology.py and sam/utils.py, using these tables in place of the identical OpenFold ones
import mdtraj
import numpy as np
import torch

aa_three_letters = ["GLN", "TRP", "GLU", "ARG", "THR",
                    "TYR", "ILE", "PRO", "ALA", "SER",
                    "ASP", "PHE", "GLY", "HIS", "LYS",
                    "LEU", "CYS", "VAL", "ASN", "MET"]

aa_one_to_three_dict = {
    "G": "GLY", "A": "ALA", "L": "LEU", "I": "ILE", "R": "ARG", "K": "LYS",
    "M": "MET", "C": "CYS", "Y": "TYR", "T": "THR", "P": "PRO", "S": "SER",
    "W": "TRP", "D": "ASP", "E": "GLU", "N": "ASN", "Q": "GLN", "F": "PHE",
    "H": "HIS", "V": "VAL", "X": "XXX"
}

# Data from OpenFold.
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

restype_atom14_mask = np.array(
    [[1.0 if n else 0.0 for n in ofo_restype_name_to_atom14_names[aa_one_to_three_dict[r]]] for r in ofo_restypes]
    + [[0.0] * 14]
)


def print_msg(msg, verbose=True, tag="verbose"):
    if verbose:
        print(f"[{tag}]:", msg)


def _get_a(structure: dict, a: torch.Tensor = None):
    if "a" in structure:
        _a = structure["a"]
    else:
        _a = a
    if _a is None:
        raise TypeError()
    return _a


def get_atom14_topology(sequence: str, verbose: bool = False):
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for i, si in enumerate(sequence):
        if verbose:
            print("---")
        res_obj = topology.add_residue(aa_one_to_three_dict[si], chain)
        ai = ofo_restype_name_to_atom14_names[aa_one_to_three_dict[si]]
        for j, aj in enumerate(ai):
            if aj:
                if verbose:
                    print(aj)
                if aj.startswith("C"):
                    elem = mdtraj.core.topology.elem.carbon
                elif aj.startswith("O"):
                    elem = mdtraj.core.topology.elem.oxygen
                elif aj.startswith("N"):
                    elem = mdtraj.core.topology.elem.nitrogen
                elif aj.startswith("S"):
                    elem = mdtraj.core.topology.elem.sulfur
                else:
                    raise NotImplementedError()
                topology.add_atom(
                    aj, elem, res_obj)
    return topology


def get_atom14_mask(aa, device=None):
    atom14_ids = []
    for res in ofo_restypes:
        vals_i = []
        for a in ofo_restype_name_to_atom14_names[aa_one_to_three_dict[res]]:
            if a:
                vals_i.append(1)
            else:
                vals_i.append(0)
        atom14_ids.append(vals_i)
    atom14_ids = torch.tensor(
        atom14_ids,
        dtype=torch.long,
        device=device if device is not None else aa.device
    )
    atom14_atom_exists = atom14_ids[aa]
    return atom14_atom_exists


def get_traj_list(
        structure: dict,
        a: torch.Tensor = None,
        verbose: bool = False,
        join: bool = False
    ):

    _a = _get_a(structure, a).cpu()

    mask = restype_atom14_mask[_a]

    traj_l = []
    for i in range(structure["positions"].shape[0]):

        one_letter_seq_i = [ofo_restypes[j] for j in _a[i]]

        xyz_atom14_i = structure["positions"][i]
        bool_mask_i = mask[i].astype(bool)
        bool_mask_i = torch.Tensor(bool_mask_i) == 1.0

        xyz_traj_i = xyz_atom14_i[bool_mask_i]

        topology_i = get_atom14_topology(one_letter_seq_i, verbose=verbose)

        traj_i = mdtraj.Trajectory(xyz=xyz_traj_i.detach().cpu().numpy()*0.1,
                                   topology=topology_i)
        if verbose:
            print(traj_i)
        traj_l.append(traj_i)

    if not join:
        return traj_l
    else:
        return mdtraj.join(traj_l)


def load_atom14(top_fp: str, ens_fp: str):
    """Ensemble -> atom14 positions (frames, L, 14, 3) in Angstrom and OpenFold residue types (L,), as aSAM's
    AllAtomProteinDataset reads it for the minimizer: heavy atoms of the 20 standard residues, atoms placed by
    name, absent atoms at the origin."""
    traj = mdtraj.load(ens_fp, top=top_fp)
    keep = [a.index for a in traj.topology.atoms if a.element.symbol != "H" and a.residue.name in aa_three_letters]
    traj = traj.atom_slice(keep)
    xyz = traj.xyz * 10.0
    residues = list(traj.topology.residues)
    positions = np.zeros((traj.n_frames, len(residues), 14, 3), dtype=np.float32)
    for m, res in enumerate(residues):
        names = ofo_restype_name_to_atom14_names[aa_one_to_three_dict[res.code]]
        for atom in res.atoms:
            if atom.name in names:
                positions[:, m, names.index(atom.name)] = xyz[:, atom.index]
    aatype = torch.tensor([ofo_restypes.index(res.code) for res in residues], dtype=torch.long)
    return torch.from_numpy(positions), aatype
