import math
from pathlib import Path

import numpy as np
import torch

_C = np.load(Path(__file__).with_name("residue_constants.npz"))
RESTYPES = [str(r) for r in _C["restypes"]]          # 20 amino acids + "X", AlphaFold order
RESNAMES = [str(r) for r in _C["resnames"]]          # three-letter names
ATOM14_NAMES = _C["atom14_names"]                    # (21, 14), "" for unused slots
UNK = RESTYPES.index("X")
# mdCATH (CHARMM) residue and atom names that differ from PDB
RESNAME_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HIE": "HIS", "HID": "HIS", "HIP": "HIS"}
ATOM_ALIASES = {("ILE", "CD"): "CD1", ("*", "OT1"): "O"}

_TABLES = {"rigid_group_default_frame": torch.float32, "atom14_to_rigid_group": torch.long,
           "atom14_mask": torch.float32, "atom14_rigid_group_positions": torch.float32,
           "chi_angles_atom14": torch.long, "chi_angles_mask": torch.float32,
           "chi_pi_periodic": torch.float32, "atom14_swap": torch.long}
_cache = {}


def table(name, device):
    key = (name, str(device))
    if key not in _cache:
        _cache[key] = torch.as_tensor(_C[name], dtype=_TABLES[name], device=device)
    return _cache[key]


def resname_to_index(resname):
    resname = RESNAME_ALIASES.get(resname, resname)
    return RESNAMES.index(resname) if resname in RESNAMES else UNK


def atom14_slot(restype, atom_name):
    """atom14 slot of a (PDB or CHARMM named) heavy atom, or None."""
    res3 = RESNAMES[restype]
    atom_name = ATOM_ALIASES.get((res3, atom_name), ATOM_ALIASES.get(("*", atom_name), atom_name))
    names = list(ATOM14_NAMES[restype])
    return names.index(atom_name) if atom_name and atom_name in names else None


# ---------------------------------------------------------------- frames

def _normalize(v, eps=1e-8):
    return v / torch.sqrt((v * v).sum(-1, keepdim=True) + eps)


def frames_from_3_points(p_neg_x, origin, p_xy):
    """OpenFold Rigid.from_3_points: x from p_neg_x towards origin, p_xy in the xy-plane."""
    e0 = _normalize(origin - p_neg_x)
    e1 = p_xy - origin
    e1 = _normalize(e1 - (e0 * e1).sum(-1, keepdim=True) * e0)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-1), origin


def backbone_frames(atom14):
    """AlphaFold backbone frames from atom14 N, CA, C: x along CA->C, N in the xy-plane."""
    N, CA, C = atom14[..., 0, :], atom14[..., 1, :], atom14[..., 2, :]
    e0 = _normalize(C - CA)
    e1 = N - CA
    e1 = _normalize(e1 - (e0 * e1).sum(-1, keepdim=True) * e0)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-1), CA


def compose(R1, t1, R2, t2):
    return R1 @ R2, (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1


def apply(R, t, x):
    return (R @ x.unsqueeze(-1)).squeeze(-1) + t


def invert_apply(R, t, x):
    return (R.transpose(-1, -2) @ (x - t).unsqueeze(-1)).squeeze(-1)


def quaternion_update(q_bcd):
    """AlphaFold backbone update: quaternion (1, b, c, d), normalised, as a rotation matrix."""
    q = torch.cat([torch.ones_like(q_bcd[..., :1]), q_bcd], dim=-1)
    a, b, c, d = _normalize(q, eps=1e-12).unbind(-1)
    return torch.stack([
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
        2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b),
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d,
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


# ---------------------------------------------------------------- torsions -> atoms

def torsion_angles_to_frames(R_bb, t_bb, sin_cos, aatype):
    """OpenFold torsion_angles_to_frames. sin_cos (..., L, 7, 2) -> 8 rigid-group frames (..., L, 8)."""
    default = table("rigid_group_default_frame", R_bb.device)[aatype]           # (..., L, 8, 4, 4)
    R_def, t_def = default[..., :3, :3], default[..., :3, 3]
    bb = torch.zeros_like(sin_cos[..., :1, :])
    bb[..., 1] = 1.0
    alpha = torch.cat([bb, sin_cos], dim=-2)                                     # (..., L, 8, 2)
    sin, cos = alpha[..., 0], alpha[..., 1]
    zero, one = torch.zeros_like(sin), torch.ones_like(sin)
    R_rot = torch.stack([one, zero, zero, zero, cos, -sin, zero, sin, cos], dim=-1).reshape(*sin.shape, 3, 3)
    R_all, t_all = R_def @ R_rot, t_def                                          # default frame, then torsion
    # chi2..chi4 frames are defined relative to the previous chi frame
    R_g, t_g = [R_all[..., i, :, :] for i in range(5)], [t_all[..., i, :] for i in range(5)]
    R_prev, t_prev = R_g[4], t_g[4]
    for i in (5, 6, 7):
        R_prev, t_prev = compose(R_prev, t_prev, R_all[..., i, :, :], t_all[..., i, :])
        R_g.append(R_prev)
        t_g.append(t_prev)
    R_g, t_g = torch.stack(R_g, dim=-3), torch.stack(t_g, dim=-2)               # (..., L, 8, 3, 3), (..., L, 8, 3)
    return compose(R_bb.unsqueeze(-3), t_bb.unsqueeze(-2), R_g, t_g)


def frames_to_atom14(R_g, t_g, aatype):
    """OpenFold frames_and_literature_positions_to_atom14_pos: (..., L, 8) frames -> (..., L, 14, 3)."""
    group = table("atom14_to_rigid_group", R_g.device)[aatype]                  # (..., L, 14)
    lit = table("atom14_rigid_group_positions", R_g.device)[aatype]              # (..., L, 14, 3)
    mask = table("atom14_mask", R_g.device)[aatype]
    R_a = torch.gather(R_g, -3, group[..., None, None].expand(*group.shape, 3, 3))
    t_a = torch.gather(t_g, -2, group[..., None].expand(*group.shape, 3))
    return apply(R_a, t_a, lit) * mask[..., None]


def atom14_from_frames_and_torsions(R_bb, t_bb, sin_cos, aatype):
    return frames_to_atom14(*torsion_angles_to_frames(R_bb, t_bb, sin_cos, aatype), aatype)


def atom14_torsions(x, x_mask, aatype, prev_bonded):
    """Ground-truth torsions as in OpenFold atom37_to_torsion_angles, from atom14 positions.
    x (..., L, 14, 3), x_mask (..., L, 14), prev_bonded (..., L): residue i-1 is bonded to i.
    Returns sin_cos (..., L, 7, 2), alt_sin_cos (pi-periodic chis flipped), mask (..., L, 7)."""
    prev_x = torch.cat([torch.zeros_like(x[..., :1, :, :]), x[..., :-1, :, :]], dim=-3)
    prev_m = torch.cat([torch.zeros_like(x_mask[..., :1, :]), x_mask[..., :-1, :]], dim=-2)
    prev_ok = prev_bonded.to(x_mask.dtype)
    pre_omega = torch.stack([prev_x[..., 1, :], prev_x[..., 2, :], x[..., 0, :], x[..., 1, :]], dim=-2)
    phi = torch.stack([prev_x[..., 2, :], x[..., 0, :], x[..., 1, :], x[..., 2, :]], dim=-2)
    psi = torch.stack([x[..., 0, :], x[..., 1, :], x[..., 2, :], x[..., 3, :]], dim=-2)
    m_pre_omega = prev_m[..., 1] * prev_m[..., 2] * x_mask[..., 0] * x_mask[..., 1] * prev_ok
    m_phi = prev_m[..., 2] * x_mask[..., 0] * x_mask[..., 1] * x_mask[..., 2] * prev_ok
    m_psi = x_mask[..., 0] * x_mask[..., 1] * x_mask[..., 2] * x_mask[..., 3]

    chi_idx = table("chi_angles_atom14", x.device)[aatype]                       # (..., L, 4, 4)
    chi_on = table("chi_angles_mask", x.device)[aatype]                          # (..., L, 4)
    idx = chi_idx.clamp(min=0)
    chi_x = torch.gather(x.unsqueeze(-3).expand(*idx.shape[:-1], 14, 3), -2,
                         idx[..., None].expand(*idx.shape, 3))                   # (..., L, 4, 4, 3)
    chi_m = torch.gather(x_mask.unsqueeze(-2).expand(*idx.shape[:-1], 14), -1, idx).prod(-1) * chi_on

    atoms = torch.cat([pre_omega[..., None, :, :], phi[..., None, :, :], psi[..., None, :, :], chi_x], dim=-3)
    mask = torch.cat([m_pre_omega[..., None], m_phi[..., None], m_psi[..., None], chi_m], dim=-1)
    R, t = frames_from_3_points(atoms[..., 1, :], atoms[..., 2, :], atoms[..., 0, :])
    rel = invert_apply(R, t, atoms[..., 3, :])
    sin_cos = _normalize(torch.stack([rel[..., 2], rel[..., 1]], dim=-1))
    sign = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0], device=x.device)   # psi is defined with O
    sin_cos = sin_cos * sign[:, None]
    mirror = torch.cat([torch.ones_like(chi_on[..., :3]), 1.0 - 2.0 * table("chi_pi_periodic", x.device)[aatype]], -1)
    return sin_cos, sin_cos * mirror[..., None], mask


def chi_rotamer_bins(sin_cos, torsion_mask, aatype):
    """Rotamer bins of chi1 and chi2 from torsions (..., L, 7, 2) -> (..., L, 2) int64, -1 where absent.
    chi1: the three staggered wells, g+ [0, 120), t [120, 240), g- [240, 360) degrees.
    chi2: six 60-degree bins of chi2 * 2 pi / period, so pi-periodic chi2 (ASP, PHE, TYR) count chi and chi + pi once."""
    chi = torch.atan2(sin_cos[..., 3:5, 0], sin_cos[..., 3:5, 1])
    factor = 1.0 + table("chi_pi_periodic", aatype.device)[aatype][..., :2]
    ang = torch.remainder(chi * factor, 2 * math.pi)
    bins = torch.stack([(ang[..., 0] / (2 * math.pi / 3)).long().clamp(max=2),
                        (ang[..., 1] / (2 * math.pi / 6)).long().clamp(max=5)], dim=-1)
    present = (torsion_mask[..., 3:5] > 0) & (table("chi_angles_mask", aatype.device)[aatype][..., :2] > 0)
    return torch.where(present, bins, torch.full_like(bins, -1))


def atom14_rotamer_bins(x, aatype, x_mask=None):
    """chi1/chi2 rotamer bins (..., L, 2) straight from atom14 coordinates (e.g. MD frames or samples)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    aatype = torch.as_tensor(aatype).expand(x.shape[:-2])
    if x_mask is None:
        x_mask = table("atom14_mask", x.device)[aatype]
    x_mask = torch.as_tensor(x_mask, dtype=torch.float32).expand(x.shape[:-1])
    no_prev = torch.zeros(x.shape[:-2], dtype=torch.bool)       # chis are intra-residue: no neighbours needed
    sin_cos, _, mask = atom14_torsions(x, x_mask, aatype, no_prev)
    return chi_rotamer_bins(sin_cos, mask, aatype)


# ---------------------------------------------------------------- losses (each returns a per-sample (B,) tensor)

def rename_symmetric(pred_x, true_x, R_true, t_true, R_pred, t_pred, aatype):
    """Per residue, the naming of symmetric atoms (ASP OD1/OD2, GLU OE1/OE2, PHE/TYR ring atoms) that is
    closer to the prediction, compared in each residue's own frame (AlphaFold's renaming, simplified)."""
    swap = table("atom14_swap", true_x.device)[aatype]                          # (B, L, 14)
    alt_x = torch.gather(true_x, -2, swap[..., None].expand(*swap.shape, 3))
    local_p = invert_apply(R_pred[..., None, :, :], t_pred[..., None, :], pred_x)
    d_true = ((invert_apply(R_true[..., None, :, :], t_true[..., None, :], true_x) - local_p) ** 2).sum((-1, -2))
    d_alt = ((invert_apply(R_true[..., None, :, :], t_true[..., None, :], alt_x) - local_p) ** 2).sum((-1, -2))
    return torch.where((d_alt < d_true)[..., None, None], alt_x, true_x)


def fape(R_p, t_p, x_p, R_t, t_t, x_t, frame_mask, atom_mask, clamp=10.0, scale=10.0, eps=1e-4):
    """Frame-aligned point error of every atom in every frame. frames (B, F), atoms (B, A)."""
    lp = invert_apply(R_p[:, :, None], t_p[:, :, None], x_p[:, None])            # (B, F, A, 3)
    lt = invert_apply(R_t[:, :, None], t_t[:, :, None], x_t[:, None])
    err = torch.sqrt(((lp - lt) ** 2).sum(-1) + eps).clamp(max=clamp)
    w = frame_mask[:, :, None] * atom_mask[:, None, :]
    return (err * w).sum((1, 2)) / (w.sum((1, 2)) + eps) / scale


def torsion_loss(raw, true, alt, mask, norm_weight=0.02, eps=1e-8):
    """AlphaFold torsion loss: nearest of the true and pi-flipped torsions, plus an angle-norm term."""
    pred = _normalize(raw)
    err = torch.minimum(((pred - true) ** 2).sum(-1), ((pred - alt) ** 2).sum(-1))
    norm = (torch.sqrt((raw ** 2).sum(-1) + eps) - 1.0).abs()
    per = mask.sum((-1, -2)) + eps
    return (err * mask).sum((-1, -2)) / per + norm_weight * (norm * mask).sum((-1, -2)) / per


def backbone_bond_loss(x, bonded):
    """Squared error of the C(i)-N(i+1) peptide bond (1.329 A) and CA(i)-CA(i+1) distance (3.80 A).
    bonded (B, L): residue i is bonded to i + 1."""
    m = bonded[:, :-1].to(x.dtype)
    cn = (x[:, :-1, 2] - x[:, 1:, 0]).norm(dim=-1)
    caca = (x[:, :-1, 1] - x[:, 1:, 1]).norm(dim=-1)
    err = (cn - 1.329) ** 2 + (caca - 3.80) ** 2
    return (err * m).sum(-1) / (m.sum(-1) + 1e-8)


# ---------------------------------------------------------------- output

def atom14_to_pdb_lines(x, aatype, mask=None, model=None):
    """PDB ATOM records for one structure. x (L, 14, 3) in Angstrom, aatype (L,)."""
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    aatype = aatype.detach().cpu().numpy() if torch.is_tensor(aatype) else np.asarray(aatype)
    atom_mask = _C["atom14_mask"][aatype] if mask is None else np.asarray(mask)
    lines = [f"MODEL     {model:4d}"] if model is not None else []
    serial = 1
    for i, (res, xyz) in enumerate(zip(aatype, x)):
        for k in range(14):
            if atom_mask[i, k] < 0.5:
                continue
            name = str(ATOM14_NAMES[res, k])
            lines.append(f"ATOM  {serial:5d} {name if len(name) == 4 else ' ' + name:<4s} {RESNAMES[res]:>3s} A"
                         f"{i + 1:4d}    {xyz[k, 0]:8.3f}{xyz[k, 1]:8.3f}{xyz[k, 2]:8.3f}  1.00  0.00"
                         f"          {name[0]:>2s}")
            serial += 1
    lines.append("TER")
    if model is not None:
        lines.append("ENDMDL")
    return lines
