# ref: https://github.com/microsoft/Graphormer.git
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import geometry, so3, torus
from . import residue_geometry as G
from .positional_encoding import RelativePositionBias
from .structure_module import StructureModule


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.dummy = nn.Parameter(torch.empty(0, dtype=torch.float), requires_grad=False)  # to detect fp16

    def forward(self, time):
        half_dim = self.dim // 2
        freq = torch.exp(torch.arange(half_dim, device=time.device) * -(math.log(self.max_period) / (half_dim - 1)))
        emb = time[:, None].float() * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1).to(self.dummy.dtype)


class TemperatureEmbedding(nn.Module):
    """Normalised temperature -> MLP, as aSAMt's 'scaler' encoding. No Lipschitz cap."""

    def __init__(self, out_dim, center=385.0, scale=50.0):
        super().__init__()
        self.center, self.scale = center, scale
        self.mlp = nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, temperature):
        return self.mlp(((temperature.float() - self.center) / self.scale)[:, None])


class MainModel(nn.Module):
    def __init__(self, input_shape=256, pair_shape=128, d_model=768, d_pair=256, n_layer=12, n_heads=32,
                 dim_feedforward=1024, dropout=0.0, n_template_bins=64, template_max_dist=22.0,
                 aux_max_sigma=2.0, w_fape=1.0, w_torsion=1.0, w_bond=1.0, w_chi=1.0, temp_smooth_weight=0.0,
                 pair_rbf_bins=0, pair_rbf_max=0.6):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.tr_sigma_min, self.tr_sigma_max = 0.1, 35.0
        self.rot_sigma_min, self.rot_sigma_max = 0.02, 1.65
        self.chi_sigma_min, self.chi_sigma_max = 0.0314, math.pi     # sigma_max = pi: close to uniform on the circle
        self.t_min = 1e-3
        self.aux_max_sigma, self.w_fape, self.w_torsion, self.w_bond = aux_max_sigma, w_fape, w_torsion, w_bond
        self.w_chi, self.temp_smooth_weight = w_chi, temp_smooth_weight

        self.x1d_proj = nn.Sequential(nn.LayerNorm(input_shape), nn.Linear(input_shape, d_model, bias=False))
        self.aatype_emb = nn.Embedding(21, d_model)
        self.chi_proj = nn.Linear(12, d_model)                        # noisy chi state: (sin, cos, present) x 4
        self.pair_rbf_bins, self.pair_rbf_max = pair_rbf_bins, pair_rbf_max
        if pair_rbf_bins:
            self.register_buffer("pair_centers", torch.linspace(0.0, pair_rbf_max, pair_rbf_bins), persistent=False)
            self.pair_width = pair_rbf_max / (pair_rbf_bins - 1)
            self.x2d_proj = nn.Linear(pair_shape * pair_rbf_bins, d_pair, bias=False)
        else:
            self.x2d_proj = nn.Sequential(nn.LayerNorm(pair_shape), nn.Linear(pair_shape, d_pair, bias=False))
        self.rp_proj = RelativePositionBias(num_buckets=64, max_distance=128, out_dim=d_pair)
        self.register_buffer("template_centers", torch.linspace(0.0, template_max_dist, n_template_bins), persistent=False)
        self.template_width = template_max_dist / (n_template_bins - 1)
        self.template_proj = nn.Linear(n_template_bins + 1, d_pair)

        self.time_emb = nn.Sequential(SinusoidalPositionEmbeddings(d_model), nn.Linear(d_model, d_model), nn.SiLU(),
                                      nn.Linear(d_model, d_model))
        self.temp_emb = TemperatureEmbedding(d_model)
        self.cond_to_x1d = nn.Linear(d_model, d_model)
        self.st_module = StructureModule(d_model=d_model, n_layer=n_layer, d_cond=d_model, d_pair=d_pair,
                                         n_head=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)

    # ------------------------------------------------------------ schedule

    def sigmas(self, t):
        tr = self.tr_sigma_min ** (1 - t) * self.tr_sigma_max ** t
        rot = self.rot_sigma_min ** (1 - t) * self.rot_sigma_max ** t
        return tr, rot

    def chi_sigma(self, t):
        return self.chi_sigma_min ** (1 - t) * self.chi_sigma_max ** t

    # ------------------------------------------------------------ chi angles

    @staticmethod
    def chi_factor(aatype):
        """2 pi / period for chi1..4: 2 for pi-periodic chis (ASP chi2, GLU chi3, PHE/TYR chi2), else 1."""
        return torus.chi_period_factor(G.table("chi_pi_periodic", aatype.device)[aatype])

    def chi_mask(self, aatype, res_mask):
        """Chis that exist for each residue type (B, L, 4)."""
        return G.table("chi_angles_mask", aatype.device)[aatype] * res_mask[..., None].float()

    def chi_to_u(self, sin_cos_chi, aatype):
        """(..., 4, 2) true chi (sin, cos) -> diffusion coordinate u = chi * 2 pi / period in [-pi, pi)."""
        return torus.wrap(torch.atan2(sin_cos_chi[..., 0], sin_cos_chi[..., 1]) * self.chi_factor(aatype))

    def u_to_sin_cos(self, u, aatype):
        chi = u / self.chi_factor(aatype)
        return torch.stack([torch.sin(chi), torch.cos(chi)], dim=-1)

    def sample_t(self, B, device):
        """Continuous antithetic times: each u is paired with 1 - u."""
        u = torch.rand((B + 1) // 2, device=device)
        return self.t_min + (1 - self.t_min) * torch.cat([u, 1 - u])[:B]

    # ------------------------------------------------------------ network

    def conditioning(self, t, temperature):
        return self.time_emb(t * 1000.0) + self.temp_emb(temperature)

    def encode(self, batch):
        """Sequence and pair features that do not depend on the noisy structure (reused across sampling steps)."""
        res_mask = batch["res_mask"]
        single, pair = batch["single"], batch["pair"]
        if not torch.is_autocast_enabled():                  # fp16 embeddings need autocast; else compute in fp32
            single, pair = single.float(), pair.float()
        x1d = self.x1d_proj(single) + self.aatype_emb(batch["aatype"])
        if self.pair_rbf_bins:                              # in fp32: bf16 resolves DI only to ~0.002
            v = pair.float().clamp(0.0, self.pair_rbf_max)[..., None]
            pair = torch.exp(-(((v - self.pair_centers) / self.pair_width) ** 2)).flatten(-2)
        x2d = self.x2d_proj(pair)
        L = x1d.shape[1]
        pos = torch.arange(L, device=x1d.device)
        x2d = x2d + self.rp_proj(pos[:, None] - pos[None, :])[None]
        ca, tm = batch["template_ca"], batch["template_mask"].to(x1d.dtype)
        d = torch.cdist(ca.float(), ca.float())
        pm = (tm[:, :, None] * tm[:, None, :])[..., None]
        rbf = torch.exp(-(((d[..., None] - self.template_centers) / self.template_width) ** 2)) * pm
        x2d = x2d + self.template_proj(torch.cat([rbf, pm], dim=-1).to(x2d.dtype))
        bias = torch.zeros(res_mask.shape, device=x1d.device).masked_fill(~res_mask, float("-inf"))[:, None, None, :]
        return {"x1d": x1d, "x2d": x2d, "bias": bias, "aatype": batch["aatype"], "res_mask": res_mask}

    def forward_step(self, T, IR, chi_u, chi_mask, t, temperature, feats):
        """Network outputs for noisy frames (T, IR) and noisy chis chi_u (B, L, 4) at times t.
        IR rows are the local axes (IR = R^T)."""
        c = self.conditioning(t, temperature)
        chi_feat = torch.cat([torch.sin(chi_u) * chi_mask, torch.cos(chi_u) * chi_mask, chi_mask], dim=-1)
        x1d = feats["x1d"] + self.cond_to_x1d(c)[:, None, :] + self.chi_proj(chi_feat.to(feats["x1d"].dtype))
        T_eps, IR_eps, q_bcd, trans, torsions, chi_eps = self.st_module((T, IR), x1d, feats["x2d"], feats["bias"], c)
        # frame math in fp32: under bf16 autocast a matmul returns bf16 (~0.1-0.25 A resolution at 30 A)
        with torch.autocast(device_type=T.device.type, enabled=False):
            R = IR.float().transpose(-1, -2)                                       # local -> global
            T_eps = (R @ T_eps.float().unsqueeze(-1)).squeeze(-1)                   # translation score, global
            R0 = R @ G.quaternion_update(q_bcd.float())
            T0 = T.float() + (R @ trans.float().unsqueeze(-1)).squeeze(-1)
        return {"T_eps": T_eps, "IR_eps": IR_eps.float(), "R0": R0, "T0": T0, "torsions": torsions.float(),
                "chi_eps": chi_eps.float()}

    def full_atoms(self, R0, T0, torsions, aatype, chi_u=None):
        """atom14 from frames and torsions. With chi_u (the diffused chis), those replace the torsion head's chis."""
        sin_cos = F.normalize(torsions, dim=-1)
        if chi_u is not None:
            sin_cos = torch.cat([sin_cos[..., :3, :], self.u_to_sin_cos(chi_u, aatype)], dim=-2)
        return G.atom14_from_frames_and_torsions(R0, T0, sin_cos, aatype)

    # ------------------------------------------------------------ noise

    def _rotation_noise(self, rot_sigma, B, L, device):
        """IGSO(3) perturbations, their scores and score norms (numpy lookup tables in so3)."""
        vec, mats, score, norm = [], [], [], []
        for b in range(B):
            eps = float(rot_sigma[b])
            v = so3.batch_sample_vec(L, eps=eps)
            vec.append(torch.tensor(v, dtype=torch.float32))
            score.append(torch.tensor(so3.batch_score_vec(L, vec=v, eps=eps), dtype=torch.float32))
            norm.append(so3.score_norm(torch.tensor([eps])).float())
        vec = torch.stack(vec).to(device)
        mats = geometry.axis_angle_to_matrix(vec.reshape(-1, 3)).reshape(B, L, 3, 3)
        return mats, torch.stack(score).to(device), torch.stack(norm).to(device).reshape(B, 1, 1)

    # ------------------------------------------------------------ training loss

    def forward(self, batch):
        atom14, atom14_mask, aatype = batch["atom14"].float(), batch["atom14_mask"].float(), batch["aatype"]
        res_mask, bonded = batch["res_mask"], batch["bonded"]
        B, L = aatype.shape
        device = atom14.device
        m = res_mask.float()

        fp32 = lambda: torch.autocast(device_type=device.type, enabled=False)
        with fp32():                                              # ground truth and noising in fp32 (see forward_step)
            R_true, T_true = G.backbone_frames(atom14)
            IR_true = R_true.transpose(-1, -2)
            prev_bonded = torch.cat([torch.zeros_like(bonded[:, :1]), bonded[:, :-1]], dim=1)
            tors_true, tors_alt, tors_mask = G.atom14_torsions(atom14, atom14_mask, aatype, prev_bonded)
            t = self.sample_t(B, device)
            tr_sigma, rot_sigma = self.sigmas(t)
            T_update = torch.randn(B, L, 3, device=device) * tr_sigma[:, None, None]
            rot_mats, rot_score, rot_norm = self._rotation_noise(rot_sigma, B, L, device)
            T_t = (T_true + T_update) * m[..., None]
            IR_t = (rot_mats @ IR_true) * m[..., None, None]
            # side chains: wrapped-normal noise on u = chi * 2 pi / period, only where the chi exists and its atoms do
            chi_m = self.chi_mask(aatype, res_mask) * tors_mask[..., 3:]
            chi_sigma = self.chi_sigma(t)[:, None, None]
            chi_noise = torch.randn(B, L, 4, device=device) * chi_sigma
            chi_u = torus.wrap(self.chi_to_u(tors_true[..., 3:, :], aatype) + chi_noise) * chi_m
            chi_target = torus.score(chi_noise, chi_sigma) / torus.score_rms(chi_sigma)

        out = self.forward_step(T_t, IR_t, chi_u, chi_m, t, batch["temperature"], self.encode(batch))

        # score matching; targets as in the original model: sigma * score (translations), score / norm (rotations),
        # and for chis the wrapped-normal score / its RMS
        per_res = lambda e: (e.sum(-1) * m).sum(-1) / (e.shape[-1] * m.sum(-1))   # each protein: mean over its residues
        loss_T = per_res((out["T_eps"] + T_update / tr_sigma[:, None, None]) ** 2)
        loss_IR = per_res((out["IR_eps"] - rot_score / rot_norm) ** 2)
        n_chi = chi_m.sum((1, 2))
        loss_chi = (((out["chi_eps"] - chi_target) ** 2) * chi_m).sum((1, 2)) / n_chi.clamp(min=1.0)
        has_chi = (n_chi > 0).float()                                              # e.g. an all-GLY/ALA crop
        loss_chi = (loss_chi * has_chi).sum() / has_chi.sum().clamp(min=1.0)

        # clean-structure losses at low noise
        with fp32():
            aux = (tr_sigma < self.aux_max_sigma).float()
            x_pred = self.full_atoms(out["R0"], out["T0"], out["torsions"], aatype)
            x_true = G.rename_symmetric(x_pred, atom14, R_true, T_true, out["R0"], out["T0"], aatype)
            fape = G.fape(out["R0"], out["T0"], x_pred.reshape(B, -1, 3), R_true, T_true, x_true.reshape(B, -1, 3),
                          m, atom14_mask.reshape(B, -1))
            tors = G.torsion_loss(out["torsions"], tors_true, tors_alt, tors_mask * m[..., None])
            bond = G.backbone_bond_loss(x_pred, bonded & res_mask)
            n_aux = aux.sum().clamp(min=1.0)
            aux_loss = (aux * (self.w_fape * fape + self.w_torsion * tors + self.w_bond * bond)).sum() / n_aux

        loss = loss_T.mean() + loss_IR.mean() + self.w_chi * loss_chi + aux_loss
        smooth = torch.zeros((), device=device)
        if self.temp_smooth_weight > 0:
            Ts = torch.empty(B, device=device).uniform_(300.0, 470.0)
            e = [self.temp_emb(Ts + d) for d in (-5.0, 0.0, 5.0)]
            smooth = ((e[0] - 2 * e[1] + e[2]) ** 2).sum(-1).mean()
            loss = loss + self.temp_smooth_weight * smooth
        return {"loss": loss, "T_diff_loss": loss_T.mean(), "IR_diff_loss": loss_IR.mean(), "chi_diff_loss": loss_chi,
                "fape": (aux * fape).sum() / n_aux, "torsion": (aux * tors).sum() / n_aux,
                "bond": (aux * bond).sum() / n_aux, "temp_smooth": smooth, "aux_fraction": aux.mean()}
