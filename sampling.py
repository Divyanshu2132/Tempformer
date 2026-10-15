import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr

from model import geometry, so3, torus
from model import residue_geometry as G


@torch.no_grad()
def sample(model, feats_batch, temperature, n_samples, n_steps=100, sampler="sde", noise_scale=1.0, seed=None,
           device=None, chi_source="diffused"):
    """feats_batch: a collated batch of ONE protein (keys single, pair, aatype, template_ca, template_mask,
    res_mask). Returns atom14 (n_samples, L, 14, 3) in Angstrom (model frame, centred)."""
    if sampler not in ("sde", "ode"):
        raise ValueError(f"unknown sampler {sampler}")
    if chi_source not in ("diffused", "predicted"):
        raise ValueError(f"unknown chi_source {chi_source}")
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feats = {k: v.to(device).expand(n_samples, *v.shape[1:]) if torch.is_tensor(v) else v
             for k, v in feats_batch.items() if k in ("single", "pair", "aatype", "template_ca", "template_mask", "res_mask")}
    feats["single"], feats["pair"] = feats["single"].float(), feats["pair"].float()
    enc = model.encode(feats)
    L = feats["aatype"].shape[1]
    T = (torch.randn(n_samples, L, 3, generator=gen) * model.tr_sigma_max).to(device)
    IR = torch.from_numpy(Rotation.random(n_samples * L, random_state=rng.integers(2**31)).as_matrix()
                          ).float().reshape(n_samples, L, 3, 3).to(device)
    chi_m = model.chi_mask(feats["aatype"], feats["res_mask"])
    chi_u = (torch.rand(n_samples, L, 4, generator=gen) * 2 * math.pi - math.pi).to(device) * chi_m   # uniform
    temp = torch.full((n_samples,), float(temperature), device=device)
    ts = np.linspace(1.0, model.t_min, n_steps + 1)
    tr_c = np.sqrt(2 * np.log(model.tr_sigma_max / model.tr_sigma_min))
    rot_c = 2 * np.sqrt(np.log(model.rot_sigma_max / model.rot_sigma_min))
    chi_c = np.sqrt(2 * np.log(model.chi_sigma_max / model.chi_sigma_min))
    k = 0.5 if sampler == "ode" else 1.0
    for i in range(n_steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        tr_sigma, rot_sigma = model.sigmas(torch.tensor(t))
        chi_sigma = model.chi_sigma(torch.tensor(t))
        out = model.forward_step(T, IR, chi_u, chi_m, torch.full((n_samples,), t, device=device), temp, enc)
        tr_score = out["T_eps"] / tr_sigma.item()
        rot_score = out["IR_eps"] * so3.score_norm(torch.tensor([rot_sigma.item()]))[0].item()
        chi_score = out["chi_eps"] * torus.score_rms(chi_sigma.reshape(1)).item()
        tr_g, rot_g, chi_g = tr_sigma.item() * tr_c, rot_sigma.item() * rot_c, chi_sigma.item() * chi_c
        tr_step = k * tr_g ** 2 * dt * tr_score
        rot_step = k * rot_g ** 2 * dt * rot_score
        chi_step = k * chi_g ** 2 * dt * chi_score
        if sampler == "sde" and i < n_steps - 1:
            tr_step = tr_step + noise_scale * tr_g * np.sqrt(dt) * torch.randn(T.shape, generator=gen).to(device)
            rot_step = rot_step + noise_scale * rot_g * np.sqrt(dt) * torch.randn(rot_step.shape, generator=gen).to(device)
            chi_step = chi_step + noise_scale * chi_g * np.sqrt(dt) * torch.randn(chi_step.shape, generator=gen).to(device)
        T = T + tr_step
        IR = geometry.axis_angle_to_matrix(rot_step.reshape(-1, 3)).reshape(n_samples, L, 3, 3) @ IR
        chi_u = torus.wrap(chi_u + chi_step) * chi_m
    out = model.forward_step(T, IR, chi_u, chi_m, torch.full((n_samples,), model.t_min, device=device), temp, enc)
    atoms = model.full_atoms(out["R0"], out["T0"], out["torsions"], feats["aatype"],
                             chi_u=chi_u if chi_source == "diffused" else None)
    model.train(was_training)
    return atoms


# ---------------------------------------------------------------- metrics

def kabsch_rmsf(xyz):
    """xyz (n, L, 3): superpose every frame on frame 0, RMSF about the mean (like md.rmsf(ca, ca))."""
    ref = xyz[0] - xyz[0].mean(0)
    out = np.empty_like(xyz)
    for k, x in enumerate(xyz):
        x = x - x.mean(0)
        u, _, vt = np.linalg.svd(x.T @ ref)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
        out[k] = x @ (u @ D @ vt)
    return np.sqrt(((out - out.mean(0)) ** 2).sum(-1).mean(0))


def native_contact_fraction(ca_frames, native_ca, cutoff=8.0, min_sep=3, tol=1.2):
    ii, jj = np.triu_indices(len(native_ca), k=min_sep)
    d0 = np.linalg.norm(native_ca[ii] - native_ca[jj], axis=-1)
    keep = d0 < cutoff
    d = np.linalg.norm(ca_frames[:, ii[keep]] - ca_frames[:, jj[keep]], axis=-1)
    return float((d < tol * d0[keep]).mean())


def dssp_fractions(atoms, aatype, workdir):
    """3-state DSSP fractions per residue (3, L) of generated full-atom structures, via mdtraj."""
    import mdtraj as md
    path = workdir / f"dssp_{np.random.randint(1 << 30)}.pdb"
    lines = []
    for k, x in enumerate(atoms):
        lines += G.atom14_to_pdb_lines(x, aatype, model=k + 1)
    path.write_text("\n".join(lines + ["END"]))
    try:
        codes = md.compute_dssp(md.load_pdb(str(path)), simplified=True)
    finally:
        path.unlink(missing_ok=True)
    return np.stack([(codes == s).mean(0) for s in "HEC"])


ROTAMER_BINS = (3, 6)          # chi1: g+ / t / g- wells; chi2: six 60-degree bins (see G.chi_rotamer_bins)


def rotamer_occupancy(bins, n_bins):
    """bins (frames, L) with -1 = absent -> occupancy (L, n_bins) and the residues that have this chi."""
    return np.stack([(bins == k).mean(0) for k in range(n_bins)], axis=-1), (bins >= 0).all(0)


def rotamer_error(gen_bins, md_bins, n_bins):
    """Per-residue total variation distance between generated and MD rotamer occupancies, averaged over
    residues that have the chi (0 = same populations, 1 = no overlap)."""
    p, ok_g = rotamer_occupancy(gen_bins, n_bins)
    q, ok_m = rotamer_occupancy(md_bins, n_bins)
    ok = ok_g & ok_m
    return float(0.5 * np.abs(p - q).sum(-1)[ok].mean()) if ok.any() else float("nan")


def rotamer_error_floor(md_bins, n, n_bins, seed=0, draws=10):
    """The same error for n random MD frames against all MD frames: what n samples of a perfect model would
    score, so the generated error can be read against it."""
    rng = np.random.default_rng(seed)
    if len(md_bins) <= n:
        return 0.0
    return float(np.nanmean([rotamer_error(md_bins[rng.choice(len(md_bins), n, replace=False)], md_bins, n_bins)
                             for _ in range(draws)]))


def ensemble_metrics(atoms, aatype, md_ref, temp, workdir):
    """Against MD at one temperature: RMSF Pearson r, |Q - Q_MD|, per-residue SS error (total variation), and
    per-residue chi1/chi2 rotamer-population errors (total variation), with their finite-sample floors."""
    atoms = atoms.detach().cpu().numpy() if torch.is_tensor(atoms) else atoms
    ca = atoms[:, :, 1]
    r = pearsonr(kabsch_rmsf(ca), md_ref["rmsf"][temp])[0]
    q_gen = native_contact_fraction(ca, md_ref["native_ca"])
    q_md = native_contact_fraction(md_ref["ca_frames"][temp], md_ref["native_ca"])
    ss = dssp_fractions(atoms, aatype, workdir)
    ss_err = float(0.5 * np.abs(ss - md_ref["ss"][temp]).sum(0).mean())
    gen_bins = G.atom14_rotamer_bins(atoms, torch.from_numpy(np.asarray(aatype))).numpy()
    md_bins = md_ref["chi_bins"][temp].astype(np.int64)
    out = {"rmsf_r": float(r), "q_err": abs(q_gen - q_md), "ss_err": ss_err, "q_gen": q_gen, "q_md": q_md}
    for k, (name, n_bins) in enumerate(zip(("chi1", "chi2"), ROTAMER_BINS)):
        out[f"{name}_err"] = rotamer_error(gen_bins[..., k], md_bins[..., k], n_bins)
        out[f"{name}_err_floor"] = rotamer_error_floor(md_bins[..., k], len(atoms), n_bins)
    return out
