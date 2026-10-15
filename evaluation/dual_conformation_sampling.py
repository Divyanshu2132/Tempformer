import argparse
import json
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

from model import residue_geometry as G

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "temp_sim_dataset"
BENCH = DATASET / "01_dual_conformation_benchmark"
OUT_ROOT = DATASET / "all_atom_tempformer/dual_conformation_benchmark"          # next to NMR_benchmark/, melting/
SAM2 = ROOT.parent / "sam2"
PRIORS = ("fold1", "fold2", "af2")
TEMPS = [320.0, 348.0, 379.0, 413.0, 450.0]                                      # mdCATH training temperatures
ASAM = {"asamt": ("mdcath", "mdcath_1.0"), "asamc": ("atlas", "atlas_1.0")}      # built-in config, weights flavor
ASAMC_T = 300.0

STANDARD = G.RESNAMES[:20]
ONE = {rn: G.RESTYPES[k] for k, rn in enumerate(STANDARD)}
THREE = {v: k for k, v in ONE.items()}
ATOMS = {rn: [str(a) for a in G.ATOM14_NAMES[k] if str(a)] for k, rn in enumerate(STANDARD)}   # atom14 order
# modified residues of PDB entries -> parent residue; atoms the parent lacks are dropped
MODIFIED = {
    "MSE": "MET", "FME": "MET", "CXM": "MET", "MHO": "MET",
    "MLY": "LYS", "MLZ": "LYS", "M3L": "LYS", "ALY": "LYS", "KCX": "LYS", "LLP": "LYS", "KPI": "LYS",
    "PTR": "TYR", "TYS": "TYR", "SEP": "SER", "TPO": "THR", "HYP": "PRO", "MEN": "ASN", "PCA": "GLU", "CGU": "GLU",
    "CSO": "CYS", "CSD": "CYS", "CME": "CYS", "OCS": "CYS", "CSX": "CYS", "CAS": "CYS", "SMC": "CYS", "SEC": "CYS",
    "NEP": "HIS", "HIC": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
}
RENAME = {("MSE", "SE"): "SD", ("SEC", "SE"): "SG"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--model", choices=["tempformer", "asamt", "asamc"], required=True)
    a("--checkpoint", type=Path, default=ROOT / "weights/tempformer1000_af.pth", help="Tempformer checkpoint")
    a("--bench", type=Path, default=BENCH, help="holds fasta/, native_structures/ and alphafold_embeddings/")
    a("--targets", nargs="+", help="default: every folder of native_structures/")
    a("--priors", nargs="+", choices=PRIORS, default=list(PRIORS))
    a("--temperatures", type=float, nargs="+", default=TEMPS,
      help="tempformer and asamt; default: the 5 mdCATH training temperatures. asamc ignores them")
    a("--num-samples", type=int, default=100)
    a("--steps", type=int, default=100)
    a("--batch", type=int, default=8, help="samples generated together (GPU memory)")
    a("--span", choices=["resolved", "full"], default="resolved")
    a("--template", choices=["filled", "masked"], default="filled", help="Tempformer template of unresolved residues")
    a("--seed", type=int, default=0)
    a("--minimize", action="store_true", help="also write ensembles minimized with aSAMt's minimizer to <tag>_min/")
    a("--device", choices=["cuda", "cpu"], help="default: cuda if available")
    a("--task-id", type=int, default=0, help="this task takes targets task-id, task-id + num-tasks, ...")
    a("--num-tasks", type=int, default=1)
    a("--out-dir", type=Path, default=OUT_ROOT)
    return p.parse_args()


# ---------------------------------------------------------------- structures

def read_fasta_seq(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


def read_structure(path):
    """Protein residues of the first model of a PDB file, in file order: [(resname, {atom: xyz})], heavy atoms of
    the 20 standard residues. Modified residues become their parent, other HETATM residues (ligands, waters,
    chromophores) are dropped, the first alternate location is kept, and residues without CA are left out."""
    residues, index = [], {}
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        orig = line[17:20].strip()
        resname = MODIFIED.get(orig, orig)
        if resname not in ATOMS:
            continue
        name = line[12:16].strip()
        name = RENAME.get((orig, name), name)
        key = (line[21], line[22:27])                                  # chain, residue number + insertion code
        if key not in index:
            index[key] = len(residues)
            residues.append((resname, {}))
        rn, atoms = residues[index[key]]
        if rn != resname or name not in ATOMS[rn] or name in atoms:     # other alternate location
            continue
        atoms[name] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return [r for r in residues if "CA" in r[1]]


def align_to_fasta(seq, fasta):
    """FASTA index of every residue of seq (-1 if unaligned): global alignment with free end gaps, so the unresolved
    loops and termini of a structure become gaps."""
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 2.0, -1.0
    al.open_gap_score, al.extend_gap_score = -4.0, -0.5
    al.end_gap_score = 0.0
    aln = al.align(fasta, seq)[0]
    idx = np.full(len(seq), -1)
    for (f0, f1), (s0, s1) in zip(*aln.aligned):
        idx[s0:s1] = np.arange(f0, f1)
    return idx


def load_target(target, bench=BENCH):
    """FASTA sequence and both experimental states, each {"path", "residues", "fasta_index", "ca", "identity"}:
    the residues aligned to the FASTA, their FASTA indices (increasing) and CA coordinates (Angstrom)."""
    fasta = read_fasta_seq(bench / "fasta" / f"{target}.fasta")
    states = {}
    for k in ("fold1", "fold2"):
        paths = sorted((bench / "native_structures" / target).glob(f"{k}_*.pdb"))
        if len(paths) != 1:
            raise FileNotFoundError(f"{target}: expected one {k}_*.pdb, found {len(paths)}")
        residues = read_structure(paths[0])
        idx = align_to_fasta("".join(ONE[rn] for rn, _ in residues), fasta)
        keep = idx >= 0
        residues = [r for r, ok in zip(residues, keep) if ok]
        idx = idx[keep]
        same = np.mean([ONE[rn] == fasta[i] for (rn, _), i in zip(residues, idx)])
        states[k] = {"path": paths[0], "residues": residues, "fasta_index": idx, "identity": float(same),
                     "n_unaligned": int((~keep).sum()), "ca": np.array([at["CA"] for _, at in residues])}
    return fasta, states


def find_colabfold(bench, target):
    """ColabFold rank_001 files of a target: {"model": pdb, "single": npy, "pair": npy}, the ones that exist."""
    d = bench / "alphafold_embeddings"
    out = {}
    for key, pattern in (("model", "unrelaxed_rank_001_*.pdb"), ("single", "single_repr_rank_001_*.npy"),
                         ("pair", "pair_repr_rank_001_*.npy")):
        hits = sorted(d.glob(f"{target}_{pattern}"))
        if hits:
            out[key] = hits[0]
    return out


def model_span(states, L, mode):
    if mode == "full":
        return 0, L
    return (int(min(s["fasta_index"].min() for s in states.values())),
            int(max(s["fasta_index"].max() for s in states.values())) + 1)


def kabsch(mobile, ref):
    """R, t with mobile @ R.T + t the least-squares superposition of mobile (n, 3) on ref (n, 3)."""
    mc, rc = mobile.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (ref - rc))
    R = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return R, rc - mc @ R.T


def gap_segments(resolved):
    """[(a, b)]: maximal runs of unresolved positions a..b-1."""
    segs, i = [], 0
    while i < len(resolved):
        if resolved[i]:
            i += 1
            continue
        j = i
        while j < len(resolved) and not resolved[j]:
            j += 1
        segs.append((i, j))
        i = j
    return segs


def build_prior(state, af2, fasta, span, flank=5):
    """FASTA residues span[0]..span[1]-1 as [(resname, {atom: xyz})] in the frame of the experimental state, and
    which of them the state resolves. Unresolved residues come from the AlphaFold model af2 (indexed like the FASTA):
    superposed on up to `flank` resolved residues before the gap and, separately, after it, and blended linearly
    from the first superposition to the second along the gap, so both ends join their anchors. A terminal gap uses
    its one side; with fewer than 2 anchor residues, the superposition on every resolved residue is used."""
    s0, s1 = span
    n = s1 - s0
    by_index = dict(zip(state["fasta_index"].tolist(), state["residues"]))
    residues, resolved = [None] * n, np.zeros(n, bool)
    for j in range(n):
        res = by_index.get(s0 + j)
        if res is None:
            continue
        name, (rn, atoms) = THREE[fasta[s0 + j]], res
        if rn != name:                                                  # mutation: backbone and CB only
            atoms = {a: x for a, x in atoms.items() if a in ("N", "CA", "C", "O") or (a == "CB" and name != "GLY")}
        residues[j], resolved[j] = (name, atoms), True
    def fit(anchors):
        pairs = [(af2[s0 + j][1][at], residues[j][1][at]) for j in anchors for at in ("N", "CA", "C")
                 if at in residues[j][1] and at in af2[s0 + j][1]]
        return kabsch(np.array([m for m, _ in pairs]), np.array([r for _, r in pairs]))

    fit_all = fit(np.flatnonzero(resolved).tolist())
    for a, b in gap_segments(resolved):
        left = [j for j in range(max(0, a - flank), a) if resolved[j]]
        right = [j for j in range(b, min(n, b + flank)) if resolved[j]]
        fits = [fit(side) for side in (left, right) if len(side) >= 2] or [fit_all]
        for j in range(a, b):
            w = (j - a + 1) / (b - a + 1) if len(fits) == 2 else 0.0          # weight of the right-side fit
            rn, atoms = af2[s0 + j]
            residues[j] = (rn, {k: (1 - w) * (x @ fits[0][0].T + fits[0][1]) + w * (x @ fits[-1][0].T + fits[-1][1])
                                for k, x in atoms.items()})
    return residues, resolved


def write_pdb(residues, path):
    lines, serial = [], 1
    for i, (rn, atoms) in enumerate(residues, 1):
        for name in ATOMS[rn]:
            if name not in atoms:
                continue
            x = atoms[name]
            lines.append(f"ATOM  {serial:5d} {name if len(name) == 4 else ' ' + name:<4s} {rn:>3s} A{i:4d}    "
                         f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {name[0]:>2s}")
            serial += 1
    path.write_text("\n".join(lines + ["TER", "END"]) + "\n")


# ---------------------------------------------------------------- models

def load_tempformer(path, device):
    import torch
    from model.main_model import MainModel
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MainModel(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])                    # EMA weights
    model.to(device).eval()
    return model, ckpt.get("epoch")


def sample_tempformer(model, colabfold, fasta, span, template_ca, template_mask, T, args, seed_key, device):
    """atom14 (n, L, 14, 3) of the span, conditioned on the rank_001 embeddings (sliced to the span, as the
    training crops are) and the given template CAs."""
    import torch
    from model.sampling import sample
    s0, s1 = span
    L = s1 - s0
    aatype = np.array([G.RESTYPES.index(c) for c in fasta[s0:s1]], dtype=np.int64)
    feats = {
        "aatype": torch.from_numpy(aatype)[None],
        "single": torch.from_numpy(np.array(np.load(colabfold["single"], mmap_mode="r")[s0:s1])).float()[None],
        "pair": torch.from_numpy(np.array(np.load(colabfold["pair"], mmap_mode="r")[s0:s1, s0:s1])).float()[None],
        "template_ca": torch.from_numpy(template_ca.astype(np.float32))[None],
        "template_mask": torch.from_numpy(template_mask)[None],
        "res_mask": torch.ones((1, L), dtype=torch.bool),
    }
    atoms = []
    for c in range(0, args.num_samples, args.batch):
        n = min(args.batch, args.num_samples - c)
        atoms.append(sample(model, feats, float(T), n, n_steps=args.steps, seed=zlib.crc32(f"{seed_key}:{c}".encode()),
                            device=device).cpu().numpy())
    return aatype, np.concatenate(atoms)


def load_asam(kind, device):
    if str(SAM2 / "src") not in sys.path:
        sys.path.append(str(SAM2 / "src"))
    import importlib.resources
    from sam.config import MODEL_FILENAMES
    from sam.model import AllAtomSAM
    config, flavor = ASAM[kind]
    return AllAtomSAM(config_fp=str(importlib.resources.files("sam.config") / MODEL_FILENAMES[config]),
                      weights_dir=SAM2 / "weights" / flavor, device=device)


def sample_asam(model, kind, prior_pdb, T, args, seed, out_stem):
    """aSAM ensemble from the prior structure, written to <out_stem>.{traj.dcd,top.pdb}."""
    import mdtraj
    import torch
    from sam.data.topology import get_seq_from_top
    torch.manual_seed(seed)
    np.random.seed(seed)
    traj = mdtraj.load(str(prior_pdb))
    out = model.sample(seq=get_seq_from_top(traj.topology), n_samples=args.num_samples, n_steps=args.steps,
                       batch_size_eps=args.batch, batch_size_dec=args.batch,
                       tbm_data={"xyz": traj.xyz, "topology": traj.topology},
                       conditions={"temperature": float(T)} if kind == "asamt" else {}, use_cache=True)
    model.save(out=out, out_path=out_stem, out_fmt="dcd")


def minimize(top, dcd, min_top, min_dcd, protocol, device):
    from minimizer.runner import Minimizer
    traj = Minimizer(name=top.stem, top_fp=str(top), ens_fp=str(dcd), protocol=protocol).run(device=device, verbose=False)
    min_top.parent.mkdir(parents=True, exist_ok=True)
    traj.save_dcd(str(min_dcd))
    traj[0].save_pdb(str(min_top))


def write_json(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ---------------------------------------------------------------- driver

def run_target(target, model, tag, protocol, temps, args, device, epoch):
    fasta, states = load_target(target, args.bench)
    colabfold = find_colabfold(args.bench, target)
    need = ("model", "single", "pair") if args.model == "tempformer" else ("model",)
    if any(k not in colabfold for k in need):
        print(f"{target}: ColabFold rank_001 {'/'.join(k for k in need if k not in colabfold)} not in "
              f"{args.bench / 'alphafold_embeddings'} yet, skipped", flush=True)
        return
    if any(c not in ONE.values() for c in fasta):
        print(f"{target}: non-standard residues in the FASTA sequence, skipped", flush=True)
        return
    af2 = read_structure(colabfold["model"])
    if "".join(ONE[rn] for rn, _ in af2) != fasta:
        print(f"{target}: {colabfold['model'].name} does not match the FASTA sequence, skipped", flush=True)
        return
    span = model_span(states, len(fasta), args.span)
    s0, s1 = span
    info = ", ".join(f"{k} {len(s['residues'])} resolved ({100 * s['identity']:.0f}% identical"
                     + (f", {s['n_unaligned']} unaligned" if s["n_unaligned"] else "") + ")" for k, s in states.items())
    print(f"{target}: FASTA {len(fasta)}, modeled {s0 + 1}-{s1} ({s1 - s0} residues); {info}", flush=True)
    for k, s in states.items():
        if s["identity"] < 0.9:
            print(f"  warning: {k} only {100 * s['identity']:.0f}% identical to the FASTA after alignment", flush=True)

    folder, min_folder = args.out_dir / tag / target, args.out_dir / f"{tag}_min" / target
    folder.mkdir(parents=True, exist_ok=True)
    for prior in args.priors:
        if prior == "af2":
            residues, resolved, source = af2[s0:s1], np.ones(s1 - s0, bool), colabfold["model"]
        else:
            residues, resolved = build_prior(states[prior], af2, fasta, span)
            source = states[prior]["path"]
        prior_pdb = folder / f"{target}.{prior}.prior.pdb"
        write_pdb(residues, prior_pdb)
        for T in temps:
            stem = f"{target}.{prior}.{T:g}"
            meta_path, min_meta = folder / f"{stem}.json", min_folder / f"{stem}.json"
            top, dcd = folder / f"{stem}.top.pdb", folder / f"{stem}.traj.dcd"
            label = f"  {stem} ({'from ' + prior if prior == 'af2' else f'{(~resolved).sum()} residues from af2'})"
            if meta_path.exists() and (not args.minimize or min_meta.exists()):
                print(f"{label}: already done", flush=True)
                continue
            start, did = time.time(), []
            if not meta_path.exists():
                seed_key = f"{args.seed}:{target}:{prior}:{T:g}"
                if args.model == "tempformer":
                    from evaluation.evaluate import write_ensemble
                    ca = np.array([atoms["CA"] for _, atoms in residues])
                    mask = resolved.copy() if args.template == "masked" else np.ones(len(residues), bool)
                    aatype, atoms = sample_tempformer(model, colabfold, fasta, span, ca * mask[:, None], mask, T, args,
                                                      seed_key, device)
                    write_ensemble(atoms, aatype, top, dcd)
                else:
                    sample_asam(model, args.model, prior_pdb, T, args, zlib.crc32(seed_key.encode()), folder / stem)
                meta = {"target": target, "model": args.model, "tag": tag, "prior": prior, "prior_source": str(source),
                        "prior_pdb": str(prior_pdb), "temperature": T, "temperature_conditioned": args.model != "asamc",
                        "n_samples": args.num_samples, "steps": args.steps, "seed": args.seed, "span": [s0, s1],
                        "fasta_length": len(fasta), "resolved_in_prior": resolved.astype(int).tolist(),
                        "n_from_af2": int((~resolved).sum()) if prior != "af2" else int(len(resolved)),
                        "template": args.template if args.model == "tempformer" else None,
                        "checkpoint": str(args.checkpoint) if args.model == "tempformer" else None, "epoch": epoch,
                        "minimized": False, "time_s": round(time.time() - start, 1)}
                write_json(meta, meta_path)
                did.append("sampled")
            if args.minimize and not min_meta.exists():
                minimize(top, dcd, min_folder / top.name, min_folder / dcd.name, protocol, device.type)
                meta = json.loads(meta_path.read_text())
                write_json({**meta, "minimized": True}, min_meta)
                did.append("minimized")
            print(f"{label}: {' + '.join(did)} in {time.time() - start:.0f}s", flush=True)


def main():
    args = parse_args()
    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    targets = args.targets or sorted(d.name for d in (args.bench / "native_structures").iterdir() if d.is_dir())
    work = targets[args.task_id::args.num_tasks]
    if args.model == "tempformer":
        model, epoch = load_tempformer(args.checkpoint, device)
        tag, protocol = args.checkpoint.stem, "mdcath"
    else:
        model, epoch = load_asam(args.model, device.type), None
        tag, protocol = args.model, model.model_cfg["minimization"]["protocol"]
    temps = [ASAMC_T] if args.model == "asamc" else args.temperatures
    print(f"{tag}{f' (epoch {epoch})' if epoch is not None else ''} on {device.type} | task {args.task_id}/"
          f"{args.num_tasks}: {len(work)} targets x priors {' '.join(args.priors)} x {' '.join(f'{T:g}' for T in temps)} K"
          f", {args.num_samples} samples each -> {args.out_dir / tag}", flush=True)
    failed = []
    for target in work:
        try:
            run_target(target, model, tag, protocol, temps, args, device, epoch)
        except Exception:
            traceback.print_exc()
            failed.append(target)
    if failed:
        raise SystemExit(f"{len(failed)} targets failed: {' '.join(failed)}")
    print("all targets of this task done", flush=True)


if __name__ == "__main__":
    main()
