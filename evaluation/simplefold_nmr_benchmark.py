import argparse
import sys
import tempfile
import time
import urllib.request
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import mdtraj as md
import torch

import simplefold
sys.path.insert(0, str(Path(simplefold.__file__).resolve().parent))  # its modules import each other top-level (cli.py);
                                                                     # first, so Tempformer's model/ does not shadow its model/
from simplefold.inference import generate_structure, initialize_esm_model, initialize_folding_model, initialize_others
from utils.boltz_utils import process_structure, save_structure
from utils.datamodule_utils import process_one_inference_structure
from utils.fasta_utils import CCD_URL, process_fastas

DATASET = Path(__file__).resolve().parents[2] / "temp_sim_dataset"
BENCH = DATASET / "03_NMR_ensemble_benchmark"
CACHE = Path(__file__).resolve().parents[3] / "simplefold_cache"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--out-dir", type=Path, required=True, help="the ensembles go to <out-dir>/<PDB>/")
    a("--bench", type=Path, default=BENCH, help="holds fasta/<PDB>.fasta")
    a("--proteins", nargs="+", help="default: every fasta/<PDB>.fasta")
    a("--model", default="simplefold_3B", help="simplefold_100M/360M/700M/1.1B/1.6B/3B")
    a("--cache", type=Path, default=CACHE, help="SimpleFold checkpoints and the CCD dictionary")
    a("--num-samples", type=int, default=50)
    a("--batch", type=int, default=25, help="structures sampled at once")
    a("--num-steps", type=int, default=500)
    a("--tau", type=float, default=0.6)
    a("--seed", type=int, default=0)
    a("--temperature", type=int, default=300, help="label of the written ensembles")
    return p.parse_args()


def finished(top, dcd):
    """As evaluation.evaluate.finished: a finished ensemble's top is written after its dcd."""
    return top.exists() and dcd.exists() and top.stat().st_mtime >= dcd.stat().st_mtime


def fasta_seq(bench, pdb):
    return "".join(l.strip() for l in open(bench / f"fasta/{pdb}.fasta") if not l.startswith(">"))


def write_ensemble(structure, record, coords, pad_mask, top, dcd, seq):
    """Sampled coordinates -> one PDB per structure (SimpleFold's writer) -> topology PDB + DCD."""
    with tempfile.TemporaryDirectory(dir=top.parent) as tmp:
        for i, c in enumerate(coords):
            save_structure(process_structure(deepcopy(structure), c, pad_mask, record), Path(tmp), f"s{i:04d}",
                           output_format="pdb")
        traj = md.load([str(Path(tmp) / f"s{i:04d}.pdb") for i in range(len(coords))])
    if "".join(r.code or "X" for r in traj.topology.residues) != seq:
        print(f"{top.parent.name}: WARNING structure sequence differs from the fasta", flush=True)
    traj.superpose(traj, 0, atom_indices=traj.topology.select("name CA"))
    traj.save_dcd(str(dcd))
    tmp_top = top.with_name(top.name + ".tmp")
    traj[0].save_pdb(str(tmp_top))
    tmp_top.replace(top)                                                   # top after dcd: finished() is true
    return traj


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    proteins = args.proteins or sorted(p.stem for p in (args.bench / "fasta").glob("*.fasta"))
    index = {pdb: k for k, pdb in enumerate(sorted(p.stem for p in (args.bench / "fasta").glob("*.fasta")))}
    paths = {pdb: (args.out_dir / pdb / f"{pdb}.{args.temperature}.top.pdb",
                   args.out_dir / pdb / f"{pdb}.{args.temperature}.traj.dcd") for pdb in proteins}
    todo = [pdb for pdb in proteins if not finished(*paths[pdb])]
    print(f"{len(proteins) - len(todo)} of {len(proteins)} ensembles finished; {len(todo)} to sample", flush=True)
    if not todo:
        return

    # inputs in SimpleFold's FASTA format (>CHAIN|ENTITY), processed into simplefold_raw/{structures,records}/<PDB>
    raw = args.out_dir / "simplefold_raw"
    (raw / "fasta").mkdir(parents=True, exist_ok=True)
    for pdb in todo:
        (raw / "fasta" / f"{pdb}.fasta").write_text(f">A|protein\n{fasta_seq(args.bench, pdb)}\n")
    args.cache.mkdir(parents=True, exist_ok=True)
    ccd = args.cache / "ccd.pkl"
    if not ccd.exists():
        urllib.request.urlretrieve(CCD_URL, str(ccd))
    process_fastas(data=[raw / "fasta" / f"{pdb}.fasta" for pdb in todo], out_dir=raw, ccd_path=ccd)

    sf = Namespace(simplefold_model=args.model, ckpt_dir=str(args.cache), backend="torch", plddt=False,
                   nsample_per_protein=args.batch, num_steps=args.num_steps, tau=args.tau)
    model, device = initialize_folding_model(sf)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(sf, device)
    tokenizer, featurizer, processor, flow, sampler = initialize_others(sf, device)

    for pdb in todo:
        t0 = time.time()
        pl.seed_everything(args.seed + index.get(pdb, 0), verbose=False)
        batch, structure, record = process_one_inference_structure(
            raw / "structures" / f"{pdb}.npz", raw / "records" / f"{pdb}.json",
            tokenizer, featurizer, processor, esm_model, esm_dict, af2_to_esm)
        coords = []
        while sum(len(c) for c in coords) < args.num_samples:
            c, pad_mask, _ = generate_structure(sf, batch, sampler, flow, processor, model, None, None, device)
            coords.append(c.cpu())
        coords = torch.cat(coords)[:args.num_samples]
        top, dcd = paths[pdb]
        top.parent.mkdir(parents=True, exist_ok=True)
        traj = write_ensemble(structure, record, coords, pad_mask[0].cpu(), top, dcd, fasta_seq(args.bench, pdb))
        print(f"{pdb}: {traj.n_frames} structures, {traj.n_residues} residues, {traj.n_atoms} atoms, "
              f"{time.time() - t0:.0f} s -> {dcd}", flush=True)
        del batch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
