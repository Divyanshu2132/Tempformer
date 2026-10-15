import argparse
import zlib
from pathlib import Path

import numpy as np
import torch

import data as D
from model import residue_geometry as G
from model.main_model import MainModel
from sampling import sample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--embedding-path", type=Path, required=True)
    p.add_argument("--temperatures", type=float, nargs="+", default=[320, 348, 379, 413, 450])
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--sampler", choices=["sde", "ode"], default="sde")
    p.add_argument("--chi-source", choices=["diffused", "predicted"], default="diffused",
                   help="side-chain chis from the diffusion (varying) or from the torsion head (one per backbone)")
    p.add_argument("--batch", type=int, default=10, help="samples generated together")
    p.add_argument("--raw-weights", action="store_true", help="use the raw instead of the EMA weights")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MainModel(**ckpt["config"])
    model.load_state_dict(ckpt["raw_model_state_dict" if args.raw_weights else "model_state_dict"])
    model.to(device).eval()
    seqs = D.read_fasta(args.fasta)
    emb = D.index_embeddings(args.embedding_path)
    for dom, seq in seqs.items():
        entry = emb.get(dom)
        if not entry or not entry["ranks"]:
            print(f"{dom}: no embeddings, skipped", flush=True)
            continue
        aatype = np.array([G.RESTYPES.index(c) if c in G.RESTYPES else G.UNK for c in seq], dtype=np.int64)
        tmpl = D.template_ca(entry["template"], len(seq)) if "template" in entry else None
        feats = D.protein_batch({"aatype": aatype, "ranks": entry["ranks"], "template_ca": tmpl})
        for T in args.temperatures:
            tag = f"{dom}_{T:g}"
            folder = args.out_dir / tag
            folder.mkdir(parents=True, exist_ok=True)
            todo = [i for i in range(args.num_samples) if not (folder / f"{tag}_{i}.pdb").exists()]
            for c in range(0, len(todo), args.batch):
                chunk = todo[c:c + args.batch]
                atoms = sample(model, feats, T, len(chunk), n_steps=args.steps, sampler=args.sampler,
                               seed=zlib.crc32(f"{dom}:{T:g}:{chunk[0]}".encode()), device=device,
                               chi_source=args.chi_source)
                for i, x in zip(chunk, atoms):
                    path = folder / f"{tag}_{i}.pdb"
                    tmp = path.with_suffix(".pdb.tmp")
                    tmp.write_text("\n".join(G.atom14_to_pdb_lines(x, aatype) + ["END"]))
                    tmp.rename(path)
            print(f"{tag}: {args.num_samples} structures in {folder}", flush=True)


if __name__ == "__main__":
    main()
