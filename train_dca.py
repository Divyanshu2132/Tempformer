import argparse
import os
import re
from pathlib import Path

import data as D
import train

DATASET = Path(__file__).resolve().parent.parent / "temp_sim_dataset"


def dca_index(af_dir, dca_dir, template=True):
    """data.index_embeddings of the AlphaFold outputs with each rank's pair file replaced by the domain's DCA
    file. Domains without one get no ranks (skipped as "no embeddings")."""
    dca = {}
    for name in os.listdir(dca_dir):
        m = re.match(r"^_?([^_.]+)_pairwise_dca\.npy$", name)
        if m:
            dca[m.group(1)] = dca_dir / name
    index = D.index_embeddings(af_dir)
    for dom, entry in index.items():
        entry["ranks"] = {r: {"single": p["single"], "pair": dca[dom]} for r, p in entry["ranks"].items()} \
            if dom in dca else {}
        if not template:
            entry.pop("template", None)
    return index


def main():
    p = train.build_parser(__doc__)
    p.add_argument("--dca-path", type=Path, default=DATASET / "dca_embeddings")
    p.add_argument("--dca-bins", type=int, default=32, help="RBF bins for the DI value")
    p.add_argument("--dca-max", type=float, default=0.6, help="centre of the last bin; larger DI is clipped to it")
    p.add_argument("--template", action=argparse.BooleanOptionalAction, default=True,
                   help="AlphaFold rank_001 CA distances as template features")
    p.set_defaults(fasta=train.ROOT / "data/splits/tempformer500_train.fasta",
                   val_fasta=train.ROOT / "data/splits/tempformer500_val.fasta",
                   embedding_path=DATASET / "alphafold_embeddings",
                   val_embedding_path=DATASET / "alphafold_embeddings",
                   save_path=train.ROOT / "weights/tempformer500_dca.pth")
    args = train.parse_args(p)
    dca_dir = train.LAUNCH_DIR / args.dca_path
    train.main(args, index_fn=lambda d: dca_index(d, dca_dir, args.template),
               model_extra=dict(pair_rbf_bins=args.dca_bins, pair_rbf_max=args.dca_max))


if __name__ == "__main__":
    main()
