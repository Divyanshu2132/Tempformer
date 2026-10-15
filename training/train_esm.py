import argparse
import os
import re
from pathlib import Path

from data import dataset as D
from training import train

DATASET = Path(__file__).resolve().parents[2] / "temp_sim_dataset"


def esm_index(esm_dir, template_dir=None):
    """Embedding index in the form of data.index_embeddings: rank 1 is the domain's ESM single/pairwise pair of
    files; the template is the AlphaFold rank_001 model from template_dir (none without it)."""
    names = set(os.listdir(esm_dir))
    af = D.index_embeddings(template_dir) if template_dir else {}
    index = {}
    for name in names:
        m = re.match(r"^single_(.+)\.npy$", name)
        if m and f"pairwise_{m.group(1)}.npy" in names:
            dom = m.group(1)
            index[dom] = {"ranks": {1: {"single": esm_dir / name, "pair": esm_dir / f"pairwise_{dom}.npy"}}}
            if "template" in af.get(dom, {}):
                index[dom]["template"] = af[dom]["template"]
    return index


def main():
    p = train.build_parser(__doc__)
    p.add_argument("--template-path", type=Path, default=DATASET / "alphafold_embeddings",
                   help="ColabFold outputs holding the <dom>_unrelaxed_rank_001_*.pdb models")
    p.add_argument("--template", action=argparse.BooleanOptionalAction, default=True,
                   help="AlphaFold rank_001 CA distances as template features")
    p.set_defaults(fasta=train.ROOT / "data/splits/tempformer500_train.fasta",
                   val_fasta=train.ROOT / "data/splits/tempformer500_val.fasta",
                   embedding_path=DATASET / "esm_embeddings/mdcath",
                   val_embedding_path=DATASET / "esm_embeddings/mdcath",
                   save_path=train.ROOT / "weights/tempformer500_esm.pth")
    args = train.parse_args(p)
    template_dir = train.LAUNCH_DIR / args.template_path if args.template else None
    train.main(args, index_fn=lambda d: esm_index(d, template_dir))


if __name__ == "__main__":
    main()
