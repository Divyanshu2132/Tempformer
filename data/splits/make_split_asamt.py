import argparse
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FASTA = ROOT / "data/alphafold_embeddings/train_data.fasta"
EMBEDDINGS = ROOT / "data/alphafold_embeddings/train_embeddings"
MDCATH = ROOT.parent / "temp_sim_dataset/filtered_train_data"
OUT = Path(__file__).resolve().parent

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n-train", type=int, default=500)
p.add_argument("--n-val", type=int, default=100)
args = p.parse_args()

seqs, cur = {}, None
for line in open(FASTA):
    line = line.strip()
    if line.startswith(">"):
        cur = line[1:].split()[0]
        seqs[cur] = ""
    elif cur:
        seqs[cur] += line

rank1 = {}
for name in os.listdir(EMBEDDINGS):
    m = re.match(r"^_?([^_.]+)_(single|pair)_repr_rank_001_.*\.npy$", name)
    if m:
        rank1.setdefault(m.group(1), set()).add(m.group(2))
h5 = {re.sub(r"^mdcath_(snapshots|dataset)_", "", f)[:-3] for f in os.listdir(MDCATH) if f.endswith(".h5")}
usable = [d for d in seqs if rank1.get(d) == {"single", "pair"} and d in h5]
print(f"{len(usable)} of {len(seqs)} domains usable; left out: {sorted(set(seqs) - set(usable))}")

drawn = random.Random(42).sample(usable, args.n_val + args.n_train)
val, train = drawn[:args.n_val], drawn[args.n_val:]
tag = f"asamt{args.n_train}"
for name, doms in (("train", train), ("val", val)):
    (OUT / f"{tag}_{name}.fasta").write_text("".join(f">{d}\n{seqs[d]}\n" for d in doms))
    print(f"{name}: {len(doms)} domains -> {OUT / f'{tag}_{name}.fasta'}")
