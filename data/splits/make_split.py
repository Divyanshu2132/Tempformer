import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT.parent / "temp_sim_dataset"
SAM_SPLITS = ROOT.parent / "sam2/data/splits/mdcath"
OUT = Path(__file__).resolve().parent

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n", type=int, default=500, help="domains drawn (at most 1369)")
p.add_argument("--n-val", type=int, default=100, help="the first n-val drawn are validation")
args = p.parse_args()

listing = [l.strip() for l in (DATASET / "domain_ids.csv").read_text().splitlines()[1:] if l.strip()]
all_files = random.Random(42).sample(listing, args.n)             # == random.seed(42); random.sample(...)
dom = lambda f: f.split("_")[-1].split(".")[0]                     # mdcath_dataset_<dom>.h5 -> <dom>
val, train = [dom(f) for f in all_files[:args.n_val]], [dom(f) for f in all_files[args.n_val:]]

seqs, cur = {}, None
for line in open(DATASET / "seq.fasta"):
    line = line.strip()
    if line.startswith(">"):
        cur = line[1:].split()[0]
        seqs[cur] = ""
    elif cur:
        seqs[cur] += line
tag = f"tempformer{args.n}"
for name, doms in (("train", train), ("val", val)):
    missing = [d for d in doms if d not in seqs]
    (OUT / f"{tag}_{name}.fasta").write_text("".join(f">{d}\n{seqs[d]}\n" for d in doms if d in seqs))
    print(f"{name}: {len(doms) - len(missing)} domains -> {OUT / f'{tag}_{name}.fasta'}"
          + (f" (no sequence for {missing})" if missing else ""))

sam_test = {l.split(".")[0] for l in (SAM_SPLITS / "test.txt").read_text().split()}
sam_val = {l.split(".")[0] for l in (SAM_SPLITS / "val.txt").read_text().split()}
for name, doms in (("train", train), ("val", val)):
    print(f"{name}: {len(set(doms) & sam_test)} aSAMt test domains, {len(set(doms) & sam_val)} aSAMt val domains")
