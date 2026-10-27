import argparse
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT.parent / "temp_sim_dataset"
OUT = Path(__file__).resolve().parent

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n", type=int, default=250, help="test domains")
p.add_argument("--skip", type=int, default=1000, help="domains of the draw the runs used (make_split.py --n)")
args = p.parse_args()


def read_fasta(path):
    seqs, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].split()[0]
            seqs[cur] = ""
        elif cur:
            seqs[cur] += line
    return seqs


listing = [l.strip() for l in (DATASET / "domain_ids.csv").read_text().splitlines()[1:] if l.strip()]
dom = lambda f: f.split("_")[-1].split(".")[0]                     # mdcath_dataset_<dom>.h5 -> <dom>
order = [dom(f) for f in random.Random(42).sample(listing, len(listing))]
used = set()
for name in ("train", "val"):
    path = OUT / f"tempformer{args.skip}_{name}.fasta"
    if path.exists():
        used |= set(read_fasta(path))
if used and used != set(order[:args.skip]):
    raise SystemExit(f"the first {args.skip} domains of the draw are not tempformer{args.skip}_{{train,val}}.fasta")

seqs = read_fasta(DATASET / "seq.fasta")
af = {}
for name in os.listdir(DATASET / "alphafold_embeddings"):
    m = re.match(r"^_?(.+?)_(single_repr|pair_repr|unrelaxed)_rank_001_", name)
    if m:
        af.setdefault(m.group(1), set()).add(m.group(2))
dca = {m.group(1) for m in (re.match(r"^_?([^_.]+)_pairwise_dca\.npy$", n) for n in os.listdir(DATASET / "dca_embeddings")) if m}
esm = set(os.listdir(DATASET / "esm_embeddings/mdcath"))
mdcath = set(os.listdir(DATASET / "filtered_train_data"))
checks = {
    "no sequence": lambda d: d in seqs,
    "no ColabFold rank_001 embeddings/model": lambda d: af.get(d, set()) >= {"single_repr", "pair_repr", "unrelaxed"},
    "no DCA": lambda d: d in dca,
    "no ESM-2": lambda d: f"single_{d}.npy" in esm and f"pairwise_{d}.npy" in esm,
    "no mdCATH file": lambda d: f"mdcath_snapshots_{d}.h5" in mdcath or f"mdcath_dataset_{d}.h5" in mdcath,
}

test, skipped = [], {k: 0 for k in checks}
for d in order[args.skip:]:
    failed = next((k for k, ok in checks.items() if not ok(d)), None)
    if failed:
        skipped[failed] += 1
    else:
        test.append(d)
    if len(test) == args.n:
        break
if len(test) < args.n:
    raise SystemExit(f"only {len(test)} usable domains after the first {args.skip}")

out = OUT / f"tempformer{args.skip}_test{args.n}.fasta"
out.write_text("".join(f">{d}\n{seqs[d]}\n" for d in test))
print(f"test: {len(test)} domains -> {out}; skipped: {', '.join(f'{v} {k}' for k, v in skipped.items() if v) or 'none'}")
