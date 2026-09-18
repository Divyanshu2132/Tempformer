from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from tqdm import tqdm
import argparse
import os
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_lock import try_claim, release

ROOT = Path("/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/")

# --- defaults ---
DEFAULT_TEMPERATURES = ["300", "320", "340", "360", "380", "400", "420", "440", "460", "480", "500"]
DEFAULT_DATASETS = ["case_study/melting_temperature/generated_data/"]
VALID_PDB_NAME = re.compile(r"^.+_[0-9]+_[0-9]+\.pdb$")
# Matches folder names like "1adnA00_348" or "Trp-cage_348" → captures prefix and temperature
FOLDER_NAME = re.compile(r"^(.+)_(\d+)$")


def write_all_atom_pdb(src_path: str) -> str:
    src = Path(src_path)

    # Ignore anything that is not exactly domain_temp_index.pdb
    # Example accepted: 1pytA00_450_36.pdb
    if not VALID_PDB_NAME.match(src.name):
        return f"skip_name:{src}"

    out = src.with_name(f"{src.stem}_all_atom.pdb")

    # Atomically claim `out` so that concurrent fixer.py runs (e.g. one per
    # GPU job scanning the same tree) never redo each other's work.
    if not try_claim(out):
        return f"skip_claimed:{out}"

    try:
        fixer = PDBFixer(filename=str(src))
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        with open(out, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        return f"written:{out}"
    finally:
        release(out)


def auto_discover(datasets):
    """Scan dataset directories to find all (domain_prefix, temperature) pairs."""
    prefixes, temperatures = set(), set()
    for dataset in datasets:
        dataset_path = ROOT / dataset
        if not dataset_path.exists():
            continue
        for folder in dataset_path.iterdir():
            if not folder.is_dir():
                continue
            m = FOLDER_NAME.match(folder.name)
            if m:
                prefixes.add(m.group(1))
                temperatures.add(m.group(2))
    return sorted(prefixes), sorted(temperatures)


def collect_tasks(domain_prefixes, datasets, temperatures):
    tasks = []

    for dataset in datasets:
        print(f"Collecting tasks for dataset: {dataset}")
        for domain_prefix in domain_prefixes:
            print(f"  Domain prefix: {domain_prefix}")
            for temp in temperatures:
                print(f"    Temperature: {temp}")
                domain_id = f"{domain_prefix}_{temp}"
                base_path = ROOT / dataset / domain_id

                if not base_path.exists():
                    continue

                for p in base_path.iterdir():
                    if not p.is_file():
                        continue
                    if p.suffix.lower() != ".pdb":
                        continue
                    if p.stem.endswith("_all_atom"):
                        continue
                    if p.stem.endswith("_minimized"):
                        continue

                    out = p.with_name(f"{p.stem}_all_atom.pdb")

                    minimized = p.with_name(f"{p.stem}_minimized.pdb")

                    if out.exists() or minimized.exists():

                        continue

                    tasks.append(str(p))

    return tasks


def main(domain_prefixes, datasets, temperatures, max_workers=16):
    tasks = collect_tasks(domain_prefixes, datasets, temperatures)
    print(f"Total jobs: {len(tasks)}")

    if not tasks:
        return

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(write_all_atom_pdb, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="All-atom conversion"):
            try:
                print(fut.result())
            except Exception as e:
                print(f"error:{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain-prefix",
        nargs="*",
        default=None,
        help="Domain prefixes; if omitted, auto-discovers from dataset directories",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to scan (relative to ROOT)",
    )
    parser.add_argument(
        "--temperatures",
        nargs="*",
        default=None,
        help="Temperatures to scan; if omitted with --domain-prefix, auto-discovers",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    domain_prefixes = args.domain_prefix
    temperatures = args.temperatures

    if not domain_prefixes or not temperatures:
        discovered_prefixes, discovered_temps = auto_discover(args.datasets)
        if not domain_prefixes:
            domain_prefixes = discovered_prefixes
            print(f"Auto-discovered domain prefixes: {domain_prefixes}")
        if not temperatures:
            temperatures = discovered_temps
            print(f"Auto-discovered temperatures: {temperatures}")

    main(
        domain_prefixes=domain_prefixes,
        datasets=args.datasets,
        temperatures=temperatures,
        max_workers=args.max_workers,
    )
