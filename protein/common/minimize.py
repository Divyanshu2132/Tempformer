import os
import sys
import argparse
from pathlib import Path

from openmm.app import *
from openmm import *
from openmm.unit import *
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_lock import try_claim, release

ROOT = Path("/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset/")

DEFAULT_DATASETS = ["fourier_temperature_generated/generated_data/"]

for k in range(Platform.getNumPlatforms()):
    p = Platform.getPlatform(k)
    print(k, p.getName())

platform = Platform.getPlatformByName('CUDA')
platformProperties = {'Precision': 'single'}

forcefield = ForceField('amber14-all.xml')


def minimize_pdb(in_pdb: str, out_pdb: str) -> str:
    out = Path(out_pdb)

    # Atomically claim `out` so that concurrent minimize.py runs (e.g. one
    # per GPU job scanning the same tree) never redo each other's GPU work.
    if not try_claim(out):
        return f"skip_claimed:{out}"

    try:
        pdb = PDBFile(in_pdb)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds
        )
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator, platform, platformProperties)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        print(f"Minimized energy for {Path(in_pdb).name}: {state.getPotentialEnergy()}")
        with open(out_pdb, 'w') as f:
            PDBFile.writeFile(simulation.topology, state.getPositions(), f)
        return f"written:{out}"
    finally:
        release(out)


def collect_tasks(datasets):
    tasks = []
    for dataset in datasets:
        dataset_path = ROOT / dataset
        if not dataset_path.exists():
            print(f"Dataset path not found, skipping: {dataset_path}")
            continue
        for subdir in sorted(dataset_path.iterdir()):
            if not subdir.is_dir():
                continue
            for p in sorted(subdir.iterdir()):
                if not p.is_file() or not p.name.endswith("_all_atom.pdb"):
                    continue
                stem = p.stem[: -len("_all_atom")]
                out = p.with_name(f"{stem}_minimized.pdb")
                if out.exists():
                    continue
                tasks.append((str(p), str(out)))
    return tasks


def main(datasets):
    tasks = collect_tasks(datasets)
    print(f"Total minimization jobs: {len(tasks)}")
    if not tasks:
        return
    for in_pdb, out_pdb in tqdm(tasks, desc="Minimizing"):
        try:
            print(minimize_pdb(in_pdb, out_pdb))
        except Exception as e:
            print(f"Error minimizing {in_pdb}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to scan for _all_atom.pdb files (relative to ROOT)",
    )
    args = parser.parse_args()
    main(datasets=args.datasets)
