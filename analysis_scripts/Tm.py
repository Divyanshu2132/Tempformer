import os
import glob
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_DIR = "/lustre/hdd/LAS/potoyan-lab/Divyanshu/git-repos/temp_sim_dataset"
threshold = 0.5
def get_native_contacts(
    native_pdb,
    contact_cutoff=1.0,   # 10 Angstrom = 1.0 nm in mdtraj
    min_sequence_separation=3,
):
    """
    Get native residue-residue contacts from a native/reference structure.

    Contact distance = shortest distance between any heavy atom in residue i and j.
    Keeps contacts with native distance <= 10 Angstrom and |i - j| >= 3.

    Returns:
        native_contact_pairs: shape (N_contacts, 2)
        native_contact_distances: shape (N_contacts,), in nm
    """
    native = md.load(native_pdb)
    topology = native.topology

    residues = list(topology.residues)
    candidate_pairs = []

    for i in range(len(residues)):
        for j in range(i + min_sequence_separation, len(residues)):
            candidate_pairs.append([i, j])

    candidate_pairs = np.asarray(candidate_pairs, dtype=int)

    distances, pairs = md.compute_contacts(
        native,
        contacts=candidate_pairs,
        scheme="closest-heavy",
    )

    native_distances = distances[0]
    keep = native_distances <= contact_cutoff

    native_contact_pairs = pairs[keep]
    native_contact_distances = native_distances[keep]

    return native_contact_pairs, native_contact_distances


def compute_q_values(
    traj,
    native_contact_pairs,
    native_contact_distances,
    beta=50.0,   # 5.0 Angstrom^-1 = 50.0 nm^-1
    lam=1.2,
):
    """
    Compute Q(x) for every frame in traj.

    Q(x) = mean over native contacts of:
        1 / (1 + exp(beta * (d_ij(x) - lambda * d_ij_native)))

    Returns:
        q_values: shape (n_frames,)
    """
    distances, _ = md.compute_contacts(
        traj,
        contacts=native_contact_pairs,
        scheme="closest-heavy",
    )

    contact_terms = 1.0 / (
        1.0 + np.exp(beta * (distances - lam * native_contact_distances))
    )

    q_values = contact_terms.mean(axis=1)

    return q_values


def folded_state_fraction_from_q(q_values, q_threshold=0.6):
    """
    Fraction of conformations with Q(x) > q_threshold.
    """
    return np.mean(q_values > q_threshold)


def folded_state_fraction_for_pdbs(
    pdb_files,
    native_contact_pairs,
    native_contact_distances,
    q_threshold=0.6,
    beta=50.0,
    lam=1.2,
):
    """
    Compute FSF for a set of generated/static PDB conformations.
    One PDB is treated as one conformation unless it has multiple frames.
    """
    all_q_values = []

    for pdb_path in pdb_files:
        traj = md.load(pdb_path)
        q_values = compute_q_values(
            traj,
            native_contact_pairs,
            native_contact_distances,
            beta=beta,
            lam=lam,
        )
        all_q_values.extend(q_values)

    all_q_values = np.asarray(all_q_values)

    fsf = folded_state_fraction_from_q(
        all_q_values,
        q_threshold=q_threshold,
    )

    return fsf, all_q_values

def generated_fsf_by_temperature(
    domain_id,
    temps,
    native_pdb,
    generated_dir,
    pdb_pattern="*_minimized.pdb",
    q_threshold=0.6,
):
    """
    Returns:
        fsf_by_temp[temp] = folded state fraction
        q_by_temp[temp] = all Q values at that temperature
    """
    native_contact_pairs, native_contact_distances = get_native_contacts(native_pdb)

    fsf_by_temp = {}
    q_by_temp = {}

    for temp in temps:
        folder = os.path.join(generated_dir, f"{domain_id}_{temp}")
        pdb_files = sorted(glob.glob(os.path.join(folder, pdb_pattern)))

        if not pdb_files:
            raise FileNotFoundError(f"No PDB files found in {folder}")

        fsf, q_values = folded_state_fraction_for_pdbs(
            pdb_files,
            native_contact_pairs,
            native_contact_distances,
            q_threshold=q_threshold,
        )

        fsf_by_temp[temp] = fsf
        q_by_temp[temp] = q_values

    return fsf_by_temp, q_by_temp
def melting_sigmoid(T, Tm, k_slope, y_min, y_max):
    """
    c(T) = y_min + (y_max - y_min) / (1 + exp((T - Tm) / k_slope))

    This decreases with temperature when k_slope > 0.
    """
    return y_min + (y_max - y_min) / (1.0 + np.exp((T - Tm) / k_slope))


def fit_melting_temperature(fsf_by_temp):
    temps = np.asarray(sorted(fsf_by_temp.keys()))
    values = np.asarray([fsf_by_temp[temp] for temp in temps], dtype=float)

    y_min = values.min()
    y_max = values.max()

    initial_guess = [
        np.median(temps),  # Tm
        10.0,              # k_slope
    ]

    def fit_func(T, Tm, k_slope):
        return melting_sigmoid(T, Tm, k_slope, y_min, y_max)

    popt, pcov = curve_fit(
        fit_func,
        temps,
        values,
        p0=initial_guess,
        maxfev=10000,
    )

    Tm, k_slope = popt

    return {
        "Tm": Tm,
        "k_slope": k_slope,
        "y_min": y_min,
        "y_max": y_max,
        "temps": temps,
        "values": values,
        "covariance": pcov,
    }
def fsf_and_tm_fit(fit_result):
    temps = fit_result["temps"]
    values = fit_result["values"]

    T_fit = np.linspace(temps.min(), temps.max(), 200)
    y_fit = melting_sigmoid(
        T_fit,
        fit_result["Tm"],
        fit_result["k_slope"],
        fit_result["y_min"],
        fit_result["y_max"],
    )
    return T_fit,y_fit,temps,values


GEN_DIR = os.path.join(DATA_DIR, "spectral_temperature_generated/melting_temperature")
domain_ids = {i.split("_")[0] for i in os.listdir(GEN_DIR)}
temps = [280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
ss_types = ("H", "E", "C")
from concurrent.futures import ProcessPoolExecutor, as_completed

OUTPUT_DIR = os.path.join(DATA_DIR, "case_study/melting_temperature/tm_fits")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_domain(domain_id):
    print(f"Processing domain {domain_id}...", flush=True)

    native_matches = sorted(glob.glob(os.path.join(
        os.path.join(DATA_DIR, "case_study/melting_temperature/alphafold_structures"),
        f"{domain_id}*rank_001_*.pdb",
    )))

    if not native_matches:
        raise FileNotFoundError(f"No native PDB found for {domain_id} in {GEN_DIR}")

    native_pdb = native_matches[0]

    fsf_by_temp, q_by_temp = generated_fsf_by_temperature(
        domain_id=domain_id,
        temps=temps,
        native_pdb=native_pdb,
        generated_dir=GEN_DIR,
        pdb_pattern="*_minimized.pdb",
        q_threshold=threshold,
    )

    fit_result = fit_melting_temperature(fsf_by_temp)

    T_fit, y_fit, fit_temps, values = fsf_and_tm_fit(fit_result)

    out_path = os.path.join(OUTPUT_DIR, f"{domain_id}_tm_fit.npz")
    np.savez_compressed(
        out_path,
        T_fit=T_fit,
        y_fit=y_fit,
        temps=fit_temps,
        values=values,
        Tm=fit_result["Tm"],
        k_slope=fit_result["k_slope"],
        y_min=fit_result["y_min"],
        y_max=fit_result["y_max"],
        covariance=fit_result["covariance"],
    )

    return domain_id, fit_result["Tm"], fsf_by_temp, out_path


print(f"Threshold for folded state (Q): {threshold}")

max_workers = min(len(domain_ids), os.cpu_count() or 1)

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(process_domain, domain_id): domain_id
        for domain_id in domain_ids
    }

    for future in as_completed(futures):
        domain_id = futures[future]

        try:
            domain_id, tm, fsf_by_temp, out_path = future.result()
            print(f"{domain_id}:")
            print(fsf_by_temp)
            print("Tm =", tm)
            print("Saved:", out_path)
        except Exception as exc:
            print(f"{domain_id} failed: {exc}", flush=True)