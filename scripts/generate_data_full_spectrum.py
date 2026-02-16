"""
Full-Spectrum GPU Data Generation for Underwater Acoustic Metamaterial

Generates 1,000,000 samples with full 1000-point absorption spectrum (1-1000 Hz)
using GPU-accelerated Transfer Matrix Method (TMM).

Output:
  - data/lhs_data_full_spectrum.parquet  (primary, compressed)
  - data/lhs_data_full_spectrum.csv      (secondary, human-readable)
  - data/lhs_data_full_spectrum.npz      (tertiary, fast ML loading)

Physics: Equivalent Medium Theory + Transfer Matrix Method
  Following: Gao et al., Ocean Engineering — "On-demand Prediction of
  Low-Frequency Average Sound Absorption Coefficient of Underwater Coating
  Using Machine Learning"

Usage:
    python scripts/generate_data_full_spectrum.py
"""

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
import sys
import gc
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# GPU Setup
# ---------------------------------------------------------------------------
assert torch.cuda.is_available(), (
    "CUDA not available! This script requires an NVIDIA GPU.\n"
    "Install PyTorch with CUDA: pip install torch --index-url "
    "https://download.pytorch.org/whl/cu121"
)
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"GPU : {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RHO_WATER = 1000.0   # kg/m^3
C_WATER = 1500.0     # m/s
Z_WATER = RHO_WATER * C_WATER  # Pa*s/m
W_METERS = 2.0       # Unit cell width (2000 mm = 2.0 m)

# 0-indexed layer indices that are hollow (paper layers 2,3,5,6,8,9)
HOLLOW_LAYER_INDICES = [1, 2, 4, 5, 7, 8]

# Parameter ranges (from paper Table 4)
PARAM_BOUNDS = {
    "d": (1.0, 20.0),         # mm, layer thickness (10 layers)
    "m": (20.0, 1980.0),      # mm, hollow diameter (6 hollow layers)
    "rho": (1000.0, 1500.0),  # kg/m^3
    "eta": (0.1, 0.8),        # loss factor (dimensionless)
    "E": (1e7, 1e8),          # Pa, real part of Young's modulus
    "nu": (0.4, 0.49),        # Poisson's ratio
}

NUM_PARAMS = 20   # 10 thicknesses + 6 hollow diameters + 4 material props
NUM_FREQS = 1000  # 1 Hz to 1000 Hz

# Column names
HOLLOW_LAYER_IDS = [2, 3, 5, 6, 8, 9]
PARAM_COL_NAMES = (
    [f"d{j+1}" for j in range(10)]
    + [f"m{j}" for j in HOLLOW_LAYER_IDS]
    + ["rho", "eta", "E", "nu"]
)
FREQ_COL_NAMES = [f"alpha_{f}" for f in range(1, NUM_FREQS + 1)]
ALL_COL_NAMES = PARAM_COL_NAMES + FREQ_COL_NAMES + ["Average_Absorption"]


# ---------------------------------------------------------------------------
# GPU-Accelerated TMM: Full Spectrum
# ---------------------------------------------------------------------------
def _compute_spectrum_chunk(param_matrix):
    """Compute full absorption spectrum for a chunk on GPU.

    Parameters
    ----------
    param_matrix : np.ndarray, shape (N, 20)
        Physical parameters: [d1..d10 (mm), m2..m9 (mm), rho, eta, E, nu]

    Returns
    -------
    np.ndarray, shape (N, 1000), dtype float32
    """
    N = param_matrix.shape[0]
    param_tensor = torch.from_numpy(param_matrix).float().to(DEVICE)

    # --- Extract and convert parameters ---
    d_vals = param_tensor[:, 0:10] / 1000.0   # mm -> m, shape (N, 10)
    m_vals = param_tensor[:, 10:16] / 1000.0  # mm -> m, shape (N, 6)
    rho_r = param_tensor[:, 16:17]             # (N, 1)
    eta = param_tensor[:, 17:18]               # (N, 1)
    E_r = param_tensor[:, 18:19]               # (N, 1)
    nu = param_tensor[:, 19:20]                # (N, 1)

    # --- Complex Young's modulus: E_c = E_r * (1 + eta*i) ---
    E_c = E_r * (1.0 + 1j * eta)

    # --- Lame constants (Eqs. 1, 2) ---
    lam = (E_c * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_c / (2.0 * (1.0 + nu))

    # --- Frequencies ---
    frequencies = torch.arange(1, NUM_FREQS + 1, device=DEVICE, dtype=torch.float32)
    omega = 2.0 * torch.pi * frequencies  # shape (1000,)

    # --- Initialize total transfer matrix as identity ---
    eye = torch.eye(2, device=DEVICE, dtype=torch.complex64)
    T_total = eye.unsqueeze(0).unsqueeze(0).expand(N, NUM_FREQS, -1, -1).clone()

    # --- Layer loop (10 layers) ---
    for lay_idx in range(10):
        d = d_vals[:, lay_idx:lay_idx + 1]  # (N, 1)

        # Perforation rate epsilon
        if lay_idx in HOLLOW_LAYER_INDICES:
            m_ptr = HOLLOW_LAYER_INDICES.index(lay_idx)
            eps = m_vals[:, m_ptr:m_ptr + 1] / W_METERS  # (N, 1)
        else:
            eps = torch.zeros((N, 1), device=DEVICE, dtype=torch.float32)

        # Equivalent density: rho' = rho * (1 - eps^2)  [Eq. 10a]
        rho_eff = rho_r * (1.0 - eps ** 2)

        # Volume longitudinal wave modulus S' [Eq. 9a]
        numerator = (mu * (lam + 2.0 * mu) * (eps ** 2 + 1.0)) + (2.0 * (eps ** 2) * lam)
        denominator = ((lam + mu) * eps ** 2) + mu
        safe_mask = torch.abs(denominator) < 1e-15
        S_eff = torch.where(safe_mask, numerator * 1e15, numerator / denominator)

        # Equivalent sound velocity: c' = sqrt(S'/rho')  [Eq. 11]
        c_eff = torch.sqrt(S_eff / rho_eff)

        # Wave number and impedance
        k_eff = omega.unsqueeze(0) / c_eff   # (N, 1000)
        Z_eff = rho_eff * c_eff               # (N, 1)

        # Transfer matrix elements [Eq. 12]
        cos_kd = torch.cos(k_eff * d)
        sin_kd = torch.sin(k_eff * d)

        Z_mask = torch.abs(Z_eff) < 1e-15
        t21 = torch.where(Z_mask, torch.full_like(cos_kd, 1e15), 1j * sin_kd / Z_eff)

        # Build per-layer transfer matrix: shape (N, NUM_FREQS, 2, 2)
        T_i = torch.zeros((N, NUM_FREQS, 2, 2), device=DEVICE, dtype=torch.complex64)
        T_i[:, :, 0, 0] = cos_kd
        T_i[:, :, 0, 1] = 1j * Z_eff * sin_kd
        T_i[:, :, 1, 0] = t21
        T_i[:, :, 1, 1] = cos_kd

        # Multiply into total transfer matrix
        T_total = torch.matmul(T_total, T_i)

    # --- Reflection coefficient [Eq. 14] ---
    T11 = T_total[:, :, 0, 0]  # (N, 1000)
    T21 = T_total[:, :, 1, 0]  # (N, 1000)

    # Rigid backing: Z_in = T11 / T21
    t21_mask = torch.abs(T21) < 1e-15
    Z_in = torch.where(t21_mask, torch.full_like(T21, 1e15 + 0j), T11 / T21)
    R = torch.where(t21_mask, torch.ones_like(T21), (Z_in - Z_WATER) / (Z_in + Z_WATER))

    # --- Absorption coefficient: alpha = 1 - |R|^2  [Eq. 16, T=0 rigid backing] ---
    alpha = 1.0 - torch.abs(R) ** 2
    alpha = torch.clamp(alpha.real, 0.0, 1.0)  # (N, 1000)

    return alpha.cpu().numpy().astype(np.float32)


def calculate_absorption_spectrum_gpu(param_matrix, chunk_size=5000):
    """
    GPU-accelerated TMM returning full absorption spectrum.

    Parameters
    ----------
    param_matrix : np.ndarray, shape (N, 20)
        Physical parameters in their natural units.
    chunk_size : int
        Number of samples per GPU chunk (tune for VRAM).

    Returns
    -------
    np.ndarray, shape (N, 1000), dtype float32
        Absorption coefficient at each frequency 1-1000 Hz.
    """
    N = param_matrix.shape[0]
    if N <= chunk_size:
        return _compute_spectrum_chunk(param_matrix)

    results = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        results.append(_compute_spectrum_chunk(param_matrix[start:end]))
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------
def generate_lhs_samples(num_samples):
    """
    Generate parameter samples using Latin Hypercube Sampling.

    Returns
    -------
    np.ndarray, shape (num_samples, 20), dtype float64
    """
    print(f"Generating {num_samples:,} LHS samples across {NUM_PARAMS} dimensions...",
          flush=True)
    sampler = qmc.LatinHypercube(d=NUM_PARAMS)
    sample_norm = sampler.random(n=num_samples)

    physical = np.zeros_like(sample_norm)

    # d1..d10 (indices 0-9)
    lo, hi = PARAM_BOUNDS["d"]
    physical[:, 0:10] = lo + (hi - lo) * sample_norm[:, 0:10]

    # m2,m3,m5,m6,m8,m9 (indices 10-15)
    lo, hi = PARAM_BOUNDS["m"]
    physical[:, 10:16] = lo + (hi - lo) * sample_norm[:, 10:16]

    # Scalar parameters (indices 16-19)
    for idx, key in zip([16, 17, 18, 19], ["rho", "eta", "E", "nu"]):
        lo, hi = PARAM_BOUNDS[key]
        physical[:, idx] = lo + (hi - lo) * sample_norm[:, idx]

    del sample_norm
    print(f"  LHS sampling complete.", flush=True)
    return physical


# ---------------------------------------------------------------------------
# Main Generation Pipeline
# ---------------------------------------------------------------------------
def generate_dataset(num_samples=1_000_000, gpu_chunk_size=5000,
                     mega_chunk_size=50_000):
    """
    Generate full-spectrum dataset with chunked GPU processing.

    Returns params (N,20) float64 and spectra (N,1000) float32 arrays.
    Does NOT build a pandas DataFrame in memory (too large for 1M samples).
    """
    t_start = time.time()

    # --- Step 1: LHS Sampling ---
    physical_samples = generate_lhs_samples(num_samples)

    # --- Step 2: GPU Spectrum Calculation (mega-chunked) ---
    print(f"\nComputing absorption spectra on GPU ({DEVICE})...", flush=True)
    print(f"  GPU chunk size:  {gpu_chunk_size:,}", flush=True)
    print(f"  Mega chunk size: {mega_chunk_size:,}", flush=True)

    # Pre-allocate full spectra array to avoid concatenation overhead
    all_spectra = np.empty((num_samples, NUM_FREQS), dtype=np.float32)
    num_mega_chunks = (num_samples + mega_chunk_size - 1) // mega_chunk_size

    for mc_idx in range(num_mega_chunks):
        mc_start = mc_idx * mega_chunk_size
        mc_end = min(mc_start + mega_chunk_size, num_samples)
        mc_size = mc_end - mc_start

        t_mc = time.time()
        all_spectra[mc_start:mc_end] = calculate_absorption_spectrum_gpu(
            physical_samples[mc_start:mc_end], chunk_size=gpu_chunk_size
        )
        elapsed = time.time() - t_mc
        speed = mc_size / elapsed

        print(f"  Mega-chunk {mc_idx + 1:>3d}/{num_mega_chunks}: "
              f"{mc_start:>8,}-{mc_end:>8,} | {elapsed:>5.1f}s | "
              f"{speed:,.0f} samples/s", flush=True)

    t_compute = time.time() - t_start
    print(f"\nGPU computation complete in {t_compute:.1f}s "
          f"({num_samples / t_compute:,.0f} samples/s)", flush=True)

    # --- Summary Statistics ---
    avg_absorption = np.mean(all_spectra, axis=1)
    print(f"\n--- Summary Statistics ---", flush=True)
    print(f"  Samples:        {num_samples:,}", flush=True)
    print(f"  Avg Absorption: mean={avg_absorption.mean():.4f}, "
          f"std={avg_absorption.std():.4f}, "
          f"min={avg_absorption.min():.4f}, max={avg_absorption.max():.4f}",
          flush=True)
    print(f"  Alpha range:    [{all_spectra.min():.6f}, {all_spectra.max():.6f}]",
          flush=True)

    return physical_samples, all_spectra


# ---------------------------------------------------------------------------
# Saving Functions (memory-efficient, streaming)
# ---------------------------------------------------------------------------
def save_npz(physical_samples, all_spectra, output_dir):
    """Save as compressed NPZ (fastest ML loading)."""
    path = os.path.join(output_dir, "lhs_data_full_spectrum.npz")
    print(f"\nSaving NPZ: {path}", flush=True)
    t0 = time.time()
    np.savez_compressed(
        path,
        params=physical_samples,   # (N, 20) float64
        spectra=all_spectra,       # (N, 1000) float32
        frequencies=np.arange(1, NUM_FREQS + 1, dtype=np.int32),
    )
    size = os.path.getsize(path) / 1e9
    print(f"  NPZ saved: {size:.2f} GB ({time.time() - t0:.1f}s)", flush=True)
    return path


def save_parquet(physical_samples, all_spectra, output_dir, row_group_size=50_000):
    """Save as Parquet using PyArrow (memory-efficient, columnar)."""
    path = os.path.join(output_dir, "lhs_data_full_spectrum.parquet")
    print(f"\nSaving Parquet: {path}", flush=True)
    t0 = time.time()

    N = physical_samples.shape[0]
    avg_absorption = np.mean(all_spectra, axis=1)

    # Build PyArrow table column by column (more memory efficient than pandas)
    arrays = []
    names = []

    # Parameter columns (float64)
    for i, col_name in enumerate(PARAM_COL_NAMES):
        arrays.append(pa.array(physical_samples[:, i]))
        names.append(col_name)

    # Spectrum columns (float32 → stored as float)
    for f_idx in range(NUM_FREQS):
        arrays.append(pa.array(all_spectra[:, f_idx].astype(np.float32)))
        names.append(f"alpha_{f_idx + 1}")

    # Average absorption
    arrays.append(pa.array(avg_absorption.astype(np.float32)))
    names.append("Average_Absorption")

    table = pa.table(arrays, names=names)
    pq.write_table(table, path, row_group_size=row_group_size)

    del table, arrays
    gc.collect()

    size = os.path.getsize(path) / 1e9
    print(f"  Parquet saved: {size:.2f} GB ({time.time() - t0:.1f}s)", flush=True)
    return path


def save_csv(physical_samples, all_spectra, output_dir, write_chunk_size=10_000):
    """Save as CSV in chunks to manage memory."""
    path = os.path.join(output_dir, "lhs_data_full_spectrum.csv")
    print(f"\nSaving CSV: {path}", flush=True)
    print(f"  Writing in chunks of {write_chunk_size:,} rows...", flush=True)
    t0 = time.time()

    N = physical_samples.shape[0]
    avg_absorption = np.mean(all_spectra, axis=1)

    # Write header
    header = ",".join(ALL_COL_NAMES)
    with open(path, "w") as f:
        f.write(header + "\n")

    # Write data in chunks to avoid building a huge DataFrame
    num_chunks = (N + write_chunk_size - 1) // write_chunk_size
    for c_idx in range(num_chunks):
        c_start = c_idx * write_chunk_size
        c_end = min(c_start + write_chunk_size, N)

        # Build small chunk DataFrame
        chunk_data = np.concatenate([
            physical_samples[c_start:c_end],
            all_spectra[c_start:c_end].astype(np.float64),
            avg_absorption[c_start:c_end, np.newaxis],
        ], axis=1)

        chunk_df = pd.DataFrame(chunk_data, columns=ALL_COL_NAMES)
        chunk_df.to_csv(path, mode="a", header=False, index=False,
                        float_format="%.6f")
        del chunk_df, chunk_data

        if (c_idx + 1) % 10 == 0 or c_idx == num_chunks - 1:
            print(f"  CSV chunk {c_idx + 1}/{num_chunks} "
                  f"({c_end:,}/{N:,} rows)", flush=True)

    gc.collect()
    size = os.path.getsize(path) / 1e9
    print(f"  CSV saved: {size:.2f} GB ({time.time() - t0:.1f}s)", flush=True)
    return path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_against_theoretical_model(physical_samples, all_spectra, n_verify=10):
    """
    Verify GPU TMM output against the reference Theoretical_model.py.
    """
    print(f"\n{'=' * 70}", flush=True)
    print(f"VERIFICATION: GPU TMM vs Theoretical_model.py ({n_verify} samples)",
          flush=True)
    print(f"{'=' * 70}", flush=True)

    # Import reference implementation
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    sys.path.insert(0, src_dir)
    from Theoretical_model import calculate_acoustic_properties

    rng = np.random.default_rng(42)
    indices = rng.choice(len(physical_samples), size=n_verify, replace=False)

    max_abs_error = 0.0
    all_passed = True

    for idx in indices:
        row = physical_samples[idx]
        params_dict = {
            "rho": float(row[16]), "eta": float(row[17]),
            "E": float(row[18]), "nu": float(row[19]),
            "W": 2000.0,
            "d": [float(row[j]) for j in range(10)],
            "m": {2: float(row[10]), 3: float(row[11]), 5: float(row[12]),
                  6: float(row[13]), 8: float(row[14]), 9: float(row[15])},
        }

        _, alpha_ref, _, _ = calculate_acoustic_properties(params_dict)
        alpha_gpu = all_spectra[idx]
        abs_diff = np.abs(alpha_ref - alpha_gpu.astype(np.float64))
        max_diff = np.max(abs_diff)
        avg_diff = np.mean(abs_diff)

        if max_diff > max_abs_error:
            max_abs_error = max_diff

        status = "PASS" if max_diff < 1e-3 else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  Sample {idx:>7d}: max_diff={max_diff:.2e}, "
              f"avg_diff={avg_diff:.2e}  [{status}]", flush=True)

    print(f"\n  Overall max absolute error: {max_abs_error:.2e}", flush=True)
    print(f"  Threshold: 1e-3", flush=True)
    print(f"  Verdict: {'ALL PASSED' if all_passed else 'SOME FAILED!'}",
          flush=True)
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_SAMPLES = 1_000_000
    GPU_CHUNK_SIZE = 5000
    MEGA_CHUNK_SIZE = 50_000
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "data")

    print("=" * 70, flush=True)
    print("FULL-SPECTRUM GPU DATA GENERATION", flush=True)
    print("=" * 70, flush=True)
    print(f"  Samples:         {NUM_SAMPLES:,}", flush=True)
    print(f"  Frequencies:     1-{NUM_FREQS} Hz", flush=True)
    print(f"  Output columns:  {NUM_PARAMS} params + {NUM_FREQS} freqs "
          f"+ 1 avg = {NUM_PARAMS + NUM_FREQS + 1}", flush=True)
    print(f"  GPU:             {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  GPU chunk size:  {GPU_CHUNK_SIZE:,}", flush=True)
    print(f"  Mega chunk size: {MEGA_CHUNK_SIZE:,}", flush=True)
    print("=" * 70, flush=True)

    t_total_start = time.time()

    # --- Generate ---
    physical_samples, all_spectra = generate_dataset(
        num_samples=NUM_SAMPLES,
        gpu_chunk_size=GPU_CHUNK_SIZE,
        mega_chunk_size=MEGA_CHUNK_SIZE,
    )

    # --- Verify ---
    verify_against_theoretical_model(physical_samples, all_spectra, n_verify=10)

    # --- Save (NPZ first since it's fastest, then Parquet, then CSV) ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_npz(physical_samples, all_spectra, OUTPUT_DIR)
    save_parquet(physical_samples, all_spectra, OUTPUT_DIR)
    save_csv(physical_samples, all_spectra, OUTPUT_DIR)

    t_total = time.time() - t_total_start
    print(f"\n{'=' * 70}", flush=True)
    print(f"COMPLETE — Total time: {t_total:.1f}s ({t_total / 60:.1f} min)",
          flush=True)
    print(f"{'=' * 70}", flush=True)
