# Plan: New Data Generation Script — GPU + Full Frequency Spectrum

## Summary

Create a **new** script `scripts/generate_data_full_spectrum.py` that:
1. Generates **1,000,000** samples (10x current) using GPU-accelerated TMM
2. Outputs full 1000-point absorption spectrum per sample (1 Hz to 1000 Hz)
3. Saves in all three formats: Parquet, CSV, NPZ

The existing `scripts/generate_data.py` and `data/lhs_data.csv` remain untouched.

---

## Task 1: Data Generation Accuracy Audit — RESULT

**Verdict: The physics in `generate_data.py` is CORRECT.** All equations verified against the paper. No changes needed.

One downstream issue: the notebook's `validate_and_clip_parameters` incorrectly clips E to `[1e7*(1+eta), 1e8*(1+eta)]` when it should be `[1e7, 1e8]`. This corrupted ~5% of training data. This is a notebook bug, not a data generation bug — to be fixed separately.

---

## Task 2 & 3: Implementation Plan

### New Script: `scripts/generate_data_full_spectrum.py`

#### Structure

```python
import torch, numpy, pandas, pyarrow.parquet, scipy.stats.qmc

# 1. GPU setup (assert CUDA)
# 2. calculate_absorption_spectrum_gpu(param_matrix, chunk_size=5000)
#    - Input: (N, 20) numpy float64 array
#    - Returns: (N, 1000) numpy float32 array
# 3. generate_dataset(num_samples=1_000_000)
#    - LHS sampling (same as current)
#    - Chunked GPU computation (write to disk progressively)
#    - Save Parquet, CSV, NPZ
# 4. Verification (compare against Theoretical_model.py)
```

#### GPU TMM Function: `calculate_absorption_spectrum_gpu`

- Takes `(N, 20)` physical parameters (d in mm, m in mm, material props)
- Converts to meters internally
- Processes in chunks of ~5000 samples to fit in 6 GB VRAM
- Vectorized across both samples AND frequencies: shape `(chunk, 1000)`
- Returns full `(N, 1000)` absorption spectrum (float32 for memory efficiency)
- Based on the validated GPU TMM pattern from the notebook

#### Memory Management for 1M Samples

At 1M samples × 1000 frequencies × 4 bytes (float32) = **4 GB** for spectra alone. Strategy:
- Generate in **mega-chunks** of 50,000 samples
- Each mega-chunk: generate params, compute spectra on GPU, write to disk
- Final concatenation via Parquet append or incremental NPZ saving
- Peak RAM usage: ~500 MB per mega-chunk (manageable)

#### Output Files

| File | Est. Size | Notes |
|------|-----------|-------|
| `data/lhs_data_full_spectrum.parquet` | ~2-3 GB | Compressed, fast I/O |
| `data/lhs_data_full_spectrum.csv` | ~10-15 GB | Very large — saved with reduced decimal precision (6 digits) |
| `data/lhs_data_full_spectrum.npz` | ~4-5 GB | Two arrays: params (1M×20 float64), spectra (1M×1000 float32) |

#### Column Layout (1021 columns)

```
d1, d2, ..., d10,           # Layer thicknesses (mm)
m2, m3, m5, m6, m8, m9,     # Hollow diameters (mm)
rho, eta, E, nu,             # Material properties
alpha_1, alpha_2, ..., alpha_1000,  # Absorption at each Hz
Average_Absorption           # Mean of alpha_1..alpha_1000 (backward compat)
```

---

## Implementation Steps

### Step 1: Create `scripts/generate_data_full_spectrum.py`

Core GPU function (adapted from notebook's validated TMM):

```python
def calculate_absorption_spectrum_gpu(param_matrix, chunk_size=5000):
    """
    GPU-accelerated TMM returning full spectrum.
    Input:  (N, 20) numpy array — physical units
    Output: (N, 1000) numpy float32 array — absorption at 1-1000 Hz
    """
    # Process in chunks to fit GPU memory
    # For each chunk:
    #   1. Move params to GPU as float32 tensors
    #   2. Compute Lame constants (complex)
    #   3. Vectorized frequency loop (all 1000 freqs at once)
    #   4. 10-layer transfer matrix multiplication
    #   5. Reflection -> absorption
    #   6. Return (chunk, 1000) without averaging
```

Main generation function:

```python
def generate_dataset(num_samples=1_000_000, mega_chunk_size=50_000):
    """
    1. LHS sampling (scipy qmc) for 20 parameters
    2. Process in mega-chunks: generate spectra, write incrementally
    3. Save Parquet, CSV, NPZ
    """
```

### Step 2: Verification

After generation, verify correctness:
1. Take 10 random samples from the new dataset
2. Compute their absorption via `Theoretical_model.py` (CPU, per-frequency)
3. Compare: max absolute difference at any frequency < 1e-4
4. Verify: mean of alpha_1..alpha_1000 == Average_Absorption column

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/generate_data_full_spectrum.py` | New GPU data generation script |
| `data/lhs_data_full_spectrum.parquet` | Primary output |
| `data/lhs_data_full_spectrum.csv` | Secondary output (human readable) |
| `data/lhs_data_full_spectrum.npz` | Tertiary output (fast ML loading) |

## Files NOT Modified

| File | Reason |
|------|--------|
| `scripts/generate_data.py` | Preserved as-is per user request |
| `data/lhs_data.csv` | Preserved as-is |
| All notebooks | Reference only, per user instruction |
| `src/Theoretical_model.py` | Reference physics engine |

---

## Verification Checklist

- [ ] CUDA GPU is used (not CPU fallback)
- [ ] All 1,000,000 samples generated
- [ ] All alpha values in [0.0, 1.0]
- [ ] Average_Absorption = mean(alpha_1..alpha_1000) for every row
- [ ] 10 random samples match Theoretical_model.py output within 1e-4
- [ ] First 100 samples' averages match existing lhs_data.csv averages (regression test with the same LHS seed)
- [ ] Parquet, CSV, NPZ all saved successfully
- [ ] Script prints timing info (expected: ~10-30 min for 1M samples on RTX 4050)
