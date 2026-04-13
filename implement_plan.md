# Master Plan: Notebook 11 (cINN Full-Spectrum) Cleanup & Migration

## Goal

Clean up `notebooks/11_cINN_1DCNN.ipynb` to match the structure, polish, and visualization quality of `notebooks/10_Advanced_Audit_and_Improved_Diffusion.ipynb`, while preserving notebook 11's **full-spectrum cINN** semantics (instead of notebook 10's single-scalar-target diffusion model).

> [!IMPORTANT]
> **Key Semantic Difference:** Notebook 10's model is conditioned on a **single scalar** (target average absorption). Notebook 11's cINN is conditioned on the **entire 1000-point absorption spectrum**. This fundamentally changes how inference works, what "error" means, and how visualizations should be labeled.

---

## Background & Context

### Notebook 10 (Reference — Diffusion, single-target)
- **Model:** Physics-Guided Conditional Diffusion (V3)
- **Conditioning:** Single scalar (target avg absorption → `y_test[:, 0]`)
- **Inference:** `diffusion_v3.sample(cond, scaler=scaler_v3)` → 20 candidate parameter vectors
- **Error metric:** `|avg_achieved - target_avg| / target_avg × 100%`
- **Data structures:**
  - `test_results_v3` dict with keys: `avg_error`, `median_error`, `max_error`, `p95`, `best_predictions` (scalars), `best_designs` (20D params), `final_errors` (per-sample %), `single_avg`, `single_median`
  - `y_test` shape `(N, 1)` — single scalar target per sample
  - `diffusion_v3`, `scaler_v3` — model & scaler objects

### Notebook 11 (Target — cINN, full-spectrum)
- **Model:** Conditional Invertible Neural Network with 1D-CNN encoder
- **Conditioning:** Full 1000-point absorption spectrum (→ `Y_test[idx]`, shape `(1000,)`)
- **Inference:** `model(z_random, target_tensor, rev=True)` → 20 candidate parameter vectors
- **Error metric:** Spectral RMSE, R², frequency-wise relative error, etc.
- **Data structures:**
  - `test_results` dict with keys: `spectrum_mse`, `spectrum_relative_error`, `spectrum_r2`, `all_predictions` (list of tuples: `(target_spectrum, best_spectrum, best_params)`), `hit_rate`, `oob_rate`
  - `Y_test` shape `(N, 1000)` — full spectrum target per sample
  - `X_test` shape `(N, 20)` — parameter vectors
  - `model`, `scaler_params` — model & scaler objects

---

## Critical Issues Found in Notebook 11

### 1. Massive Dead Code (11 fully-commented cells)
| Cell | Lines | Content |
|------|-------|---------|
| 2 | 42 | Duplicate imports |
| 5 | 71 | Commented-out GPU TMM function |
| 11 | 66 | Old `SpectrumEncoder` (replaced by `SpectrumEncoder1DCNN`) |
| 14 | 84 | Old training code |
| 22 | 126 | Old evaluation (TMM-based) |
| 23 | 699 | Full duplicate evaluation #1 |
| 24 | 655 | Full duplicate evaluation #2 |
| 25 | 109 | Per-frequency accuracy deep dive |
| 27 | 83 | Uncertainty quantification |
| 28 | 374 | Forward-inverse consistency test |
| 29 | 101 | Interactive Plotly visualization |
| 57 | 115 | Commented spaghetti+heatmap code |

**Total: ~2,525 lines of 100% commented dead code.**

### 2. Diffusion-Era Variable References (will crash at runtime)
Cells 46, 47, 48, 50, 54 use `diffusion_v3`, `scaler_v3`, `test_results_v3` — these objects **do not exist** in notebook 11. They are notebook 10's diffusion model objects.

### 3. Semantic Mismatches in Visualization Cells (31–58)
- **Final Summary** (Cell 32): Uses cINN's `test_results` correctly ✅
- **TMM Candidate Eval** (Cell 34): Uses `test_results['all_predictions']` correctly ✅
- **Panel (b)** (Cells 35-36): Markdown references `test_results_v3`, code partially adapted but title still says "Physics-Guided Conditional Diffusion (V3)" ⚠️
- **Panel (c)** (Cells 37-38): Already adapted to use cINN inference (finds closest test spectrum as target) ✅ but markdown still says "diffusion model"
- **Panel (d)** (Cells 39-40): Still references `audit_results_09`, `nb09_metrics` — notebook 11 has no notebook 09 audit ⚠️
- **Frequency-Targeted** (Cells 41-42): Already adapted for full-spectrum cINN ✅
- **Top-5** (Cells 43-44): Works correctly (depends on cell 42) ✅
- **Best Test Sample / All 20 Candidates** (Cell 46): Uses `test_results_v3`, `diffusion_v3.sample()`, `scaler_v3` — **WILL CRASH** ❌
- **Top-4 / Top-6 Diagnostics** (Cells 47-48): Same as above — **WILL CRASH** ❌
- **Sweep** (Cell 50): Uses `diffusion_v3.sample()`, `scaler_v3` — **WILL CRASH** ❌
- **Pareto/t-SNE** (Cell 54): Same — **WILL CRASH** ❌

### 4. Missing Section Headers
Notebook 11 lacks proper markdown headers matching notebook 10's structure for sections 1–7 and has inconsistent numbering.

---

## Implementation Plan

### Phase 1: Dead Code Removal
**Goal:** Delete all 100%-commented cells to reduce clutter. No behavioral change.

| Step | Cell(s) | Action |
|------|---------|--------|
| 1.1 | Cell 2 | DELETE — duplicate imports (42 lines) |
| 1.2 | Cell 5 | DELETE — commented GPU TMM (71 lines) |
| 1.3 | Cell 11 | DELETE — old SpectrumEncoder (66 lines) |
| 1.4 | Cell 14 | DELETE — old training code (84 lines) |
| 1.5 | Cells 22-25 | DELETE — 4 dead evaluation blocks (~1,589 lines) |
| 1.6 | Cells 27-29 | DELETE — uncertainty/consistency/plotly (558 lines) |
| 1.7 | Cell 30 | DELETE — empty cell |
| 1.8 | Cell 57 | DELETE — commented spaghetti+heatmap (115 lines) |

**Checkpoint:** Notebook has ~47 cells (down from 59). Only active code remains. No runtime behavior change.

---

### Phase 2: Section Restructuring & Markdown Polish
**Goal:** Mirror notebook 10's section flow with full-spectrum language.

| Notebook 10 Section | Notebook 11 Equivalent | Action |
|---------------------|----------------------|--------|
| `# 10 — Advanced Audit & Improved...` | `# Full-Spectrum Inverse Design...` | ✅ Keep (already appropriate) |
| `## Parameter Constraints & Physics Engine (TMM)` | `# 1. Physics engine (TMM)` | Rename to `## Parameter Constraints & Physics Engine (TMM)` |
| `## Load Data & Train/Val/Test Split` | `# 2. Data loading...` | Rename to `## Load Data & Train/Val/Test Split` |
| `## AUDIT: Notebook 09 Model Integrity Check` | N/A | Skip (not applicable to cINN) |
| `## Forward Surrogate` | N/A | Skip (cINN doesn't use surrogate) |
| `## Improved Diffusion Model (V3)` | `# 3. Invertible architecture definition` | Rename to `## cINN Architecture — 1D-CNN Encoder + Invertible Coupling Blocks` |
| `## Save/Load Utilities & Training` | `# 4. Training the network` | Rename to `## Save/Load Utilities & Training` |
| `## Comprehensive Testing on Entire Test Set` | `# 6. Comprehensive evaluation...` | Rename to `## Comprehensive Testing on Entire Test Set` |
| N/A | `# 5. Inverse Inference...` | Rename to `## Inverse Inference — cINN Candidate Generation` |

**Additional cleanups:**
- Add a top-level configuration cell with runtime toggles (following Plan 1's approach):
  ```python
  # ═══════════════════════════════════════════════════════════════
  # RUNTIME CONFIGURATION
  # ═══════════════════════════════════════════════════════════════
  RUN_FULL_EVALUATION = False  # True → 100K test samples; False → 1000 samples (fast)
  RUN_SWEEP = False            # True → 901-level sweep; False → skip
  RUN_MANIFOLD = False         # True → t-SNE/UMAP; False → skip
  ```

**Checkpoint:** Notebook section headers match notebook 10's style. Runtime toggles added.

---

### Phase 3: Final Summary Migration (Cells 31-32)
**Goal:** Adapt the Final Summary to cINN full-spectrum semantics.

#### Cell 31 (Markdown) — Keep as-is ✅

#### Cell 32 (Code) — Enhance to match NB10 depth

Current NB11 Final Summary is **sparse** (36 lines, just prints basic stats). NB10's is **rich** (83 lines, includes audit conclusions, V3 performance, manual target, key improvements, FAQ).

**Adaptation:**
- Keep the cINN metrics block (spectrum MSE, R², relative error, OOB %, hit rate)
- Add: Architecture details section (encoder type, coupling blocks, conditioning dimension)
- Add: Key design decisions section (full-spectrum conditioning vs scalar, 1D-CNN vs MLP encoder)
- Add: Comparison framing — how this differs from NB10's diffusion approach
- Add: Physical constraints printout (same as NB10)
- Remove: Any FAQ about "cheating" / audit (not relevant here)
- **Error framing:** Change from "avg absorption error %" to "spectral RMSE / R² / per-frequency relative error %"

---

### Phase 4: Visualization Panel Migration (Cells 33–58)

This is the core work. Each panel must be adapted from NB10's diffusion/scalar semantics to NB11's cINN/full-spectrum semantics.

#### 4.1 — TMM Candidate Evaluation (Cells 33-34)
**Status:** Already correctly adapted ✅
**Minor fix:** Cell 34 uses `CURRENT_DIR` variable — verify it's defined. If not, replace with `'..'`.

---

#### 4.2 — Panel (b): Target vs Achieved Scatter Plot (Cells 35-36)
**Status:** Partially adapted ⚠️

**NB10 approach:** `targets = y_test[:, 0]` (scalar), `predictions = test_results_v3['best_predictions']` (scalar). Scatter of scalar→scalar.

**NB11 adaptation needed:**
- `targets = np.mean(Y_test, axis=1)` → average of target spectrum
- `predictions = np.mean(best_preds, axis=1)` → average of best-of-20 predicted spectrum
- `errors = test_results['spectrum_relative_error']` → spectral relative error
- **Title:** Change from "Physics-Guided Conditional Diffusion (V3)" → "Full-Spectrum cINN Inverse Design (1D-CNN)"
- **Markdown Cell 35:** Remove reference to `test_results_v3['best_predictions']`, update description to explain that this shows avg-absorption correlation as a summary statistic even though the model works on full spectra.

> [!NOTE]
> NB11's cell 36 has already been partially adapted (uses `np.mean(Y_test, axis=1)` and `test_results`). The main fix is the **title string** on line 371-373 which still says "Physics-Guided Conditional Diffusion (V3)".

---

#### 4.3 — Panel (c): Full-Curve Verification (Cells 37-38)
**Status:** Already adapted to use cINN inference ✅

**NB10 approach:** Uses `diffusion_v3.sample(cond, scaler=scaler_v3)` with scalar conditioning.
**NB11 current code:** Finds closest test spectrum to target avg, uses cINN's `model(z_random, cond, rev=True)` with full-spectrum conditioning. ✅

**Fixes needed:**
- **Markdown Cell 37:** Change "the diffusion model generates design parameters" → "the cINN generates design parameters"
- Titles/labels in plot: Already says "Predicted design" which is model-agnostic ✅
- The concept (Low/Medium/High target → find test spectrum → generate → verify via TMM) is correctly adapted

---

#### 4.4 — Panel (d): Method Comparison (Cells 39-40)
**Status:** Major rewrite needed ❌

**NB10 approach:** Compares V3 vs NB09 baseline vs Gao et al. using `test_results_v3` and `audit_results_09`.

**NB11 problem:** 
- `audit_results_09` does not exist in NB11 (no audit was run)
- `test_results_v3` does not exist
- The comparison should be cINN vs Gao et al. (and optionally vs NB10's diffusion if those results are available)

**Adaptation:**
- **Metrics source:** Use `test_results['spectrum_relative_error']` for cINN
- **Comparison methods:**
  - cINN (this work) — from `test_results`
  - Gao et al. — hardcoded reference (same: 0.026% and 0.33%, 2 cases)
  - NB10 Diffusion V3 — OPTIONAL, only if user has those results available. Otherwise, just cINN vs Gao et al.
- **Table:** Update "Physics in loss" → "No" for cINN, "Training data" → adjust
- **Labels:** "cINN — This Work (250 epochs)" instead of "V3 — This Work"
- **Markdown Cell 39:** Remove references to "V3", "Notebook 09 baseline"

> [!IMPORTANT]
> **Decision needed:** Should Panel (d) compare cINN vs Gao only, or also include NB10's diffusion results as a third method? If yes, we need to either hardcode NB10's metrics or create a mechanism to load them.

---

#### 4.5 — Frequency-Targeted Inverse Design (Cells 41-42)
**Status:** Conceptual rewrite needed ⚠️

**NB10 approach:** Since the diffusion model only takes avg absorption as input, it sweeps across multiple avg-absorption conditioning levels, generates candidates at each, then ranks by target-frequency match.

**NB11 reality:** The cINN takes the **full spectrum** as input. This means:
- We can construct a **synthetic target spectrum** with a Gaussian peak at the desired frequency (e.g., Gaussian centered at 600 Hz with absorption 0.8)
- Feed that directly to the cINN as conditioning
- No need to sweep across conditioning levels!

**Adaptation options:**
1. **Simple approach:** Keep the current NB10-style sweep logic (it works because it generates candidates and ranks them). The code already does this and works.
2. **Better approach (from Plan 2):** Add a configurable Gaussian target spectrum demo — construct a Gaussian-shaped absorption curve peaking at the target frequency, feed to cINN, and verify via TMM.

> [!IMPORTANT]
> **Decision needed:** Keep the current sweep approach (simpler, already works), or rewrite to use direct Gaussian-spectrum conditioning (more elegant, leverages cINN's full-spectrum capability)?

---

#### 4.6 — Best Test Sample — All 20 Candidates (Cell 46)
**Status:** Will crash, uses `diffusion_v3`, `scaler_v3`, `test_results_v3` ❌

**Rewrite:**
- Replace `test_results_v3` → `test_results`
- Replace `best_sample_idx = np.argmin(test_results_v3['final_errors'])` → `best_sample_idx = np.argmin(np.array(test_results['spectrum_mse']))`
- Replace `target_avg = y_test[best_sample_idx, 0]` → `target_spectrum = Y_test[best_sample_idx]`
- Replace `cond = torch.FloatTensor([[target_avg]]).repeat(...)` → `cond = torch.FloatTensor(target_spectrum).unsqueeze(0).repeat(NUM_CANDIDATES, 1).to(device)`
- Replace `diffusion_v3.sample(cond, scaler=scaler_v3)` → `model(z_random, cond, rev=True)` + `scaler_params.inverse_transform()`
- The target line changes from a **horizontal line** (scalar avg) to the **full target spectrum curve** (1000-point)
- Error calculation: curve-level RMSE instead of `|avg - target|`

---

#### 4.7 — Top-4 / Top-6 Best Test Samples (Cells 47-48)
**Status:** Will crash, same diffusion-era variables ❌

**Same rewrite pattern as 4.6:**
- `test_results_v3` → `test_results`
- `y_test[si, 0]` (scalar target) → `Y_test[si]` (1000-point spectrum)
- `diffusion_v3.sample()` → cINN inference
- Target visualization: full spectrum curve instead of horizontal line
- Error: RMSE of full curve instead of avg-absorption error

---

#### 4.8 — Design-Space Sweep (Cell 50)
**Status:** Will crash, uses `diffusion_v3`, `scaler_v3` ❌

**NB10 approach:** Sweeps scalar target absorption 0.1→1.0, generates 20 candidates per level via `diffusion_v3.sample()`.

**NB11 adaptation:**
- Instead of scalar conditioning, use **test set spectra** bucketed by average absorption level
- For each of the 901 target levels, find the closest test spectrum, use it as conditioning for cINN
- Or: construct synthetic flat-spectrum targets at each level (simpler)
- Replace `diffusion_v3.sample(chunk, scaler=scaler_v3)` → cINN inference
- Gate behind `RUN_SWEEP` toggle (this is expensive)

---

#### 4.9 — Sweep Metrics & CSV (Cell 52)
**Status:** Depends on sweep cell output arrays ⚠️
No variable fixes needed if sweep cell is fixed — this cell only uses `best_spectra`, `target_values`, `best_avgs`, `best_indices` which are computed in the sweep cell.

---

#### 4.10 — Pareto Front & t-SNE/UMAP (Cell 54)
**Status:** Will crash, uses `diffusion_v3`, `scaler_v3` ❌

**Same pattern as sweep:** Replace diffusion sampling with cINN inference.
Gate behind `RUN_MANIFOLD` toggle.

---

#### 4.11 — t-SNE/UMAP Re-render (Cell 55)
**Status:** Depends on Cell 54 output ⚠️
No fixes needed if Cell 54 is fixed — uses `emb_tsne`, `target_values`, `NUM_CAND`.

---

#### 4.12 — Publication Spaghetti+Heatmap (Cell 58 markdown, no code cell)
**Status:** Orphaned markdown with no code ⚠️
**Action:** Either delete or uncomment Cell 57 (the spaghetti+heatmap code).

---

### Phase 5: Consistency Pass & Verification

#### 5.1 — Variable Audit
Grep the entire notebook for these terms and ensure **zero active references**:
- `diffusion_v3` → replace with cINN inference
- `scaler_v3` → replace with `scaler_params`
- `test_results_v3` → replace with `test_results`
- `audit_results_09` → remove or make optional
- `y_test` (when used as `y_test[:, 0]`) → replace with `Y_test` (full spectrum)

#### 5.2 — Label & Title Consistency
- All plot titles: "cINN" not "Diffusion"
- All model references: "1D-CNN cINN" not "V3"
- Error language: "spectral RMSE" / "curve-level" not "average absorption error"
- Figure save paths: prefix with `cinn_` to avoid overwriting NB10 figures

#### 5.3 — Shape Assertions
Add lightweight checks before visualization cells:
```python
assert 'test_results' in globals(), "Run evaluation cell first"
assert len(test_results['spectrum_mse']) > 0, "No test results available"
```

#### 5.4 — Runtime Verification
- Fast-mode run (RUN_FULL_EVALUATION=False, 1000 samples): should complete in ~5 minutes
- All cells from Final Summary to end should execute without NameError
- All figures should save to `../figures/cinn_*`

---

## Summary: What Each Plan Got Right/Missing

| Aspect | Plan 1 | Plan 2 | This Master Plan |
|--------|--------|--------|-----------------|
| Dead code removal | ✅ Mentioned | ✅ Specific line refs | ✅ Full cell-by-cell inventory |
| Runtime toggles | ✅ Explicit | ✅ Mentioned | ✅ Detailed config cell |
| Diffusion→cINN variable mapping | ⚠️ Implied | ✅ Specific variables | ✅ Full mapping table |
| Panel (b) scatter fix | ⚠️ General | ⚠️ General | ✅ Exact line refs for title |
| Panel (c) concept | ⚠️ General | ✅ Correct | ✅ Already adapted |
| Panel (d) comparison | ⚠️ General | ✅ Good detail | ✅ + user decision needed |
| Freq-targeted design | ⚠️ General | ✅ Gaussian demo idea | ✅ Both options presented |
| Cells 46-48 crash fix | ⚠️ Implied | ✅ Identified | ✅ Full rewrite spec |
| Sweep cell crash fix | ⚠️ Implied | ✅ Identified | ✅ Full rewrite spec |
| Semantic difference (scalar vs spectrum) | ⚠️ Brief | ⚠️ Brief | ✅ Detailed analysis |
| `test_results` vs `test_results_v3` structure | ❌ Missing | ❌ Missing | ✅ Full API diff documented |

---

## Open Questions

> [!IMPORTANT]
> 1. **Panel (d) comparison:** Should we compare cINN vs Gao et al. only, or also include NB10's diffusion model results? If including diffusion, should we hardcode the metrics or load from a cache file?
Ans : This should be compared to Gao etal only and no mention of notebook 10 , infact i dont even want a comparision at all, 
all i want is this 2 images generated in the las cell of Notebook 10 "sweep_heatmap_TNR.png/pdf, sweep_all_curves_TNR.png/pdf" and this 3rd image ../figures/tsne_umap_design_space_TNR.png/pdf which is the end goal of this migration  
> [!IMPORTANT]
> 2. **Frequency-Targeted Design:** Keep the existing multi-level sweep approach (simpler, works now), or rewrite to use direct Gaussian-spectrum conditioning (leverages cINN's full-spectrum capability)? 
Ans : leverage full capability 
> [!IMPORTANT]
> 3. **Sweep cell approach:** For the 901-level sweep, should we use (a) closest test spectra as conditioning, (b) synthetic flat spectra, or (c) some other target construction method? 
Ans : i dont know anything reason deeply on this and choos the best based on yout preference 

> [!WARNING]
> 4. **Test evaluation size:** The current code runs on the **entire** test set with `num_samples=None`. This could be 100K+ samples × 20 candidates each = 2M+ inferences. Should we keep this behind a toggle or change the default? 
Ans : i think behind the toggle, 

---

## Verification Plan

### Automated Checks
1. `python -c "import json; nb = json.load(open('notebooks/11_cINN_1DCNN.ipynb')); [print('CRASH:', i) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and any(v in ''.join(c['source']) for v in ['diffusion_v3', 'scaler_v3', 'test_results_v3']) and not ''.join(c['source']).strip().startswith('#')]"` — Zero matches expected.
2. Count cells with >80% commented lines → should be 0.
3. All figure save paths use `cinn_` prefix.

### Manual Verification
- Run notebook from top to Final Summary in fast mode → no errors
- Spot-check 3 visualization panels for correct labels/titles
- Verify figure output in `../figures/`
