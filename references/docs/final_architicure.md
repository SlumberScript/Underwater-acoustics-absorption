# Physics-Guided Conditional Diffusion Framework  
## Inverse Design of Underwater Acoustic Metamaterials

---

# ════════════════════════════════════════════════════════════════════════
# PHASE 0 — FORWARD SURROGATE TRAINING
# ════════════════════════════════════════════════════════════════════════

## Objective

Train a differentiable neural surrogate that approximates the GPU-based
Transfer Matrix Method (TMM) acoustic simulation.

This surrogate replaces expensive, non-differentiable physics with a fast,
gradient-compatible model that can later guide inverse design.

---

## Architecture (As Implemented in Notebook)

```
  Training data (1M samples)
  ┌──────────────────┐
  │  x_0 [batch, 20] │   20 normalised design parameters
  │  (from LHS data) │
  └────────┬─────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────┐
  │           NETWORK 1: Forward Surrogate MLP          │
  │                                                    │
  │  Linear(20→256) → LeakyReLU(0.01) → BatchNorm1d   │
  │         ▼                                          │
  │  Linear(256→256) → LeakyReLU(0.01) → BatchNorm1d  │
  │         ▼                                          │
  │  Linear(256→128) → LeakyReLU(0.01)                 │
  │         ▼                                          │
  │  Linear(128→1)  → Sigmoid                          │
  └────────────────────────┬───────────────────────────┘
                           │
                    α̂_surr [batch, 1]   ← surrogate predicted absorption
                           │
                           │   compare against
                           │
                    α_tmm  [batch, 1]   ← true absorption from GPU TMM
                           │
                           ▼
              L_surr = MSE(α̂_surr, α_tmm)
                           │
                     Adam optimizer
                     CosineAnnealing LR  (T_max=30 per cycle)
                     150 epochs total
                           │
                           ▼
              ✓ NETWORK 1 TRAINED & FROZEN
                (requires_grad = False for all weights)
```

---

## Key Details

### Training Data
- 1,000,000 samples generated using Latin Hypercube Sampling (LHS)
- Each sample contains 20 normalised parameters in range `[0,1]`

### Why Normalisation?
Parameters such as:
- Thickness: 1–20 mm
- Young’s modulus: 1e7–1e8 Pa

Without scaling, large-magnitude values dominate gradients.

### Loss Function

\[
L_{surr} = MSE(\hat{\alpha}_{surr}, \alpha_{tmm})
\]

### Optimisation
- Adam optimiser
- Cosine Annealing LR  (T_max=30 per cycle, 5 cycles total)
- 150 epochs total
- Final model frozen (requires_grad=False)

Network 1 now acts as a **Physics Oracle**.

---

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — CONDITIONAL DIFFUSION TRAINING
# ════════════════════════════════════════════════════════════════════════

## Objective

Train Network 2 to generate parameter vectors conditioned on a target
absorption value.

Network 1 remains frozen and provides differentiable physics supervision.

---

## ONE TRAINING STEP

```
════════════════════════════════════════════════════════════════════════
                         ONE TRAINING STEP
════════════════════════════════════════════════════════════════════════

INPUTS FROM DATASET:
  x_0 [batch, 20]  ←── clean normalised design parameters
  c   [batch,  1]  ←── target average absorption (the label)
  ε   [batch, 20]  ←── random Gaussian noise  ~ N(0, I)
  t   [batch,  1]  ←── random integer 0..99   ~ Uniform

────────────────────────────────────────────────────────────────────────
STEP 1 — FORWARD DIFFUSION  (not a neural net, just math)
────────────────────────────────────────────────────────────────────────

  ᾱ_t  = product of (1 − βs) for s=1..t      (pre-computed cosine schedule)

        √ᾱ_t × x_0       √(1−ᾱ_t) × ε
             │                   │
             └─────────┬─────────┘
                       │  add together
                       ▼
               x_t [batch, 20]   ← noisy version of x_0

  (at t=0:  x_t ≈ x_0  — barely noisy)
  (at t=99: x_t ≈ ε    — pure noise)

────────────────────────────────────────────────────────────────────────
STEP 2 — NETWORK 2  FORWARD PASS
────────────────────────────────────────────────────────────────────────

 t [batch,1]                                 c [batch,1]
      │                                            │
      │  ╔══════════════════════════╗              │  ╔═══════════════════════╗
      │  ║  time_mlp (PATH A)       ║              │  ║  cond_mlp (PATH B)    ║
      │  ║                          ║              │  ║                       ║
      └─►║  SinusoidalEmbeddings    ║              └─►║  Linear(1→512)        ║
         ║  → [batch, 512]          ║                 ║  SiLU                 ║
         ║  Linear(512→512)         ║                 ║  Linear(512→512)      ║
         ║  SiLU                   ║                 ╚══════════╤════════════╝
         ║  Linear(512→512)         ║                           │
         ╚══════════╤══════════════╝                            │
                    │  t_emb [batch,512]         c_emb [batch,512]
                    │                                           │
                    └────────────────┬──────────────────────────┘
                                     │  element-wise  +
                                     ▼
                              global_cond [batch, 512]
                         "how noisy + what target absorption"
                                     │
                        ┌────────────┼──────────────────────────────────────┐
                        │            │  flows into ALL 4 FiLMBlocks          │
                        │            │                                       │
 x_t [batch, 20]        │            │                                       │
      │                 │            │                                       │
      │  ╔══════════════╪╗           │                                       │
      │  ║  input_proj   ║           │                                       │
      └─►║  Linear(20→512)║          │                                       │
         ╚══════╤════════╝           │                                       │
                │  h [batch, 512]    │                                       │
                ▼                                               │
         ╔════════════════════════════════════════════╗                     │
         ║  FiLMBlock 1                               ║                     │
         ╚══════════════════╤═════════════════════════╝
         ║  (structure identical for all 4 blocks)    ║
         ╚════════════════════════════════════════════╝

                     LayerNorm → SiLU → Linear(512→20)

                       ε_pred [batch, 20]   ← predicted noise

────────────────────────────────────────────────────────────────────────
STEP 3 — DUAL LOSS COMPUTATION
────────────────────────────────────────────────────────────────────────

 L_noise = MSE(ε_pred, ε)
 L_phys  = MSE(α̂_pred, c)

 L_total = L_noise  +  λ × L_phys

 λ ramps 0 → 0.1 over first 15 epochs
 (physics warmup — avoids early instability)

────────────────────────────────────────────────────────────────────────
STEP 4 — BACKWARD PASS
────────────────────────────────────────────────────────────────────────

 gradients flow ONLY through Network 2
 Network 1 weights: FROZEN, no update

 clip_grad_norm(Network2, max_norm=1.0)

 Adam + CosineAnnealingLR  (1e-3 → 1e-5, T_max=50 per cycle, 250 epochs total)

 EMA update:
 ema_weights ← 0.999 × ema_weights + 0.001 × current_weights

════════════════════════════════════════════════════════════════════════
```

---

# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — INFERENCE (PARAMETER GENERATION)
# ════════════════════════════════════════════════════════════════════════

## Objective

Generate physically valid metamaterial parameters for a desired absorption.

---

```
════════════════════════════════════════════════════════════════════════
                    INFERENCE  (generating parameters)
════════════════════════════════════════════════════════════════════════

  User provides:  c = 0.325678  (desired absorption)

  START:  x_99 ~ N(0, I)   shape [batch, 20]   ← pure Gaussian noise
               │
               │  repeat for t = 99, 98, 97, ... 2, 1, 0
               ▼
    ╔═══════════════════════════════════════════╗
    ║  EMA Network 2 (more stable than raw net) ║
    ║                                           ║
    ║  inputs:  x_t,  t,  c                     ║
    ║  output:  ε_pred [batch, 20]              ║
    ╚═══════════════════╤═══════════════════════╝
                        │
                        ▼
    DDPM reverse step:
    mean = (1/√α_t) × (x_t − (1−α_t)/√(1−ᾱ_t) × ε_pred)
    x_{t−1} = mean + √β_t × z      z~N(0,I)  if t>0, else z=0
                        │
                        │   every 10 steps:
                        ├─────────────────────────────┐
                        │   CONSTRAINT ENFORCEMENT    │
                        │   inverse_scaler(x)         │
                        │   clip to physical bounds   │
                        │   (d: 1-20mm, E: 1e7-1e8Pa) │
                        │   scaler(x)  back to [0,1]  │
                        └─────────────────────────────┘
                        │
               (loop 100 times total)
                        ▼
    x_0 [batch, 20]   ← denoised parameter vector
                        ▼
    inverse_scaler → 20 physical parameters
                        ▼
    GPU TMM physics engine  (NOT Network 1 — the real physics)
                        ▼
    select best candidate from batch of 20

════════════════════════════════════════════════════════════════════════
```

---

# Final Pipeline Summary

PHASE 0 → Learn differentiable physics  
PHASE 1 → Learn conditional reverse diffusion  
PHASE 2 → Generate & validate physically consistent designs  

---