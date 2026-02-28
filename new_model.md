# Complete End-to-End Architecture: `FullSpectrum_cINN` (1D-CNN Encoder Variant)

> **One diagram, every component, both directions.**  
> Total parameters: **2,946,272 (~2.9M)**  
> 3 learnable components: 1 SpectrumEncoder1DCNN + 8 CondAffineCoupling SubNets

---

## Single Integrated Flow Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║  DATA GENERATION (offline, before training)                                              ║
║                                                                                          ║
║  Latin Hypercube Sampling → 1,000,000 parameter sets (20 dims each)                      ║
║                                                                                          ║
║  20 Physical Parameters per sample:                                                      ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │ idx 0-9:   d1–d10  layer thicknesses    (1–20 mm)                                  │  ║
║  │ idx 10-15: m2,m3,m5,m6,m8,m9  cavity ∅  (20–1980 mm)                               │  ║
║  │ idx 16:    ρ  density                    (1000–1500 kg/m³)                          │  ║
║  │ idx 17:    η  loss factor                (0.1–0.8)                                  │  ║
║  │ idx 18:    E  Young's modulus            (1e7–1e8 Pa)  constrained by η             │  ║
║  │ idx 19:    ν  Poisson's ratio            (0.4–0.49)                                 │  ║
║  └─────────────────────────────────────────────────────────────────────────────────────┘  ║
║         │                                                                                ║
║         ▼                                                                                ║
║  ┌──────────────────────────────────────────────────┐                                    ║
║  │  TMM Physics Engine (GPU-accelerated)            │                                    ║
║  │  10-layer structure, rigid backing, 1–1000 Hz    │                                    ║
║  │                                                  │                                    ║
║  │  For each layer (10 total):                      │                                    ║
║  │    E_c = E(1 + jη)   complex modulus             │                                    ║
║  │    λ, μ = Lamé params from E_c, ν               │                                    ║
║  │    ρ_eff, S_eff = f(ρ, λ, μ, ε)                │                                    ║
║  │    c_eff = √(S_eff/ρ_eff)                      │                                    ║
║  │    k = ω/c_eff,  Z = ρ_eff·c_eff              │                                    ║
║  │    T_i = [[cos(kd), jZ·sin(kd)],               │                                    ║
║  │           [j·sin(kd)/Z, cos(kd)]]              │                                    ║
║  │    T_total = T_total × T_i                      │                                    ║
║  │                                                  │                                    ║
║  │  Z_in = T11/T21                                  │                                    ║
║  │  R = (Z_in - Z_w)/(Z_in + Z_w)                  │                                    ║
║  │  α = 1 - |R|²          absorption ∈ [0, 1]      │                                    ║
║  └──────────────────────┬───────────────────────────┘                                    ║
║                         │                                                                ║
║                         ▼                                                                ║
║  params (1M, 20) ──────────── spectra (1M, 1000)                                        ║
║         │                           │                                                    ║
║         ▼                           │                                                    ║
║  validate_and_clip_parameters()     │                                                    ║
║         │                           │                                                    ║
║         ▼                           │                                                    ║
║  StandardScaler.fit_transform()     │  (spectra stay raw — no scaling)                   ║
║  (fit on TRAIN set only)            │                                                    ║
║         │                           │                                                    ║
║         ▼                           ▼                                                    ║
║  80/10/10 split → X_train(B,20)  Y_train(B,1000)  → DataLoader(batch=2048)              ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║   ██████████████████████████████████████████████████████████████████████████████████████   ║
║   █                    TRAINING  (Forward Direction)                                 █   ║
║   ██████████████████████████████████████████████████████████████████████████████████████   ║
║                                                                                          ║
║   x_params(B,20)                           y_spectrum(B,1000)                            ║
║   (scaled params)                          (raw absorption curve)                        ║
║       │                                          │                                       ║
║       │                                          ▼                                       ║
║       │                               ╔════════════════════════════════╗                 ║
║       │                               ║  SPECTRUM ENCODER (1D-CNN)     ║  2,094,144 p    ║
║       │                               ║                                ║                 ║
║       │                               ║  (B,1000) → unsqueeze(1)       ║                 ║
║       │                               ║         → (B,1,1000)           ║                 ║
║       │                               ║                                ║                 ║
║       │                               ║  Conv1d(1→16, k=5, s=2, p=2)  ║                 ║
║       │                               ║  LeakyReLU(0.01)    → (B,16,500)               ║
║       │                               ║                                ║                 ║
║       │                               ║  Conv1d(16→32, k=5, s=2, p=2) ║                 ║
║       │                               ║  LeakyReLU(0.01)    → (B,32,250)               ║
║       │                               ║                                ║                 ║
║       │                               ║  Conv1d(32→64, k=5, s=2, p=2) ║                 ║
║       │                               ║  LeakyReLU(0.01)    → (B,64,125)               ║
║       │                               ║                                ║                 ║
║       │                               ║  Flatten()           → (B,8000)║                 ║
║       │                               ║  Linear(8000→256)              ║                 ║
║       │                               ║  LeakyReLU(0.01)              ║                 ║
║       │                               ║  Linear(256→128)              ║                 ║
║       │                               ╚═══════════════╤════════════════╝                 ║
║       │                                               │                                  ║
║       │                                         cond (B,128)                             ║
║       │                                         (shared by ALL 8 blocks)                 ║
║       │                                               │                                  ║
║       ▼                                               ▼                                  ║
║  ┌────────────────────────────────────────────────────────────────────────────────────┐   ║
║  │                                                                                    │   ║
║  │   ╔═══════════════════════  COUPLING BLOCK k  ═══════════════════════════════╗     │   ║
║  │   ║                                                                          ║     │   ║
║  │   ║   x (B,20)                                                               ║     │   ║
║  │   ║    │                                                                      ║     │   ║
║  │   ║    ├──────── x1 = x[:, :10]          x2 = x[:, 10:]                      ║     │   ║
║  │   ║    │         (B,10) UNCHANGED         (B,10) TO BE TRANSFORMED            ║     │   ║
║  │   ║    │              │                          │                            ║     │   ║
║  │   ║    │              ▼                          │                            ║     │   ║
║  │   ║    │     cat([x1, cond]) → (B, 138)          │                            ║     │   ║
║  │   ║    │              │                          │                            ║     │   ║
║  │   ║    │              ▼                          │                            ║     │   ║
║  │   ║    │     ╔════════════════════╗              │                            ║     │   ║
║  │   ║    │     ║   SubNet_k         ║              │  106,516 params per block  ║     │   ║
║  │   ║    │     ║   Linear(138→256)  ║              │  (each block has its OWN   ║     │   ║
║  │   ║    │     ║   LeakyReLU(0.01)  ║              │   independent weights)     ║     │   ║
║  │   ║    │     ║   Linear(256→256)  ║              │                            ║     │   ║
║  │   ║    │     ║   LeakyReLU(0.01)  ║              │                            ║     │   ║
║  │   ║    │     ║   Linear(256→20)   ║              │                            ║     │   ║
║  │   ║    │     ╚════════╤═══════════╝              │                            ║     │   ║
║  │   ║    │              │                          │                            ║     │   ║
║  │   ║    │         st (B,20)                       │                            ║     │   ║
║  │   ║    │          │       │                      │                            ║     │   ║
║  │   ║    │     s=st[:,:10]  t=st[:,10:]            │                            ║     │   ║
║  │   ║    │     (B,10)       (B,10)                 │                            ║     │   ║
║  │   ║    │          │                              │                            ║     │   ║
║  │   ║    │     s = clamp(s, -3, +3)                │                            ║     │   ║
║  │   ║    │          │       │                      │                            ║     │   ║
║  │   ║    │          ▼       ▼                      ▼                            ║     │   ║
║  │   ║    │     ┌──────────────────────────────────────┐                         ║     │   ║
║  │   ║    │     │  AFFINE TRANSFORM (element-wise)     │  0 params               ║     │   ║
║  │   ║    │     │  y2[i] = x2[i] × exp(s[i]) + t[i]   │  for i = 0..9           ║     │   ║
║  │   ║    │     └──────────────────┬───────────────────┘                         ║     │   ║
║  │   ║    │                        │                                             ║     │   ║
║  │   ║    │     y1 = x1            y2 (B,10)                                     ║     │   ║
║  │   ║    │          │              │                                            ║     │   ║
║  │   ║    │          ▼              ▼                                            ║     │   ║
║  │   ║    │     output = cat([y1, y2]) → (B, 20)                                 ║     │   ║
║  │   ║    │                                                                      ║     │   ║
║  │   ║    │     log_det_J += sum(s, dim=1)     (B,) scalar per sample            ║     │   ║
║  │   ║    │                                                                      ║     │   ║
║  │   ╚════╩══════════════════════════════════════════════════════════════════════╝     │   ║
║  │                        │                                                           │   ║
║  │                        ▼                                                           │   ║
║  │              ╔════════════════════╗                                                 │   ║
║  │              ║   FLIP (0 params)  ║                                                 │   ║
║  │              ║   torch.flip(x, 1) ║   reverses all 20 columns                      │   ║
║  │              ║   old x1 → new x2  ║   so NEXT block transforms the OTHER half      │   ║
║  │              ║   old x2 → new x1  ║                                                 │   ║
║  │              ╚═════════╤══════════╝                                                 │   ║
║  │                        │                                                           │   ║
║  │                        ▼                                                           │   ║
║  │              REPEAT for k = 0, 1, 2, 3, 4, 5, 6, 7                                │   ║
║  │                                                                                    │   ║
║  │   Block 0: transforms dims 10-19  ──flip──  Block 1: transforms dims 0-9           │   ║
║  │   Block 2: transforms dims 10-19  ──flip──  Block 3: transforms dims 0-9           │   ║
║  │   Block 4: transforms dims 10-19  ──flip──  Block 5: transforms dims 0-9           │   ║
║  │   Block 6: transforms dims 10-19  ──flip──  Block 7: transforms dims 0-9           │   ║
║  │                                                                                    │   ║
║  │   Each half transformed 4 times ✓                                                  │   ║
║  └────────────────────────────────────────────────────────────────────────────────────┘   ║
║                        │                                                                 ║
║                        ▼                                                                 ║
║              z (B, 20)              log_det_J_total (B,)                                  ║
║              (latent space)         (sum of 8 blocks)                                     ║
║                   │                        │                                             ║
║                   ▼                        ▼                                             ║
║          ╔═════════════════════════════════════════════╗                                  ║
║          ║              NLL LOSS                       ║                                  ║
║          ║  L = mean( 0.5 × Σ(z²) - log_det_J_total ) ║                                  ║
║          ║       ↑                    ↑                ║                                  ║
║          ║  push z toward N(0,I)    reward volume      ║                                  ║
║          ╚══════════════════╤══════════════════════════╝                                  ║
║                             │                                                            ║
║                             ▼                                                            ║
║                    loss.backward()                                                       ║
║                             │                                                            ║
║              ┌──────────────┴──────────────┐                                             ║
║              ▼                             ▼                                             ║
║    gradients → 8 SubNets          gradients → 1D-CNN Encoder                             ║
║    (8 × 3 Linear layers)          (3 Conv1d + 2 Linear layers)                           ║
║              │                             │                                             ║
║              └──────────────┬──────────────┘                                             ║
║                             ▼                                                            ║
║              clip_grad_norm_(max_norm=1.0)                                               ║
║                             │                                                            ║
║                             ▼                                                            ║
║              Adam(lr=1e-3, weight_decay=1e-5)                                            ║
║              CosineAnnealingLR(T_max=250)                                                ║
║              250 epochs, batch=2048                                                      ║
║                                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║   ██████████████████████████████████████████████████████████████████████████████████████   ║
║   █                   INFERENCE  (Reverse Direction)                                 █   ║
║   ██████████████████████████████████████████████████████████████████████████████████████   ║
║                                                                                          ║
║   target_spectrum (1, 1000)                                                              ║
║       │                                                                                  ║
║       ├──→ repeat(num_candidates, 1) → (C, 1000)     z ~ N(0,I)  → (C, 20)             ║
║       │                                                     │                            ║
║       ▼                                                     │                            ║
║   ╔════════════════════════════════════╗                     │                            ║
║   ║  SPECTRUM ENCODER (1D-CNN)         ║                     │                            ║
║   ║  (SAME weights as training)        ║                     │                            ║
║   ║                                    ║                     │                            ║
║   ║  (C,1000) → unsqueeze → (C,1,1000)║                     │                            ║
║   ║  Conv1d(1→16) → (C,16,500)        ║                     │                            ║
║   ║  Conv1d(16→32) → (C,32,250)       ║                     │                            ║
║   ║  Conv1d(32→64) → (C,64,125)       ║                     │                            ║
║   ║  Flatten → Linear(8000→256)        ║                     │                            ║
║   ║  Linear(256→128)                   ║                     │                            ║
║   ╚═════════════╤══════════════════════╝                     │                            ║
║                 │                                            │                            ║
║            cond (C, 128)                                     │                            ║
║                 │                                            │                            ║
║                 └────────────────────┬───────────────────────┘                            ║
║                                      │                                                   ║
║                                      ▼                                                   ║
║   ┌──────────────────────────────────────────────────────────────────────────────────┐   ║
║   │                                                                                  │   ║
║   │   for block in REVERSED(blocks):   (block 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0)      │   ║
║   │                                                                                  │   ║
║   │       ╔════════════════════╗                                                     │   ║
║   │       ║   UN-FLIP first   ║   torch.flip(x, 1)  — undo the forward's flip       │   ║
║   │       ╚═════════╤════════╝                                                       │   ║
║   │                 │                                                                │   ║
║   │                 ▼                                                                │   ║
║   │       ╔═════════════════════════════════════════════════╗                         │   ║
║   │       ║   REVERSE COUPLING BLOCK k                     ║                         │   ║
║   │       ║                                                ║                         │   ║
║   │       ║   x1 = x[:, :10]        x2 = x[:, 10:]        ║                         │   ║
║   │       ║        │                       │               ║                         │   ║
║   │       ║        ▼                       │               ║                         │   ║
║   │       ║   cat([x1, cond])              │               ║                         │   ║
║   │       ║        │                       │               ║                         │   ║
║   │       ║        ▼                       │               ║                         │   ║
║   │       ║   SubNet_k → st (B,20)         │               ║                         │   ║
║   │       ║   s=st[:,:10]  t=st[:,10:]     │               ║                         │   ║
║   │       ║   s = clamp(s, -3, +3)         │               ║                         │   ║
║   │       ║        │       │               │               ║                         │   ║
║   │       ║        ▼       ▼               ▼               ║                         │   ║
║   │       ║   ┌────────────────────────────────────┐       ║                         │   ║
║   │       ║   │ INVERSE AFFINE (element-wise)      │       ║                         │   ║
║   │       ║   │ y2[i] = (x2[i] - t[i]) × exp(-s[i])│       ║                         │   ║
║   │       ║   └──────────────────┬─────────────────┘       ║                         │   ║
║   │       ║                      │                         ║                         │   ║
║   │       ║   output = cat([x1, y2]) → (B, 20)            ║                         │   ║
║   │       ╚══════════════════════╤══════════════════════════╝                         │   ║
║   │                              │                                                   │   ║
║   │                              ▼                                                   │   ║
║   │              REPEAT for k = 7, 6, 5, 4, 3, 2, 1, 0                              │   ║
║   │                                                                                  │   ║
║   └──────────────────────────────┬───────────────────────────────────────────────────┘   ║
║                                  │                                                       ║
║                                  ▼                                                       ║
║                    x_scaled (C, 20)    (StandardScaler space)                            ║
║                                  │                                                       ║
║                                  ▼                                                       ║
║                 ╔══════════════════════════════════╗                                     ║
║                 ║  scaler.inverse_transform()      ║                                     ║
║                 ║  x_physical = x_scaled × σ + μ   ║                                     ║
║                 ╚═══════════════╤══════════════════╝                                     ║
║                                 │                                                        ║
║                                 ▼                                                        ║
║                 ╔══════════════════════════════════╗                                     ║
║                 ║  validate_and_clip_parameters()  ║                                     ║
║                 ║  • Clip d1-d10 to [1, 20] mm     ║                                     ║
║                 ║  • Clip m's to [20, 1980] mm     ║                                     ║
║                 ║  • Clip ρ,η,E,ν to bounds        ║                                     ║
║                 ║  • Dynamic E constraint:          ║                                     ║
║                 ║    E ∈ [1e7(1+η), 1e8(1+η)]     ║                                     ║
║                 ╚═══════════════╤══════════════════╝                                     ║
║                                 │                                                        ║
║                                 ▼                                                        ║
║                 ╔══════════════════════════════════╗                                     ║
║                 ║  TMM Physics Engine              ║                                     ║
║                 ║  (same engine used for data gen) ║                                     ║
║                 ║  params → predicted_spectrum     ║                                     ║
║                 ╚═══════════════╤══════════════════╝                                     ║
║                                 │                                                        ║
║                                 ▼                                                        ║
║                  predicted_spectra (C, 1000)                                             ║
║                                 │                                                        ║
║                                 ▼                                                        ║
║                 ╔══════════════════════════════════╗                                     ║
║                 ║  CANDIDATE SELECTION             ║                                     ║
║                 ║  RMSE = √mean((pred - target)²)  ║                                     ║
║                 ║  best_idx = argmin(RMSE)         ║                                     ║
║                 ╚═══════════════╤══════════════════╝                                     ║
║                                 │                                                        ║
║                                 ▼                                                        ║
║               best_params (1, 20)  +  best_spectrum (1, 1000)                            ║
║                                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Spectrum Encoder: MLP vs 1D-CNN Comparison

| Property | Old (MLP) | New (1D-CNN) |
|----------|-----------|--------------|
| **Class** | `SpectrumEncoder` | `SpectrumEncoder1DCNN` |
| **Input** | `(B, 1000)` | `(B, 1000)` → reshaped to `(B, 1, 1000)` |
| **Feature extraction** | 3 fully-connected layers | 3 Conv1d layers + 2 fully-connected layers |
| **Local pattern detection** | None (global mixing only) | Yes — kernel size 5 captures local spectral features |
| **Spatial downsampling** | Immediate 1000→256 | Gradual 1000→500→250→125 via stride-2 convolutions |
| **Output** | `cond ∈ ℝ^(B×128)` | `cond ∈ ℝ^(B×128)` (identical interface) |
| **Parameters** | 354,944 | 2,094,144 |
| **Total model params** | 1,207,072 (~1.2M) | 2,946,272 (~2.9M) |

**Why 1D-CNN?** Absorption spectra are spatially structured signals — neighbouring frequency bins are highly correlated. Convolutional layers exploit this local structure via shared-weight sliding kernels, extracting multi-scale spectral features (peaks, slopes, resonance patterns) that an MLP must learn from scratch. The hierarchical feature extraction (1→16→32→64 channels) mirrors the multi-scale physics of acoustic absorption.

---

## 1D-CNN Spectrum Encoder — Detailed Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        SPECTRUM ENCODER (1D-CNN variant)                                  │
│                                                                                          │
│   Function: Compress 1000-dim absorption spectrum into 128-dim condition embedding       │
│   Key change: Replaces MLP with Conv1d stack for local feature extraction                │
│                                                                                          │
│   y_spectra ∈ ℝ^(B×1000)                                                                │
│        │                                                                                 │
│        ▼                                                                                 │
│   unsqueeze(dim=1)  →  (B, 1, 1000)          ◄── add channel dim for Conv1d             │
│        │                                                                                 │
│        ▼                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                                │     │
│   │  ╔══════════════════════  CONV BLOCK 1  ══════════════════════════════════╗    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  Conv1d(in=1, out=16, kernel=5, stride=2, padding=2)                   ║    │     │
│   │  ║    • 16 filters, each slides across the 1000-point spectrum            ║    │     │
│   │  ║    • receptive field: 5 frequency bins                                 ║    │     │
│   │  ║    • stride=2 halves spatial dimension: 1000 → 500                     ║    │     │
│   │  ║    • output: (B, 16, 500)                                              ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  LeakyReLU(negative_slope=0.01)                                        ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ╚════════════════════════════════════════════════════════════════════════╝    │     │
│   │       │                                                                        │     │
│   │       ▼                                                                        │     │
│   │  ╔══════════════════════  CONV BLOCK 2  ══════════════════════════════════╗    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  Conv1d(in=16, out=32, kernel=5, stride=2, padding=2)                  ║    │     │
│   │  ║    • 32 filters operating on 16-channel feature maps                   ║    │     │
│   │  ║    • receptive field: 5 × 2 = 10 original freq bins (effective)        ║    │     │
│   │  ║    • stride=2 halves spatial dimension: 500 → 250                      ║    │     │
│   │  ║    • output: (B, 32, 250)                                              ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  LeakyReLU(negative_slope=0.01)                                        ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ╚════════════════════════════════════════════════════════════════════════╝    │     │
│   │       │                                                                        │     │
│   │       ▼                                                                        │     │
│   │  ╔══════════════════════  CONV BLOCK 3  ══════════════════════════════════╗    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  Conv1d(in=32, out=64, kernel=5, stride=2, padding=2)                  ║    │     │
│   │  ║    • 64 filters operating on 32-channel feature maps                   ║    │     │
│   │  ║    • receptive field: 5 × 4 = 20 original freq bins (effective)        ║    │     │
│   │  ║    • stride=2 halves spatial dimension: 250 → 125                      ║    │     │
│   │  ║    • output: (B, 64, 125)                                              ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  LeakyReLU(negative_slope=0.01)                                        ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ╚════════════════════════════════════════════════════════════════════════╝    │     │
│   │       │                                                                        │     │
│   │       ▼                                                                        │     │
│   │  Flatten()  →  (B, 8000)          ◄── 64 channels × 125 spatial = 8000        │     │
│   │       │                                                                        │     │
│   │       ▼                                                                        │     │
│   │  ╔══════════════════════  FC HEAD  ═══════════════════════════════════════╗    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  Linear(8000 → 256)                                                    ║    │     │
│   │  ║  LeakyReLU(negative_slope=0.01)                                        ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ║  Linear(256 → 128)                                                     ║    │     │
│   │  ║                                                                        ║    │     │
│   │  ╚════════════════════════════════════════════════════════════════════════╝    │     │
│   │                                                                                │     │
│   └────────────────────────────────────────────────────────────────────────────────┘     │
│        │                                                                                 │
│        ▼                                                                                 │
│   cond ∈ ℝ^(B×128)                                                                      │
│                                                                                          │
│═════════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                          │
│   SPATIAL DIMENSION PROGRESSION:                                                         │
│                                                                                          │
│   Input:    ───────────────────────────────────────────── 1000 freq bins                 │
│   Conv1d-1: ─────────────────────────────── 500 (stride 2, ÷2)                           │
│   Conv1d-2: ──────────────────── 250 (stride 2, ÷2)                                     │
│   Conv1d-3: ────────── 125 (stride 2, ÷2)                                               │
│   Flatten:  8000 (64 ch × 125)                                                          │
│   FC-1:     256                                                                          │
│   FC-2:     128  ← final condition embedding                                            │
│                                                                                          │
│   CHANNEL PROGRESSION:                                                                   │
│                                                                                          │
│   1 ──→ 16 ──→ 32 ──→ 64                                                                │
│   (raw signal)  (low-level)  (mid-level)  (high-level spectral features)                │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Network Component Details and Parameter Counts

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                       1D-CNN SPECTRUM ENCODER — LAYER-BY-LAYER                            │
│                                                                                          │
│   y_spectra ∈ ℝ^(B×1000) → unsqueeze → (B, 1, 1000)                                    │
│        │                                                                                 │
│        ▼                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────┐              │
│   │  Layer                │ Shape              │ Weights  │ Biases│ Total  │              │
│   │  ─────────────────────┼────────────────────┼──────────┼───────┼────────│              │
│   │  Conv1d_1             │ (1→16, k=5, s=2)   │ 80       │ 16    │ 96     │              │
│   │  LeakyReLU(0.01)      │ —                  │ 0        │ 0     │ 0      │              │
│   │  Conv1d_2             │ (16→32, k=5, s=2)  │ 2,560    │ 32    │ 2,592  │              │
│   │  LeakyReLU(0.01)      │ —                  │ 0        │ 0     │ 0      │              │
│   │  Conv1d_3             │ (32→64, k=5, s=2)  │ 10,240   │ 64    │ 10,304 │              │
│   │  LeakyReLU(0.01)      │ —                  │ 0        │ 0     │ 0      │              │
│   │  Flatten              │ (B,64,125)→(B,8000)│ 0        │ 0     │ 0      │              │
│   │  Linear_1 (FC)        │ 8000 → 256         │2,048,000 │ 256   │2,048,256│             │
│   │  LeakyReLU(0.01)      │ —                  │ 0        │ 0     │ 0      │              │
│   │  Linear_2 (FC)        │ 256 → 128          │ 32,768   │ 128   │ 32,896 │              │
│   │  ─────────────────────┼────────────────────┼──────────┼───────┼────────│              │
│   │  SUBTOTAL             │                    │2,093,648 │ 496   │2,094,144│             │
│   └────────────────────────────────────────────────────────────────────────┘              │
│        │                                                                                 │
│        ▼                                                                                 │
│   cond ∈ ℝ^(B×128)                                                                      │
│                                                                                          │
│─────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                          │
│                     SINGLE COUPLING BLOCK SUBNET (unchanged)                             │
│                                                                                          │
│   Function: Predict affine parameters (scale s, translation t) from                      │
│             unchanged half x₁ concatenated with condition embedding                      │
│                                                                                          │
│   [x₁ ∈ ℝ^10 ; cond ∈ ℝ^128] = input ∈ ℝ^138                                          │
│        │                                                                                 │
│        ▼                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────┐              │
│   │  Layer                │ Shape          │ Weights   │ Biases │ Total   │              │
│   │  ─────────────────────┼────────────────┼───────────┼────────┼─────────│              │
│   │  Linear_1             │ 138 → 256      │ 35,328    │ 256    │ 35,584  │              │
│   │  LeakyReLU(0.01)      │ —              │ 0         │ 0      │ 0       │              │
│   │  Linear_2             │ 256 → 256      │ 65,536    │ 256    │ 65,792  │              │
│   │  LeakyReLU(0.01)      │ —              │ 0         │ 0      │ 0       │              │
│   │  Linear_3             │ 256 → 20       │ 5,120     │ 20     │ 5,140   │              │
│   │  ─────────────────────┼────────────────┼───────────┼────────┼─────────│              │
│   │  SUBTOTAL (per block) │                │ 105,984   │ 532    │ 106,516 │              │
│   └────────────────────────────────────────────────────────────────────────┘              │
│        │                                                                                 │
│        ▼                                                                                 │
│   [s ∈ ℝ^10 ; t ∈ ℝ^10] = output ∈ ℝ^20                                                │
│                                                                                          │
│─────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                          │
│                       PARAMETER COUNT SUMMARY                                            │
│                                                                                          │
│   ┌─────────────────────────────┬────────────────┬───────────────────┐                   │
│   │  Component                  │ Count          │ Parameters        │                   │
│   │  ───────────────────────────┼────────────────┼───────────────────│                   │
│   │  SpectrumEncoder1DCNN       │ 1              │ 2,094,144         │                   │
│   │  Coupling Block SubNets     │ 8 × 106,516   │ 852,128           │                   │
│   │  ───────────────────────────┼────────────────┼───────────────────│                   │
│   │  GRAND TOTAL                │                │ 2,946,272 (~2.9M) │                   │
│   └─────────────────────────────┴────────────────┴───────────────────┘                   │
│                                                                                          │
│   Note: Coupling block operations (exp, affine, flip) have ZERO learnable parameters.    │
│         All learning occurs in SubNets (predict s, t) and CNN Encoder (predict cond).    │
│         The encoder now dominates the parameter budget (71% vs 29% for SubNets).         │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Forward Pass: Detailed Data Flow Through a Single Coupling Block

During training, design parameters **x** flow forward through 8 coupling blocks to produce latent vector **z** and log-determinant of the Jacobian. Each block splits the 20-dim vector into two halves, transforms one half conditioned on the other, and swaps them.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                    FORWARD PASS — SINGLE COUPLING BLOCK (DETAIL)                         │
│                                                                                          │
│   x ∈ ℝ^(B×20)          cond ∈ ℝ^(B×128)                                                │
│       │                       │                                                          │
│       │    SPLIT              │                                                          │
│       ├──────────┐            │                                                          │
│       │          │            │                                                          │
│       ▼          ▼            │                                                          │
│   x₁ = x[:,:10]  x₂ = x[:,10:]                                                         │
│   (B×10)          (B×10)      │                                                          │
│       │              │        │                                                          │
│       │              │        │                                                          │
│       │    CONCATENATE        │                                                          │
│       ├───────────────────────┤                                                          │
│       │                       │                                                          │
│       ▼                       │                                                          │
│   [x₁ ; cond] ∈ ℝ^(B×138)   │                                                          │
│       │                       │                                                          │
│       ▼                       │                                                          │
│   ┌─────────────────────┐     │                                                          │
│   │      SubNet          │     │                                                          │
│   │  138→256→256→20      │     │                                                          │
│   │  (106,516 params)    │     │                                                          │
│   └──────────┬──────────┘     │                                                          │
│              │                │                                                          │
│         st ∈ ℝ^(B×20)        │                                                          │
│              │                │                                                          │
│       ┌──────┴──────┐        │                                                          │
│       │             │        │                                                          │
│       ▼             ▼        │                                                          │
│    s = st[:,:10]  t = st[:,10:]                                                          │
│    (B×10)         (B×10)     │                                                          │
│       │                      │                                                          │
│       ▼                      │                                                          │
│    s = clamp(s, −3, 3)       │   ◄── hard clamp to prevent extreme scaling               │
│       │                      │                                                          │
│       │     AFFINE TRANSFORM │                                                          │
│       │          ┌───────────┘                                                          │
│       │          │                                                                       │
│       ▼          ▼                                                                       │
│    y₂ = x₂ · exp(s) + t        ◄── invertible affine coupling                           │
│    (B×10)                                                                                │
│       │                                                                                  │
│       │     ACCUMULATE JACOBIAN                                                          │
│       │                                                                                  │
│    log_det_J += Σ(s, dim=1)     ◄── log|det(J)| for this block                          │
│       │                                                                                  │
│       │     RECOMBINE                                                                    │
│       │                                                                                  │
│    output = cat([x₁, y₂])  ∈ ℝ^(B×20)                                                  │
│       │                                                                                  │
│       │     FLIP (swap halves for next block)                                            │
│       │                                                                                  │
│    output = output[:, [10:20, 0:10]]  ∈ ℝ^(B×20)                                        │
│       │                                                                                  │
│       ▼                                                                                  │
│   → next coupling block                                                                  │
│                                                                                          │
│   After 8 blocks:                                                                        │
│       z ∈ ℝ^(B×20)  ,   total_log_det_J ∈ ℝ^B                                           │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                    FORWARD PASS — FULL 8-BLOCK CHAIN                                     │
│                                                                                          │
│   x_params ∈ ℝ^(B×20)     y_spectra ∈ ℝ^(B×1000) ──►1D-CNN Encoder──► cond ∈ ℝ^(B×128)│
│       │                                                                │                 │
│       ▼                                                                │                 │
│   ┌──────────────────────────────────────────────────────────────────┼──────────────────┐│
│   │  Block 1:  x₁=[0:10] identity  │  x₂=[10:20] transformed   │  → FLIP          ││
│   ├──────────────────────────────────────────────────────────────────┼──────────────────┤│
│   │  Block 2:  x₁=[10:20] identity │  x₂=[0:10]  transformed   │  → FLIP          ││
│   ├──────────────────────────────────────────────────────────────────┼──────────────────┤│
│   │  Block 3:  x₁=[0:10] identity  │  x₂=[10:20] transformed   │  → FLIP          ││
│   ├──────────────────────────────────────────────────────────────────┼──────────────────┤│
│   │  Block 4:  x₁=[10:20] identity │  x₂=[0:10]  transformed   │  → FLIP          ││
│   ├──────────────────────────────────────────────────────────────────┼──────────────────┤│
│   │  Block 5:  x₁=[0:10] identity  │  x₂=[10:20] transformed   │  → FLIP          ││
│   ├──────────────────────────────────────────────────────────────────┼──────────────────┤│
│   │  Block 6:  x₁=[10:20] identity │  x₂=[0:10]  transformed   │  → FLIP          ││
│   ├──────────────────────────────────────────────────────────────────┼──────────────────┤│
│   │  Block 7:  x₁=[0:10] identity  │  x₂=[10:20] transformed   │  → FLIP          ││
│   ├──────────────────────────────────────────────────────────────────┼──────────────────┤│
│   │  Block 8:  x₁=[10:20] identity │  x₂=[0:10]  transformed   │  → FLIP          ││
│   └──────────────────────────────────────────────────────────────────┴──────────────────┘│
│       │                                                                                  │
│       ▼                                                                                  │
│   z ∈ ℝ^(B×20)     log_det_J ∈ ℝ^B                                                      │
│                                                                                          │
│   Dims [0:10]  transformed in blocks: 2, 4, 6, 8  (4 times each)                        │
│   Dims [10:20] transformed in blocks: 1, 3, 5, 7  (4 times each)                        │
│   ✓ SYMMETRIC — every dimension transformed equally                                      │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Reverse Pass (Inference): Latent Sample → Physical Parameters → Spectrum

During inference, latent samples **z ~ N(0, I)** are pushed backward through the 8 coupling blocks in reverse order. Each block applies the **inverse** affine transform. The output is then unscaled, clipped to physical bounds, and validated through the TMM physics engine.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                    REVERSE PASS — INFERENCE PIPELINE                                     │
│                                                                                          │
│   STEP 1: ENCODE TARGET SPECTRUM (1D-CNN)                                                │
│   ───────────────────────────────────────                                                │
│                                                                                          │
│   y_target ∈ ℝ^(1×1000)                                                                 │
│   (desired absorption spectrum)                                                          │
│        │                                                                                 │
│        ▼                                                                                 │
│   ┌──────────────────────────────────────┐                                               │
│   │  1D-CNN SPECTRUM ENCODER              │  ◄── same trained weights as training phase   │
│   │  ─────────────────────────            │                                              │
│   │  unsqueeze → (1, 1, 1000)             │                                              │
│   │  Conv1d(1→16, k=5, s=2) → (1,16,500) │                                              │
│   │  Conv1d(16→32, k=5, s=2) → (1,32,250)│                                              │
│   │  Conv1d(32→64, k=5, s=2) → (1,64,125)│                                              │
│   │  Flatten → (1, 8000)                  │                                              │
│   │  Linear(8000→256) → LeakyReLU         │                                              │
│   │  Linear(256→128)                      │                                              │
│   └──────────────────┬───────────────────┘                                               │
│                      │                                                                   │
│                 cond ∈ ℝ^(1×128)                                                         │
│                      │                                                                   │
│                                                                                          │
│   STEP 2: SAMPLE LATENT SPACE                                                            │
│   ────────────────────────────                                                           │
│                                                                                          │
│   z ~ N(0, I) ∈ ℝ^(N×20)      (N = number of candidate solutions, e.g., 5000)           │
│        │                                                                                 │
│        │                                                                                 │
│   STEP 3: REVERSE FLOW (8 blocks in reverse order: block 8 → block 1)                   │
│   ─────────────────────────────────────────────────────────────────────                   │
│        │                                                                                 │
│        ▼                                                                                 │
│   For each block i = 8, 7, 6, 5, 4, 3, 2, 1:                                            │
│                                                                                          │
│      ┌──────────────────────────────────────────────────────────────────────────┐         │
│      │  STEP 3a: UN-FLIP                                                        │         │
│      │     x = x[:, [10:20, 0:10]]         (swap halves back)                   │         │
│      │                                                                          │         │
│      │  STEP 3b: SPLIT                                                          │         │
│      │     x₁ = x[:, :10]                  (unchanged half)                     │         │
│      │     x₂ = x[:, 10:]                  (previously transformed half)        │         │
│      │                                                                          │         │
│      │  STEP 3c: COMPUTE AFFINE PARAMETERS                                     │         │
│      │     [x₁ ; cond] → SubNet_i → s, t                                       │         │
│      │     s = clamp(s, −3, 3)                                                  │         │
│      │                                                                          │         │
│      │  STEP 3d: INVERSE AFFINE TRANSFORM                                      │         │
│      │     x₂_recovered = (x₂ − t) · exp(−s)     ◄── exact inverse             │         │
│      │                                                                          │         │
│      │  STEP 3e: RECOMBINE                                                      │         │
│      │     output = cat([x₁, x₂_recovered])                                    │         │
│      └──────────────────────────────────────────────────────────────────────────┘         │
│        │                                                                                 │
│        ▼                                                                                 │
│   x_scaled ∈ ℝ^(N×20)   (StandardScaler space)                                          │
│        │                                                                                 │
│                                                                                          │
│   STEP 4: CONVERT TO PHYSICAL UNITS                                                      │
│   ─────────────────────────────────                                                      │
│        │                                                                                 │
│        ▼                                                                                 │
│   x_physical = x_scaled · σ + μ           (inverse StandardScaler)                       │
│        │                                                                                 │
│        ▼                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐             │
│   │  VALIDATE AND CLIP TO PHYSICAL BOUNDS                                   │             │
│   │                                                                         │             │
│   │  d₁–d₁₀    ∈ [1×10⁻³, 20×10⁻³] m     (layer thicknesses)              │             │
│   │  m₂,₃,₅,₆,₈,₉ ∈ [20×10⁻³, 1980×10⁻³] m  (hollow cavity dims)        │             │
│   │  ρ         ∈ [1000, 1500] kg/m³        (density)                        │             │
│   │  η         ∈ [0.1, 0.8]               (loss factor)                     │             │
│   │  E         ∈ [E_min(η), E_max(η)]     (Young's modulus, η-dependent)    │             │
│   │  ν         ∈ [0.4, 0.49]              (Poisson's ratio)                 │             │
│   │                                                                         │             │
│   │  E_min = 1×10⁷·(1+η),  E_max = 1×10⁸·(1+η)                            │             │
│   └─────────────────────────────────────────────────────────────────────────┘             │
│        │                                                                                 │
│        ▼                                                                                 │
│   x_physical ∈ ℝ^(N×20)   (valid metamaterial configurations)                            │
│        │                                                                                 │
│                                                                                          │
│   STEP 5: PHYSICS VALIDATION VIA TMM                                                     │
│   ───────────────────────────────────                                                    │
│        │                                                                                 │
│        ▼                                                                                 │
│   ┌──────────────────────────────────────────────────────┐                                │
│   │  TRANSFER MATRIX METHOD (TMM)                        │                                │
│   │  ─────────────────────────────                       │                                │
│   │  For each of N candidates:                           │                                │
│   │    • Build 10-layer impedance stack                  │                                │
│   │    • Compute complex transfer matrices               │                                │
│   │    • Calculate absorption at 1000 frequencies        │                                │
│   │    • α_pred ∈ ℝ^(N×1000)                            │                                │
│   └──────────────────────┬───────────────────────────────┘                                │
│                          │                                                                │
│                          ▼                                                                │
│   STEP 6: CANDIDATE SELECTION                                                             │
│   ───────────────────────────                                                             │
│                                                                                          │
│   RMSE_i = √( mean( (α_pred_i − y_target)² ) )    for i = 1…N                           │
│                                                                                          │
│   best = argmin(RMSE)                                                                     │
│                                                                                          │
│   OUTPUT: x_physical[best] ∈ ℝ^20                                                        │
│           (optimal metamaterial configuration for desired spectrum)                       │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Loop: NLL-Only with Validation Tracking

This variant uses **pure NLL training** (no physics loss), relying on TMM physics validation only at inference time for candidate selection.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│         NLL-ONLY TRAINING LOOP WITH VALIDATION TRACKING                                                  │
│                                                                                                          │
│   FOR epoch = 1 to 250:                                                                                  │
│     FOR each mini-batch (x_batch ∈ ℝ^(2048×20), y_batch ∈ ℝ^(2048×1000)):                               │
│                                                                                                          │
│  ╔════════════════════════════════════════════════════════════════════════════════════════════════════╗    │
│  ║  NEGATIVE LOG-LIKELIHOOD LOSS  (all 250 epochs)                                                   ║    │
│  ╚════════════════════════════════════════════════════════════════════════════════════════════════════╝    │
│                                                                                                          │
│     y_batch ──────────────────────┐                                                                      │
│     (B×1000)                      │                                                                      │
│                                   ▼                                                                      │
│                        ┌──────────────────────────┐                                                      │
│                        │  1D-CNN SPECTRUM ENCODER   │                                                      │
│                        │  ───────────────────────   │                                                      │
│                        │  (B,1000) → unsqueeze      │                                                      │
│                        │  3× Conv1d(k=5, s=2)       │                                                      │
│                        │  Flatten + 2× Linear       │                                                      │
│                        │  → cond ∈ ℝ^(B×128)        │                                                      │
│                        └──────────┬─────────────────┘                                                     │
│                                   │                                                                      │
│     x_batch ──────────►┌──────────┴─────────────┐                                                        │
│     (B×20)             │    cINN FORWARD          │                                                        │
│                        │    ──────────────        │◄── cond from 1D-CNN Encoder                            │
│                        │    8 coupling blocks     │                                                        │
│                        │    split→subnet→         │                                                        │
│                        │    affine→flip           │                                                        │
│                        └──────────┬───────────────┘                                                       │
│                                   │                                                                      │
│                          ┌────────┴────────┐                                                             │
│                          │                 │                                                              │
│                    z ∈ ℝ^(B×20)    log_det_J ∈ ℝ^B                                                       │
│                          │                 │                                                              │
│                          ▼                 ▼                                                              │
│                    ┌────────────────────────────────┐                                                     │
│                    │         L_NLL                   │                                                     │
│                    │                                │                                                     │
│                    │  = mean( 0.5·Σ(z²) − log|J| ) │                                                     │
│                    │                                │                                                     │
│                    │  Purpose: map x → z ~ N(0,I)   │                                                     │
│                    └───────────────┬────────────────┘                                                     │
│                                   │                                                                      │
│                            L_NLL.backward()                                                               │
│                                   │                                                                      │
│          ┌────────────────────────┴────────────────────────┐                                              │
│          │              GRADIENT FLOW                       │                                              │
│          │                                                  │                                              │
│          │  ∂L/∂z → ∂z/∂(SubNet_W)                         │                                              │
│          │  ∂L/∂log|J| → ∂|J|/∂(SubNet_W)                  │                                              │
│          │  → via cond → ∂/∂(CNN_Encoder_W)                 │                                              │
│          │                                                  │                                              │
│          └────────────────────────┬────────────────────────┘                                              │
│                                   │                                                                      │
│                                   ▼                                                                      │
│             ┌──────────────────────────────┐                                                               │
│             │  GRADIENT CLIPPING            │                                                               │
│             │  clip_grad_norm_(θ, max=1.0)  │                                                               │
│             └──────────────┬───────────────┘                                                               │
│                            │                                                                              │
│                            ▼                                                                              │
│          ┌──────────────────────────────────────────┐                                                     │
│          │  ADAM OPTIMIZER STEP                      │                                                     │
│          │  ──────────────────                      │                                                     │
│          │  Updates ALL 2,946,272 parameters:       │                                                     │
│          │                                          │                                                     │
│          │  ┌─────────────────────────────────────┐ │                                                     │
│          │  │  1D-CNN Encoder (2,094,144 params)  │ │                                                     │
│          │  │    Conv1d(1→16, k=5)    (96)        │ │                                                     │
│          │  │    Conv1d(16→32, k=5)   (2,592)     │ │                                                     │
│          │  │    Conv1d(32→64, k=5)   (10,304)    │ │                                                     │
│          │  │    Linear 8000→256      (2,048,256) │ │                                                     │
│          │  │    Linear 256→128       (32,896)    │ │                                                     │
│          │  └─────────────────────────────────────┘ │                                                     │
│          │                                          │                                                     │
│          │  ┌─────────────────────────────────────┐ │                                                     │
│          │  │  8 Coupling SubNets (852,128 params)│ │                                                     │
│          │  │    SubNet 1: 138→256→256→20          │ │                                                     │
│          │  │    SubNet 2: 138→256→256→20          │ │                                                     │
│          │  │    SubNet 3: 138→256→256→20          │ │                                                     │
│          │  │    SubNet 4: 138→256→256→20          │ │                                                     │
│          │  │    SubNet 5: 138→256→256→20          │ │                                                     │
│          │  │    SubNet 6: 138→256→256→20          │ │                                                     │
│          │  │    SubNet 7: 138→256→256→20          │ │                                                     │
│          │  │    SubNet 8: 138→256→256→20          │ │                                                     │
│          │  └─────────────────────────────────────┘ │                                                     │
│          │                                          │                                                     │
│          │  lr_scheduler: CosineAnnealing(T=250)    │                                                     │
│          │  weight_decay: 1e-5                      │                                                     │
│          │  optimizer.zero_grad()                   │                                                     │
│          └──────────────────────────────────────────┘                                                     │
│                                                                                                          │
│     --- VALIDATION PASS (every epoch, no gradient) ---                                                    │
│                                                                                                          │
│     model.eval()                                                                                          │
│     with torch.no_grad():                                                                                │
│       for val_x, val_cond in val_loader:                                                                 │
│         z, log_det_J = model(val_x, val_cond, rev=False)                                                 │
│         val_loss += nll_loss(z, log_det_J)                                                               │
│                                                                                                          │
│     Track: train_losses[], val_losses[] for learning curve plots                                         │
│                                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Differentiable Transfer Matrix Method (TMM) — Physics Validation Engine

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│              DIFFERENTIABLE TMM — 10-LAYER ACOUSTIC METAMATERIAL STACK                   │
│                                                                                          │
│   x_physical ∈ ℝ^(N×20)   (from inverse scaler + clipping, at inference time)           │
│       │                                                                                  │
│   ┌───┴──────────────────────────────────────────────────────────────────────────────┐   │
│   │  STEP 1: EXTRACT PHYSICAL PARAMETERS                                             │   │
│   │                                                                                   │   │
│   │  d₁–d₁₀ = params[:, 0:10] / 1000     (mm → m)                                   │   │
│   │  m₂,₃,₅,₆,₈,₉ = params[:, 10:16] / 1000                                        │   │
│   │  ρ = params[:, 16],  η = params[:, 17]                                           │   │
│   │  E = params[:, 18],  ν = params[:, 19]                                           │   │
│   │                                                                                   │   │
│   └───┬──────────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                                  │
│       ▼                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│   │  STEP 2: MATERIAL PROPERTY COMPUTATION                                           │   │
│   │                                                                                   │   │
│   │  E_c = E · (1 + jη)                    complex Young's modulus                    │   │
│   │  λ = (E_c · ν)/((1+ν)(1−2ν))          first Lamé parameter                      │   │
│   │  μ = E_c / (2(1+ν))                   shear modulus                              │   │
│   │                                                                                   │   │
│   └───┬──────────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                                  │
│       ▼                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│   │  STEP 3: 10-LAYER TRANSFER MATRIX COMPUTATION                                    │   │
│   │                                                                                   │
│   │  For each layer i = 1..10, at each frequency f = 1..1000 Hz:                     │   │
│   │                                                                                   │   │
│   │    ε = m_cavity / W  (porosity, 0 for solid layers)                              │   │
│   │    ρ_eff = ρ · (1 − ε²)                                                         │   │
│   │    S_eff = μ(λ+2μ)(ε²+1) + 2ε²λ  /  ((λ+μ)ε² + μ)                             │   │
│   │    c_eff = √(S_eff / ρ_eff)                                                     │   │
│   │    k = ω / c_eff                                                                 │   │
│   │    Z = ρ_eff · c_eff                                                             │   │
│   │                                                                                   │   │
│   │    T_i = ┌ cos(kd)      jZ·sin(kd)  ┐                                           │   │
│   │          │ j·sin(kd)/Z   cos(kd)     │                                           │   │
│   │          └                            ┘                                           │   │
│   │                                                                                   │   │
│   │    T_total = T₁ · T₂ · T₃ · … · T₁₀                                            │   │
│   │                                                                                   │   │
│   │    Z_in = T₁₁/T₂₁                                                               │   │
│   │    R = (Z_in − Z_w)/(Z_in + Z_w)                                                │   │
│   │    α(f) = clamp(1 − |R|², 0, 1)                                                 │   │
│   │                                                                                   │   │
│   └───┬──────────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                                  │
│       ▼                                                                                  │
│   α_pred ∈ ℝ^(N×1000)                                                                   │
│   (predicted absorption spectrum for each candidate)                                     │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Schedule and Hyperparameter Summary

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING SCHEDULE                                               │
│                                                                                          │
│   Epoch:  1                                                                  250         │
│           │                                                                   │           │
│           ▼                                                                   ▼           │
│   ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│   │                          PURE NLL TRAINING                                         │ │
│   │                          (all 250 epochs)                                          │ │
│   │                                                                                    │ │
│   │  L = L_NLL = mean( 0.5·‖z‖² − log|J| )                                           │ │
│   │                                                                                    │ │
│   │  Enforces:  x_params ↔ z ~ N(0, I)  bijective mapping                             │ │
│   │  conditioned on absorption spectrum via 1D-CNN encoder                             │ │
│   │                                                                                    │ │
│   │  Physics validation applied ONLY at inference time via TMM + RMSE selection        │ │
│   └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│   Learning Rate (CosineAnnealing):                                                       │
│                                                                                          │
│   lr  1e-3 ┐                                                                             │
│            │╲                                                                            │
│            │  ╲                                                                          │
│            │    ╲                                                                        │
│            │      ╲                                                                      │
│            │        ╲  ╱ cosine decay                                                    │
│            │         ╲╱                                                                  │
│   lr  ~0   └─────────────────────────────────────────────────────────────────────        │
│            1                                                          250                 │
│                                                                                          │
│─────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                          │
│                      HYPERPARAMETER SUMMARY                                              │
│                                                                                          │
│   ┌─────────────────────────┬───────────────────────────────────────┐                    │
│   │  Hyperparameter         │ Value                                 │                    │
│   │  ───────────────────────┼─────────────────────────────────────  │                    │
│   │  Epochs                 │ 250                                   │                    │
│   │  Batch size             │ 2048                                  │                    │
│   │  Optimizer              │ Adam                                  │                    │
│   │  Learning rate          │ 1×10⁻³                                │                    │
│   │  Weight decay           │ 1×10⁻⁵                                │                    │
│   │  LR scheduler           │ CosineAnnealingLR (T_max=250)        │                    │
│   │  Gradient clipping      │ max_norm = 1.0                       │                    │
│   │  Training loss          │ NLL only (no physics loss)           │                    │
│   │  Coupling blocks        │ 8                                    │                    │
│   │  Encoder type           │ 1D-CNN (3 Conv1d + 2 Linear)        │                    │
│   │  Encoder output dim     │ 128                                  │                    │
│   │  Conv kernel size       │ 5                                    │                    │
│   │  Conv stride            │ 2                                    │                    │
│   │  Conv channels          │ 1 → 16 → 32 → 64                    │                    │
│   │  FC hidden dim          │ 256                                  │                    │
│   │  SubNet hidden dim      │ 256                                  │                    │
│   │  Activation             │ LeakyReLU(0.01)                      │                    │
│   │  Clamp range (s)        │ [−3, 3]  (hard clamp)               │                    │
│   │  Latent distribution    │ N(0, I₂₀)                            │                    │
│   └─────────────────────────┴───────────────────────────────────────┘                    │
│                                                                                          │
│─────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                          │
│                         DATASET SUMMARY                                                  │
│                                                                                          │
│   ┌─────────────────────────┬───────────────────────────────────────┐                    │
│   │  Property               │ Value                                 │                    │
│   │  ───────────────────────┼─────────────────────────────────────  │                    │
│   │  Total samples          │ 1,000,000 (Latin Hypercube Sampling) │                    │
│   │  Training set           │ ~800,000 (80%)                       │                    │
│   │  Validation set         │ ~100,000 (10%)                       │                    │
│   │  Test set               │ 100,000  (10%)                       │                    │
│   │  Input (x)              │ 20 design parameters (scaled)        │                    │
│   │  Output (y)             │ 1000-point absorption spectrum       │                    │
│   │  Frequency range        │ 1–1000 Hz                            │                    │
│   │  Scaling                │ StandardScaler (per-feature)         │                    │
│   │  Precision              │ float32                              │                    │
│   └─────────────────────────┴───────────────────────────────────────┘                    │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Count Summary

| Component | Architecture | Params |
|-----------|-------------|--------|
| **SpectrumEncoder1DCNN** | Conv1d(1→16→32→64) + Linear(8000→256→128) | 2,094,144 |
| **SubNet_0** | 138→256→256→20 | 106,516 |
| **SubNet_1** | 138→256→256→20 | 106,516 |
| **SubNet_2** | 138→256→256→20 | 106,516 |
| **SubNet_3** | 138→256→256→20 | 106,516 |
| **SubNet_4** | 138→256→256→20 | 106,516 |
| **SubNet_5** | 138→256→256→20 | 106,516 |
| **SubNet_6** | 138→256→256→20 | 106,516 |
| **SubNet_7** | 138→256→256→20 | 106,516 |
| **Affine ops** | `exp(s)`, `+t` | 0 |
| **Flip** | `torch.flip` | 0 |
| | | **2,946,272** |

---

## Key Design Decisions

| Decision | Value | Why |
|----------|-------|-----|
| **1D-CNN encoder** (replacing MLP) | 3 Conv1d layers + FC head | Absorption spectra are spatially ordered signals; convolutions exploit local frequency correlations and extract multi-scale features (peaks, slopes, resonances) that MLP layers must learn individually |
| **Kernel size = 5** | 5 frequency bins per filter | Captures local spectral structure without excessive receptive field early on |
| **Stride = 2, all layers** | Progressive 8× spatial downsampling | Gradual compression (1000→500→250→125) vs. MLP's abrupt 1000→256 — preserves spatial structure during compression |
| **Channel doubling (1→16→32→64)** | Standard CNN design principle | Increasing feature channels while reducing spatial dimension maintains information capacity at each layer |
| 8 coupling blocks | 4 transforms per half | Rule-of-thumb minimum is 2; 4 provides good expressivity for 20-dim |
| Split at dim 10 | 10 + 10 | Equal halves = symmetric transforms |
| `clamp(s, -3, 3)` | `exp(s) ∈ [0.05, 20.1]` | Prevents extreme scaling that destabilises training |
| No batch norm in SubNets | — | Flow networks typically avoid batchnorm (breaks invertibility guarantees) |
| NLL-only training | No physics loss | Simple maximum likelihood; physics consistency checked at inference via TMM + RMSE selection |
| Spectra not scaled | raw [0,1] | Already bounded; scaling would lose physical meaning |
| `num_candidates=10` | 10 random z samples | Exploits the one-to-many nature of the inverse problem |

---

### Key Equations

$$\mathcal{L}_{NLL} = \mathbb{E}\left[ \frac{1}{2} \|z\|^2 - \log |\det J| \right]$$

where $G$ is the forward flow mapping design parameters $x$ to latent space $z$, conditioned on absorption spectrum $y$ via the 1D-CNN encoder:

$$z = G(x \mid \text{CNN}(y))$$

$$x_{recovered} = G^{-1}(z \mid \text{CNN}(y_{target}))$$

The 1D-CNN encoder compresses the spectrum through hierarchical local feature extraction:

$$\text{CNN}: \mathbb{R}^{1000} \xrightarrow{\text{Conv1d} \times 3} \mathbb{R}^{64 \times 125} \xrightarrow{\text{Flatten+FC}} \mathbb{R}^{128}$$

---

*This architecture reference documents the 1D-CNN encoder variant of the FullSpectrum_cINN, as implemented in `CiNN-1DCNN-FULL-SPECTRUM.ipynb`. All dimensions, parameter counts, and data flow directions are exact.*
