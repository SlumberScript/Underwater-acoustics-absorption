# Complete End-to-End Architecture: `FullSpectrum_cINN`

> **One diagram, every component, both directions.**  
> Total parameters: **1,207,072 (~1.2M)**  
> 3 learnable components: 1 SpectrumEncoder + 8 CondAffineCoupling SubNets

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
║       │                               ╔══════════════════════╗                           ║
║       │                               ║   SPECTRUM ENCODER   ║  354,944 params           ║
║       │                               ║                      ║                           ║
║       │                               ║  Linear(1000→256)    ║                           ║
║       │                               ║  LeakyReLU(0.01)     ║                           ║
║       │                               ║  Linear(256→256)     ║                           ║
║       │                               ║  LeakyReLU(0.01)     ║                           ║
║       │                               ║  Linear(256→128)     ║                           ║
║       │                               ╚═════════╤════════════╝                           ║
║       │                                          │                                       ║
║       │                                    cond (B,128)                                  ║
║       │                                    (shared by ALL 8 blocks)                      ║
║       │                                          │                                       ║
║       ▼                                          ▼                                       ║
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
║    gradients → 8 SubNets          gradients → Encoder                                    ║
║    (8 × 3 Linear layers)          (3 Linear layers)                                      ║
║              │                             │                                             ║
║              └──────────────┬──────────────┘                                             ║
║                             ▼                                                            ║
║              clip_grad_norm_(max_norm=1.0)                                               ║
║                             │                                                            ║
║                             ▼                                                            ║
║              Adam(lr=1e-3, weight_decay=1e-5)                                            ║
║              CosineAnnealingLR(T_max=100)                                                ║
║              100 epochs, batch=2048                                                      ║
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
║   ╔══════════════════════╗                                  │                            ║
║   ║   SPECTRUM ENCODER   ║  (SAME weights as training)      │                            ║
║   ║   1000 → 256 → 128  ║                                  │                            ║
║   ╚═════════╤════════════╝                                  │                            ║
║             │                                               │                            ║
║        cond (C, 128)                                        │                            ║
║             │                                               │                            ║
║             └────────────────────┬──────────────────────────┘                            ║
║                                  │                                                       ║
║                                  ▼                                                       ║
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

## Parameter Count Summary

| Component | Architecture | Params |
|-----------|-------------|--------|
| **SpectrumEncoder** | 1000→256→256→128 | 354,944 |
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
| | | **1,207,072** |

## Key Design Decisions

| Decision | Value | Why |
|----------|-------|-----|
| 8 coupling blocks | 4 transforms per half | Rule-of-thumb minimum is 2; 4 provides good expressivity for 20-dim |
| Split at dim 10 | 10 + 10 | Equal halves = symmetric transforms |
| `clamp(s, -3, 3)` | `exp(s) ∈ [0.05, 20.1]` | Prevents extreme scaling that destabilises training |
| No batch norm in SubNets | — | Flow networks typically avoid batchnorm (breaks invertibility guarantees) |
| Spectra not scaled | raw [0,1] | Already bounded; scaling would lose physical meaning |
| `num_candidates=10` | 10 random z samples | Exploits the one-to-many nature of the inverse problem |
