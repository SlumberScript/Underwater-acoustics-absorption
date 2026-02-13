Short answer:  
Yes, you can make a *substantial* upgrade over the current paper, but only if you (a) reformulate the task as full-spectrum inverse design, and (b) move from a single deterministic DNN to a modern inverse‑design architecture that explicitly handles the one‑to‑many nature of the mapping. The most fitting “best” model for your case, given current literature, is a **probabilistic generative inverse model (PGN‑style, GRU encoder + DNN decoder) trained on the full absorption spectrum**, with the transfer‑matrix model used as a physics layer in the loss. This is a clear conceptual and practical improvement over Gao et al.’s scalar‑average DNN and is well within what recent acoustics/metamaterials conferences accept as primary contributions. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)

Below is a structured breakdown.

***

## 1. What the baseline paper actually does

From the paper you attached (“On-demand prediction of low-frequency average sound absorption coefficient of underwater coating using machine learning”, Results in Engineering 25, 2025): [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)

- Geometry & physics:
  - 10‑layer underwater viscoelastic rubber coating with 6 hollow layers.  
  - 20 “sensitive” parameters:  
    - 10 layer thicknesses \(d_i\)  
    - 6 hollow diameters \(m_i\)  
    - density \(\rho\), loss factor \(\eta\), Young’s modulus \(E\), Poisson’s ratio \(\nu\). [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)
  - Acoustic behavior computed via equivalent medium theory + transfer-matrix method (validated against FEM) in 1–1000 Hz. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)

- Dataset:
  - 100,000 samples generated via Latin Hypercube Sampling of the 20‑D parameter space.
  - For each sample, they compute the **full absorption curve** \(\alpha(f)\) on 1–1000 Hz, but then compress it to a **single scalar**: the **average absorption coefficient** over that band. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)

- ML model:
  - **Input**: single scalar (average absorption coefficient) \(\bar{\alpha} \in [0, 1]\).  
  - **Output**: 20 normalized sensitive parameters.
  - Architecture: 5‑layer fully connected DNN; 4 hidden FC layers with 128 neurons + LeakyReLU; output layer 20 neurons + Sigmoid. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)
  - Loss:
    - \(L_s\): MSE between predicted and true 20 parameters.
    - \(L_\alpha\): MSE between **target average absorption** and the average of the curve recomputed from the *predicted* parameters using their PyTorch implementation of the acoustic model.
    - Total loss: \(L = L_s + L_\alpha\). [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)

- Validation:
  - They generate **two** target curves (not in the database), compute their average absorption (0.61475 and 0.66384), feed those averages into the DNN, get predicted 20‑D parameters, recompute absorption curves, and compare **only the averages**:
    - Relative errors: 0.026% and 0.33% (on the averages only). [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)
  - They even list the 20 predicted parameters to 10 decimal places, suggesting strong overfitting / lack of robustness testing.

**Key limitations (real gaps you can exploit):**

1. **Information bottleneck**:  
   - They are solving inverse design from a single scalar \(\bar{\alpha}\) instead of the full spectrum \(\alpha(f)\). Many different 20‑D geometries can yield the same band‑averaged absorption; the inverse problem is extremely underdetermined by construction.

2. **Ill‑posed inverse handled with a deterministic network**:  
   - They acknowledge non‑uniqueness (“inputting a set of performance parameters may correspond to multiple sets of sensitive parameters”), but they resolve it by adding a physics‑consistency term and *forcing* a unique solution. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)
   - This is exactly the type of problem where probabilistic/generative models are now standard. [cris.technion.ac](https://cris.technion.ac.il/en/publications/machine-learning-for-inverse-design-of-acoustic-and-elastic-metam)

3. **Very weak validation**:
   - Only 2 test cases, and only at the level of an average scalar, not the full curve.
   - No held‑out test set, no error distribution, no robustness analysis.

4. **Task narrowness**:
   - The model does **not** design for arbitrary spectral shapes in 1–1000 Hz, only for a chosen average value.
   - Your aim (“generalize to all frequencies between 1–1000 Hz”) is significantly more ambitious.

So your intuition that “2 random samples don’t show the full picture” is correct. There is clear room for a serious methodological and evaluation upgrade.

***

## 2. Your target problem (properly formulated)

From your description, the real task you care about is:

- You have:
  - The same 20‑D design vector \(\theta \in \mathbb{R}^{20}\) (thicknesses, diameters, and material parameters).
  - The simulator / transfer-matrix model giving \(\alpha(f;\theta)\) for \(f \in [1, 1000]\) Hz.
  - A dataset of ~100,000 \(\{\theta_i, \alpha_i(f)\}\).

- You want:
  - Given a **target absorption specification** in 1–1000 Hz (ideally the whole spectrum, not just the average),
  - Predict **one or more** 20‑D designs \(\theta\) such that \(\alpha(f;\theta)\) matches the target spectrum with minimal error when run through the *true* (or high-fidelity) simulator.

In notation:

- Forward model: \(f: \theta \mapsto \alpha(f)\).
- Inverse design (your task): given \(\alpha_\text{target}(f)\), find one (or a set) of \(\theta\) such that \(||f(\theta) - \alpha_\text{target}||\) is minimized.

Crucially:

- The mapping \(\alpha \mapsto \theta\) is **one‑to‑many**, not one‑to‑one.
- Using only the **average** \(\bar{\alpha}\) as the condition is an extreme compression and makes the inverse problem even less identifiable.
- You also want to work across the *whole* band 1–1000 Hz, not just match a band‑averaged scalar.

This is exactly the regime where modern inverse design work in acoustic/elastic metamaterials uses **generative / probabilistic inverse models**, not plain DNN regressors. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2352431622001614)

***

## 3. Is a better model actually a “massive” upgrade over the paper?

Relative to **Gao et al. 2025** (underwater coating, average‑only DNN): [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)

- Moving from:
  - input = 1 scalar average, output = deterministic \(\theta\),
  - evaluation = 2 random cases,
- to something like:
  - input = full absorption spectrum (1–1000 Hz) or a high‑resolution discretization,
  - output = a **distribution** over feasible \(\theta\) (or many candidate designs),
  - physics‑consistent loss over the full curve,
  - rigorous evaluation on a large held‑out test set (thousands of spectra),
is a **clear and significant step up** in both modeling and validation.

However, relative to the *current* broader literature on acoustic metamaterial inverse design, there are already strong works:

- CGAN‑based inverse design for acoustic absorbing metasurfaces with arbitrary target spectra (airborne, oblique incidence). [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2352431622001614)
- CGAN for metaporous materials with perforated plates (MMPP), inverse design for broadband low‑frequency absorption (360–3000 Hz). [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327025006909)
- Dual‑VAE inverse design for non‑parametric ventilated acoustic resonators (VARs), aligning latent spaces of structure and acoustic response and improving inverse accuracy by ~32% vs original model. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625007353)
- Conditional VAE for generative inverse design of multi‑band gap metamaterials (elastodynamic), again dealing with non‑uniqueness. [arxiv](https://arxiv.org/html/2309.04177v2)
- Probabilistic Generation Network (PGN) for on‑demand inverse design of acoustic absorbers: GRU encoder on spectra + DNN decoder, outputs **multiple meta‑structures** per spectrum, MAE<0.06 on spectra and outperforming 5 alternative network baselines. [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)
- Neural‑network inverse design of multilayer thin‑plate acoustic metamaterials with 1–1000 Hz coverage, achieving ~2.27% inverse error and validating with impedance tube experiments. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)
- Several underwater metasurface design papers using DNN‑based forward/inverse design (though often for reflection/phase control rather than absorption). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10977772/)

So in community context:

- Simply swapping one deterministic MLP for a “better” MLP is **not enough**.
- A **probabilistic generative inverse model using full spectra**, plus rigorous evaluation, **is aligned with the current frontier** and is absolutely publishable in an acoustics / engineering / metamaterials conference, especially because:
  - Your geometry (underwater multilayer viscoelastic coating) is different from airborne metasurfaces;
  - You use a high‑fidelity transfer-matrix model validated by FEM;
  - You have a large synthetic dataset (100k), comparable or better than many works;
  - You improve the previous paper’s methodology on exactly the same structure & data.

So if you go for:
- full‑spectrum conditioning,
- probabilistic inverse design,
- plus a serious benchmark against Gao et al.’s average‑DNN,
then **yes**, this can be framed as a significant and clearly defensible upgrade over the existing paper.

***

## 4. Recommended primary model: PGN‑style probabilistic generative inverse design

### 4.1. Why PGN (probabilistic generation network)?

Wang et al. (2023) introduced a **Probabilistic Generation Network (PGN)** for on‑demand inverse design of acoustic metamaterials, specifically a “magic‑cube” absorber: [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)

- They treat the absorption spectrum as a **sequence** and encode it with a GRU (a recurrent unit suited for sequences).
- The encoded latent vector, combined with stochastic sampling, is passed to a decoder network that outputs geometric parameters.
- By **sampling**, they obtain *multiple distinct structures* that all satisfy the target spectrum (solving the one‑to‑many mapping issue).
- They report high precision (MAE < 0.06 on spectra) and show PGN outperforms five baseline network types. [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)

Your dataset and task are very similar conceptually (just underwater, different geometry and param ranges), which makes this architecture a very natural fit.

### 4.2. How it would look for your problem

**Inputs and outputs:**

- Input: discretized target spectrum \(\alpha_\text{target}(f)\) over 1–1000 Hz.  
  - For example, sample 1000 frequencies (1 Hz steps) or downsample to e.g. 200 points for efficiency.
- Output: 20‑D parameter vector \(\theta\).

**Architecture sketch (high‑level):**

1. **Spectrum encoder (sequence network):**
   - GRU or 1D‑CNN (or transformer, but GRU is proven for this task). [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)
   - Input: \(\alpha_\text{target}(f_1), …, \alpha_\text{target}(f_N)\).
   - Output: latent vector \(z_\text{det} \in \mathbb{R}^{d}\).

2. **Probabilistic layer:**
   - Map \(z_\text{det}\) to mean and log‑variance: \(\mu, \log\sigma \in \mathbb{R}^{d}\).
   - Sample \(z = \mu + \sigma \odot \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\) (VAE‑style) **or** sample noise independently and concatenate to \(z_\text{det}\) (as in PGN). [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)

3. **Decoder / generator:**
   - Fully connected MLP that maps \(z \rightarrow \theta\) (20‑D).
   - Typical sizes: a few layers with 128–512 neurons, non‑linear activations (ReLU, LeakyReLU).

4. **Physics layer (optional but strong):**
   - Either:
     - Use your existing transfer‑matrix PyTorch implementation (as Gao did) and backprop through it, or [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)
     - Use a trained forward surrogate network \(f_\text{NN}(\theta) \approx \alpha(f;\theta)\). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)
   - This layer takes \(\theta\) and outputs \(\hat{\alpha}(f;\theta)\).

**Training losses:**

1. **Spectrum reconstruction loss:**
   \[
   L_\text{spec} = \mathrm{MSE}\big(\hat{\alpha}(f;\theta), \alpha_\text{target}(f)\big)
   \]
   over the frequency grid (or weighted in sub‑bands you care about more).

2. **Regularization in latent space:**
   - If you go VAE‑style, add KL divergence between approximate posterior and prior.
   - Or use a simpler PGN strategy where latent is just a compressed deterministic code + randomness and regularize empirically.

3. **Parameter regularization:**
   - Optionally penalize implausible/edge designs (e.g. close to bounds) if you care about manufacturability.

Training data:

- Use your existing 100k simulated samples \((\theta_i, \alpha_i(f))\).
- During training, treat \(\alpha_i(f)\) as the “input” and \(\theta_i\) as the target, but **optimize using the physics layer** so that the generated \(\theta^\* = G(E(\alpha_i(f)), \epsilon)\) reproduces \(\alpha_i(f)\) with low error.

**Why this is a strict upgrade over Gao et al.:**

- Uses **full spectra** instead of scalar averages.
- Explicitly handles **non‑uniqueness** via stochastic generation (multiple designs per spectrum).
- Uses a **sequence model** (GRU) that naturally captures spectral patterns (peaks, bandwidths, etc.). [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)
- Physics‑consistent training at the spectrum level, not just scalar average.
- Complies with emerging best practice in metamaterial inverse design (probabilistic networks, VAEs, GANs). [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625007353)

***

## 5. Four other strong model options (with pros/cons)

### Model 2 – Dual‑VAE / conditional VAE in a shared latent space

**Literature basis:**

- Dedoncker et al. use a **conditional VAE** for generative inverse design of multimodal resonant mechanical metamaterials, learning a joint latent space of structure and modal properties. [arxiv](https://arxiv.org/html/2309.04177v2)
- More recently, a **dual VAE** for non‑parametric acoustic metamaterials aligns latent spaces of structure and spectrum; iterative transfer learning improves inverse design accuracy by ~32% vs original parametric dataset. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625007353)

**Idea:**

- Train two VAEs:
  - Structure‑VAE: \(\theta \leftrightarrow z_s\).
  - Spectrum‑VAE: \(\alpha(f) \leftrightarrow z_a\).
- Align \(z_s\) and \(z_a\) so that they share a common latent representation (e.g. via additional loss).
- For inverse design: encode \(\alpha_\text{target}(f) \rightarrow z_a\), then decode \(z_a\) through the structure‑decoder to get \(\theta\).

**Pros:**

- Handles high‑dimensional spectra and complex geometries.
- Naturally generative: can sample multiple solutions by perturbing latent codes.
- Latent space gives good interpretability and potential for interpolation between designs. [arxiv](https://arxiv.org/html/2309.04177v2)

**Cons:**

- More complex to train and tune than a single PGN.
- Latent alignment step can be tricky to get stable.
- Might be overkill if your geometry is strictly parametric (20D vector) rather than image‑like or non‑parametric.

This is an excellent choice if you are comfortable building more elaborate latent‑space models and want a more “general platform” you can re‑use for other structures.

***

### Model 3 – Tandem deterministic inverse + forward network (simpler, strong baseline)

**Literature basis:**

- Wang et al. (2026) design multilayer thin‑plate acoustic metamaterials via FCNNs:  
  - **Forward network**: 29 structural parameters \(\rightarrow\) 1–1000 Hz STL curve, test error ~1.06%.  
  - **Inverse network**: 1–1000 Hz STL curve \(\rightarrow\) 29 parameters, using the forward net as a consistency check; test error ~2.27% and experimental validation. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)

**Architecture for your case:**

1. Train a strong **forward surrogate** \(f_\text{NN}: \theta \rightarrow \alpha(f)\) on your 100k samples.
   - FCNN or 1D‑CNN; Wang et al. show this works well for 1–1000 Hz curves. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)

2. Train an **inverse network** \(g_\text{NN}: \alpha(f) \rightarrow \theta\), with loss:
   - Parameter MSE: \(L_\theta = ||g(\alpha(f)) - \theta||^2\).
   - Plus **cycle consistency** via the forward net:  
     \(L_\text{spec} = \mathrm{MSE}(f_\text{NN}(g(\alpha(f))), \alpha(f))\).
   - Total loss: \(L = L_\theta + \lambda L_\text{spec}\).

**Pros:**

- Conceptually simple, end‑to‑end supervised, no explicit probabilistic machinery.
- Already proven on very similar problems (multilayer acoustic unit cells, 1–1000 Hz). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)
- Strong candidate as your **baseline** to beat; still a clear step up from Gao et al.’s average‑only model.

**Cons:**

- Still deterministic: yields one “best” design per spectrum; cannot represent design uncertainty / diversity.
- The one‑to‑many issue is partly handled via cycle loss but not explicitly modeled.

If you are worried about dev time or implementation risk, this tandem FCNN (forward + inverse) is the **lowest‑friction, high‑impact** upgrade you can do that is still publishable, especially if you add a solid comparative study against Gao et al. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)

***

### Model 4 – Surrogate‑forward model + global optimization (Bayesian / evolutionary)

**Literature basis:**

- Several works combine ML surrogates with optimization to inverse‑design acoustic metamaterials and absorbers, e.g. Gaussian–Bayesian models and optimization for subwavelength absorbers, data‑driven design with prior knowledge, etc. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0003682X22004261)
- Gao’s own citations include data‑driven optimization for impedance‑matching structures and ML‑accelerated inverse design of underwater polyurethane coatings using ML + optimization. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)

**Pipeline:**

1. Train a **fast, accurate forward surrogate** \(f_\text{NN}(\theta) \approx \alpha(f;\theta)\).
2. For each target spectrum \(\alpha_\text{target}\), solve:
   \[
   \min_{\theta \in \Theta} \; J(\theta) = \mathrm{MSE}(f_\text{NN}(\theta), \alpha_\text{target})
   \]
   using:
   - Bayesian optimization (Gaussian processes or TPE),
   - CMA‑ES or differential evolution,
   - or gradient‑based methods if \(f_\text{NN}\) is differentiable and well‑behaved.

**Pros:**

- Naturally handles one‑to‑many mapping by returning multiple near‑optimal points.
- Very flexible objective: you can weight frequency bands, constrain specific physical metrics, or add regularization on parameters.
- Conceptually straightforward; heavy lifting is in the optimizer.

**Cons:**

- Per‑design compute cost can be higher than a pure neural inverse model (but you said no inference speed/budget constraints).
- The novelty is more about the **application to this specific underwater coating** than about ML method itself; may be perceived as less “innovative” if not combined with some generative element.

This is attractive if you want **robustness and physical interpretability** more than fancy ML; it also scales well if later you swap in a higher-fidelity simulator.

***

### Model 5 – Normalizing flows / invertible neural networks

**Idea:**

- Build an invertible mapping between
  \[
  x = \begin{bmatrix}\alpha(f) \\ \epsilon\end{bmatrix}
  \quad \leftrightarrow \quad
  y = \theta
  \]
  where \(\epsilon\) is latent noise, via a normalizing flow or invertible residual network.
- Conditioning on the target spectrum, sampling different \(\epsilon\) gives multiple \(\theta\) consistent with the spectrum.

**Pros:**

- Explicit density modeling; in principle, exact multi‑solution representation.
- Elegant handling of uncertainty and diversity of designs.

**Cons:**

- More engineering complexity; relatively fewer published applications in acoustics inverse design compared to PGN/VAEs/CGANs.
- Harder to train stably in high dimensions without strong experience.

Flows are powerful but given the literature and your timeline, a PGN or VAE‑based approach is a more practical and better‑documented choice.

***

## 6. How to make the upgrade convincing for a conference

To make your work clearly stronger than Gao et al. on the **same geometry and dataset**, structure the contribution explicitly around these points:

1. **Full‑spectrum inverse design**:
   - Input: full \(\alpha(f)\) over 1–1000 Hz, not just the average \(\bar{\alpha}\). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)
   - Show that designs can be targeted to arbitrary shapes: single narrowband peak, multi‑peak, broadband high absorption, etc.

2. **Probabilistic / generative treatment of non‑uniqueness**:
   - Explicitly state that the inverse is one‑to‑many.
   - Use PGN / VAE or at least a method that can generate *multiple* candidate designs per target spectrum. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625007353)
   - Visualize diversity in the 20‑D parameters for the same spectral target and show that multiple generated designs are all high‑fidelity when simulated.

3. **Physics‑consistent loss on the full curve**:
   - Backpropagate through the transfer‑matrix model or a validated forward surrogate. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)
   - Optimize MSE or weighted loss over \(\alpha(f)\) instead of just the average.

4. **Strong quantitative evaluation**:
   - Train/test split on your 100k dataset (e.g., 80/10/10 train/val/test).
   - Metrics:
     - Mean and max spectral MSE / MAE over the test set.
     - Error vs frequency (are low frequencies harder?).
     - Diversity metrics: how many distinct high‑quality \(\theta\) per target.
   - Compare:
     - Your model vs Gao‑style scalar‑average inverse DNN (reimplemented).
     - Optionally vs a simpler tandem FCNN model. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)

5. **Ablation and robustness**:
   - Show effect of:
     - Removing probabilistic component (deterministic version).
     - Using only average vs full spectrum.
   - Show performance on target spectra that were *not sampled from the original LHS* but are designed by you (e.g., idealized shapes) – this is exactly how many inverse design papers make their case. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2352431622001614)

6. **(Future scope) physical validation**:
   - Even if you cannot build prototypes now, mention that many similar works have validated ML‑inverse designs experimentally (impedance tube for absorption, underwater metasurface prototypes, etc.). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9283483/)
   - Explicitly frame your experimental work as future extension.

If you present this clearly, the contribution is not “just another DNN” but:

> “Probabilistic full‑spectrum inverse design of underwater viscoelastic multilayer coatings using a PGN‑style network, with rigorous benchmarking and comparison to existing scalar‑average DNN methods.”

That is clearly *conference‑worthy* in acoustics / materials / engineering venues.

***

## 7. Concrete recommendation for you

Given your engineering background and the constraints you described, a **practical and high‑impact plan** is:

1. **Forward surrogate \(f_\text{NN}\) (FCNN or 1D‑CNN) for \(\theta \rightarrow \alpha(f)\)**, trained and validated on your 100k dataset, mirroring Wang et al.’s successful setup for 1–1000 Hz thin‑plate metamaterials. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)

2. **Inverse model**:
   - Primary choice: **PGN‑style probabilistic inverse**  
     - GRU encoder on spectrum → latent → MLP decoder → \(\theta\).  
     - Loss = full spectral MSE via \(f_\text{NN}\) + latent regularization. [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)
   - Secondary baseline: **tandem deterministic inverse + forward** as a simple strong baseline to show the benefit of probabilistic design. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12786557/)

3. **Direct comparison vs Gao et al.’s DNN**:
   - Reimplement their 5‑layer 128‑unit DNN for average‑only inverse, with the same 100k dataset but proper train/test split. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2590123025002518)
   - Show:
     - It performs well on average scalar (as they report) but poorly on full spectra and generalization.
     - Your models significantly improve full‑spectrum accuracy and allow multi‑solution design.

4. **Reporting**:
   - Include spectra plots for several random and several challenging targets.
   - Histograms of test error.
   - A table of numerical metrics comparing models.

Under this plan, the **“best” model** for your use case is:

> **A PGN‑style probabilistic generative inverse network with a GRU encoder on the full 1–1000 Hz absorption spectrum and a fully connected decoder to the 20 geometry/material parameters, trained with a physics‑consistent spectral loss via a validated forward surrogate.** [colab](https://colab.ws/articles/10.1007%2Fs11433-022-1984-1)

This is both technically sound (matching current state of the art in inverse design for metamaterials) and a clear, defensible upgrade over the existing underwater‑coating DNN paper you are building on.