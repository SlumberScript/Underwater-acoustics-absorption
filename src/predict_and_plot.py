"""
Predict & Plot — Inverse Design via Diffusion Model V3
=======================================================
Usage:
    python src/predict_and_plot.py                     # default target = 0.4
    python src/predict_and_plot.py 0.35                # single target
    python src/predict_and_plot.py 0.25 0.35 0.45 0.55 # multiple targets

Workflow:
    1. User specifies one or more target average-absorption coefficients
    2. V3 diffusion model generates candidate design parameters
    3. TMM physics engine computes the FULL absorption curve (1–1000 Hz)
    4. Plots all curves together with the base-case reference
"""

import sys, os, warnings, math, copy, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths (relative to repo root)
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_V3_PATH = os.path.join(REPO_ROOT, "models", "diffusion_model_v3_physics.pth")
SCALER_V3_PATH = os.path.join(REPO_ROOT, "models", "diffusion_model_v3_physics_scaler.pkl")
SURROGATE_PATH = os.path.join(REPO_ROOT, "models", "forward_surrogate.pth")

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ──────────────────────────────────────────────
# Parameter constraints
# ──────────────────────────────────────────────
PARAM_NAMES = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
               'm2','m3','m5','m6','m8','m9','rho','eta','E','nu']

PARAM_RANGES = {
    'd1': (1,20), 'd2': (1,20), 'd3': (1,20), 'd4': (1,20), 'd5': (1,20),
    'd6': (1,20), 'd7': (1,20), 'd8': (1,20), 'd9': (1,20), 'd10': (1,20),
    'm2': (20,1980), 'm3': (20,1980), 'm5': (20,1980),
    'm6': (20,1980), 'm8': (20,1980), 'm9': (20,1980),
    'rho': (1000,1500), 'eta': (0.1,0.8), 'E': (1e7,1e8), 'nu': (0.4,0.49),
}

def get_param_bounds():
    lower = np.array([PARAM_RANGES[n][0] for n in PARAM_NAMES], dtype=np.float64)
    upper = np.array([PARAM_RANGES[n][1] for n in PARAM_NAMES], dtype=np.float64)
    return lower, upper

def validate_and_clip_parameters(params):
    lower, upper = get_param_bounds()
    clipped = np.clip(params, lower, upper)
    eta_vals = clipped[:, 17]
    E_vals   = clipped[:, 18]
    clipped[:, 18] = np.clip(E_vals, 1e7*(1+eta_vals), 1e8*(1+eta_vals))
    return clipped

# ──────────────────────────────────────────────
# TMM physics engine (GPU-accelerated)
# ──────────────────────────────────────────────
def calculate_absorption_curve(param_row):
    """
    Given a single 20-parameter row, return (frequencies, alpha_per_freq)
    where alpha_per_freq is the absorption at every Hz from 1–1000.
    """
    param_matrix = param_row.reshape(1, -1)
    param_tensor = torch.from_numpy(param_matrix).float().to(device)
    W = 2.0  # 2000 mm in metres
    rho_w, c_w = 1000.0, 1500.0
    Z_w = rho_w * c_w

    frequencies = torch.arange(1, 1001, 1, device=device, dtype=torch.float32)
    omega = 2 * np.pi * frequencies
    num_freq = len(frequencies)
    N = 1
    hollow_layer_indices = [1, 2, 4, 5, 7, 8]

    d_vals = param_tensor[:, 0:10] / 1000.0
    m_vals = param_tensor[:, 10:16] / 1000.0
    rho_r  = param_tensor[:, 16:17]
    eta    = param_tensor[:, 17:18]
    E_r    = param_tensor[:, 18:19]
    nu     = param_tensor[:, 19:20]

    E_c = E_r * (1 + 1j * eta)
    lam = (E_c * nu) / ((1 + nu) * (1 - 2*nu))
    mu  = E_c / (2 * (1 + nu))

    eye_matrix = torch.eye(2, device=device, dtype=torch.complex64)
    T_total = eye_matrix.unsqueeze(0).unsqueeze(0).repeat(N, num_freq, 1, 1)

    for lay_idx in range(10):
        d   = d_vals[:, lay_idx:lay_idx+1]
        eps = torch.zeros((N, 1), device=device, dtype=torch.float32)
        if lay_idx in hollow_layer_indices:
            m_ptr = hollow_layer_indices.index(lay_idx)
            eps = m_vals[:, m_ptr:m_ptr+1] / W

        rho_eff = rho_r * (1 - eps**2)
        numerator   = (mu*(lam+2*mu)*(eps**2+1)) + (2*(eps**2)*lam)
        denominator = ((lam+mu)*eps**2) + mu
        mask_denom  = torch.abs(denominator) < 1e-15
        S_eff = torch.where(mask_denom, numerator*1e15, numerator/denominator)
        c_eff = torch.sqrt(S_eff / rho_eff)
        k_eff = omega.unsqueeze(0) / c_eff
        Z_eff = rho_eff * c_eff

        cos_kd = torch.cos(k_eff * d)
        sin_kd = torch.sin(k_eff * d)
        Z_mask = torch.abs(Z_eff) < 1e-15
        t21    = torch.where(Z_mask, torch.full_like(cos_kd, 1e15), 1j*sin_kd/Z_eff)

        T_i = torch.zeros((N, num_freq, 2, 2), device=device, dtype=torch.complex64)
        T_i[:, :, 0, 0] = cos_kd
        T_i[:, :, 0, 1] = 1j * Z_eff * sin_kd
        T_i[:, :, 1, 0] = t21
        T_i[:, :, 1, 1] = cos_kd
        T_total = torch.matmul(T_total, T_i)

    T11 = T_total[:, :, 0, 0]
    T21 = T_total[:, :, 1, 0]
    mask = torch.abs(T21) < 1e-15
    Z_in = torch.where(mask, torch.full_like(T21, 1e15), T11/T21)
    R    = torch.where(mask, torch.ones_like(T21), (Z_in - Z_w)/(Z_in + Z_w))
    alpha = 1 - torch.abs(R)**2
    alpha = torch.clamp(alpha.real, 0.0, 1.0)

    freq_np  = frequencies.cpu().numpy()
    alpha_np = alpha[0].cpu().numpy()
    return freq_np, alpha_np


def calculate_absorption_tmm_batch(param_matrix, chunk_size=2000):
    """Return average absorption for each row."""
    N = param_matrix.shape[0]
    if N > chunk_size:
        results = []
        for i in range(0, N, chunk_size):
            results.append(calculate_absorption_tmm_batch(param_matrix[i:min(i+chunk_size, N)]))
        return np.concatenate(results)

    param_tensor = torch.from_numpy(param_matrix).float().to(device)
    W = 2.0
    rho_w, c_w = 1000.0, 1500.0
    Z_w = rho_w * c_w
    frequencies = torch.arange(1, 1001, 1, device=device, dtype=torch.float32)
    omega = 2 * np.pi * frequencies
    num_freq = len(frequencies)
    hollow_layer_indices = [1, 2, 4, 5, 7, 8]

    d_vals = param_tensor[:, 0:10] / 1000.0
    m_vals = param_tensor[:, 10:16] / 1000.0
    rho_r  = param_tensor[:, 16:17]
    eta    = param_tensor[:, 17:18]
    E_r    = param_tensor[:, 18:19]
    nu     = param_tensor[:, 19:20]
    E_c = E_r * (1 + 1j * eta)
    lam = (E_c * nu) / ((1 + nu) * (1 - 2*nu))
    mu  = E_c / (2 * (1 + nu))

    eye_matrix = torch.eye(2, device=device, dtype=torch.complex64)
    T_total = eye_matrix.unsqueeze(0).unsqueeze(0).repeat(N, num_freq, 1, 1)

    for lay_idx in range(10):
        d   = d_vals[:, lay_idx:lay_idx+1]
        eps = torch.zeros((N, 1), device=device, dtype=torch.float32)
        if lay_idx in hollow_layer_indices:
            m_ptr = hollow_layer_indices.index(lay_idx)
            eps = m_vals[:, m_ptr:m_ptr+1] / W
        rho_eff = rho_r * (1 - eps**2)
        numerator   = (mu*(lam+2*mu)*(eps**2+1)) + (2*(eps**2)*lam)
        denominator = ((lam+mu)*eps**2) + mu
        mask_d = torch.abs(denominator) < 1e-15
        S_eff = torch.where(mask_d, numerator*1e15, numerator/denominator)
        c_eff = torch.sqrt(S_eff / rho_eff)
        k_eff = omega.unsqueeze(0) / c_eff
        Z_eff = rho_eff * c_eff
        cos_kd = torch.cos(k_eff * d)
        sin_kd = torch.sin(k_eff * d)
        Z_mask = torch.abs(Z_eff) < 1e-15
        t21 = torch.where(Z_mask, torch.full_like(cos_kd, 1e15), 1j*sin_kd/Z_eff)
        T_i = torch.zeros((N, num_freq, 2, 2), device=device, dtype=torch.complex64)
        T_i[:, :, 0, 0] = cos_kd
        T_i[:, :, 0, 1] = 1j * Z_eff * sin_kd
        T_i[:, :, 1, 0] = t21
        T_i[:, :, 1, 1] = cos_kd
        T_total = torch.matmul(T_total, T_i)

    T11 = T_total[:, :, 0, 0]
    T21 = T_total[:, :, 1, 0]
    mask = torch.abs(T21) < 1e-15
    Z_in = torch.where(mask, torch.full_like(T21, 1e15), T11/T21)
    R = torch.where(mask, torch.ones_like(T21), (Z_in - Z_w)/(Z_in + Z_w))
    alpha = 1 - torch.abs(R)**2
    alpha = torch.clamp(alpha.real, 0.0, 1.0)
    return torch.mean(alpha, dim=1).cpu().numpy()


# ──────────────────────────────────────────────
# Model architecture (must match training)
# ──────────────────────────────────────────────
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=time.device) * -emb)
        emb = time * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.film_gen = nn.Linear(cond_dim, hidden_dim * 2)
    def forward(self, x, cond):
        residual = x
        x = self.fc1(x)
        x = self.norm(x)
        scale, shift = self.film_gen(self.act(cond)).chunk(2, dim=1)
        x = x * (1 + scale) + shift
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual

class ConditionalDiffusionNetV3(nn.Module):
    def __init__(self, param_dim=20, cond_dim=1, hidden_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(param_dim, hidden_dim)
        self.blocks = nn.ModuleList([FiLMBlock(hidden_dim, hidden_dim) for _ in range(4)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_act = nn.SiLU()
        self.output_proj = nn.Linear(hidden_dim, param_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    def forward(self, x, t, cond):
        g = self.time_mlp(t) + self.cond_mlp(cond)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, g)
        h = self.final_norm(h)
        h = self.final_act(h)
        return self.output_proj(h)

class ForwardSurrogate(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.01), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.01), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

class DiffusionModelV3:
    def __init__(self, network, surrogate, num_timesteps=100, device='cpu'):
        self.net = network.to(device)
        self.surrogate = surrogate
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = cosine_beta_schedule(num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.ema_net = copy.deepcopy(network).to(device)
        self.ema_net.eval()

    @torch.no_grad()
    def sample(self, cond, scaler=None, use_ema=True):
        net = self.ema_net if use_ema else self.net
        net.eval()
        bs = cond.shape[0]
        x = torch.randn(bs, 20, device=self.device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((bs, 1), i, device=self.device).float()
            eps = net(x, t, cond)
            beta = self.betas[i]; alpha = self.alphas[i]; ab = self.alpha_bars[i]
            noise = torch.randn_like(x) if i > 0 else 0
            mean = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/torch.sqrt(1-ab)) * eps)
            x = mean + torch.sqrt(beta) * noise
            if i % 10 == 0 and scaler is not None:
                xnp = scaler.inverse_transform(x.cpu().numpy())
                xnp = validate_and_clip_parameters(xnp)
                x = torch.FloatTensor(scaler.transform(xnp)).to(self.device)
        if scaler is not None:
            xnp = scaler.inverse_transform(x.cpu().numpy())
            xnp = validate_and_clip_parameters(xnp)
            x = torch.FloatTensor(scaler.transform(xnp)).to(self.device)
        return x


# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
def load_models():
    # Surrogate
    surrogate = ForwardSurrogate(input_dim=20, hidden_dim=256).to(device)
    surrogate.load_state_dict(torch.load(SURROGATE_PATH, map_location=device, weights_only=True))
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    # Diffusion V3
    ckpt = torch.load(MODEL_V3_PATH, map_location=device, weights_only=False)
    net = ConditionalDiffusionNetV3(hidden_dim=512).to(device)
    net.load_state_dict(ckpt['model_state_dict'])
    diff = DiffusionModelV3(net, surrogate, num_timesteps=ckpt['num_timesteps'], device=device)
    if 'ema_state_dict' in ckpt:
        diff.ema_net.load_state_dict(ckpt['ema_state_dict'])

    with open(SCALER_V3_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print("Diffusion Model V3 loaded successfully")
    return diff, scaler


# ──────────────────────────────────────────────
# Base-case reference (from Theoretical_model.py)
# ──────────────────────────────────────────────
def get_base_case_curve():
    """Compute absorption curve for the base case (Table 1 & 2 in paper)."""
    base_params = np.array([[
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10,       # d1–d10
        1000, 1000, 1000, 1000, 1000, 1000,             # m2,m3,m5,m6,m8,m9
        1130, 0.4, 5e7, 0.44                             # rho, eta, E, nu
    ]])
    freq, alpha = calculate_absorption_curve(base_params[0])
    return freq, alpha


# ──────────────────────────────────────────────
# Predict parameters for a target absorption
# ──────────────────────────────────────────────
def predict_for_target(diffusion, scaler, target_absorption, num_candidates=20):
    """Generate candidates and pick the best one."""
    cond = torch.FloatTensor([[target_absorption]]).repeat(num_candidates, 1).to(device)
    with torch.no_grad():
        gen = diffusion.sample(cond, scaler=scaler)
    gen_real = scaler.inverse_transform(gen.cpu().numpy())
    gen_real = validate_and_clip_parameters(gen_real)

    # Evaluate with TMM
    avg_abs = calculate_absorption_tmm_batch(gen_real)
    errors = np.abs(avg_abs - target_absorption) / (target_absorption + 1e-8) * 100
    best_idx = np.argmin(errors)

    return gen_real[best_idx], avg_abs[best_idx], errors[best_idx], gen_real, avg_abs, errors


# ──────────────────────────────────────────────
# Pretty-print parameters
# ──────────────────────────────────────────────
def print_parameters(params, target, actual_avg, error):
    print(f"\n{'='*60}")
    print(f"  Target absorption:  {target:.4f}")
    print(f"  Actual (TMM) avg:   {actual_avg:.6f}")
    print(f"  Relative error:     {error:.2f}%")
    print(f"{'='*60}")
    print(f"  {'Parameter':<8} {'Value':>14}")
    print(f"  {'-'*24}")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:<8} {params[i]:>14.4f}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#17becf']

def plot_curves(target_results, base_freq, base_alpha):
    """
    target_results: list of dicts with keys:
        target, params, actual_avg, freq, alpha_curve
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Base case
    ax.plot(base_freq, base_alpha, color='black', linestyle='--', linewidth=1.5,
            label=f'Base Case (avg={np.mean(base_alpha):.4f})')

    # Each target
    for i, res in enumerate(target_results):
        col = COLOURS[i % len(COLOURS)]
        ax.plot(res['freq'], res['alpha_curve'], color=col, linewidth=2.0,
                label=f"Target {res['target']:.4f} → actual {res['actual_avg']:.4f}  (err {res['error']:.2f}%)")

    ax.set_xlabel('Frequency / Hz', fontsize=12)
    ax.set_ylabel('Sound Absorption Coefficient', fontsize=12)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.set_title('Diffusion Model V3 — Inverse Design Verification via TMM', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Main — EDIT YOUR TARGETS HERE
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # ╔══════════════════════════════════════════╗
    # ║  ADD YOUR TARGET ABSORPTION VALUES HERE  ║
    # ╚══════════════════════════════════════════╝
    targets = [0.4, 0.34, 0.25, 0.55]

    NUM_CANDIDATES = 20   # more candidates = better best-of-N selection

    print(f"\nTarget absorption coefficients: {targets}")
    print(f"Number of candidates per target: {NUM_CANDIDATES}")

    # Load model
    diffusion, scaler = load_models()

    # Base case
    print("\nComputing base-case curve...")
    base_freq, base_alpha = get_base_case_curve()
    print(f"Base-case average absorption: {np.mean(base_alpha):.4f}")

    # Predict for each target
    all_results = []
    for target in targets:
        print(f"\n--- Generating design for target = {target:.4f} ---")
        best_params, actual_avg, error, all_params, all_avg, all_errors = \
            predict_for_target(diffusion, scaler, target, num_candidates=NUM_CANDIDATES)

        print_parameters(best_params, target, actual_avg, error)

        # Full absorption curve for the best candidate
        freq, alpha_curve = calculate_absorption_curve(best_params)

        all_results.append({
            'target': target,
            'params': best_params,
            'actual_avg': actual_avg,
            'error': error,
            'freq': freq,
            'alpha_curve': alpha_curve,
        })

    # Plot everything
    print("\nPlotting curves...")
    plot_curves(all_results, base_freq, base_alpha)
    print("Done!")
