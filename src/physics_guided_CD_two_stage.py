import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import time
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# PARAMETER CONSTRAINTS
# ==========================================
PARAM_RANGES = {
    # Layer thicknesses d1-d10 (mm)
    'd1': (1.0, 20.0), 'd2': (1.0, 20.0), 'd3': (1.0, 20.0), 'd4': (1.0, 20.0), 'd5': (1.0, 20.0),
    'd6': (1.0, 20.0), 'd7': (1.0, 20.0), 'd8': (1.0, 20.0), 'd9': (1.0, 20.0), 'd10': (1.0, 20.0),
    # Hollow diameters m2, m3, m5, m6, m8, m9 (mm)
    'm2': (20.0, 1980.0), 'm3': (20.0, 1980.0), 'm5': (20.0, 1980.0),
    'm6': (20.0, 1980.0), 'm8': (20.0, 1980.0), 'm9': (20.0, 1980.0),
    # Material properties
    'rho': (1000.0, 1500.0),      # Density (kg/m³)
    'eta': (0.1, 0.8),             # Loss factor
    'E': (1e7, 1e8),               # Young's modulus (Pa) - will adjust based on eta
    'nu': (0.4, 0.49)              # Poisson's ratio
}

def get_param_bounds():
    """Get parameter bounds as numpy arrays."""
    param_names = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
                   'm2','m3','m5','m6','m8','m9','rho','eta','E','nu']
    lower_bounds = np.array([PARAM_RANGES[name][0] for name in param_names])
    upper_bounds = np.array([PARAM_RANGES[name][1] for name in param_names])
    return lower_bounds, upper_bounds

def validate_and_clip_parameters(params):
    """
    Validate and clip parameters to physical constraints.
    Also enforces E constraint based on eta: 1E7(1+eta) <= E <= 1E8(1+eta)
    
    Args:
        params: (N, 20) numpy array of parameters
    Returns:
        clipped_params: (N, 20) numpy array with valid parameters
    """
    lower_bounds, upper_bounds = get_param_bounds()
    
    # Clip all parameters to their basic ranges
    clipped = np.clip(params, lower_bounds, upper_bounds)
    
    # Special handling for Young's modulus E based on eta
    eta_values = clipped[:, 17]  # eta is at index 17
    E_values = clipped[:, 18]    # E is at index 18
    
    # Enforce E constraint: 1E7(1+eta) <= E <= 1E8(1+eta)
    E_min = 1e7 * (1 + eta_values)
    E_max = 1e8 * (1 + eta_values)
    clipped[:, 18] = np.clip(E_values, E_min, E_max)
    
    return clipped

# ==========================================
# 1. PHYSICS ENGINE (Transfer Matrix Method)
# ==========================================
def calculate_absorption_tmm_batch(param_matrix):
    """
    Calculates average absorption for a batch of parameters using TMM.
    OPTIMIZED VERSION with vectorized frequency calculations.
    
    Args:
        param_matrix: (N, 20) array [d1..d10, m2..m9, rho, eta, E, nu]
    """
    N = param_matrix.shape[0]
    results = []

    # Constants (SI Units)
    W = 2000.0 / 1000.0 # Width in meters
    rho_w, c_w = 1000.0, 1500.0
    Z_w = rho_w * c_w
    
    # Vectorize frequencies (all at once)
    frequencies = np.arange(1, 1001, 1)
    omega = 2 * np.pi * frequencies  # Shape: (1000,)
    
    # Layer mapping
    hollow_layer_indices = [1, 2, 4, 5, 7, 8] 
    
    for i in range(N):
        p = param_matrix[i]
        d_vals = p[0:10] / 1000.0 # mm to m
        m_vals = p[10:16] / 1000.0 # mm to m
        rho_r = p[16]
        eta = p[17]
        E_r = p[18]
        nu = p[19]
        
        # Complex Modulus
        E_c = E_r * (1 + 1j * eta)
        
        # Lame Constants
        lam = (E_c * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E_c / (2 * (1 + nu))
        
        # Initialize transfer matrix for all frequencies at once
        # Shape: (num_freq, 2, 2)
        T_total = np.tile(np.eye(2, dtype=complex), (len(frequencies), 1, 1))
        
        for lay_idx in range(10):
            d = d_vals[lay_idx]
            
            # Check if this layer is hollow
            eps = 0.0
            if lay_idx in hollow_layer_indices:
                m_ptr = hollow_layer_indices.index(lay_idx)
                eps = m_vals[m_ptr] / W
            
            # Equivalent Medium Theory (EMT)
            rho_eff = rho_r * (1 - eps**2)
            
            # Corrected Bulk Modulus Equation
            numerator = (mu * (lam + 2*mu) * (eps**2 + 1)) + (2 * (eps**2) * lam)
            denominator = ((lam + mu) * eps**2) + mu
            
            if abs(denominator) < 1e-15: 
                S_eff = numerator * 1e15
            else: 
                S_eff = numerator / denominator
            
            c_eff = np.sqrt(S_eff / rho_eff)
            k_eff = omega / c_eff  # Vectorized: Shape (1000,)
            Z_eff = rho_eff * c_eff
            
            # Transfer Matrix - vectorized for all frequencies
            cos_kd = np.cos(k_eff * d)  # Shape: (1000,)
            sin_kd = np.sin(k_eff * d)  # Shape: (1000,)
            
            if abs(Z_eff) < 1e-15: 
                t21 = np.full_like(cos_kd, 1e15)
            else: 
                t21 = 1j * sin_kd / Z_eff
            
            # Build transfer matrix for all frequencies
            # T_i shape: (1000, 2, 2)
            T_i = np.zeros((len(frequencies), 2, 2), dtype=complex)
            T_i[:, 0, 0] = cos_kd
            T_i[:, 0, 1] = 1j * Z_eff * sin_kd
            T_i[:, 1, 0] = t21
            T_i[:, 1, 1] = cos_kd
            
            # Matrix multiplication for all frequencies at once
            T_total = np.matmul(T_total, T_i)
        
        # Reflection Coefficient for all frequencies
        T11 = T_total[:, 0, 0]  # Shape: (1000,)
        T21 = T_total[:, 1, 0]  # Shape: (1000,)
        
        # Avoid division by zero
        mask = np.abs(T21) < 1e-15
        R = np.ones_like(T21, dtype=complex)
        
        Z_in = np.divide(T11, T21, where=~mask)
        R[~mask] = (Z_in[~mask] - Z_w) / (Z_in[~mask] + Z_w)
        
        alpha = 1 - np.abs(R)**2
        alpha = np.clip(alpha.real, 0.0, 1.0)
        
        results.append(np.mean(alpha))
        
    return np.array(results)

# ==========================================
# 2. INITIAL PARAMETER PREDICTOR
# ==========================================

class InitialParameterPredictor(nn.Module):
    """
    Predicts initial noisy parameters from target absorption.
    This gives the diffusion model a better starting point.
    
    Architecture:
        - Shared encoder for target absorption
        - Multiple heads for diversity (5 candidates)
        - Each head produces different predictions with controlled noise
    """
    def __init__(self, num_candidates=5, param_dim=20, hidden_dim=256):
        super().__init__()
        self.num_candidates = num_candidates
        self.param_dim = param_dim
        
        # Shared encoder: Learns general mapping from target to parameter space
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Diversity heads: Each produces different predictions
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(128, param_dim)
            ) for _ in range(num_candidates)
        ])
        
        # Learnable noise scales for each head (for diversity)
        self.noise_scales = nn.Parameter(torch.linspace(0.05, 0.15, num_candidates))
    
    def forward(self, target):
        """
        Args:
            target: (batch_size, 1) - Target absorption values
        Returns:
            predictions: (batch_size * num_candidates, 20) - Noisy parameter predictions
        """
        batch_size = target.shape[0]
        
        # Encode target into shared representation
        encoded = self.encoder(target)
        
        # Generate diverse predictions from each head
        predictions = []
        for i, head in enumerate(self.heads):
            base_pred = head(encoded)
            noise = torch.randn_like(base_pred) * self.noise_scales[i]
            pred_with_noise = base_pred + noise
            predictions.append(pred_with_noise)
        
        # Stack all candidates: (batch_size * 5, 20)
        return torch.cat(predictions, dim=0)

# ==========================================
# 3. DIFFUSION MODEL ARCHITECTURE
# ==========================================

class ResidualBlock(nn.Module):
    """Residual block with Swish activation."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU() 
    
    def forward(self, x):
        return x + self.fc(self.act(x))

class ConditionalDiffusionNet(nn.Module):
    """Neural Network that predicts noise in parameters."""
    def __init__(self, param_dim=20, cond_dim=1, hidden_dim=256):
        super().__init__()
        
        # Embeddings for inputs
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_embed = nn.Linear(param_dim, hidden_dim)
        
        # Main Backbone (ResNet-like MLP)
        self.layers = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_dim)
        )
        
    def forward(self, x, t, cond):
        emb = self.input_embed(x) + self.time_embed(t) + self.cond_embed(cond)
        return self.layers(emb)

class DiffusionModel:
    """Manages the forward (noise) and reverse (denoise) diffusion processes."""
    def __init__(self, network, num_timesteps=100, device='cpu'):
        self.net = network.to(device)
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define Noise Schedule (Beta Schedule)
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def train_step(self, x0, cond, optimizer):
        """Standard diffusion training step."""
        batch_size = x0.shape[0]
        
        t = torch.randint(0, self.num_timesteps, (batch_size, 1), device=self.device).float()
        epsilon = torch.randn_like(x0)
        
        alpha_bar_t = self.alpha_bars[t.long().squeeze()].unsqueeze(1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        
        t_norm = t / self.num_timesteps
        eps_pred = self.net(xt, t_norm, cond)
        
        loss = nn.MSELoss()(eps_pred, epsilon)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(self, dataloader):
        """Calculate validation loss."""
        self.net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                batch_size = x_batch.shape[0]
                
                t = torch.randint(0, self.num_timesteps, (batch_size, 1), device=self.device).float()
                epsilon = torch.randn_like(x_batch)
                
                alpha_bar_t = self.alpha_bars[t.long().squeeze()].unsqueeze(1)
                xt = torch.sqrt(alpha_bar_t) * x_batch + torch.sqrt(1 - alpha_bar_t) * epsilon
                
                t_norm = t / self.num_timesteps
                eps_pred = self.net(xt, t_norm, y_batch)
                
                loss = nn.MSELoss()(eps_pred, epsilon)
                val_loss += loss.item()
        
        self.net.train()
        return val_loss / len(dataloader)
    
    @torch.no_grad()
    def sample_from_noisy(self, x_start, cond, start_timestep=50, scaler=None):
        """
        Sample from noisy initial predictions (partial denoising).
        """
        x = x_start.clone()
        
        # Denoise from start_timestep down to 0
        for i in reversed(range(start_timestep)):
            batch_size = x.shape[0]
            t = torch.full((batch_size, 1), i, device=self.device).float()
            t_norm = t / self.num_timesteps
            
            eps_pred = self.net(x, t_norm, cond)
            
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_bars[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            mean = (1 / torch.sqrt(alpha_t)) * \
                   (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
            std = torch.sqrt(beta_t)
            x = mean + std * noise
            
            # Apply constraints periodically
            if i % 10 == 0 and scaler is not None:
                x_original = scaler.inverse_transform(x.cpu().numpy())
                x_constrained = validate_and_clip_parameters(x_original)
                x = torch.FloatTensor(scaler.transform(x_constrained)).to(self.device)
        
        # Final constraint
        if scaler is not None:
            x_original = scaler.inverse_transform(x.cpu().numpy())
            x_constrained = validate_and_clip_parameters(x_original)
            x = torch.FloatTensor(scaler.transform(x_constrained)).to(self.device)
        
        return x

# ==========================================
# 4. TWO-STAGE PREDICTOR
# ==========================================

class TwoStagePredictor:
    """
    Combines initial predictor with diffusion denoising.
    
    Pipeline:
        1. Predictor generates initial noisy parameters
        2. Diffusion denoises from partial noise level
    """
    def __init__(self, predictor, diffusion, device='cpu', start_timestep=50):
        self.predictor = predictor.to(device)
        self.diffusion = diffusion
        self.device = device
        self.start_timestep = start_timestep
    
    @torch.no_grad()
    def predict(self, target, scaler=None):
        """Two-stage prediction process."""
        batch_size = target.shape[0]
        num_candidates = self.predictor.num_candidates
        
        # Stage 1: Initial Prediction
        x_noisy = self.predictor(target)
        
        # Expand target for all candidates
        target_expanded = target.repeat_interleave(num_candidates, dim=0)
        
        # Stage 2: Partial Denoising
        x_final = self.diffusion.sample_from_noisy(
            x_noisy, 
            target_expanded, 
            start_timestep=self.start_timestep,
            scaler=scaler
        )
        
        return x_final

# ==========================================
# 5. TRAINING FUNCTIONS
# ==========================================

def train_predictor(predictor, X_train, y_train, scaler_x, device, epochs=15, batch_size=256):
    """Train the initial predictor network."""
    print("\n" + "="*70)
    print("STAGE 1: TRAINING INITIAL PREDICTOR")
    print("="*70)
    
    optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    X_train_norm = scaler_x.transform(X_train)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        predictor.train()
        
        epoch_loss = 0
        num_batches = 0
        
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            y_batch = torch.FloatTensor(y_train[batch_indices]).to(device)
            x_true = torch.FloatTensor(X_train_norm[batch_indices]).to(device)
            
            x_pred = predictor(y_batch)
            x_pred_reshaped = x_pred.reshape(len(batch_indices), predictor.num_candidates, 20)
            
            distances = torch.norm(x_pred_reshaped - x_true.unsqueeze(1), dim=2)
            min_distances = torch.min(distances, dim=1)[0]
            loss = min_distances.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.5f} | Time: {epoch_time:.2f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  → ✓ New best predictor!")
    
    print("="*70)
    print(f"✓ Predictor training completed! Best Loss: {best_loss:.5f}")
    
    return predictor

def train_diffusion(diffusion, train_loader, val_loader, optimizer, epochs=10):
    """Train the diffusion denoiser."""
    print("\n" + "="*70)
    print("STAGE 2: TRAINING DIFFUSION DENOISER")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        diffusion.net.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(diffusion.device), y_batch.to(diffusion.device)
            loss = diffusion.train_step(x_batch, y_batch, optimizer)
            train_loss += loss
        
        train_loss /= len(train_loader)
        val_loss = diffusion.validate(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.5f} | "
              f"Val Loss: {val_loss:.5f} | "
              f"Time: {epoch_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  → ✓ New best diffusion model!")
    
    print("="*70)
    print(f"✓ Diffusion training completed! Best Val Loss: {best_val_loss:.5f}")
    
    return best_val_loss

# ==========================================
# 6. MODEL SAVING/LOADING
# ==========================================

def save_two_stage_model(predictor, diffusion, scaler_x, filepath='two_stage_model.pth'):
    """Save both predictor and diffusion models."""
    torch.save({
        'predictor_state_dict': predictor.state_dict(),
        'diffusion_state_dict': diffusion.net.state_dict(),
        'num_timesteps': diffusion.num_timesteps,
        'num_candidates': predictor.num_candidates,
    }, filepath)
    
    scaler_path = filepath.replace('.pth', '_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_x, f)
    
    print(f"✓ Model saved to {filepath}")
    print(f"✓ Scaler saved to {scaler_path}")

def load_two_stage_model(filepath='two_stage_model.pth', device='cpu', start_timestep=50):
    """Load both predictor and diffusion models."""
    if not os.path.exists(filepath):
        return None, None, None
    
    scaler_path = filepath.replace('.pth', '_scaler.pkl')
    if not os.path.exists(scaler_path):
        return None, None, None
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    
    predictor = InitialParameterPredictor(num_candidates=checkpoint['num_candidates']).to(device)
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    
    net = ConditionalDiffusionNet().to(device)
    net.load_state_dict(checkpoint['diffusion_state_dict'])
    diffusion = DiffusionModel(net, num_timesteps=checkpoint['num_timesteps'], device=device)
    
    two_stage = TwoStagePredictor(predictor, diffusion, device, start_timestep)
    
    with open(scaler_path, 'rb') as f:
        scaler_x = pickle.load(f)
    
    print(f"✓ Two-stage model loaded from {filepath}")
    print(f"✓ Scaler loaded from {scaler_path}")
    
    return two_stage, diffusion, scaler_x

# ==========================================
# 7. COMPREHENSIVE TESTING
# ==========================================

def comprehensive_test(two_stage, X_test_norm, y_test, scaler_x, device, num_candidates=5):
    """
    Test on entire test set and provide comprehensive statistics.
    SAME FORMAT AS physics_guided_CD.py
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE TESTING ON ENTIRE TEST SET")
    print("="*70)
    
    N_test = len(X_test_norm)
    print(f"\nTest samples: {N_test}")
    print(f"Candidates per target: {num_candidates}")
    
    targets_batch = torch.FloatTensor(y_test).to(device)
    
    print(f"\nGenerating {num_candidates} candidates per target...")
    start_time = time.time()
    
    with torch.no_grad():
        generated_norm = two_stage.predict(targets_batch, scaler=scaler_x)
    
    generation_time = time.time() - start_time
    print(f"✓ Generation completed in {generation_time:.2f}s")
    
    generated_norm = generated_norm.cpu().numpy()
    generated_real = scaler_x.inverse_transform(generated_norm)
    generated_real = validate_and_clip_parameters(generated_real)
    
    print("\nRunning Physics Engine (TMM) Validation...")
    validation_start = time.time()
    actual_absorptions = calculate_absorption_tmm_batch(generated_real)
    validation_time = time.time() - validation_start
    print(f"✓ Validation completed in {validation_time:.2f}s")
    
    actual_abs_reshaped = actual_absorptions.reshape(N_test, num_candidates)
    targets_reshaped = y_test.reshape(N_test, 1)
    
    errors = np.abs(actual_abs_reshaped - targets_reshaped) / (targets_reshaped + 1e-8)
    best_indices = np.argmin(errors, axis=1)
    
    final_errors = []
    best_predictions = []
    best_designs = []
    
    for i in range(N_test):
        best_idx = best_indices[i]
        best_error = errors[i, best_idx]
        final_errors.append(best_error)
        best_predictions.append(actual_abs_reshaped[i, best_idx])
        best_designs.append(generated_real[i * num_candidates + best_idx])
    
    final_errors = np.array(final_errors) * 100
    best_predictions = np.array(best_predictions)
    best_designs = np.array(best_designs)
    
    avg_error = np.mean(final_errors)
    min_error = np.min(final_errors)
    max_error = np.max(final_errors)
    median_error = np.median(final_errors)
    std_error = np.std(final_errors)
    
    # Find min and max error samples
    min_error_idx = np.argmin(final_errors)
    max_error_idx = np.argmax(final_errors)
    
    print("\nVerifying parameter constraints...")
    violations = 0
    for design in best_designs:
        lower_bounds, upper_bounds = get_param_bounds()
        if np.any(design < lower_bounds) or np.any(design > upper_bounds):
            violations += 1
        eta = design[17]
        E = design[18]
        E_min = 1e7 * (1 + eta)
        E_max = 1e8 * (1 + eta)
        if E < E_min or E > E_max:
            violations += 1
    
    if violations == 0:
        print("✓ All parameters within valid ranges!")
    else:
        print(f"⚠ Warning: {violations}/{N_test} samples had constraint violations (auto-corrected)")
    
    # Print statistics ONCE
    print("\n" + "="*70)
    print("TEST SET STATISTICS")
    print("="*70)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-"*70)
    print(f"{'Average Relative Error':<30} {avg_error:>15.4f} %")
    print(f"{'Minimum Relative Error':<30} {min_error:>15.4f} %")
    print(f"{'Maximum Relative Error':<30} {max_error:>15.4f} %")
    print(f"{'Median Relative Error':<30} {median_error:>15.4f} %")
    print(f"{'Std Dev Relative Error':<30} {std_error:>15.4f} %")
    print("="*70)
    
    # Performance rating
    if avg_error < 5:
        rating = "EXCELLENT (<5%) ⭐⭐⭐⭐⭐"
    elif avg_error < 10:
        rating = "GOOD (<10%) ⭐⭐⭐⭐"
    elif avg_error < 15:
        rating = "ACCEPTABLE (<15%) ⭐⭐⭐"
    else:
        rating = "NEEDS IMPROVEMENT (>15%) ⭐⭐"
    
    print(f"\nPerformance Rating: {rating}")
    
    # Print MIN ERROR sample details
    param_names = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
                   'm2','m3','m5','m6','m8','m9','rho','eta','E','nu']
    
    print("\n" + "="*70)
    print("SAMPLE WITH MINIMUM ERROR")
    print("="*70)
    print(f"Target Absorption:     {y_test[min_error_idx, 0]:.6f}")
    print(f"Calculated Absorption: {best_predictions[min_error_idx]:.6f}")
    print(f"Relative Error:        {final_errors[min_error_idx]:.2f}%")
    print(f"\n{'Parameter':<10} {'Value':>15}")
    print("-"*70)
    for i, name in enumerate(param_names):
        print(f"{name:<10} {best_designs[min_error_idx][i]:>15.4f}")
    print("="*70)
    
    # Print MAX ERROR sample details
    print("\n" + "="*70)
    print("SAMPLE WITH MAXIMUM ERROR")
    print("="*70)
    print(f"Target Absorption:     {y_test[max_error_idx, 0]:.6f}")
    print(f"Calculated Absorption: {best_predictions[max_error_idx]:.6f}")
    print(f"Relative Error:        {final_errors[max_error_idx]:.2f}%")
    print(f"\n{'Parameter':<10} {'Value':>15}")
    print("-"*70)
    for i, name in enumerate(param_names):
        print(f"{name:<10} {best_designs[max_error_idx][i]:>15.4f}")
    print("="*70)
    
    return {
        'avg_error': avg_error,
        'min_error': min_error,
        'max_error': max_error,
        'median_error': median_error,
        'std_error': std_error,
        'best_predictions': best_predictions,
        'best_designs': best_designs,
        'final_errors': final_errors
    }

# ==========================================
# 8. MANUAL TARGET PREDICTION
# ==========================================
def predict_manual_target(two_stage, target_absorption, scaler_x, device, num_candidates=5):
    """
    Generate design for manually specified absorption target using two-stage approach.
    
    Args:
        two_stage: Trained two-stage predictor
        target_absorption: Float - Desired absorption coefficient (e.g., 0.75)
        scaler_x: Scaler for parameters
        device: Computation device
        num_candidates: Number of candidates to generate
    """
    print("\n" + "="*70)
    print("MANUAL TARGET PREDICTION (TWO-STAGE)")
    print("="*70)
    
    print(f"\nTarget Absorption (Manual): {target_absorption:.6f}")
    
    target_tensor = torch.FloatTensor([[target_absorption]]).to(device)
    
    print(f"\nGenerating {num_candidates} candidate designs...")
    with torch.no_grad():
        generated_norm = two_stage.predict(target_tensor, scaler=scaler_x)
    
    generated_norm = generated_norm.cpu().numpy()
    generated_real = scaler_x.inverse_transform(generated_norm)
    generated_real = validate_and_clip_parameters(generated_real)
    
    print("Validating with Physics Engine...")
    actual_absorptions = calculate_absorption_tmm_batch(generated_real)
    
    errors = np.abs(actual_absorptions - target_absorption) / (target_absorption + 1e-8) * 100
    best_idx = np.argmin(errors)
    best_design = generated_real[best_idx]
    best_absorption = actual_absorptions[best_idx]
    best_error = errors[best_idx]
    
    param_names = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
                   'm2','m3','m5','m6','m8','m9','rho','eta','E','nu']
    
    # Print summary of all candidates
    print("\n" + "="*70)
    print(f"SUMMARY: ALL {num_candidates} CANDIDATES")
    print("="*70)
    print(f"{'Candidate':<12} {'Absorption':>15} {'Relative Error %':>20}")
    print("-"*70)
    for i in range(num_candidates):
        marker = " ← BEST" if i == best_idx else ""
        print(f"Candidate {i+1:<2} {actual_absorptions[i]:>15.6f} {errors[i]:>20.2f}{marker}")
    print("="*70)
    
    # Print ONLY the best candidate's parameters
    print("\n" + "="*70)
    print(f"BEST CANDIDATE (Candidate {best_idx + 1})")
    print("="*70)
    print(f"Target Absorption:     {target_absorption:.6f}")
    print(f"Calculated Absorption: {best_absorption:.6f}")
    print(f"Relative Error:        {best_error:.2f}%")
    print(f"\n{'Parameter':<10} {'Value':>15}")
    print("-"*70)
    for i, name in enumerate(param_names):
        print(f"{name:<10} {best_design[i]:>15.4f}")
    print("="*70)
    
    return {
        'target': target_absorption,
        'error': best_error,
        'design': best_design,
        'absorption': best_absorption,
        'all_candidates': generated_real,
        'all_absorptions': actual_absorptions,
        'all_errors': errors
    }

# ==========================================
# 9. MAIN EXECUTION PIPELINE
# ==========================================

def main():
    print("\n" + "="*70)
    print("TWO-STAGE PHYSICS-GUIDED CONDITIONAL DIFFUSION MODEL")
    print("="*70)
    
    print("\nLoading Data...")
    try:
        df = pd.read_csv('lhs_data.csv')
        print(f"✓ Dataset loaded: {df.shape}")
    except FileNotFoundError:
        print("Error: 'lhs_data.csv' not found. Please upload the dataset first.")
        return
    
    param_cols = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
                  'm2','m3','m5','m6','m8','m9','rho','eta','E','nu']
    target_col = 'Average_Absorption'
    
    X = df[param_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    print("\nValidating input data constraints...")
    X_validated = validate_and_clip_parameters(X)
    violations = np.sum(np.abs(X - X_validated) > 1e-6)
    if violations > 0:
        print(f"⚠ Warning: {violations} parameters were outside valid ranges (corrected)")
        X = X_validated
    else:
        print("✓ All input data within valid parameter ranges")
    
    # SAME DATA SPLIT AS physics_guided_CD.py
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\nData Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model_path = 'two_stage_model.pth'
    two_stage, diffusion, scaler_x = load_two_stage_model(model_path, device, start_timestep=50)
    
    if two_stage is not None:
        print("\n💡 TIP: To retrain the model, delete 'two_stage_model.pth' and 'two_stage_model_scaler.pkl' then run again.")
    else:
        print("\n" + "="*70)
        print("NO EXISTING MODEL FOUND - STARTING TRAINING")
        print("="*70)
        
        scaler_x = MinMaxScaler()
        X_train_norm = scaler_x.fit_transform(X_train)
        X_val_norm = scaler_x.transform(X_val)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_norm), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        predictor = InitialParameterPredictor(num_candidates=5).to(device)
        diffusion_net = ConditionalDiffusionNet().to(device)
        diffusion = DiffusionModel(diffusion_net, num_timesteps=100, device=device)
        
        print(f"\nPredictor Parameters: {sum(p.numel() for p in predictor.parameters()):,}")
        print(f"Diffusion Parameters: {sum(p.numel() for p in diffusion_net.parameters()):,}")
        
        predictor = train_predictor(predictor, X_train, y_train, scaler_x, device, epochs=15)
        
        diffusion_optimizer = optim.Adam(diffusion_net.parameters(), lr=1e-3)
        train_diffusion(diffusion, train_loader, val_loader, diffusion_optimizer, epochs=10)
        
        two_stage = TwoStagePredictor(predictor, diffusion, device, start_timestep=50)
        
        save_two_stage_model(predictor, diffusion, scaler_x, model_path)
    
    X_test_norm = scaler_x.transform(X_test)
    
    test_results = comprehensive_test(two_stage, X_test_norm, y_test, scaler_x, device, num_candidates=5)
    
    # --- MANUAL TARGET PREDICTION ---
    print("\n" + "="*70)
    print("MANUAL TARGET ABSORPTION INPUT")
    print("="*70)
    
    # CHANGE THIS VALUE TO YOUR DESIRED TARGET ABSORPTION
    manual_target = 0.325678  # Example: Change this to any value between 0 and 1
    
    manual_results = predict_manual_target(two_stage, manual_target, scaler_x, device, num_candidates=5)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\n📊 TEST SET PERFORMANCE:")
    print(f"  ✓ Average Relative Error: {test_results['avg_error']:.4f}%")
    print(f"  ✓ Minimum Relative Error: {test_results['min_error']:.4f}%")
    print(f"  ✓ Maximum Relative Error: {test_results['max_error']:.4f}%")
    
    print("\n🎯 MANUAL TARGET:")
    print(f"  ✓ Target Absorption: {manual_results['target']:.6f}")
    print(f"  ✓ Best Candidate Error: {manual_results['error']:.2f}%")
    
    print("\n💾 MODEL INFO:")
    print(f"  ✓ Model saved as: {model_path}")
    print(f"  ✓ Scaler saved as: two_stage_model_scaler.pkl")
    print(f"  ✓ To retrain: Delete both files and run again")
    
    print("\n✅ PARAMETER CONSTRAINTS:")
    print(f"  ✓ d1-d10: [1.0, 20.0] mm")
    print(f"  ✓ m2,m3,m5,m6,m8,m9: [20.0, 1980.0] mm")
    print(f"  ✓ rho: [1000.0, 1500.0] kg/m³")
    print(f"  ✓ eta: [0.1, 0.8]")
    print(f"  ✓ E: [1E7(1+eta), 1E8(1+eta)] Pa")
    print(f"  ✓ nu: [0.4, 0.49]")
    print("="*70)

if __name__ == "__main__":
    main()