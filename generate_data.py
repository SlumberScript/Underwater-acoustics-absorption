import numpy as np
import pandas as pd
import warnings
from scipy.stats import qmc
from numba import jit, prange
import multiprocessing as mp
from functools import partial

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

@jit(nopython=True, cache=True)
def calculate_single_absorption_fast(d_vals, m_vals, rho_r, eta, E_r, nu, W):
    """
    Optimized absorption calculation using Numba JIT compilation.
    """
    # Convert to complex
    E_complex = E_r * (1.0 + 1j * eta)
    
    # Standard Water Properties
    rho_w, c_w = 1000.0, 1500.0
    Z_w = rho_w * c_w
    
    # Lame Constants
    lam = (E_complex * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_complex / (2.0 * (1.0 + nu))
    
    # Hollow layer mapping
    hollow_layers = np.array([1, 2, 4, 5, 7, 8], dtype=np.int32)
    
    alpha_sum = 0.0
    
    # Vectorized frequency loop
    for f in range(1, 1001):
        omega = 2.0 * np.pi * f
        
        # Initialize identity matrix
        T_total = np.eye(2, dtype=np.complex128)
        
        for i in range(10):
            layer_idx = i
            d = d_vals[i]
            
            # Perforation Rate epsilon
            eps = 0.0
            for j in range(6):
                if hollow_layers[j] == i:
                    eps = m_vals[j] / W
                    break
            
            # Effective Density
            rho_eff = rho_r * (1.0 - eps**2)
            
            # Effective Volume Modulus S'
            numerator = (mu * (lam + 2.0*mu) * (eps**2 + 1.0)) + (2.0 * (eps**2) * lam)
            denominator = ((lam + mu) * eps**2) + mu
            
            if abs(denominator) < 1e-15:
                S_eff = numerator * 1e15
            else:
                S_eff = numerator / denominator
            
            # Effective Sound Velocity
            c_eff = np.sqrt(S_eff / rho_eff)
            
            k_eff = omega / c_eff
            Z_eff = rho_eff * c_eff
            
            # Transfer Matrix
            cos_kd = np.cos(k_eff * d)
            sin_kd = np.sin(k_eff * d)
            
            if abs(Z_eff) < 1e-15:
                t21 = 1e15 + 0j
            else:
                t21 = 1j * sin_kd / Z_eff
            
            T_i = np.array([
                [cos_kd, 1j * Z_eff * sin_kd],
                [t21, cos_kd]
            ], dtype=np.complex128)
            
            T_total = np.dot(T_total, T_i)
        
        # Calculate Reflection
        T11, T21 = T_total[0,0], T_total[1,0]
        
        if abs(T21) < 1e-15:
            R = 1.0 + 0j
        else:
            Z_in = T11 / T21
            if abs(Z_in) > 1e15:
                R = 1.0 + 0j
            else:
                R = (Z_in - Z_w) / (Z_in + Z_w)
        
        # Absorption Coefficient
        alpha = 1.0 - abs(R)**2
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        else:
            alpha = alpha.real
        
        alpha_sum += alpha
    
    return alpha_sum / 1000.0

def process_batch(batch_data):
    """Process a batch of samples."""
    results = []
    for row in batch_data:
        d_vals = row[0:10] / 1000.0  # Convert mm to m
        m_vals = row[10:16] / 1000.0
        rho_r = row[16]
        eta = row[17]
        E_r = row[18]
        nu = row[19]
        W = 2.0  # Already in meters
        
        avg_alpha = calculate_single_absorption_fast(d_vals, m_vals, rho_r, eta, E_r, nu, W)
        results.append(avg_alpha)
    
    return results

def generate_dataset(num_samples=100000):
    """
    Generates a dataset using optimized parallel processing.
    """
    print(f"Generating {num_samples} samples using Latin Hypercube Sampling (LHS)...")
    
    # --- 1. Define Parameter Ranges ---
    bounds = {
        'd': (1.0, 20.0),
        'm': (20.0, 1980.0),
        'rho': (1000.0, 1500.0),
        'eta': (0.1, 0.8),
        'E': (1e7, 1e8),
        'nu': (0.4, 0.49)
    }
    
    # --- 2. Latin Hypercube Sampling ---
    num_params = 20
    sampler = qmc.LatinHypercube(d=num_params)
    sample_norm = sampler.random(n=num_samples)
    
    # --- 3. Scale Samples to Physical Ranges ---
    physical_samples = np.zeros_like(sample_norm)
    
    idx_d = slice(0, 10)
    idx_m = slice(10, 16)
    idx_rho = 16
    idx_eta = 17
    idx_E = 18
    idx_nu = 19
    
    lower, upper = bounds['d']
    physical_samples[:, idx_d] = lower + (upper - lower) * sample_norm[:, idx_d]
    
    lower, upper = bounds['m']
    physical_samples[:, idx_m] = lower + (upper - lower) * sample_norm[:, idx_m]
    
    for idx, key in zip([idx_rho, idx_eta, idx_E, idx_nu], ['rho', 'eta', 'E', 'nu']):
        lower, upper = bounds[key]
        physical_samples[:, idx] = lower + (upper - lower) * sample_norm[:, idx]
    
    # --- 4. Parallel Processing ---
    num_cores = mp.cpu_count()
    batch_size = max(1, num_samples // (num_cores * 4))
    
    print(f"Using {num_cores} CPU cores for parallel processing...")
    print(f"Batch size: {batch_size}")
    
    # Split data into batches
    batches = [physical_samples[i:i+batch_size] for i in range(0, num_samples, batch_size)]
    
    # Process batches in parallel
    with mp.Pool(processes=num_cores) as pool:
        results = []
        for i, batch_result in enumerate(pool.imap(process_batch, batches)):
            results.extend(batch_result)
            print(f"Processed batch {i+1}/{len(batches)} ({len(results)}/{num_samples} samples)")
    
    avg_absorptions = results
    
    # --- 5. Format Output DataFrame ---
    hollow_layer_ids = [2, 3, 5, 6, 8, 9]
    col_names = [f'd{j+1}' for j in range(10)] + \
                [f'm{j}' for j in hollow_layer_ids] + \
                ['rho', 'eta', 'E', 'nu']
    
    df = pd.DataFrame(physical_samples, columns=col_names)
    df['Average_Absorption'] = avg_absorptions
    
    return df

# --- Main Execution ---
if __name__ == "__main__":
    df = generate_dataset(num_samples=100000)
    
    print("\n--- Data Generation Complete ---")
    print("Method used: Latin Hypercube Sampling (LHS)")
    print(f"Columns generated: {len(df.columns)}")
    print("Columns d1..d10, m2..m9, rho, eta, E, nu are the parameters.")
    print("Column 'Average_Absorption' is the target value.")
    print("\nFirst 5 rows of generated data:")
    print(df.head())
    
    filename = "lhs_data.csv"
    df.to_csv(filename, index=False)
    print(f"\nData saved to '{filename}'")