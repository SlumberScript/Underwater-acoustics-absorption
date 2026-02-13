import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings

# Suppress runtime warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_base_case_parameters():
    """
    Returns parameters for the 'Base Case' (Tables 1 & 2 in the article).
    """
    # Material Parameters (Table 2)
    rho = 1130.0
    eta = 0.4
    E_real = 5e7
    nu = 0.44
    
    # Geometric Parameters (Table 1)
    W = 2000.0
    d_layers = [10.0] * 10
    # Hollow layers (2,3,5,6,8,9) have m = 1000mm
    m_values = {2: 1000.0, 3: 1000.0, 5: 1000.0, 6: 1000.0, 8: 1000.0, 9: 1000.0}

    return {'rho': rho, 'eta': eta, 'E': E_real, 'nu': nu, 'W': W, 'd': d_layers, 'm': m_values}

def calculate_acoustic_properties(params):
    """
    Calculates Alpha, R, and Zin using TMM with correct physics.
    """
    rho_rubber = params['rho']
    eta = params['eta']
    # Complex Modulus
    E_rubber = complex(params['E'] * (1 + 1j * eta))
    nu = params['nu']
    W = params['W'] / 1000.0
    d_list = [x / 1000.0 for x in params['d']]
    m_dict = {k: v / 1000.0 for k, v in params['m'].items()}
    
    rho_water = 1000.0
    c_water = 1500.0
    Z_water = rho_water * c_water
    
    frequencies = np.arange(1, 1001, 1)
    
    alpha_arr = []
    R_arr = []
    Z_in_arr = []
    
    # Lame Constants
    lam = (E_rubber * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_rubber / (2 * (1 + nu))
    
    for f in frequencies:
        omega = 2 * np.pi * f
        T_total = np.eye(2, dtype=complex)
        
        for i in range(10):
            layer_num = i + 1
            d_i = d_list[i]
            
            # Perforation Rate
            if layer_num in m_dict:
                m_i = m_dict[layer_num]
                epsilon = m_i / W
            else:
                epsilon = 0.0 
            
            # Equivalent Medium Theory
            rho_eff = rho_rubber * (1 - epsilon**2)
            
            # Modulus with Dimensional Fix (Eq 9a corrected)
            numerator = (mu * (lam + 2*mu) * (epsilon**2 + 1)) + (2 * (epsilon**2) * lam)
            denominator = ((lam + mu) * epsilon**2) + (mu)
            
            if abs(denominator) < 1e-15: S_eff = numerator * 1e15
            else: S_eff = numerator / denominator
            
            c_eff = np.sqrt(complex(S_eff / rho_eff))
            k_eff = omega / c_eff
            Z_eff = rho_eff * c_eff
            
            # Transfer Matrix
            cos_kd = np.cos(k_eff * d_i)
            sin_kd = np.sin(k_eff * d_i)
            
            if abs(Z_eff) < 1e-15: term_21 = 1e15
            else: term_21 = 1j * sin_kd / Z_eff

            T_i = np.array([[cos_kd, 1j * Z_eff * sin_kd], [term_21, cos_kd]], dtype=complex)
            T_total = np.dot(T_total, T_i)
        
        # Calculate Input Impedance (Z_in)
        # Rigid Backing Condition (V_out = 0) implies Z_in = T11 / T21
        T11 = T_total[0, 0]
        T21 = T_total[1, 0]
        
        if np.abs(T21) < 1e-15:
            Z_in = 1e15 + 0j
            R = 1.0 + 0j
        else:
            Z_in = T11 / T21
            if np.abs(Z_in) > 1e15: R = 1.0 + 0j
            else: R = (Z_in - Z_water) / (Z_in + Z_water)
        
        alpha = 1 - np.abs(R)**2
        alpha = max(0.0, min(1.0, alpha.real))
        
        alpha_arr.append(alpha)
        R_arr.append(R)
        Z_in_arr.append(Z_in)
        
    return frequencies, np.array(alpha_arr), np.array(R_arr), np.array(Z_in_arr)

def plot_figure_3_reproduction(freq, alpha, R, Z_in):
    """
    Plots the results matching Figure 3 with non-intersecting legend.
    """
    fig, ax1 = plt.subplots(figsize=(9, 7))
    
    # --- LEFT AXIS: Coefficients ---
    line1, = ax1.plot(freq, alpha, color='black', linestyle='-', linewidth=2.0, label='Sound absorption coefficient')
    line2, = ax1.plot(freq, np.abs(R), color='black', linestyle='--', linewidth=2.0, label='Sound reflection coefficient')
    
    ax1.set_xlabel('Frequency/Hz', fontsize=12)
    ax1.set_ylabel('Coefficient', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, 1000)
    
    # Ticks for left axis
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.tick_params(direction='in', which='both', top=True)

    # --- RIGHT AXIS: Impedance ---
    ax2 = ax1.twinx()
    
    # 3. Real Impedance (Solid Red)
    line3, = ax2.semilogy(freq, np.real(Z_in), color='red', linestyle='-', linewidth=1.5, label='Real(Surface impedance)')
    
    # 4. Imaginary Impedance (Dashed Red)
    # Using Magnitude to handle negative reactance on log plot
    line4, = ax2.semilogy(freq, np.abs(np.imag(Z_in)), color='red', linestyle='--', linewidth=1.5, label='Imag(Surface impedance)')
    
    # 5. Water Impedance (Purple Dash-Dot)
    ax2.axhline(y=1.5e6, color='blueviolet', linestyle='-.', linewidth=2.0, label='Water Impedance')

    ax2.set_ylabel('Surface impedance/Pa·s/m²', fontsize=12, color='red')
    ax2.set_ylim(1e2, 1e8)
    ax2.tick_params(axis='y', colors='red', direction='in', which='both')
    
    # Legend - Positioned at Center Right to avoid curve intersection
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def get_custom_parameters():
    """
    Returns custom design parameters for comparison.
    EDIT THESE VALUES to test any parameter set.
    """
    rho = 1254.90951538086
    eta = 0.700865012407303
    E_real = 7.96552193164825e7   # E_real only — code computes E_c = E*(1+ηi) internally
    nu = 0.472864060401917
    
    W = 2000.0
    d_layers = [10.4116438329220, 9.59269911050797, 9.48593208193779, 10.2778358161449,
                9.56153133511543, 9.52644741535187, 10.2548802793026, 9.57734596729279,
                9.54049196839333, 10.3517957925797]
    m_values = {2: 422.048478722572, 3: 415.016191601753, 5: 421.283624768257,
                6: 428.282207846642, 8: 437.154577970505, 9: 433.320407271385}

    return {'rho': rho, 'eta': eta, 'E': E_real, 'nu': nu, 'W': W, 'd': d_layers, 'm': m_values}

def plot_custom_absorption(freq_base, alpha_base, freq_custom, alpha_custom):
    """
    Plots custom parameters absorption vs base case absorption.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot(freq_base, alpha_base, color='black', linestyle='--', linewidth=1.5,
            label=f'Base Case (avg={np.mean(alpha_base):.4f})')
    ax.plot(freq_custom, alpha_custom, color='blue', linestyle='-', linewidth=2.0,
            label=f'Custom Parameters (avg={np.mean(alpha_custom):.4f})')
    
    ax.set_xlabel('Frequency/Hz', fontsize=12)

    ax.set_ylabel('Sound Absorption Coefficient', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1000)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    ax.set_title('Custom Design vs Base Case — Absorption Coefficient', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    # --- Custom Parameters Plot ---
    custom_params = get_custom_parameters()
    print("Calculating custom parameters absorption...")
    freq_c, alpha_c, _, _ = calculate_acoustic_properties(custom_params)
    avg_alpha_c = np.mean(alpha_c)
    print(f"Custom Average Absorption Coefficient: {avg_alpha_c:.6f}")

    # --- Base Case ---
    params = get_base_case_parameters()
    print("Calculating base case absorption...")
    freq, alpha, R, Z_in = calculate_acoustic_properties(params)
    avg_alpha = np.mean(alpha)
    print(f"Base Case Average Absorption Coefficient: {avg_alpha:.6f}")
    
    print("-" * 60)
    print("Plotting Custom vs Base Case comparison...")
    plot_custom_absorption(freq, alpha, freq_c, alpha_c)

    # # Print Calculated Complex Surface Impedance
    # print("\nCalculated Complex Surface Impedance (Z_in) Samples:")
    # print("First 5 values (1-5 Hz):")
    # print(Z_in[:5])
    # print("\nLast 5 values (996-1000 Hz):")
    # print(Z_in[-5:])
    # print("-" * 60)
    
    print("\nPlotting Figure 3 (Base Case full analysis)...")
    plot_figure_3_reproduction(freq, alpha, R, Z_in)