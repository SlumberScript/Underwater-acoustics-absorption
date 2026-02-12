import numpy as np
import matplotlib.pyplot as plt
from Theoretical_model import calculate_acoustic_properties, plot_figure_3_reproduction

def get_case1_parameters():
    """
    Returns parameters for Case 1 prediction results.
    """
    # Your predicted parameters
    d_layers = [
        10.7398927211761,   # d1
        9.19723930954933,   # d2
        9.03188413381577,   # d3
        10.3210442066193,   # d4
        9.02042958140373,   # d5
        9.03090962767601,   # d6
        10.1851529777050,   # d7
        9.20814347267151,   # d8
        9.10563176870346,   # d9
        10.3094107210636    # d10
    ]
    
    # Hollow layers (2,3,5,6,8,9) with their m values in mm
    m_values = {
        2: 353.101807236671,
        3: 354.557039141655,
        5: 371.865028738976,
        6: 382.696526646614,
        8: 404.766833186150,
        9: 400.110272169113
    }
    
    # Material properties
    rho = 1268.43154430389        # kg/m³
    eta = 0.758901304006577       # loss factor
    E_real = 8.49030417203903e7   # Pa (the real part, imaginary part is handled by eta)
    nu = 0.476384522914887        # Poisson's ratio
    
    # Width (assuming same as base case, typically constant)
    W = 2000.0  # mm
    
    return {
        'rho': rho, 
        'eta': eta, 
        'E': E_real, 
        'nu': nu, 
        'W': W, 
        'd': d_layers, 
        'm': m_values
    }

def get_case2_parameters():
    """
    Returns parameters for Case 2 prediction results.
    """
    # Your predicted parameters
    d_layers = [
        10.4116438329220,   # d1
        9.59269911050797,   # d2
        9.48593208193779,   # d3
        10.2778358161449,   # d4
        9.56153133511543,   # d5
        9.52644741535187,   # d6
        10.2548802793026,   # d7
        9.57734596729279,   # d8
        9.54049196839333,   # d9
        10.3517957925797    # d10
    ]
    
    # Hollow layers (2,3,5,6,8,9) with their m values in mm
    m_values = {
        2: 422.048478722572,
        3: 415.016191601753,
        5: 421.283624768257,
        6: 428.282207846642,
        8: 437.154577970505,
        9: 433.320407271385
    }
    
    # Material properties
    rho = 1254.90951538086        # kg/m³
    eta = 0.700865012407303       # loss factor
    E_real = 7.96552193164825e7   # Pa (the real part, imaginary part is handled by eta)
    nu = 0.472864060401917        # Poisson's ratio
    
    # Width (assuming same as base case, typically constant)
    W = 2000.0  # mm
    
    return {
        'rho': rho, 
        'eta': eta, 
        'E': E_real, 
        'nu': nu, 
        'W': W, 
        'd': d_layers, 
        'm': m_values
    }

if __name__ == "__main__":
    print("="*70)
    print("CASE 2 - Absorption Curve Calculation")
    print("="*70)
    
    # Get your parameters
    params = get_case2_parameters()
    
    # Display parameters
    print("\nInput Parameters:")
    print("-" * 70)
    print("Thickness layers (d1-d10) [mm]:")
    for i, d in enumerate(params['d'], 1):
        print(f"  d{i} = {d:.6f}")
    
    print("\nHollow layer masses (m2,3,5,6,8,9) [mm]:")
    for i, m in params['m'].items():
        print(f"  m{i} = {m:.6f}")
    
    print(f"\nMaterial Properties:")
    print(f"  ρ (density) = {params['rho']:.6f} kg/m³")
    print(f"  η (loss factor) = {params['eta']:.6f}")
    print(f"  E (Young's modulus) = {params['E']:.6e} Pa (1+ηi)")
    print(f"  ν (Poisson's ratio) = {params['nu']:.6f}")
    print(f"  W (width) = {params['W']:.1f} mm")
    print("-" * 70)
    
    # Calculate acoustic properties
    print("\nCalculating absorption curve...")
    freq, alpha, R, Z_in = calculate_acoustic_properties(params)
    
    # Calculate statistics
    avg_alpha = np.mean(alpha)
    max_alpha = np.max(alpha)
    max_alpha_freq = freq[np.argmax(alpha)]
    
    print("\nResults:")
    print("-" * 70)
    print(f"Average Absorption Coefficient: {avg_alpha:.6f}")
    print(f"Maximum Absorption Coefficient: {max_alpha:.6f} at {max_alpha_freq:.0f} Hz")
    
    # Find absorption peaks (values > 0.9)
    peak_indices = np.where(alpha > 0.9)[0]
    if len(peak_indices) > 0:
        print(f"\nHigh absorption regions (α > 0.9):")
        # Group consecutive frequencies
        freq_ranges = []
        start_idx = peak_indices[0]
        for i in range(1, len(peak_indices)):
            if peak_indices[i] != peak_indices[i-1] + 1:
                freq_ranges.append((freq[start_idx], freq[peak_indices[i-1]]))
                start_idx = peak_indices[i]
        freq_ranges.append((freq[start_idx], freq[peak_indices[-1]]))
        
        for f_start, f_end in freq_ranges:
            print(f"  {f_start:.0f} - {f_end:.0f} Hz")
    
    print("="*70)
    
    # Save results to file
    output_file = "case2_absorption_results.csv"
    np.savetxt(output_file, 
               np.column_stack([freq, alpha, np.abs(R), np.real(Z_in), np.imag(Z_in)]),
               delimiter=',',
               header='Frequency(Hz),Absorption_Coefficient,Reflection_Coefficient,Real(Z_in),Imag(Z_in)',
               comments='')
    print(f"\nResults saved to: {output_file}")
    
    # Plot the results
    print("\nGenerating plot...")
    plot_figure_3_reproduction(freq, alpha, R, Z_in)
