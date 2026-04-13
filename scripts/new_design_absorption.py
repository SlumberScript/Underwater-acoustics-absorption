"""
New Design Absorption Spectrum

Computes and plots the per-frequency absorption coefficient for the
new optimized design parameters, compared against the base case.

Parameters:
    d1–d10  : layer thicknesses (mm)
    m2,m3,m5,m6,m8,m9 : hollow-layer widths (mm)
    ρ   = 1268.43  kg/m³
    η   = 0.7589
    E   = 8.490e7 Pa  (complex modulus: E*(1+ηi))
    ν   = 0.4764

Usage:
    python scripts/new_design_absorption.py
    python scripts/new_design_absorption.py --save          # saves to figures/
    python scripts/new_design_absorption.py --save --csv    # also saves CSV
"""

import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from Theoretical_model import calculate_acoustic_properties, get_base_case_parameters


# ── New optimised design parameters ──────────────────────────────────────────
def get_new_design_parameters():
    """
    New optimised design parameters.
    E is the real part only — calculate_acoustic_properties applies E*(1+ηi).
    """
    rho = 1268.43154430389
    eta = 0.758901304006577
    E_real = 8.49030417203903e7
    nu = 0.476384522914887

    W = 2000.0
    d_layers = [
        10.7398927211761,   # d1
         9.19723930954933,  # d2
         9.03188413381577,  # d3
        10.3210442066193,   # d4
         9.02042958140373,  # d5
         9.03090962767601,  # d6
        10.1851529777050,   # d7
         9.20814347267151,  # d8
         9.10563176870346,  # d9
        10.3094107210636,   # d10
    ]
    m_values = {
        2: 353.101807236671,
        3: 354.557039141655,
        5: 371.865028738976,
        6: 382.696526646614,
        8: 404.766833186150,
        9: 400.110272169113,
    }

    return {
        'rho': rho, 'eta': eta, 'E': E_real, 'nu': nu,
        'W': W, 'd': d_layers, 'm': m_values,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_comparison(freq_base, alpha_base, freq_new, alpha_new, save_path=None):
    """Per-frequency absorption: New Design vs Base Case."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(freq_base, alpha_base, color='black', linestyle='--', linewidth=1.5,
            label=f'Base Case (avg = {np.mean(alpha_base):.4f})')
    ax.plot(freq_new, alpha_new, color='red', linestyle='-', linewidth=2.0,
            label=f'New Design (avg = {np.mean(alpha_new):.4f})')

    ax.set_xlabel('Frequency / Hz', fontsize=13)
    ax.set_ylabel('Sound Absorption Coefficient', fontsize=13)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    ax.set_title('New Design vs Base Case — Absorption Coefficient', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_full_analysis(freq, alpha, R, Z_in, save_path=None):
    """Full analysis plot (absorption, reflection, impedance) for the new design."""
    fig, ax1 = plt.subplots(figsize=(10, 7))

    line1, = ax1.plot(freq, alpha, 'k-',  lw=2.0, label='Absorption coeff.')
    line2, = ax1.plot(freq, np.abs(R), 'k--', lw=2.0, label='Reflection coeff.')

    ax1.set_xlabel('Frequency / Hz', fontsize=13)
    ax1.set_ylabel('Coefficient', fontsize=13)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.tick_params(direction='in', which='both', top=True, labelsize=11)

    ax2 = ax1.twinx()
    line3, = ax2.semilogy(freq, np.real(Z_in),        'r-',  lw=1.5, label='Re(Z_in)')
    line4, = ax2.semilogy(freq, np.abs(np.imag(Z_in)), 'r--', lw=1.5, label='|Im(Z_in)|')
    ax2.axhline(y=1.5e6, color='blueviolet', ls='-.', lw=2.0, label='Z_water')

    ax2.set_ylabel('Surface impedance / Pa·s·m⁻²', fontsize=13, color='red')
    ax2.set_ylim(1e2, 1e8)
    ax2.tick_params(axis='y', colors='red', direction='in', which='both')

    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', frameon=True, fontsize=10)

    ax1.set_title('New Design — Full Acoustic Analysis', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='New Design absorption spectrum.')
    parser.add_argument('--save', action='store_true',
                        help='Save figures to figures/ instead of displaying')
    parser.add_argument('--csv', action='store_true',
                        help='Also export per-frequency results to CSV')
    args = parser.parse_args()

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ── Base case ──────────────────────────────────────────────────────────
    print("Calculating base case...")
    base_params = get_base_case_parameters()
    freq_b, alpha_b, R_b, Z_in_b = calculate_acoustic_properties(base_params)
    print(f"  Base Case  avg absorption: {np.mean(alpha_b):.6f}")

    # ── New design ─────────────────────────────────────────────────────────
    print("Calculating new design...")
    new_params = get_new_design_parameters()
    freq_n, alpha_n, R_n, Z_in_n = calculate_acoustic_properties(new_params)
    print(f"  New Design avg absorption: {np.mean(alpha_n):.6f}")

    # ── Per-frequency summary (first & last 10) ───────────────────────────
    print("\n{:<10s}  {:>12s}  {:>12s}".format("Freq (Hz)", "Base α", "New α"))
    print("-" * 38)
    show = list(range(0, 10)) + list(range(990, 1000))
    for i in show:
        if i == 990:
            print("  ...")
        print(f"{freq_n[i]:>8.0f}    {alpha_b[i]:>10.6f}    {alpha_n[i]:>10.6f}")

    # ── Plots ──────────────────────────────────────────────────────────────
    save1 = os.path.join(fig_dir, 'new_design_vs_base.png') if args.save else None
    save2 = os.path.join(fig_dir, 'new_design_full_analysis.png') if args.save else None

    print("\nPlotting comparison...")
    plot_comparison(freq_b, alpha_b, freq_n, alpha_n, save_path=save1)

    print("Plotting full analysis...")
    plot_full_analysis(freq_n, alpha_n, R_n, Z_in_n, save_path=save2)

    # ── CSV export ─────────────────────────────────────────────────────────
    if args.csv:
        csv_path = os.path.join(fig_dir, '..', 'new_design_absorption_results.csv')
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_Hz', 'Alpha_Base', 'Alpha_NewDesign',
                             'R_NewDesign_real', 'R_NewDesign_imag'])
            for i in range(len(freq_n)):
                writer.writerow([
                    int(freq_n[i]),
                    f"{alpha_b[i]:.8f}",
                    f"{alpha_n[i]:.8f}",
                    f"{R_n[i].real:.8f}",
                    f"{R_n[i].imag:.8f}",
                ])
        print(f"CSV saved → {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
