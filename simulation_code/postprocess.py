"""
Post-processing and plotting for FEM absorption results.
Compares FEM results with the TMM (Transfer Matrix Method) baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import csv
import os
import sys

# Add parent directory to path so we can import the TMM model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_fem_results(csv_file="fem_absorption_results.csv"):
    """Load FEM results from CSV."""
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def compute_tmm_baseline(params_raw):
    """Compute TMM absorption using the existing Theoretical_model.py."""
    from src.Theoretical_model import calculate_acoustic_properties

    # Convert to the format expected by Theoretical_model.py
    tmm_params = {
        "rho": params_raw["rho"],
        "eta": params_raw["eta"],
        "E": params_raw["E"],
        "nu": params_raw["nu"],
        "W": params_raw["W"],
        "d": params_raw["d"],
        "m": params_raw["m"],
    }
    freq, alpha, R, Z_in = calculate_acoustic_properties(tmm_params)
    return freq, alpha


def plot_fem_vs_tmm(freq_fem, alpha_fem, freq_tmm, alpha_tmm,
                    title="FEM vs TMM — Sound Absorption Coefficient",
                    save_path=None):
    """
    Plot FEM and TMM absorption curves side by side.
    Replicates Fig. 2b from the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(freq_tmm, alpha_tmm, "k-", linewidth=2.0,
            label=f"TMM (avg = {np.mean(alpha_tmm):.4f})")
    ax.plot(freq_fem, alpha_fem, "r--", linewidth=1.5,
            label=f"FEM / FEniCSx (avg = {np.mean(alpha_fem):.4f})")

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Sound Absorption Coefficient", fontsize=12)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_absorption_only(freq, alpha, title="Absorption Coefficient",
                         save_path=None):
    """Simple single-curve absorption plot."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(freq, alpha, "b-", linewidth=2.0,
            label=f"avg = {np.mean(alpha):.4f}")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Sound Absorption Coefficient", fontsize=12)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def print_summary(freq_fem, alpha_fem, freq_tmm=None, alpha_tmm=None):
    """Print a summary of results."""
    print("=" * 60)
    print("FEM Results Summary")
    print("=" * 60)
    print(f"  Frequency range: {freq_fem[0]:.0f} – {freq_fem[-1]:.0f} Hz")
    print(f"  Number of points: {len(freq_fem)}")
    print(f"  Average absorption: {np.mean(alpha_fem):.6f}")
    print(f"  Max absorption: {np.max(alpha_fem):.6f} at {freq_fem[np.argmax(alpha_fem)]:.0f} Hz")
    print(f"  Min absorption: {np.min(alpha_fem):.6f} at {freq_fem[np.argmin(alpha_fem)]:.0f} Hz")

    if alpha_tmm is not None:
        # Interpolate TMM to FEM frequencies for comparison
        alpha_tmm_interp = np.interp(freq_fem, freq_tmm, alpha_tmm)
        diff = np.abs(alpha_fem - alpha_tmm_interp)
        print(f"\n  TMM Average absorption: {np.mean(alpha_tmm):.6f}")
        print(f"  Mean |FEM - TMM|: {np.mean(diff):.6f}")
        print(f"  Max  |FEM - TMM|: {np.max(diff):.6f} at {freq_fem[np.argmax(diff)]:.0f} Hz")
        rel_err = np.abs(np.mean(alpha_fem) - np.mean(alpha_tmm)) / np.mean(alpha_tmm) * 100
        print(f"  Relative error (avg): {rel_err:.3f}%")
    print("=" * 60)


if __name__ == "__main__":
    from config import get_base_case_params

    # Load FEM results
    fem_file = "fem_absorption_results.csv"
    if not os.path.exists(fem_file):
        print(f"FEM results file not found: {fem_file}")
        print("Run the solver first: python run_simulation.py")
        sys.exit(1)

    freq_fem, alpha_fem = load_fem_results(fem_file)

    # Compute TMM baseline
    params = get_base_case_params()
    freq_tmm, alpha_tmm = compute_tmm_baseline(params)

    # Print summary
    print_summary(freq_fem, alpha_fem, freq_tmm, alpha_tmm)

    # Plot comparison
    save_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(save_dir, exist_ok=True)
    plot_fem_vs_tmm(
        freq_fem, alpha_fem, freq_tmm, alpha_tmm,
        save_path=os.path.join(save_dir, "fem_vs_tmm_comparison.png")
    )
