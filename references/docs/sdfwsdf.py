# ============================================================
# SPAGHETTI + HEATMAP INSET (Times New Roman, publication quality)
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator

# Validate required arrays from previous sweep cell
if 'best_spectra' not in globals() or 'target_values' not in globals():
    raise RuntimeError(
        "Required arrays not found. Please run the full sweep cell first "
        "(the one that creates 'best_spectra' and 'target_values')."
    )

if 'frequencies' not in globals() or len(frequencies) != best_spectra.shape[1]:
    frequencies = np.arange(1, best_spectra.shape[1] + 1)

# Figure style for publication
plot_rc = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.titleweight': 'bold',
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
}

with plt.rc_context(plot_rc):
    fig, ax = plt.subplots(figsize=(16, 10))

    # Main panel: full sweep spaghetti plot
    norm_target = Normalize(vmin=target_values.min(), vmax=target_values.max())
    cmap_target = cm.viridis

    for i in range(len(target_values)):
        ax.plot(
            frequencies,
            best_spectra[i],
            color=cmap_target(norm_target(target_values[i])),
            linewidth=0.45,
            alpha=0.55
        )

    sm = cm.ScalarMappable(norm=norm_target, cmap=cmap_target)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.025)
    cbar.set_label('Target Avg Absorption', fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Absorption Coefficient')
    ax.set_xlim(1, 1000)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction='in', which='both', top=True, right=True)
    ax.grid(True, alpha=0.16)
    ax.set_title('Design-Space Sweep with Heatmap Inset\n901 targets, Best-of-20 GPU TMM')

    # Inset: top-left in sparse region (no second colorbar)
    # Rows are target levels, columns are frequency bins
    heatmap_data = best_spectra[::-1] if target_values[0] > target_values[-1] else best_spectra
    y_min = float(np.min(target_values))
    y_max = float(np.max(target_values))

    axins = ax.inset_axes([0.055, 0.57, 0.41, 0.37])  # [x0, y0, w, h] in axes fraction
    axins.imshow(
        heatmap_data,
        aspect='auto',
        origin='lower',
        extent=[1, 1000, y_min, y_max],
        cmap='inferno',
        vmin=0,
        vmax=1,
        interpolation='bilinear'
    )

    axins.set_title('Sweep Heatmap (Inset)', fontsize=12, pad=3)
    axins.set_xlabel('f (Hz)', fontsize=10)
    axins.set_ylabel('Target', fontsize=10)
    axins.tick_params(axis='both', labelsize=9, direction='in')
    axins.set_xticks([1, 250, 500, 750, 1000])
    axins.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    # Thin border around inset
    for spine in axins.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color('black')

    # Subtle cue for inset colormap meaning (without second colorbar)
    axins.text(
        0.98, 0.02,
        'Color = α',
        transform=axins.transAxes,
        ha='right', va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.tight_layout()
    os.makedirs('../figures', exist_ok=True)
    out_png = '../figures/sweep_spaghetti_with_heatmap_inset_TNR.png'
    out_pdf = '../figures/sweep_spaghetti_with_heatmap_inset_TNR.pdf'

    # High-quality export
    plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved high-quality figure:\n  {out_png}\n  {out_pdf}")
    plt.show()