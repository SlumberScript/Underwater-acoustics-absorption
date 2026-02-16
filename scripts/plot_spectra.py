"""
Interactive Absorption Spectrum Plotter

Plot absorption spectra from the full-spectrum dataset by specifying
sample indices, random counts, or parameter filters.

Usage:
    # Plot 10 random samples
    python scripts/plot_spectra.py --random 10

    # Plot specific sample indices
    python scripts/plot_spectra.py --indices 0 100 5000 99999

    # Plot N random samples with a specific seed
    python scripts/plot_spectra.py --random 5 --seed 123

    # Plot samples with highest average absorption
    python scripts/plot_spectra.py --top 5

    # Plot samples with lowest average absorption
    python scripts/plot_spectra.py --bottom 5

    # Plot samples closest to a target average absorption
    python scripts/plot_spectra.py --target-avg 0.6 --count 5

    # Save to a custom path
    python scripts/plot_spectra.py --random 10 --output my_plot.png

    # Use a different data file
    python scripts/plot_spectra.py --random 5 --data data/lhs_data_full_spectrum.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
import sys

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "lhs_data_full_spectrum.npz"
)


def load_data(path):
    data = np.load(path)
    return data["params"], data["spectra"], data["frequencies"]


def select_indices(args, spectra):
    """Resolve which sample indices to plot based on CLI args."""
    N = spectra.shape[0]

    if args.indices is not None:
        indices = np.array(args.indices, dtype=int)
        bad = indices[(indices < 0) | (indices >= N)]
        if len(bad) > 0:
            print(f"Warning: indices {bad.tolist()} out of range [0, {N-1}], skipping.")
            indices = indices[(indices >= 0) & (indices < N)]
        return indices

    if args.top is not None:
        avg = np.mean(spectra, axis=1)
        return np.argsort(avg)[-args.top:][::-1]

    if args.bottom is not None:
        avg = np.mean(spectra, axis=1)
        return np.argsort(avg)[:args.bottom]

    if args.target_avg is not None:
        avg = np.mean(spectra, axis=1)
        count = args.count if args.count else 5
        dist = np.abs(avg - args.target_avg)
        return np.argsort(dist)[:count]

    # Default: random
    count = args.random if args.random else 10
    rng = np.random.default_rng(args.seed)
    return rng.choice(N, size=min(count, N), replace=False)


def plot_spectra(frequencies, spectra, indices, title=None, output=None):
    """Plot absorption spectra for the given indices."""
    n = len(indices)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n, 10)))
    if n > 10:
        colors = plt.cm.turbo(np.linspace(0.05, 0.95, n))

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, idx in enumerate(indices):
        alpha = spectra[idx]
        avg = np.mean(alpha)
        ax.plot(frequencies, alpha, color=colors[i % len(colors)],
                linewidth=1.5,
                label=f"Sample {idx:,} (avg={avg:.3f})")

    ax.set_xlabel("Frequency/Hz", fontsize=14)
    ax.set_ylabel("Sound absorption coefficient", fontsize=14)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=12)
    ax.grid(True, alpha=0.3)

    ncol = 1 if n <= 6 else 2
    ax.legend(loc="upper right", fontsize=9, frameon=True, ncol=ncol)

    if title:
        ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot absorption spectra from the full-spectrum dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--indices", type=int, nargs="+",
                       help="Specific sample indices to plot")
    group.add_argument("--random", type=int, default=None,
                       help="Number of random samples to plot (default: 10)")
    group.add_argument("--top", type=int,
                       help="Plot the N samples with highest avg absorption")
    group.add_argument("--bottom", type=int,
                       help="Plot the N samples with lowest avg absorption")
    group.add_argument("--target-avg", type=float,
                       help="Plot samples closest to this target avg absorption")

    parser.add_argument("--count", type=int, default=5,
                        help="Number of samples for --target-avg (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save plot to file instead of displaying")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="Path to .npz data file")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")

    args = parser.parse_args()

    # Load
    print(f"Loading: {args.data}", flush=True)
    params, spectra, frequencies = load_data(args.data)
    print(f"  {params.shape[0]:,} samples, {spectra.shape[1]} frequencies",
          flush=True)

    # Select
    indices = select_indices(args, spectra)
    print(f"  Plotting {len(indices)} sample(s): {indices.tolist()}", flush=True)

    # Title
    title = args.title
    if not title:
        if args.indices is not None:
            title = f"Absorption Spectra — {len(indices)} Selected Samples"
        elif args.top is not None:
            title = f"Top {args.top} Highest Average Absorption"
        elif args.bottom is not None:
            title = f"Bottom {args.bottom} Lowest Average Absorption"
        elif args.target_avg is not None:
            title = f"Samples Closest to Avg Absorption = {args.target_avg:.3f}"
        else:
            title = f"Absorption Spectra — {len(indices)} Random Samples"

    # Plot
    plot_spectra(frequencies, spectra, indices, title=title, output=args.output)


if __name__ == "__main__":
    main()
