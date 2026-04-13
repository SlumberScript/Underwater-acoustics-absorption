"""
Main runner: end-to-end FEniCSx simulation of the underwater acoustic coating.

Steps:
  1. Load parameters (base case or custom)
  2. Generate mesh with gmsh
  3. Solve Helmholtz equation at each frequency
  4. Save and plot results

Usage (from WSL Ubuntu terminal):
  cd /mnt/d/Research/Under-water-metamaterial/Underwater-acoustics-absorption/simulation_code

  # Quick test (every 50 Hz → ~20 frequency points):
  python3 run_simulation.py --quick

  # Full run (every 1 Hz → 1000 points):
  python3 run_simulation.py --full

  # Use Case 1 or Case 2 predictions:
  python3 run_simulation.py --case 1
  python3 run_simulation.py --case 2

  # Skip mesh generation (if mesh already exists):
  python3 run_simulation.py --skip-mesh
"""

import argparse
import os
import sys
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="FEniCSx FEM simulation of underwater acoustic coating"
    )
    parser.add_argument("--case", type=int, default=0,
                        help="Parameter set: 0=base, 1=case1, 2=case2")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: every 50 Hz (20 points)")
    parser.add_argument("--full", action="store_true",
                        help="Full run: every 1 Hz (1000 points)")
    parser.add_argument("--skip-mesh", action="store_true",
                        help="Skip mesh generation (use existing mesh)")
    parser.add_argument("--mesh-file", default="coating_mesh.msh",
                        help="Mesh filename")
    parser.add_argument("--output", default="fem_absorption_results.csv",
                        help="Output CSV filename")
    args = parser.parse_args()

    # ─── 1. Load parameters ───
    from config import (get_base_case_params, get_case1_params,
                        get_case2_params, compute_derived, FREQUENCIES)

    if args.case == 1:
        raw_params = get_case1_params()
        case_name = "Case 1"
    elif args.case == 2:
        raw_params = get_case2_params()
        case_name = "Case 2"
    else:
        raw_params = get_base_case_params()
        case_name = "Base Case"

    params = compute_derived(raw_params)
    print(f"\n{'='*60}")
    print(f"  Underwater Acoustic Coating FEM Simulation")
    print(f"  Parameter set: {case_name}")
    print(f"{'='*60}")
    print(f"  Total coating thickness: {params['total_thickness']*1000:.2f} mm")
    print(f"  Unit cell radius: {params['R_outer']*1000:.1f} mm")
    print(f"  Rubber density: {params['rho']:.1f} kg/m³")
    print(f"  Loss factor: {raw_params['eta']:.4f}")
    print(f"{'='*60}\n")

    # ─── 2. Generate mesh ───
    if not args.skip_mesh:
        print("Step 1: Generating mesh ...")
        t0 = time.time()
        from mesh_generation import create_mesh
        mesh_result = create_mesh(params, mesh_file=args.mesh_file)
        print(f"  Mesh generated in {time.time() - t0:.1f}s\n")
    else:
        if not os.path.exists(args.mesh_file):
            print(f"ERROR: Mesh file not found: {args.mesh_file}")
            print("Run without --skip-mesh first.")
            sys.exit(1)
        print(f"Step 1: Using existing mesh: {args.mesh_file}\n")

    # ─── 3. Determine frequencies ───
    if args.quick:
        freqs = np.arange(1, 1001, 50)
        print(f"Step 2: Quick mode — {len(freqs)} frequency points (every 50 Hz)")
    elif args.full:
        freqs = FREQUENCIES
        print(f"Step 2: Full mode — {len(freqs)} frequency points (every 1 Hz)")
    else:
        freqs = np.arange(1, 1001, 10)
        print(f"Step 2: Default mode — {len(freqs)} frequency points (every 10 Hz)")

    # ─── 4. Solve ───
    print("\nStep 3: Solving (this may take a while) ...")
    t0 = time.time()
    from fem_solver import solve_absorption
    freq_out, alpha_out = solve_absorption(
        params, frequencies=freqs, msh_file=args.mesh_file, verbose=True
    )
    solve_time = time.time() - t0
    print(f"\n  Solved in {solve_time:.1f}s "
          f"({solve_time/len(freqs):.2f}s per frequency point)")

    # ─── 5. Save results ───
    np.savetxt(args.output, np.column_stack([freq_out, alpha_out]),
               delimiter=",", header="frequency_Hz,absorption_coefficient",
               comments="")
    print(f"\nStep 4: Results saved to {args.output}")
    print(f"  Average absorption coefficient: {np.mean(alpha_out):.6f}")

    # ─── 6. Plot comparison with TMM ───
    print("\nStep 5: Generating comparison plot ...")
    from postprocess import compute_tmm_baseline, plot_fem_vs_tmm, print_summary

    freq_tmm, alpha_tmm = compute_tmm_baseline(raw_params)
    print_summary(freq_out, alpha_out, freq_tmm, alpha_tmm)

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_fem_vs_tmm(
        freq_out, alpha_out, freq_tmm, alpha_tmm,
        title=f"FEM vs TMM — {case_name}",
        save_path=os.path.join(fig_dir, f"fem_vs_tmm_{case_name.lower().replace(' ', '_')}.png")
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
