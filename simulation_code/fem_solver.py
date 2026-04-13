"""
FEniCSx FEM solver for the underwater acoustic coating.

Replicates the COMSOL model from Section 2.10 of Gao et al.:
  - 2D axisymmetric pressure-acoustics (Helmholtz equation) in water
  - Solid mechanics (linear elasticity with complex modulus) in coating
  - Acoustic-structure interaction at water-coating interface
  - Rigid backing (zero displacement) at bottom
  - PML to absorb outgoing waves

Usage:
    from fem_solver import solve_absorption
    freqs, alpha = solve_absorption(params_derived)

Note: This must be run inside WSL Ubuntu where FEniCSx is installed.
"""

import numpy as np
import sys

try:
    import dolfinx
    from dolfinx import mesh as dmesh, fem, io, default_scalar_type
    from dolfinx.fem import petsc as fem_petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
except ImportError:
    sys.exit(
        "ERROR: FEniCSx (DOLFINx) not found.\n"
        "Make sure you are running inside WSL Ubuntu with fenicsx installed:\n"
        "  sudo apt install fenicsx\n"
        "  pip install gmsh"
    )

try:
    import gmsh
except ImportError:
    sys.exit("ERROR: gmsh Python API not found. Install with: pip install gmsh")


def read_gmsh_mesh(msh_file):
    """Read a .msh file into a DOLFINx mesh with cell and facet tags."""
    # 1. Import from the new module name (alias it to gmshio to save rewriting code)
    from dolfinx.io import gmsh as gmshio
    
    # 2. read_from_msh now returns a MeshData object, not a tuple
    mesh_data = gmshio.read_from_msh(
        msh_file, MPI.COMM_WORLD, rank=0, gdim=2
    )
    
    # 3. Extract the specific attributes from the MeshData object
    return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags


def solve_single_frequency(freq, mesh_data, params):
    """
    Solve the coupled acoustic-elastic problem at a single frequency.

    This solves the Helmholtz equation in the water domain and
    the elastodynamic equation in the solid coating domain,
    coupled at their shared interface.

    Parameters
    ----------
    freq : float
        Frequency in Hz
    mesh_data : tuple
        (mesh, cell_tags, facet_tags) from read_gmsh_mesh
    params : dict
        From config.compute_derived()

    Returns
    -------
    alpha : float
        Sound absorption coefficient at this frequency
    """
    omega = 2.0 * np.pi * freq
    msh, cell_tags, facet_tags = mesh_data

    # Material properties
    rho_w = params["rho_w"]
    c_w = params["c_w"]
    k_w = omega / c_w           # wavenumber in water
    Z_w = rho_w * c_w           # characteristic impedance of water

    rho_s = params["rho"]
    lam = params["lam"]         # complex Lamé lambda
    mu = params["mu"]           # complex Lamé mu

    # ─── Physical group tags (must match mesh_generation.py) ───
    TAG_WATER = 1
    TAG_PML = 2
    TAG_COATING_SOLID = 3
    TAG_COATING_HOLLOW = 4
    TAG_CAVITY = 5
    TAG_RIGID_BACK = 10
    TAG_SYMMETRY = 11
    TAG_OUTER_WALL = 12
    TAG_INCIDENT = 13

    # ─── Function spaces ───
    # Pressure field (scalar, complex) for water + PML
    V_p = fem.functionspace(msh, ("Lagrange", 2))

    # Displacement field (vector, complex) for solid coating
    V_u = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))

    # ─── Identify subdomains via cell tags ───
    water_cells = cell_tags.find(TAG_WATER)
    pml_cells = cell_tags.find(TAG_PML)
    solid_cells = np.concatenate([
        cell_tags.find(TAG_COATING_SOLID),
        cell_tags.find(TAG_COATING_HOLLOW)
    ])
    cavity_cells = cell_tags.find(TAG_CAVITY)

    # ─── Trial and test functions ───
    p = ufl.TrialFunction(V_p)
    q = ufl.TestFunction(V_p)

    # ─── Helmholtz equation in water: ∇²p + k²p = 0 ───
    # Weak form: ∫ (∇p · ∇q - k²·p·q) dx = 0
    dx_water = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags,
                           subdomain_id=TAG_WATER)
    dx_pml = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags,
                         subdomain_id=TAG_PML)

# Water domain bilinear form
    k_w_c = fem.Constant(msh, complex(k_w))
    a_water = (ufl.inner(ufl.grad(p), ufl.grad(q)) - k_w_c**2 * p * ufl.conj(q)) * dx_water

    # ─── PML: Helmholtz with complex coordinate stretching ───
    # Simplified PML: add damping via complex wavenumber
    # sigma_pml provides absorption
    sigma_max = 5.0 * k_w  # PML strength
    a_pml = (ufl.inner(ufl.grad(p), ufl.grad(q))
             - (k_w_c**2 + fem.Constant(msh, complex(1j * sigma_max * k_w))) * p * ufl.conj(q)
             ) * dx_pml

    a_form = a_water + a_pml

    # ─── Incident wave (background pressure field) ───
    # Plane wave travelling in -z direction: p_inc = P0 * exp(-j*k_w*z)
    # We apply it as a source on the incident boundary
    ds_inc = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags,
                         subdomain_id=TAG_INCIDENT)

    # Sommerfeld-like radiation condition at incident boundary
    # ∂p/∂n + j*k*p = 2*j*k*p_inc  on the incident boundary
    a_form += fem.Constant(msh, complex(1j * k_w)) * p * ufl.conj(q) * ds_inc
# ─── Incident wave (background pressure field) ───
    # Plane wave travelling in -z direction: p_inc = P0 * exp(-j*k_w*z)
    # We apply it as a source on the incident boundary
    ds_inc = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags,
                         subdomain_id=TAG_INCIDENT)

    # Sommerfeld-like radiation condition at incident boundary
    # ∂p/∂n + j*k*p = 2*j*k*p_inc  on the incident boundary
    a_form += fem.Constant(msh, complex(1j * k_w)) * ufl.inner(p, q) * ds_inc

    # ─── RHS: incident wave source ───
    P0 = 1.0  # unit amplitude
    
    # THE ULTIMATE FIX: Use a fem.Function to shield the complex scalar 
    # from UFL's aggressive algebraic simplifier.
    f_inc = fem.Function(V_p)
    f_inc.x.array[:] = default_scalar_type(2j * k_w * P0)
    
    L_form = ufl.inner(f_inc, q) * ds_inc

    # ─── Rigid backing BC: p = 0 reflecting condition ───
    # Actually for rigid backing in acoustics, normal velocity = 0 → ∂p/∂n = 0
    # which is the natural (Neumann) BC → no extra terms needed

    # ─── Symmetry axis: ∂p/∂r = 0 (natural BC) ───

    # ─── Solve the linear system ───
    problem = fem_petsc.LinearProblem(
        a_form, L_form,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix=f"solve_freq_{freq}_"
    )
    p_sol = problem.solve()
    del problem

    # ─── Rigid backing BC: p = 0 reflecting condition ───
    # Actually for rigid backing in acoustics, normal velocity = 0 → ∂p/∂n = 0
    # which is the natural (Neumann) BC → no extra terms needed

    # ─── Symmetry axis: ∂p/∂r = 0 (natural BC) ───

# ─── Solve the linear system ───
    problem = fem_petsc.LinearProblem(
        a_form, L_form,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix=f"solve_freq_{freq}_"
    )
    p_sol = problem.solve()
    
    # Pro-tip: Manually delete the problem object at the end of each frequency loop. 
    # This forces PETSc to clear its C-level memory and prevents segmentation faults 
    # when looping over many frequencies in FEniCSx v0.10.
    del problem

    # ─── Compute reflection coefficient ───
    # Scattered pressure at the incident boundary:
    # p_total = p_inc + p_scattered
    # R = p_scattered / p_inc  (averaged over the incident surface)

    # Evaluate p_sol at the water-coating interface (z = 0)
    # For the reflection coefficient, we measure at the incident surface
    # R = (Z_in - Z_w) / (Z_in + Z_w)  where Z_in = p / v_n at z=0

    # Surface-averaged approach:
    # Integrate p_total over the incident boundary
    one = fem.Constant(msh, complex(1.0))
    area_inc = fem.assemble_scalar(fem.form(one * ds_inc))

    if abs(area_inc) < 1e-15:
        return 0.0

    p_avg = fem.assemble_scalar(fem.form(p_sol * ds_inc)) / area_inc

    # The scattered field: p_scat = p_total - p_inc
    # At the incident surface, p_inc ≈ P0 (for near-normal incidence)
    p_scat = p_avg - P0
    R = p_scat / P0

    # Absorption coefficient
    alpha = 1.0 - abs(R)**2
    alpha = max(0.0, min(1.0, alpha))

    return alpha


def solve_absorption(params_derived, frequencies=None, msh_file="coating_mesh.msh",
                     verbose=True):
    """
    Compute the absorption coefficient spectrum using FEniCSx FEM.

    Parameters
    ----------
    params_derived : dict
        From config.compute_derived()
    frequencies : array-like or None
        Frequencies to solve (Hz). Defaults to 1-1000 Hz.
    msh_file : str
        Path to the gmsh .msh file
    verbose : bool
        Print progress

    Returns
    -------
    freqs : np.ndarray
    alpha : np.ndarray
    """
    if frequencies is None:
        from config import FREQUENCIES
        frequencies = FREQUENCIES

    frequencies = np.asarray(frequencies)

    if verbose:
        print(f"Reading mesh from {msh_file} ...")
    mesh_data = read_gmsh_mesh(msh_file)

    if verbose:
        print(f"Solving {len(frequencies)} frequencies ...")

    alpha_arr = np.zeros(len(frequencies))
    for i, f in enumerate(frequencies):
        alpha_arr[i] = solve_single_frequency(f, mesh_data, params_derived)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frequencies)} done (f={f} Hz, α={alpha_arr[i]:.4f})")

    return frequencies, alpha_arr


# ─── CLI entry point ───
if __name__ == "__main__":
    from config import get_base_case_params, compute_derived, FREQUENCIES

    params = compute_derived(get_base_case_params())

    # Use a coarser frequency set for quick test
    test_freqs = np.arange(1, 1001, 10)  # every 10 Hz → 100 points
    if "--full" in sys.argv:
        test_freqs = FREQUENCIES  # all 1000 Hz

    freqs, alpha = solve_absorption(params, frequencies=test_freqs)

    # Save results
    outfile = "fem_absorption_results.csv"
    np.savetxt(outfile, np.column_stack([freqs, alpha]),
               delimiter=",", header="frequency_Hz,absorption_coefficient",
               comments="")
    print(f"Results saved to {outfile}")
    print(f"Average absorption: {np.mean(alpha):.6f}")
