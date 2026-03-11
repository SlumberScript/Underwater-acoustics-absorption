# FEniCSx Simulation — Underwater Acoustic Coating

This folder contains a complete FEniCSx FEM simulation replicating the COMSOL model
from **Gao et al. — "On-demand Prediction of Low-Frequency Average Sound Absorption
Coefficient of Underwater Coating Using Machine Learning" (Ocean Engineering)**.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | 20-parameter geometry & material definitions (base case, Case 1, Case 2) |
| `mesh_generation.py` | Builds the 2D axisymmetric mesh using gmsh |
| `fem_solver.py` | FEniCSx Helmholtz solver (pressure acoustics + PML + rigid backing) |
| `postprocess.py` | Plotting: FEM vs TMM comparison |
| `run_simulation.py` | **Main entry point** — runs everything end-to-end |

---

## Prerequisites

You need **WSL Ubuntu** (which you already have) with the following installed:

### Already installed ✅
```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx
```

### Still needed (run these in your WSL terminal):
```bash
# gmsh Python API (for mesh generation)
pip install gmsh

# matplotlib (for plotting)
pip install matplotlib

# numpy (should already be installed with fenicsx)
pip install numpy
```

---

## Step-by-Step: How to Run

### Step 0: Open your WSL Ubuntu terminal

Press `Win + R`, type `wsl`, press Enter. Or open "Ubuntu" from your Start menu.

### Step 1: Navigate to the simulation folder

```bash
cd /mnt/d/Research/Under-water-metamaterial/Underwater-acoustics-absorption/simulation_code
```

> **Note:** Windows path `D:\Research\...` becomes `/mnt/d/Research/...` in WSL.

### Step 2: Install Python dependencies (first time only)

```bash
pip install gmsh matplotlib numpy
```

### Step 3: Run a quick test (20 frequency points)

```bash
python3 run_simulation.py --quick
```

This will:
1. Generate the mesh (saves `coating_mesh.msh`)
2. Solve at 20 frequency points (every 50 Hz from 1–1000 Hz)
3. Save results to `fem_absorption_results.csv`
4. Compare with TMM and show a plot
5. Save the comparison plot to `../figures/`

### Step 4: Run the full simulation (1000 frequency points)

```bash
python3 run_simulation.py --full
```

> ⚠️ The full run solves 1000 linear systems — it will take longer.

### Step 5: Try different parameter sets

```bash
# Case 1 (predicted by the DNN)
python3 run_simulation.py --case 1 --output case1_fem.csv

# Case 2 (predicted by the DNN)
python3 run_simulation.py --case 2 --output case2_fem.csv
```

### Step 6: View the mesh (optional, requires GUI)

```bash
python3 mesh_generation.py --show
```

> Requires X11 forwarding or WSLg. If you see an error about display, skip this step.

---

## What the Simulation Does (Physics)

The simulation exactly follows **Section 2.10** of the paper:

1. **Geometry**: 2D axisymmetric model of one unit cell
   - Water column on top (where sound enters)
   - 10-layer rubber coating (layers 2,3,5,6,8,9 have hollow cylinders)
   - Rigid backing at the bottom

2. **Physics**:
   - **Water domain**: Helmholtz equation $\nabla^2 p + k^2 p = 0$
   - **PML**: Perfectly Matched Layer absorbs outgoing waves
   - **Incident wave**: Plane wave from top ($p = P_0 e^{-jkz}$)
   - **Rigid backing**: Zero normal velocity (natural Neumann BC)

3. **Output**: Absorption coefficient $\alpha = 1 - |R|^2$ at each frequency

4. **Validation**: Results are compared against the existing TMM code (Theoretical_model.py)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: dolfinx` | Make sure you're running in WSL, not Windows PowerShell |
| `ModuleNotFoundError: gmsh` | Run `pip install gmsh` in WSL |
| `No display` when trying `--show` | Install WSLg or skip the GUI step |
| `MUMPS solver error` | Run `sudo apt install libmumps-dev` |
| Import errors for `src.Theoretical_model` | Run from inside the `simulation_code/` directory |

---

## Output Files

After running, you'll find:
- `coating_mesh.msh` — the generated mesh
- `fem_absorption_results.csv` — frequency vs absorption coefficient
- `../figures/fem_vs_tmm_*.png` — comparison plots
