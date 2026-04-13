"""
Configuration: 20-parameter geometry and material properties for the
underwater acoustic coating from Gao et al. (Ocean Engineering).

Base case = Table 1 & Table 2 of the paper.
Modify get_params() to use any parameter set (e.g. Case 1 or Case 2 predictions).
"""

import numpy as np


def get_base_case_params():
    """Base case from the paper (Tables 1 & 2)."""
    return {
        # --- Geometric (all in mm, converted to m inside the solver) ---
        "W": 2000.0,           # unit cell width (mm)
        "d": [10.0] * 10,     # layer thicknesses d1..d10 (mm)
        "m": {                 # hollow diameters for layers 2,3,5,6,8,9 (mm)
            2: 1000.0, 3: 1000.0,
            5: 1000.0, 6: 1000.0,
            8: 1000.0, 9: 1000.0,
        },
        # --- Material (rubber) ---
        "rho": 1130.0,         # density (kg/m^3)
        "E": 5e7,              # real part of Young's modulus (Pa)
        "eta": 0.4,            # loss factor
        "nu": 0.44,            # Poisson's ratio
    }


def get_case1_params():
    """Predicted parameters for Case 1 (Table 5 in the paper)."""
    return {
        "W": 2000.0,
        "d": [10.7398927211761, 9.19723930954933, 9.03188413381577,
              10.3210442066193, 9.02042958140373, 9.03090962767601,
              10.1851529777050, 9.20814347267151, 9.10563176870346,
              10.3094107210636],
        "m": {
            2: 353.101807236671, 3: 354.557039141655,
            5: 371.865028738976, 6: 382.696526646614,
            8: 404.766833186150, 9: 400.110272169113,
        },
        "rho": 1268.43154430389,
        "E": 8.49030417203903e7,
        "eta": 0.758901304006577,
        "nu": 0.476384522914887,
    }


def get_case2_params():
    """Predicted parameters for Case 2 (Table 5 in the paper)."""
    return {
        "W": 2000.0,
        "d": [10.4116438329220, 9.59269911050797, 9.48593208193779,
              10.2778358161449, 9.56153133511543, 9.52644741535187,
              10.2548802793026, 9.57734596729279, 9.54049196839333,
              10.3517957925797],
        "m": {
            2: 422.048478722572, 3: 415.016191601753,
            5: 421.283624768257, 6: 428.282207846642,
            8: 437.154577970505, 9: 433.320407271385,
        },
        "rho": 1254.90951538086,
        "E": 7.96552193164825e7,
        "eta": 0.700865012407303,
        "nu": 0.472864060401917,
    }


# ---------- Derived quantities ----------

def compute_derived(params):
    """
    From raw 20-param dict, return everything the FEM solver needs
    (all in SI units: metres, Pa, kg/m^3).
    """
    W_m = params["W"] / 1000.0                       # half-width used as radius in axi-symmetric
    R_outer = W_m / 2.0                               # outer radius (m)
    d_m = [di / 1000.0 for di in params["d"]]         # layer thicknesses (m)
    m_m = {k: v / 1000.0 for k, v in params["m"].items()}  # hollow diameters (m)

    rho = params["rho"]
    eta = params["eta"]
    E_complex = params["E"] * (1.0 + 1j * eta)        # complex Young's modulus
    nu = params["nu"]

    # Lamé constants (complex)
    lam = E_complex * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E_complex / (2 * (1 + nu))

    # Water
    rho_w = 1000.0
    c_w = 1500.0

    total_thickness = sum(d_m)

    return {
        "R_outer": R_outer,
        "d_m": d_m,
        "m_m": m_m,
        "rho": rho,
        "E_complex": E_complex,
        "nu": nu,
        "lam": lam,
        "mu": mu,
        "eta": eta,
        "rho_w": rho_w,
        "c_w": c_w,
        "total_thickness": total_thickness,
    }


# ---------- Frequency range ----------

FREQ_START = 1       # Hz
FREQ_END = 1000      # Hz
FREQ_STEP = 1        # Hz
FREQUENCIES = np.arange(FREQ_START, FREQ_END + 1, FREQ_STEP)


# ---------- 20-parameter ranges (Section 2.13) ----------

PARAM_RANGES = {
    "d":   (1.0, 20.0),         # mm, each of 10 layers
    "m":   (20.0, 1980.0),      # mm, each of 6 hollow diameters
    "rho": (1000.0, 1500.0),    # kg/m^3
    "eta": (0.1, 0.8),
    "E":   (1e7, 1e8),          # Pa (real part only; complex = E*(1+eta*i))
    "nu":  (0.40, 0.49),
}
