"""Stage 1: solve a simple 2-NFP fixed-boundary stellarator equilibrium.

Configuration: R0=3.0 m, a≈0.5 m, NFP=2, parabolic pressure, linear iota.
Output: eq_fixed.h5
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Boundary: simple 2-field-period helical stellarator
#   R(θ,ζ) = 3.0 + 0.5 cos θ + 0.10 cos(ζ_NFP) + 0.02 cos(θ - ζ_NFP)
#   Z(θ,ζ) = 0.5 sin θ + 0.10 sin(ζ_NFP) + 0.02 sin(θ - ζ_NFP)
# where ζ_NFP = NFP·ζ  (ζ = geometric toroidal angle, period 2π/NFP)
# ---------------------------------------------------------------------------
NFP = 2

surface = FourierRZToroidalSurface(
    R_lmn=np.array([3.0, 0.5, 0.10, 0.02]),
    Z_lmn=np.array([0.5, 0.10, 0.02]),
    modes_R=[[0, 0], [1, 0], [0, 1], [1, 1]],
    modes_Z=[[1, 0], [0, 1], [1, 1]],
    NFP=NFP,
    sym=True,
)

# p(s) = 1e4 (1 - s²)  Pa,   iota(s) = 0.4 + 0.1 s²
pressure = PowerSeriesProfile(params=[1e4, -1e4], modes=[0, 2])
iota = PowerSeriesProfile(params=[0.4, 0.1], modes=[0, 2])

eq = Equilibrium(
    NFP=NFP,
    sym=True,
    L=6,
    M=6,
    N=4,
    L_grid=12,
    M_grid=12,
    N_grid=8,
    surface=surface,
    pressure=pressure,
    iota=iota,
    Psi=1.0,
)

print("Solving fixed-boundary equilibrium …")
eqs = solve_continuation_automatic(eq, objective="force", verbose=3)
eq_solved = eqs[-1]

out = HERE / "eq_fixed.h5"
eq_solved.save(str(out))
print(f"Saved → {out}")
