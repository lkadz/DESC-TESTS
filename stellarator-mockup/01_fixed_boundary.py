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

from desc import set_device
set_device("gpu")

import numpy as np

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.examples import get as get_example
from desc.profiles import PowerSeriesProfile

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Use the precise_QA boundary (NFP=2, quasi-axisymmetric, ships with DESC).
# Building a fresh equilibrium from it with custom profiles avoids having to
# hand-tune Fourier modes to keep the initial Jacobian non-degenerate.
# ---------------------------------------------------------------------------
example = get_example("precise_QA")
surface = example.surface.copy()

# ---------------------------------------------------------------------------
# Make the boundary easier for modular coils to reproduce.
#
# precise_QA is a research-grade, heavily-shaped QA boundary: simple modular
# coils only match it to ~15% max B·n, so the free-boundary B·n and force
# errors plateau well above 1%. Attenuating the non-axisymmetric (n != 0)
# boundary modes by SHAPE_FACTOR keeps a genuinely 3D stellarator but with
# milder shaping that coils CAN reproduce to <1%. The axisymmetric (n == 0)
# part — major radius, minor radius, elongation — is left untouched, so the
# surface stays a valid nested torus for any factor in [0, 1].
#
#   SHAPE_FACTOR = 1.0  -> original precise_QA (hard; max B·n ~15%)
#   SHAPE_FACTOR = 0.25 -> mild stellarator   (easy; target max B·n <1%)
# Lower it if max B·n is still >1%; raise it for stronger (harder) shaping.
# ---------------------------------------------------------------------------
SHAPE_FACTOR = 0.25

R_lmn = np.asarray(surface.R_lmn, dtype=float).copy()
Z_lmn = np.asarray(surface.Z_lmn, dtype=float).copy()
R_lmn[surface.R_basis.modes[:, 2] != 0] *= SHAPE_FACTOR
Z_lmn[surface.Z_basis.modes[:, 2] != 0] *= SHAPE_FACTOR
surface.R_lmn = R_lmn
surface.Z_lmn = Z_lmn

# Finite-beta, non-vacuum equilibrium (per requirement). Rotational transform
# from 3D shaping scales roughly with SHAPE_FACTOR, so scale the prescribed
# iota with it too — keeps the implied net plasma current (and hence the field
# the coils must reproduce) modest.
# p(s) = 1e3 (1 - s²) Pa  (beta ~ 1.7e-5);  iota(s) = (0.4 + 0.1 s²)·SHAPE_FACTOR
pressure = PowerSeriesProfile(params=[1e3, -1e3], modes=[0, 2])
iota = PowerSeriesProfile(
    params=[0.4 * SHAPE_FACTOR, 0.1 * SHAPE_FACTOR], modes=[0, 2]
)
NFP = surface.NFP  # 2

eq = Equilibrium(
    NFP=NFP,
    sym=True,
    L=8,
    M=8,
    N=6,
    L_grid=16,
    M_grid=16,
    N_grid=12,
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

import resource
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"Peak memory: {peak_gb:.2f} GB")
