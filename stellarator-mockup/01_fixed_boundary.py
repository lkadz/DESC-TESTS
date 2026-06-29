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

# p(s) = 1e4 (1 - s²)  Pa,   iota(s) = 0.4 + 0.1 s²
pressure = PowerSeriesProfile(params=[1e4, -1e4], modes=[0, 2])
iota = PowerSeriesProfile(params=[0.4, 0.1], modes=[0, 2])
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
