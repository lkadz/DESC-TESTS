"""Stage 3: free-boundary equilibrium using the optimised coil set.

Loads eq_fixed.h5 and coilset.h5.  Increases spectral resolution, then
minimises VacuumBoundaryError (B·n on the LCFS) while ForceBalance, iota,
pressure, and total flux are held as constraints — the plasma boundary is
free to deform.  VacuumBoundaryError is a valid approximation at low β.
Output: eq_free.h5
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from desc.coils import CoilSet
from desc.equilibrium import Equilibrium
from desc.objectives import (
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    VacuumBoundaryError,
)
from desc.optimize import Optimizer

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
eq = Equilibrium.load(str(HERE / "eq_fixed.h5"))
coilset = CoilSet.load(str(HERE / "coilset.h5"))
print(f"Loaded equilibrium: NFP={eq.NFP}, L={eq.L}, M={eq.M}, N={eq.N}")
print(f"Loaded coilset: {len(coilset.coils)} coils")

# ---------------------------------------------------------------------------
# Increase resolution for the free-boundary solve
# ---------------------------------------------------------------------------
eq = eq.copy()
eq.change_resolution(L=10, M=10, N=8, L_grid=20, M_grid=20, N_grid=16)
print(f"Resolution after increase: L={eq.L}, M={eq.M}, N={eq.N}")

# ---------------------------------------------------------------------------
# Free-boundary optimisation
# The plasma boundary (R_lmn, Z_lmn) is the free variable.
# We minimise B·n on that surface while the interior stays in equilibrium.
# ---------------------------------------------------------------------------
objective = ObjectiveFunction(
    VacuumBoundaryError(eq=eq, field=coilset, field_fixed=True)
)

constraints = (
    ForceBalance(eq=eq),
    FixPressure(eq=eq),
    FixIota(eq=eq),
    FixPsi(eq=eq),
)

optimizer = Optimizer("proximal-lsq-exact")
eq_free, result = optimizer.optimize(
    things=eq,
    objective=objective,
    constraints=constraints,
    verbose=3,
    copy=True,
)

out = HERE / "eq_free.h5"
eq_free.save(str(out))
print(f"Saved → {out}")

import resource
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"Peak memory: {peak_gb:.2f} GB")
