"""Stage 3: free-boundary equilibrium using the optimised coil set.

Loads eq_fixed.h5 and coilset.h5. Minimises VacuumBoundaryError (B·n on
the LCFS). Valid approximation for this low-beta plasma (beta ~ 1.7e-4).
Output: eq_free.h5
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from desc import set_device
set_device("gpu")

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
# Resolution for the free-boundary solve
# ---------------------------------------------------------------------------
eq = eq.copy()
eq.change_resolution(L=8, M=8, N=6, L_grid=16, M_grid=16, N_grid=12)
print(f"Resolution: L={eq.L}, M={eq.M}, N={eq.N}")

# ---------------------------------------------------------------------------
# Free-boundary optimisation
# VacuumBoundaryError is valid here: beta ~ 1.7e-4, plasma currents negligible.
# Tight tolerances prevent the default ftol=1e-2 from stopping too early.
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
eq_free, _ = optimizer.optimize(
    things=eq,
    objective=objective,
    constraints=constraints,
    verbose=3,
    copy=True,
    ftol=1e-8,
    gtol=1e-8,
    xtol=1e-8,
)
eq_free = eq_free[0] if isinstance(eq_free, list) else eq_free

out = HERE / "eq_free.h5"
eq_free.save(str(out))
print(f"Saved → {out}")

import resource
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"Peak memory: {peak_gb:.2f} GB")
