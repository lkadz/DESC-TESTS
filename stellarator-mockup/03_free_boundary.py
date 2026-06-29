"""Stage 3: free-boundary equilibrium using the optimised coil set.

Loads eq_fixed.h5 and coilset.h5.  Two-step solve:
  1. Warm-start with VacuumBoundaryError (cheap, brings B·n close to zero)
  2. Refine with BoundaryError (full virtual casing, correct for finite beta)
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
    BoundaryError,
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

constraints = (
    ForceBalance(eq=eq),
    FixPressure(eq=eq),
    FixIota(eq=eq),
    FixPsi(eq=eq),
)

optimizer = Optimizer("proximal-lsq-exact")

# ---------------------------------------------------------------------------
# Step 1 — warm start with VacuumBoundaryError (fast, no virtual casing)
# Brings the boundary close to zero B·n so step 2 starts from a good point.
# ---------------------------------------------------------------------------
print("\n=== Step 1: VacuumBoundaryError warm start ===")
obj_vacuum = ObjectiveFunction(
    VacuumBoundaryError(eq=eq, field=coilset, field_fixed=True)
)

eq_warm, _ = optimizer.optimize(
    things=eq,
    objective=obj_vacuum,
    constraints=constraints,
    verbose=3,
    copy=True,
)

# ---------------------------------------------------------------------------
# Step 2 — refine with BoundaryError (full virtual casing, finite-beta correct)
# Starts from the warm-started boundary so convergence is fast.
# ---------------------------------------------------------------------------
print("\n=== Step 2: BoundaryError refinement (finite-beta) ===")
obj_full = ObjectiveFunction(
    BoundaryError(eq=eq_warm, field=coilset, field_fixed=True)
)

constraints_full = (
    ForceBalance(eq=eq_warm),
    FixPressure(eq=eq_warm),
    FixIota(eq=eq_warm),
    FixPsi(eq=eq_warm),
)

eq_free, _ = optimizer.optimize(
    things=eq_warm,
    objective=obj_full,
    constraints=constraints_full,
    verbose=3,
    copy=True,
)

out = HERE / "eq_free.h5"
eq_free.save(str(out))
print(f"Saved → {out}")

import resource
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"Peak memory: {peak_gb:.2f} GB")
