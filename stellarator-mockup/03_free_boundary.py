"""Stage 3: free-boundary equilibrium — co-optimise plasma boundary + coil shapes.

Loads eq_fixed.h5 and coilset.h5 from the previous stages.

Strategy: pass things=[eq, coilset] so the optimizer simultaneously adjusts
the plasma boundary Fourier modes AND the coil XYZ Fourier modes. BoundaryError
(full virtual casing, correct for finite beta) is the objective. ForceBalance
is a hard constraint on the equilibrium. CoilLength + CoilCurvature keep the
coil shapes physical during the co-optimisation.

This is how real stellarator free-boundary problems are solved: stage 2 coils
are the initial guess, and stage 3 finds the self-consistent configuration.

Output: eq_free.h5, coilset_free.h5
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

from desc.coils import CoilSet
from desc.equilibrium import Equilibrium
from desc.objectives import (
    BoundaryError,
    CoilCurvature,
    CoilLength,
    FixCoilCurrent,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.optimize import Optimizer

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load — warm-restart: if eq_free.h5 / coilset_free.h5 already exist from a
# previous run, start from them so repeated submissions continue from wherever
# the optimizer stopped last time.
# ---------------------------------------------------------------------------
eq_free_path = HERE / "eq_free.h5"
coilset_free_path = HERE / "coilset_free.h5"

if eq_free_path.exists() and coilset_free_path.exists():
    eq = Equilibrium.load(str(eq_free_path))
    coilset = CoilSet.load(str(coilset_free_path))
    print("Warm restart: loaded eq_free.h5 + coilset_free.h5")
else:
    eq = Equilibrium.load(str(HERE / "eq_fixed.h5"))
    coilset = CoilSet.load(str(HERE / "coilset.h5"))
    print("Cold start: loaded eq_fixed.h5 + coilset.h5")
print(f"Equilibrium: NFP={eq.NFP}, L={eq.L}, M={eq.M}, N={eq.N}")
print(f"Coilset: {len(coilset.coils)} coils")

eq = eq.copy()
eq.change_resolution(L=8, M=8, N=6, L_grid=16, M_grid=16, N_grid=12)
print(f"Resolution: L={eq.L}, M={eq.M}, N={eq.N}")

# ---------------------------------------------------------------------------
# Coil regularisation bounds (same as stage 2)
# ---------------------------------------------------------------------------
mean_len = float(np.mean([c.compute("length")["length"] for c in coilset.coils]))
print(f"Mean coil length: {mean_len:.2f} m")

# ---------------------------------------------------------------------------
# Co-optimisation: plasma boundary + coil shapes, coil currents fixed
# ---------------------------------------------------------------------------
objective = ObjectiveFunction((
    BoundaryError(
        eq=eq,
        field=coilset,
        field_fixed=False,      # coilset is now a free variable
        bs_chunk_size=512,
        B_plasma_chunk_size=64,
    ),
    CoilLength(coilset, bounds=(0, 3.0 * mean_len)),
    CoilCurvature(coilset, bounds=(0, 5.0)),
))

constraints = (
    ForceBalance(eq=eq),
    FixPressure(eq=eq),
    FixIota(eq=eq),
    FixPsi(eq=eq),
    FixCoilCurrent(coilset),    # only shapes vary, not currents
)

optimizer = Optimizer("proximal-lsq-exact")
result, _ = optimizer.optimize(
    things=[eq, coilset],
    objective=objective,
    constraints=constraints,
    verbose=3,
    copy=True,
    ftol=1e-8,
    gtol=1e-8,
    xtol=1e-8,
)
# copy=True with multiple things returns a list of lists: [[eq_opt], [coilset_opt]]
eq_free = result[0][0] if isinstance(result[0], list) else result[0]
coilset_free = result[1][0] if isinstance(result[1], list) else result[1]

eq_free.save(str(HERE / "eq_free.h5"))
coilset_free.save(str(HERE / "coilset_free.h5"))
print(f"Saved → eq_free.h5, coilset_free.h5")

import resource
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"Peak memory: {peak_gb:.2f} GB")
