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
    CoilSetMinDistance,
    FixIota,
    FixPressure,
    FixPsi,
    FixSumCoilCurrent,
    ForceBalance,
    ObjectiveFunction,
    PlasmaCoilSetMinDistance,
)
from desc.optimize import Optimizer

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load — warm-restart with staleness guards. If eq_free.h5 / coilset_free.h5
# exist from a previous run, continue from them — UNLESS they predate the
# current stage-1 output or their coil count no longer matches stage 2, in
# which case they are leftovers from an older pipeline configuration and
# restarting from them would silently optimise the wrong problem.
# ---------------------------------------------------------------------------
eq_fixed_path = HERE / "eq_fixed.h5"
coilset_path = HERE / "coilset.h5"
eq_free_path = HERE / "eq_free.h5"
coilset_free_path = HERE / "coilset_free.h5"

warm = eq_free_path.exists() and coilset_free_path.exists()
if warm and eq_fixed_path.stat().st_mtime > eq_free_path.stat().st_mtime:
    print(
        "WARNING: eq_fixed.h5 is newer than eq_free.h5 — stale free-boundary "
        "files from an older pipeline run. Forcing cold start."
    )
    warm = False

if warm:
    eq = Equilibrium.load(str(eq_free_path))
    coilset = CoilSet.load(str(coilset_free_path))
    n_stage2 = len(CoilSet.load(str(coilset_path)).coils)
    if len(coilset.coils) != n_stage2:
        print(
            f"WARNING: coilset_free.h5 has {len(coilset.coils)} coils but "
            f"stage 2 produced {n_stage2} — stale files. Forcing cold start."
        )
        warm = False

if warm:
    print("Warm restart: loaded eq_free.h5 + coilset_free.h5")
else:
    eq = Equilibrium.load(str(eq_fixed_path))
    coilset = CoilSet.load(str(coilset_path))
    print("Cold start: loaded eq_fixed.h5 + coilset.h5")
print(f"Equilibrium: NFP={eq.NFP}, L={eq.L}, M={eq.M}, N={eq.N}")
print(f"Coilset: {len(coilset.coils)} coils")

eq = eq.copy()
eq.change_resolution(L=8, M=8, N=6, L_grid=16, M_grid=16, N_grid=12)
print(f"Resolution: L={eq.L}, M={eq.M}, N={eq.N}")

# ---------------------------------------------------------------------------
# Coil regularisation bounds (length/curvature match stage 2; clearance 0.2 a).
# Anchored to the STAGE-1/2 outputs, not the loaded (possibly warm-restarted)
# state — otherwise each restart re-bases "3x mean length" on already-grown
# coils and the bounds ratchet outward run over run.
# ---------------------------------------------------------------------------
eq_ref = Equilibrium.load(str(eq_fixed_path))
minor_radius = float(eq_ref.compute("a")["a"])
coilset_ref = CoilSet.load(str(coilset_path))
mean_len = float(np.mean([c.compute("length")["length"] for c in coilset_ref.coils]))
print(f"Minor radius: {minor_radius:.3f} m, reference mean coil length: {mean_len:.2f} m")

# ---------------------------------------------------------------------------
# Co-optimisation: plasma boundary + coil shapes + coil current distribution.
# Only the NET coil current is fixed (FixSumCoilCurrent) — individual currents
# redistribute freely, which is the dominant lever for driving B·n toward zero.
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
    CoilCurvature(coilset, bounds=(0, 12.0)),  # match stage 2's relaxed bound
    # Collision/clearance guards: conservative (~0.2 minor radii) so they're
    # inactive for the rung-1 geometry but stop the co-opt from pushing coils
    # into each other or into the plasma as it reshapes them.
    CoilSetMinDistance(coilset, bounds=(0.2 * minor_radius, np.inf)),
    PlasmaCoilSetMinDistance(eq, coilset, bounds=(0.2 * minor_radius, np.inf)),
))

constraints = (
    ForceBalance(eq=eq),
    FixPressure(eq=eq),
    FixIota(eq=eq),
    FixPsi(eq=eq),
    FixSumCoilCurrent(coilset),   # net current fixed; distribution + shapes vary
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
