"""Stage 2: find modular coils for the fixed-boundary equilibrium.

Loads eq_fixed.h5, initialises 4 FourierXYZCoils per half-field-period,
then minimises QuadraticFlux + CoilLength + CoilCurvature.
Output: coilset.h5
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

from desc.coils import initialize_modular_coils
from desc.equilibrium import Equilibrium
from desc.objectives import (
    CoilCurvature,
    CoilLength,
    FixSumCoilCurrent,
    ObjectiveFunction,
    QuadraticFlux,
)
from desc.optimize import Optimizer

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load equilibrium
# ---------------------------------------------------------------------------
eq = Equilibrium.load(str(HERE / "eq_fixed.h5"))
print(f"Loaded equilibrium: NFP={eq.NFP}, L={eq.L}, M={eq.M}, N={eq.N}")

# ---------------------------------------------------------------------------
# Initialise modular coils (4 unique coils for the half-period with stell sym)
# ---------------------------------------------------------------------------
NUM_COILS = 6   # unique coils (stellarator symmetry fills the rest)
R_OVER_A = 2.0  # coil-to-plasma aspect ratio (coils sit at ~2× the minor radius)

coilset = initialize_modular_coils(eq, num_coils=NUM_COILS, r_over_a=R_OVER_A)
coilset = coilset.to_FourierXYZ(N=12)
print(f"Coilset: {len(coilset.coils)} coils (including stell-sym images)")

# Rough length of an initial circular coil — used as a soft upper bound
mean_len = float(np.mean([c.compute("length")["length"] for c in coilset.coils]))
print(f"Initial mean coil length: {mean_len:.2f} m")

# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
objective = ObjectiveFunction(
    (
        QuadraticFlux(eq=eq, field=coilset, vacuum=False),
        CoilLength(coilset, bounds=(0, 3.0 * mean_len)),
        CoilCurvature(coilset, bounds=(0, 5.0)),
    )
)

# Fix only the SUM of coil currents (the net poloidal linking current), so the
# toroidal field strength is preserved and there's no trivial zero-current
# solution — but the optimizer is free to redistribute current between coils.
# The current distribution is the dominant lever for reducing B·n, so unlocking
# it (vs FixCoilCurrent which pins every coil to the same current) is what lets
# the field error drop well below the ~16% floor of the fixed-current solve.
constraints = (FixSumCoilCurrent(coilset),)

optimizer = Optimizer("lsq-exact")
coilset_opt, result = optimizer.optimize(
    things=coilset,
    objective=objective,
    constraints=constraints,
    verbose=3,
    copy=True,
    ftol=1e-8,
    gtol=1e-8,
    xtol=1e-8,
)

# optimizer returns a list of optimized things when multiple things are passed
coilset_opt = coilset_opt[0] if isinstance(coilset_opt, list) else coilset_opt

out = HERE / "coilset.h5"
coilset_opt.save(str(out))
print(f"Saved → {out}")

import resource
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"Peak memory: {peak_gb:.2f} GB")
