from __future__ import annotations
import numpy as np
import jax.numpy as jnp
try:
    from desc.grid import LinearGrid
except ImportError:
    LinearGrid = None

def _apply_nfp_symmetry(NFP, X, Y, Z, Vx, Vy, Vz, w):
    """Helper to replicate points and vectors based on toroidal symmetry (NFP)."""
    w = w / NFP
    X_list, Y_list, Z_list = [X], [Y], [Z]
    Vx_list, Vy_list, Vz_list = [Vx], [Vy], [Vz]
    w_list = [w]

    for k in range(1, NFP):
        phi = 2.0 * jnp.pi * k / NFP
        c, s = jnp.cos(phi), jnp.sin(phi)
        
        X_new = X * c - Y * s
        Y_new = X * s + Y * c
        Z_new = Z
        
        Vx_new = Vx * c - Vy * s
        Vy_new = Vx * s + Vy * c
        Vz_new = Vz

        X_list.append(X_new)
        Y_list.append(Y_new)
        Z_list.append(Z_new)
        Vx_list.append(Vx_new)
        Vy_list.append(Vy_new)
        Vz_list.append(Vz_new)
        w_list.append(w)
    
    return (
        jnp.concatenate(X_list),
        jnp.concatenate(Y_list),
        jnp.concatenate(Z_list),
        jnp.concatenate(Vx_list),
        jnp.concatenate(Vy_list),
        jnp.concatenate(Vz_list),
        jnp.concatenate(w_list)
    )


def _rho_midpoints(N_rho: int):
    return (np.arange(N_rho, dtype=float) + 0.5) / N_rho


def edge_taper(rho, rho0: float = 0.95, shape: str = "smoothstep"):
    """Smooth window in [0, 1] that tapers the plasma current near the LCFS.

    The window is 1 for ``rho <= rho0`` and rolls smoothly to 0 at ``rho = 1``,
    using the normalized edge coordinate ``t = (rho - rho0) / (1 - rho0)``:

    - ``smoothstep``  : ``1 - (3 t**2 - 2 t**3)``          (C1: zero 1st deriv at both ends)
    - ``smootherstep``: ``1 - (6 t**5 - 15 t**4 + 10 t**3)`` (C2: zero 1st AND 2nd deriv at both ends)
    - ``cosine``      : ``0.5 (1 + cos(pi t))``            (Hann; zero slope at both ends)
    - ``quadratic``   : ``(1 - t)**2``                     (zero slope only at the LCFS)

    Higher-order smoothness (``smootherstep``) makes the window's own transition
    contribute less spectral ringing, but it does not change how much current is
    removed -- the resulting field bias is set by ``rho0`` (the width of the
    tapered shell), not by the window shape.

    Multiplying J by this removes the current discontinuity at rho=1 (and so the
    Gibbs ringing of the spectral Biot-Savart field). NOTE: this changes the
    physical source, so the resulting B is the field of a reduced edge current,
    not of the true DESC equilibrium current.
    """
    rho = np.asarray(rho, dtype=float)
    w = np.ones_like(rho)
    if not (0.0 <= rho0 < 1.0):
        raise ValueError(f"edge taper rho0 must be in [0, 1); got {rho0}.")
    edge = rho > rho0
    t = np.clip((rho[edge] - rho0) / (1.0 - rho0), 0.0, 1.0)
    if shape == "smoothstep":
        w[edge] = 1.0 - (3.0 * t**2 - 2.0 * t**3)
    elif shape == "smootherstep":
        w[edge] = 1.0 - (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3)
    elif shape == "cosine":
        w[edge] = 0.5 * (1.0 + np.cos(np.pi * t))
    elif shape == "quadratic":
        w[edge] = (1.0 - t) ** 2
    else:
        raise ValueError(
            f"Unknown edge taper shape {shape!r}; expected 'smoothstep', "
            "'smootherstep', 'cosine', or 'quadratic'."
        )
    return w


def desc_volume_current(
    eq,
    *,
    L_grid: int | None = None,
    M_grid: int | None = None,
    N_grid: int | None = None,
    taper_rho0: float | None = None,
    taper_shape: str = "smoothstep",
):
    """
    Extracts plasma volume current density (J) and integration weights (dV)
    from a DESC equilibrium object.

    If ``taper_rho0`` is given, the current is multiplied by ``edge_taper`` so it
    rolls smoothly to zero at the LCFS (see ``edge_taper`` for caveats).
    """
    if LinearGrid is None:
        raise ImportError("DESC is not installed. Please install it to use this feature.")

    L = L_grid if L_grid else eq.L_grid + 4
    M = M_grid if M_grid else eq.M_grid + 4
    N = N_grid if N_grid else (eq.N_grid * 2 if eq.N_grid else 32)

    grid = LinearGrid(L=L, M=M, N=N, sym=False, NFP=eq.NFP, axis=False)

    return _desc_volume_current_from_grid(
        eq, grid, replicate_nfp=True,
        taper_rho0=taper_rho0, taper_shape=taper_shape,
    )


def desc_volume_current_on_grid(
    eq,
    *,
    N_rho: int,
    N_theta: int,
    N_zeta: int,
    replicate_nfp: bool = True,
    taper_rho0: float | None = None,
    taper_shape: str = "smoothstep",
):
    """
    Extract DESC plasma current density on a linear grid with explicit node counts.

    This helper is intended for benchmarks where ``N_rho``, ``N_theta``, and
    ``N_zeta`` refer to actual quadrature point counts rather than DESC spectral
    resolution parameters.

    If ``taper_rho0`` is given, the current is multiplied by ``edge_taper`` so it
    rolls smoothly to zero at the LCFS (see ``edge_taper`` for caveats).
    """
    if LinearGrid is None:
        raise ImportError("DESC is not installed. Please install it to use this feature.")

    grid = LinearGrid(
        rho=_rho_midpoints(N_rho),
        theta=N_theta,
        zeta=N_zeta,
        sym=False,
        NFP=eq.NFP,
        axis=False,
    )

    return _desc_volume_current_from_grid(
        eq, grid, replicate_nfp=replicate_nfp,
        taper_rho0=taper_rho0, taper_shape=taper_shape,
    )


def desc_equivalent_boundary_current(
    eq,
    *,
    N_theta: int,
    N_zeta: int,
    sign: float = -1.0,
    replicate_nfp: bool = True,
):
    """
    Extract an equivalent LCFS sheet current for reconstructing DESC ``B`` inside.

    DESC's virtual-casing current is ``K_vc = n x B / mu0``. To represent the
    interior equilibrium field with zero exterior field, the jump condition gives
    the opposite sheet current, ``K = -K_vc`` by default.
    """
    if LinearGrid is None:
        raise ImportError("DESC is not installed. Please install it to use this feature.")

    grid = LinearGrid(
        rho=np.array([1.0]),
        theta=N_theta,
        zeta=N_zeta,
        sym=False,
        NFP=eq.NFP,
        axis=False,
    )

    keys = ["K_vc", "X", "Y", "Z", "|e_theta x e_zeta|"]
    data = eq.compute(keys, grid=grid, basis="xyz")

    X = jnp.array(data["X"])
    Y = jnp.array(data["Y"])
    Z = jnp.array(data["Z"])

    K_vec = sign * jnp.array(data["K_vc"])
    Kx = K_vec[:, 0]
    Ky = K_vec[:, 1]
    Kz = K_vec[:, 2]

    jacobian_surf = jnp.array(data["|e_theta x e_zeta|"])
    grid_weights = jnp.array(grid.weights)
    w = jacobian_surf * grid_weights

    if replicate_nfp and eq.NFP > 1:
        X, Y, Z, Kx, Ky, Kz, w = _apply_nfp_symmetry(
            eq.NFP, X, Y, Z, Kx, Ky, Kz, w
        )

    return X, Y, Z, Kx, Ky, Kz, w


def _desc_volume_current_from_grid(
    eq, grid, *, replicate_nfp: bool,
    taper_rho0: float | None = None, taper_shape: str = "smoothstep",
):
    keys = ["J", "X", "Y", "Z", "sqrt(g)"]
    data = eq.compute(keys, grid=grid, basis="xyz")

    X = jnp.array(data["X"])
    Y = jnp.array(data["Y"])
    Z = jnp.array(data["Z"])

    Jx = jnp.array(data["J"][:, 0])
    Jy = jnp.array(data["J"][:, 1])
    Jz = jnp.array(data["J"][:, 2])

    if taper_rho0 is not None:
        # rho per node from the source grid; the window is invariant under the
        # NFP rotation, so applying it here (before replication) is exact.
        rho = np.asarray(grid.nodes)[:, 0]
        taper = jnp.asarray(edge_taper(rho, rho0=taper_rho0, shape=taper_shape))
        Jx = Jx * taper
        Jy = Jy * taper
        Jz = Jz * taper

    sqrt_g = jnp.array(data["sqrt(g)"])
    grid_weights = jnp.array(grid.weights)
    w = sqrt_g * grid_weights

    if replicate_nfp and eq.NFP > 1:
        X, Y, Z, Jx, Jy, Jz, w = _apply_nfp_symmetry(
            eq.NFP, X, Y, Z, Jx, Jy, Jz, w
        )

    return X, Y, Z, Jx, Jy, Jz, w


def desc_surface_current(
    field, 
    surface,
    *,
    M_grid: int = 120, 
    N_grid: int = 120
):
    """
    Extracts surface current density (K) and integration weights (dA)
    from a DESC FourierCurrentPotentialField and a corresponding surface.

    Parameters
    ----------
    field : FourierCurrentPotentialField
        The DESC field object containing the current potential phi.
    surface : Surface
        The DESC surface object (e.g. ConstantOffsetSurface) where the current lies.
    M_grid : int
        Poloidal grid resolution.
    N_grid : int
        Toroidal grid resolution.

    Returns
    -------
    X, Y, Z : jnp.ndarray
        Cartesian coordinates of the surface points.
    Kx, Ky, Kz : jnp.ndarray
        Cartesian components of the surface current density K.
    w : jnp.ndarray
        Integration weights (Area elements dA) for Biot-Savart integration.
    """
    if LinearGrid is None:
        raise ImportError("DESC is not installed.")

    grid = LinearGrid(M=M_grid, N=N_grid, NFP=surface.NFP, sym=False)
    
    keys = ["X", "Y", "Z", "K", "|e_theta x e_zeta|"]
    
    data = field.compute(keys, grid=grid, basis="xyz")
    
    X = jnp.array(data["X"])
    Y = jnp.array(data["Y"])
    Z = jnp.array(data["Z"])
    
    K_vec = jnp.array(data["K"])
    Kx = K_vec[:, 0]
    Ky = K_vec[:, 1]
    Kz = K_vec[:, 2]

    jacobian_surf = jnp.array(data["|e_theta x e_zeta|"])
    grid_weights = jnp.array(grid.weights)
    
    w = jacobian_surf * grid_weights

    if surface.NFP > 1:
        X, Y, Z, Kx, Ky, Kz, w = _apply_nfp_symmetry(
            surface.NFP, X, Y, Z, Kx, Ky, Kz, w
        )

    return X, Y, Z, Kx, Ky, Kz, w
