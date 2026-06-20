"""Spectral filters for the computed Biot-Savart field coefficients.

The plasma current density J has a discontinuity at the LCFS (it jumps from a
finite value to zero). The spectral Biot-Savart solver represents the field on a
periodic Fourier grid, so that discontinuity produces Gibbs ringing in B,
concentrated near the plasma boundary.

A spectral filter multiplies the field's Fourier coefficients ``B_hat(k)`` by a
smooth response ``sigma(|k|/k_max)`` that is ~1 for the resolved low-|k| modes
and rolls smoothly to zero toward the Nyquist wavenumber, damping the
under-resolved high-|k| modes that carry the ringing.

Unlike the real-space edge taper (``desc_interface.edge_taper``), this operates
on the *field*, not the *source*: it removes no physical current and therefore
introduces no current bias. The Fourier coefficients of a discontinuous source
still contain the full information; the ringing is a reconstruction artifact,
and filtering trades a small, controlled smoothing for the removal of the
oscillations.

References: Gottlieb & Hesthaven, "Spectral methods for hyperbolic problems"
(J. Comput. Appl. Math. 2001); Hou & Li, JCP 2007 (exponential filter).
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import BoxParams

FILTER_KINDS = ("none", "exponential", "lanczos", "cesaro", "raised_cosine")

# sigma(1) ~ machine epsilon for the exponential filter, so the Nyquist mode is
# annihilated to round-off while lower modes are essentially untouched.
_MACHINE_EPS = 2.220446049250313e-16


def _sigma_1d(eta, kind: str, order: int):
    """Filter response on the normalized wavenumber ``eta = |k| / k_max`` in
    ``[0, 1]``. All kinds satisfy ``sigma(0) = 1`` (the DC / low-|k| field is
    preserved) and decay toward ``eta = 1`` (the Nyquist)."""
    if kind == "exponential":
        # exp(-alpha eta**order): order p sets the knee sharpness. Higher p keeps
        # more low-|k| modes intact before cutting off near the Nyquist.
        alpha = -jnp.log(_MACHINE_EPS)
        return jnp.exp(-alpha * eta ** order)
    if kind == "lanczos":
        # sinc(eta) = sin(pi eta) / (pi eta), with the eta=0 limit set to 1.
        pe = jnp.pi * eta
        safe = jnp.where(eta > 0.0, pe, 1.0)
        return jnp.where(eta > 0.0, jnp.sin(safe) / safe, 1.0)
    if kind == "cesaro":
        # Fejer / Cesaro averaging: 1 - eta.
        return jnp.clip(1.0 - eta, 0.0, 1.0)
    if kind == "raised_cosine":
        # Hann window in spectral space.
        return 0.5 * (1.0 + jnp.cos(jnp.pi * eta))
    raise ValueError(
        f"Unknown spectral filter {kind!r}; expected one of {FILTER_KINDS}."
    )


def spectral_filter_array(box: BoxParams, kind: str | None, order: int = 8):
    """Separable spectral-filter multiplier on the box K-grid.

    Returns an array shaped ``(Nx, Ny, Nz)`` (broadcast from three 1-D axis
    factors) to multiply against ``B_hat``, or ``None`` when no filtering is
    requested so callers can skip the multiply entirely.

    The multiplier is the tensor product
    ``sigma(|kx|/kx_max) sigma(|ky|/ky_max) sigma(|kz|/kz_max)``, each axis
    normalized by its own Nyquist wavenumber. ``order`` applies to the
    ``exponential`` kind only.
    """
    if kind is None or kind == "none":
        return None

    def axis_factor(k1d):
        kmax = jnp.max(jnp.abs(k1d))
        eta = jnp.abs(k1d) / jnp.where(kmax > 0.0, kmax, 1.0)
        return _sigma_1d(eta, kind, order)

    fx = axis_factor(box.kx)[:, None, None]
    fy = axis_factor(box.ky)[None, :, None]
    fz = axis_factor(box.kz)[None, None, :]
    return fx * fy * fz
