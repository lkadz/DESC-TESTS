from __future__ import annotations
import jax.numpy as jnp
from jax_finufft import nufft1, nufft2

from .types import BoxParams
from .filters import spectral_filter_array


def compute_B_hat(
    X, Y, Z, Jx, Jy, Jz, w, box, eps=1e-12, spectral_filter=None, filter_order=8
):
    mu0 = 4.0 * jnp.pi * 1e-7

    Nx, Ny, Nz = box.Nx, box.Ny, box.Nz
    shape = (Nx, Ny, Nz)

    tx = 2.0 * jnp.pi * X / box.Lx
    ty = 2.0 * jnp.pi * Y / box.Ly
    tz = 2.0 * jnp.pi * Z / box.Lz

    c_x = (Jx * w).astype(jnp.complex128) 
    c_y = (Jy * w).astype(jnp.complex128)
    c_z = (Jz * w).astype(jnp.complex128)

    Jx_hat = nufft1(shape, c_x, tx, ty, tz, eps=eps, iflag=-1)
    Jy_hat = nufft1(shape, c_y, tx, ty, tz, eps=eps, iflag=-1)
    Jz_hat = nufft1(shape, c_z, tx, ty, tz, eps=eps, iflag=-1)

    KX, KY, KZ = box.KX, box.KY, box.KZ
    K2 = KX**2 + KY**2 + KZ**2
    mask = K2 > 0.0
    K2 = jnp.where(mask, K2, 1.0)

    k_dot_J = KX * Jx_hat + KY * Jy_hat + KZ * Jz_hat
    Jx_hat = jnp.where(mask, Jx_hat - KX * k_dot_J / K2, 0.0)
    Jy_hat = jnp.where(mask, Jy_hat - KY * k_dot_J / K2, 0.0)
    Jz_hat = jnp.where(mask, Jz_hat - KZ * k_dot_J / K2, 0.0)

    kxJx = KY * Jz_hat - KZ * Jy_hat
    kxJy = KZ * Jx_hat - KX * Jz_hat
    kxJz = KX * Jy_hat - KY * Jx_hat

    R_cut = min(box.Lx, box.Ly, box.Lz) / 2.0
    trunc = 1.0 - jnp.cos(jnp.sqrt(K2) * R_cut)

    Bx_hat = jnp.where(mask, (1j * mu0 * kxJx / K2) * trunc, 0.0)
    By_hat = jnp.where(mask, (1j * mu0 * kxJy / K2) * trunc, 0.0)
    Bz_hat = jnp.where(mask, (1j * mu0 * kxJz / K2) * trunc, 0.0)

    # Optionally damp the under-resolved high-|k| modes that carry the Gibbs
    # ringing from the current discontinuity. This filters the field, not the
    # source, so it removes no physical current (no current bias).
    filt = spectral_filter_array(box, spectral_filter, filter_order)
    if filt is not None:
        Bx_hat = Bx_hat * filt
        By_hat = By_hat * filt
        Bz_hat = Bz_hat * filt

    return Bx_hat / box.V, By_hat / box.V, Bz_hat / box.V


def eval_B(
    Bx_hat,
    By_hat,
    Bz_hat,
    target_pos,
    box,
    eps=1e-12,
):
    x = (target_pos[:, 0] + 0.5 * box.Lx) % box.Lx - 0.5 * box.Lx
    y = (target_pos[:, 1] + 0.5 * box.Ly) % box.Ly - 0.5 * box.Ly
    z = (target_pos[:, 2] + 0.5 * box.Lz) % box.Lz - 0.5 * box.Lz

    tx = 2.0 * jnp.pi * x / box.Lx
    ty = 2.0 * jnp.pi * y / box.Ly
    tz = 2.0 * jnp.pi * z / box.Lz
    
    Bx = nufft2(Bx_hat, tx, ty, tz, eps=eps, iflag=1).real
    By = nufft2(By_hat, tx, ty, tz, eps=eps, iflag=1).real
    Bz = nufft2(Bz_hat, tx, ty, tz, eps=eps, iflag=1).real

    return Bx, By, Bz

