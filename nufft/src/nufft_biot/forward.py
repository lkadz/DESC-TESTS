from __future__ import annotations
import jax.numpy as jnp

from .types import BoxParams
from .field import compute_B_hat, eval_B
from .current_models import torus_volume_current, torus_axis_filament_current
from .embedding import embed_geometry_in_box


def forward_B(
    x: jnp.ndarray,
    box: BoxParams,
    *,
    I: float,
    major_radius: float,
    minor_radius: float,
    N_rho: int = 8,
    N_theta: int = 16,
    N_zeta: int = 32,
    current_model: str = "volume",
    desc_eq=None,
    spectral_filter=None,
    filter_order: int = 8,
):
    if current_model == "desc":
        from .desc_interface import desc_volume_current
        if desc_eq is None:
            raise ValueError("Must provide desc_eq")
        X, Y, Z, Jx, Jy, Jz, w = desc_volume_current(desc_eq)
    elif current_model == "volume":
        X, Y, Z, Jx, Jy, Jz, w = torus_volume_current(
            I=I,
            major_radius=major_radius,
            minor_radius=minor_radius,
            N_rho=N_rho,
            N_theta=N_theta,
            N_zeta=N_zeta,
        )
    elif current_model == "filament":
        X, Y, Z, Jx, Jy, Jz, w = torus_axis_filament_current(
            I=I,
            major_radius=major_radius,
            N_zeta=N_zeta,
        )
    else:
        raise ValueError(current_model)

    Xb, Yb, Zb, center = embed_geometry_in_box(X, Y, Z, box)

    Bx_hat, By_hat, Bz_hat = compute_B_hat(
        Xb, Yb, Zb, Jx, Jy, Jz, w, box,
        spectral_filter=spectral_filter, filter_order=filter_order,
    )

    target_pos = jnp.stack([Xb, Yb, Zb], axis=1)

    Bx, By, Bz = eval_B(
        Bx_hat, By_hat, Bz_hat, target_pos, box
    )

    return Bx, By, Bz, center
