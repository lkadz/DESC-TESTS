from __future__ import annotations
import jax
import jax.numpy as jnp
import interpax
from functools import partial

@partial(jax.jit, static_argnames=["n_steps", "order"])
def trace_field_line_jax(
    Bx_grid: jnp.ndarray,
    By_grid: jnp.ndarray,
    Bz_grid: jnp.ndarray,
    x0: jnp.ndarray,
    Lx: float,
    Ly: float,
    Lz: float,
    ds: float = 0.05,
    n_steps: int = 4000,
    order: int = 1, 
):
    """
    JIT-compiled RK4 field line tracer using Interpax for high-order accuracy.
    
    Args:
        order: Interpolation order. 1=Linear, 3=Cubic.
    """
    # 1. Setup Grid Axes (Physical Coordinates)
    Nx, Ny, Nz = Bx_grid.shape
    xg = jnp.linspace(0, Lx, Nx, endpoint=False)
    yg = jnp.linspace(0, Ly, Ny, endpoint=False)
    zg = jnp.linspace(0, Lz, Nz, endpoint=False)
    
    # 2. Stack B-fields for single-pass interpolation: (Nx, Ny, Nz, 3)
    B_stack = jnp.stack([Bx_grid, By_grid, Bz_grid], axis=-1)
    
    # 3. Determine Method
    method = 'cubic' if order >= 3 else 'linear'
    period = (Lx, Ly, Lz) # Periodic boundaries for interpax

    def get_field(pos):
        # interp3d expects query points as 1D arrays
        # pos is (3,), so we slice to get (1,)
        xq = pos[0:1]
        yq = pos[1:2]
        zq = pos[2:3]
        
        # Interpolate vector field
        # Output shape will be (1, 3) -> squeeze to (3,)
        b_vec = interpax.interp3d(
            xq, yq, zq, 
            xg, yg, zg, 
            B_stack, 
            method=method, 
            period=period
        )
        return b_vec[0]

    def scan_body(curr_pos, _):
        # RK4 Integration Step
        b1 = get_field(curr_pos)
        n1 = jnp.linalg.norm(b1) + 1e-12
        k1 = (b1 / n1) * ds

        b2 = get_field(curr_pos + 0.5 * k1)
        n2 = jnp.linalg.norm(b2) + 1e-12
        k2 = (b2 / n2) * ds

        b3 = get_field(curr_pos + 0.5 * k2)
        n3 = jnp.linalg.norm(b3) + 1e-12
        k3 = (b3 / n3) * ds

        b4 = get_field(curr_pos + k3)
        n4 = jnp.linalg.norm(b4) + 1e-12
        k4 = (b4 / n4) * ds

        next_pos = curr_pos + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        
        return next_pos, next_pos

    _, trajectory = jax.lax.scan(scan_body, x0, None, length=n_steps)
    
    # Prepend start point
    return jnp.concatenate([x0[None, :], trajectory], axis=0)