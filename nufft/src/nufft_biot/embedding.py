from __future__ import annotations
import jax.numpy as jnp

from .types import BoxParams


def make_optimal_box(
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    Z: jnp.ndarray, 
    n_cells: int = 256, 
    padding: float = 2.0
) -> BoxParams:
  
    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())
    zmin, zmax = float(Z.min()), float(Z.max())
    
    Lx_geom = xmax - xmin
    Ly_geom = ymax - ymin
    Lz_geom = zmax - zmin
    
    max_L = max(Lx_geom, Ly_geom, Lz_geom)
    
    L_final = max_L * padding
    
    print(f"Geometry Extent: [{Lx_geom:.2f}, {Ly_geom:.2f}, {Lz_geom:.2f}]")
    print(f"Auto-sized Box:  {L_final:.2f} (cubic)")
    
    return BoxParams(
        Lx=L_final,
        Ly=L_final,
        Lz=L_final,
        Nx=n_cells,
        Ny=n_cells,
        Nz=n_cells,
    )


def embed_geometry_in_box(X, Y, Z, box: BoxParams):
    xc = 0.5 * (X.min() + X.max())
    yc = 0.5 * (Y.min() + Y.max())
    zc = 0.5 * (Z.min() + Z.max())

    shift = jnp.array([xc, yc, zc], dtype=X.dtype)

    Xb = X - shift[0]
    Yb = Y - shift[1]
    Zb = Z - shift[2]

    return Xb, Yb, Zb, shift

