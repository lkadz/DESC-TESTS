from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class BoxParams:
    Lx: float
    Ly: float
    Lz: float
    Nx: int
    Ny: int
    Nz: int

    def __post_init__(self):
        self.V = self.Lx * self.Ly * self.Lz
        self.N_total = self.Nx * self.Ny * self.Nz

        kx_int = jnp.arange(-self.Nx // 2, self.Nx // 2)
        ky_int = jnp.arange(-self.Ny // 2, self.Ny // 2)
        kz_int = jnp.arange(-self.Nz // 2, self.Nz // 2)

        self.kx = (2.0 * jnp.pi / self.Lx) * kx_int
        self.ky = (2.0 * jnp.pi / self.Ly) * ky_int
        self.kz = (2.0 * jnp.pi / self.Lz) * kz_int

        self.KX, self.KY, self.KZ = jnp.meshgrid(self.kx, self.ky, self.kz, indexing="ij")
