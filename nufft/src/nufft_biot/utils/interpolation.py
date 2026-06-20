from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class PeriodicFieldInterpolator:
    def __init__(self, Bx, By, Bz, box, method="cubic"): 
        self.Lx = box.Lx
        self.Ly = box.Ly
        self.Lz = box.Lz

        self.xg = np.linspace(0.0, box.Lx, box.Nx, endpoint=False)
        self.yg = np.linspace(0.0, box.Ly, box.Ny, endpoint=False)
        self.zg = np.linspace(0.0, box.Lz, box.Nz, endpoint=False)

        self._Bx = RegularGridInterpolator(
            (self.xg, self.yg, self.zg),
            Bx,
            bounds_error=False,
            fill_value=None,
            method=method, 
        )
        self._By = RegularGridInterpolator(
            (self.xg, self.yg, self.zg),
            By,
            bounds_error=False,
            fill_value=None,
            method=method,
        )
        self._Bz = RegularGridInterpolator(
            (self.xg, self.yg, self.zg),
            Bz,
            bounds_error=False,
            fill_value=None,
            method=method,
        )

    def __call__(self, pos):
        p = np.asarray(pos, dtype=float)

        p[0] %= self.Lx
        p[1] %= self.Ly
        p[2] %= self.Lz

        p2 = p[None, :]

        bx = self._Bx(p2)[0]
        by = self._By(p2)[0]
        bz = self._Bz(p2)[0]

        return np.array([bx, by, bz], dtype=float)