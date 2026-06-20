from desc.objectives import Objective
from desc.grid import LinearGrid
from desc.backend import jnp

class NUFFTBoundaryError(Objective):
    def __init__(self, eq, field, box, weight=1.0, grid=None, name="NUFFT Boundary Error"):
        # 1. Setup the evaluation grid (Boundary surface)
        if grid is None:
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=1.0)
        
        self.grid = grid
        self.field = field   # The external field (coils)
        self.box = box       # Your pre-defined NUFFT box parameters
        super().__init__(eq=eq, target=0, weight=weight, name=name)
        
    def build(self, eq, use_jit=True, verbose=1):
        # 2. Pre-compute things that don't change (like external field B_vac)
        #    We only need to re-compute B_plasma during optimization.
        
        # Get boundary coordinates for external field eval
        xyz = eq.compute("xyz", grid=self.grid)
        self.coords = jnp.stack([xyz["X"], xyz["Y"], xyz["Z"]], axis=1)
        
        # Compute B_external (Coils) once
        # (Assuming field.compute_magnetic_field exists and takes coords)
        self.B_vac = self.field.compute_magnetic_field(self.coords)
        
        return super().build(eq, use_jit, verbose)

    def compute(self, *args, **kwargs):
        # 3. This is the loop that runs every optimization step
        
        # A. Extract Volume Currents (J) and Coordinates (R, Z -> X, Y, Z)
        #    DESC calculates these from the current spectral coefficients
        params = self._parse_args(*args, **kwargs)
        data = self.things[0].compute(
            ["J", "X", "Y", "Z", "sqrt(g)", "n_rho"], 
            params=params, 
            grid=self.grid # Note: You might need a Volume grid for J, and Surface grid for B
        )
        
        # (CRITICAL NOTE: You actually need two grids: 
        #  - A Volume Grid to get J source
        #  - A Surface Grid (self.grid) to evaluate B target)
        #  For simplicity, let's assume you fetch volume data here similarly to desc_volume_current
        
        # B. Run NUFFT (Your custom code)
        #    This is pure JAX, so gradients propagate back to 'params'
        Bx_hat, By_hat, Bz_hat = compute_B_hat(
            data["X_vol"], data["Y_vol"], data["Z_vol"], 
            data["Jx"], data["Jy"], data["Jz"], data["w_vol"], 
            self.box
        )
        
        bx_n, by_n, bz_n = eval_B(
            Bx_hat, By_hat, Bz_hat, 
            self.coords, # Target: Boundary
            self.box
        )
        B_plasma = jnp.stack([bx_n, by_n, bz_n], axis=1)
        
        # C. Compute Total B
        B_total = B_plasma + self.B_vac
        
        # D. Project onto Surface Normal (n)
        #    Condition: B_total dot n = 0
        normal = data["n_rho"] # Surface normal vector
        Bn = jnp.sum(B_total * normal, axis=1)
        
        # E. Return the error (residuals)
        return Bn * self.weight