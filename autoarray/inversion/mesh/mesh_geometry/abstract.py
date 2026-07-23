import numpy as np


class AbstractMeshGeometry:

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map=None,
        kernel_bandwidth=None,
        kernel_knots=None,
        xp=np,
    ):

        self.mesh = mesh
        self.mesh_grid = mesh_grid
        self.data_grid = data_grid
        self.mesh_weight_map = mesh_weight_map
        # Kernel-density-CDF parameters for the rectangular geometry's
        # areas / edges transforms; None falls back to the kernel defaults.
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_knots = kernel_knots
        self._use_jax = xp is not np

    @property
    def _xp(self):
        if self._use_jax:
            import jax.numpy as jnp

            return jnp
        return np
