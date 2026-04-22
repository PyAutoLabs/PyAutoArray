import numpy as np


class AbstractMeshGeometry:

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map=None,
        spline_deg=None,
        xp=np,
    ):

        self.mesh = mesh
        self.mesh_grid = mesh_grid
        self.data_grid = data_grid
        self.mesh_weight_map = mesh_weight_map
        # When non-None, rectangular geometry uses the spline-CDF helpers
        # instead of the linear-interp CDF (areas / edges transforms only).
        self.spline_deg = spline_deg
        self._xp = xp
