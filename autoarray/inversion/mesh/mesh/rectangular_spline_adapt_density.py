import numpy as np
from typing import Optional, Tuple

from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    RectangularAdaptDensity,
    overlay_grid_from,
)
from autoarray.inversion.mesh.interpolator.rectangular_spline import (
    SPLINE_CDF_DEFAULT_DEG,
)
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class RectangularSplineAdaptDensity(RectangularAdaptDensity):
    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        spline_deg: int = SPLINE_CDF_DEFAULT_DEG,
    ):
        """
        Density-adaptive rectangular mesh using a polynomial + Hermite-spline
        CDF transform in place of the empirical-CDF linear-interp transform
        used by `RectangularAdaptDensity`.

        The spline CDF produces C¹-continuous gradients end-to-end (no
        piecewise-constant kinks at knot crossings) and avoids the
        ``1 / Δknot`` gradient-magnitude spike when two traced points crowd
        together. This is the gradient-robust variant intended for use with
        HMC / NUTS / variational samplers.

        Parameters
        ----------
        shape
            The 2D dimensions of the rectangular pixel grid
            ``(total_y_pixels, total_x_pixels)``.
        spline_deg
            Degree of the polynomial fit to the inverse CDF. Default 11
            (per RSE guidance). Higher degrees give smoother CDFs at the cost
            of more expensive polyfit SVDs.
        """
        super().__init__(shape=shape)
        self.spline_deg = int(spline_deg)

    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.rectangular_spline import (
            InterpolatorRectangularSpline,
        )

        return InterpolatorRectangularSpline

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """See ``RectangularAdaptDensity.interpolator_from``; forwards
        ``spline_deg`` to the spline interpolator."""
        relocated_grid = self.relocated_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            xp=xp,
        )

        mesh_grid = overlay_grid_from(
            shape_native=self.shape,
            grid=relocated_grid.over_sampled,
            xp=xp,
        )

        mesh_weight_map = self.mesh_weight_map_from(adapt_data=adapt_data, xp=xp)

        return self.interpolator_cls(
            mesh=self,
            data_grid=relocated_grid,
            mesh_grid=Grid2DIrregular(mesh_grid),
            mesh_weight_map=mesh_weight_map,
            adapt_data=adapt_data,
            spline_deg=self.spline_deg,
            xp=xp,
        )
