import numpy as np
from typing import Optional, Tuple

from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    RectangularAdaptDensity,
    overlay_grid_from,
)
from autoarray.inversion.mesh.interpolator.rectangular_kernel import (
    KERNEL_CDF_DEFAULT_BANDWIDTH,
    KERNEL_CDF_DEFAULT_KNOTS,
)
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class RectangularKernelAdaptDensity(RectangularAdaptDensity):
    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
        n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    ):
        """
        Density-adaptive rectangular mesh using a smooth kernel-density CDF
        transform in place of the empirical-CDF linear-interp transform used
        by `RectangularAdaptDensity`.

        The kernel CDF is strictly monotone by construction, C-infinity in
        both the interp queries and the traced-point positions (no ranks, no
        sorts), and duplicate-safe. It therefore carries correct smooth
        mass/shear gradients in every configuration — including imaging at
        pixelization over-sampling 1 and the interferometer sparse path,
        where the empirical CDF makes the likelihood exactly
        piecewise-constant.

        Parameters
        ----------
        shape
            The 2D dimensions of the rectangular pixel grid
            ``(total_y_pixels, total_x_pixels)``.
        bandwidth
            Kernel bandwidth in units of the mesh pixel scale
            (data span / mesh pixels per axis). Smaller values track the
            point density more sharply (approaching the empirical CDF);
            larger values smooth the mesh geometry towards uniform.
        n_knots
            Size of the fixed knot table used to invert the CDF.
        """
        super().__init__(shape=shape)
        self.bandwidth = float(bandwidth)
        self.n_knots = int(n_knots)

    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.rectangular_kernel import (
            InterpolatorRectangularKernel,
        )

        return InterpolatorRectangularKernel

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """See ``RectangularAdaptDensity.interpolator_from``; forwards
        ``bandwidth`` / ``n_knots`` to the kernel interpolator."""
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
            bandwidth=self.bandwidth,
            n_knots=self.n_knots,
            xp=xp,
        )
