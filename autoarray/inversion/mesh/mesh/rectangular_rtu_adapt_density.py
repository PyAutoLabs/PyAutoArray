import numpy as np
from typing import Optional, Tuple

from autoarray.settings import Settings
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.inversion.mesh.border_relocator import BorderRelocator

from autoarray.structures.grids import grid_2d_util
from autoarray.util.dataset_util import cap_mesh_shape_for_small_datasets

from autoarray import exc


def overlay_grid_from(
    shape_native: Tuple[int, int],
    grid: np.ndarray,
    buffer: float = 1e-8,
    xp=np,
) -> np.ndarray:
    """
    Creates a `Grid2DRecntagular` by overlaying the rectangular pixelization over an input grid of (y,x)
    coordinates.

    This is performed by first computing the minimum and maximum y and x coordinates of the input grid. A
    rectangular pixelization with dimensions `shape_native` is then laid over the grid using these coordinates,
    such that the extreme edges of this rectangular pixelization overlap these maximum and minimum (y,x) coordinates.

    A a `buffer` can be included which increases the size of the rectangular pixelization, placing additional
    spacing beyond these maximum and minimum coordinates.

    Parameters
    ----------
    shape_native
        The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
    grid
        A grid of (y,x) coordinates which the rectangular pixelization is laid-over.
    buffer
        The size of the extra spacing placed between the edges of the rectangular pixelization and input grid.
    """
    grid = grid.array

    y_min = xp.min(grid[:, 0]) - buffer
    y_max = xp.max(grid[:, 0]) + buffer
    x_min = xp.min(grid[:, 1]) - buffer
    x_max = xp.max(grid[:, 1]) + buffer

    pixel_scales = xp.array(
        (
            (y_max - y_min) / shape_native[0],
            (x_max - x_min) / shape_native[1],
        )
    )
    origin = xp.array(((y_max + y_min) / 2.0, (x_max + x_min) / 2.0))

    grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_not_mask_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin, xp=xp
    )

    return grid_slim


class RectangularRTUAdaptDensity(AbstractMesh):
    # Rectangular meshes do not support split regularization -- their interpolators provide no
    # split-cross mappings. Inherited by `RectangularUniform` and `RectangularRTUAdaptImage`.
    supports_split_regularization = False

    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        bandwidth: Optional[float] = None,
        n_knots: Optional[int] = None,
        respect_small_datasets: bool = True,
    ):
        """
        A rectangular mesh of pixels used to reconstruct a source on a regular
        grid, whose pixels adapt to the density of the traced source-plane
        coordinates through the smooth RTU kernel-density CDF transform
        (Enzi et al., arXiv:2606.30620).

        The mesh is defined by a 2D shape `(total_y_pixels, total_x_pixels)` and
        is indexed in row-major order:

            - Index 0 corresponds to the top-left pixel.
            - Indices increase from left to right across each row,
              and from top to bottom across rows.

        When to use RTU vs Bilinear
        ---------------------------
        The RTU meshes are the recommended advanced option: on GPU, for JAX
        gradient-based samplers, and on the interferometer sparse path (where
        they are the only adaptive rectangular meshes with usable gradients).
        Their kernel-CDF evaluation is an O(M_sub x N_data) erf sum that
        dominates the CPU likelihood at production scale, so for CPU-only
        fitting the `RectangularBilinearAdaptDensity` /
        `RectangularBilinearAdaptImage` meshes (empirical rank-CDF transform,
        no hyperparameters) are the faster default.

        Adaptive behaviour
        ------------------
        The mesh adapts to the spatial density of the traced points through a
        smooth per-axis kernel-density CDF transform: mesh pixels shrink where
        many coordinates land (e.g. regions of high magnification in
        gravitational lensing) and grow where sampling is sparse. The
        inversion therefore achieves higher effective resolution in these
        regions without changing the fixed rectangular topology.

        The kernel CDF is strictly monotone by construction, C-infinity in
        both the interp queries and the traced-point positions (no ranks, no
        sorts), and duplicate-safe. It therefore carries correct smooth
        mass/shear gradients in every configuration — including imaging at
        pixelization over-sampling 1 and the interferometer sparse path.

        Edge handling
        -------------
        Boundary (edge) pixels are automatically identified through the mesh
        neighbour structure. These edge pixels may be internally excluded
        (zeroed) during inversion to improve numerical stability and reduce
        edge artefacts. This zeroing is determined by the mesh connectivity
        and does not require manual specification of boundary indices.

        Parameters
        ----------
        shape : Tuple[int, int]
            The 2D dimensions of the rectangular pixel grid
            `(total_y_pixels, total_x_pixels)`.
        bandwidth
            Kernel bandwidth in units of the mesh pixel scale
            (data span / mesh pixels per axis). Smaller values track the
            point density more sharply; larger values smooth the mesh
            geometry towards uniform. Defaults to the kernel default.
        n_knots
            Size of the fixed knot table used to invert the CDF. Defaults to
            the kernel default.
        respect_small_datasets
            When ``PYAUTO_SMALL_DATASETS=1`` is set, `shape` is capped per axis
            to the small-datasets cap, matching the cap `Grid2D.uniform` and
            `Mask2D.circular` apply to the data. Pass ``False`` to opt out for a
            mesh whose resolution is load-bearing for the script.

        Raises
        ------
        MeshException
            If either dimension is less than 3, as a minimum of 3×3 pixels
            is required to define interior and boundary structure.
        """
        from autoarray.inversion.mesh.interpolator.rectangular import (
            KERNEL_CDF_DEFAULT_BANDWIDTH,
            KERNEL_CDF_DEFAULT_KNOTS,
        )

        shape = cap_mesh_shape_for_small_datasets(
            shape, respect_small_datasets=respect_small_datasets
        )

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.MeshException(
                "The rectangular pixelization must be at least dimensions 3x3"
            )

        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        self.bandwidth = float(
            bandwidth if bandwidth is not None else KERNEL_CDF_DEFAULT_BANDWIDTH
        )
        self.n_knots = int(n_knots if n_knots is not None else KERNEL_CDF_DEFAULT_KNOTS)
        super().__init__()

    @property
    def zeroed_pixels(self):
        """
        Return the **positive** 1D pixel indices of the edge pixels in a rectangular mesh.

        Indices are in row-major (C-order) flattened form for the rectangular pixel grid:
            - 0 corresponds to the top-left pixel (row=0, col=0)
            - indices increase across rows

        These indices are defined purely within the rectangular mesh's pixel indexing
        scheme (size = rows * cols) and are intended to be shifted / mapped to the full
        inversion indexing inside the inversion logic.

        Returns
        -------
        np.ndarray
            A 1D array of positive indices corresponding to edge pixels.
        """
        from autoarray.inversion.mesh.mesh_geometry.rectangular import (
            rectangular_edge_pixel_list_from,
        )

        edge_pixel_list = rectangular_edge_pixel_list_from(shape_native=self.shape)

        return np.array(edge_pixel_list, dtype=int)

    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.rectangular import (
            InterpolatorRectangular,
        )

        return InterpolatorRectangular

    @property
    def interpolator_kwargs(self) -> dict:
        """
        Extra keyword arguments `interpolator_from` forwards to
        `interpolator_cls` — the kernel-CDF parameters for the adaptive
        meshes; overridden to `{}` by `RectangularUniform`, whose
        interpolator takes no kernel arguments.
        """
        return {"bandwidth": self.bandwidth, "n_knots": self.n_knots}

    def mesh_weight_map_from(self, adapt_data, xp=np) -> np.ndarray:
        """
        The weight map of a rectangular pixelization is None, because magnificaiton adaption uses
        the distribution and density of traced (y,x) coordinates in the source plane and
        not weights or the adapt data.

        Parameters
        ----------
        xp
            The array library to use.
        """
        return None

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """
        Mapper objects describe the mappings between pixels in the masked 2D data and the pixels in a pixelization,
        in both the `data` and `source` frames.

        This function returns a `MapperRectangular` as follows:

        1) If the bordr relocator is input, the border of the input `source_plane_data_grid` is used to relocate all of the
           grid's (y,x) coordinates beyond the border to the edge of the border.

        2) Determine the (y,x) coordinates of the pixelization's rectangular pixels, by laying this rectangular grid
           over the 2D grid of relocated (y,x) coordinates computed in step 1 (or the input `source_plane_data_grid` if step 1
           is bypassed).

        3) Return the `MapperRectangular`.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_plane_data_grid` with the rectangular pixelization.
        image_plane_mesh_grid
            Not used for a rectangular pixelization.
        adapt_data
            Not used for a rectangular pixelization.
        """
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
            **self.interpolator_kwargs,
            xp=xp,
        )
