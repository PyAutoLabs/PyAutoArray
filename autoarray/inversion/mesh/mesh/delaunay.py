import numpy as np
from typing import Optional

from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class Delaunay(AbstractMesh):
    def __init__(
        self, pixels: int, zeroed_pixels: Optional[int] = 0, areas_factor: float = 0.5
    ):
        """
        A Delaunay mesh composed of irregular triangular pixels used to reconstruct
        a source on an unstructured grid.

        The mesh consists of `pixels` vertices in the source plane, which are
        connected via a Delaunay triangulation to form triangular elements.
        Each vertex represents a linear parameter in the inversion.

        Source-plane coordinates are interpolated onto this mesh using barycentric
        interpolation within the enclosing triangle. For each coordinate, the three
        vertices of the containing Delaunay triangle are identified and weighted
        according to their barycentric distances, providing a smooth, piecewise-linear
        reconstruction.

        **JAX & gradient support** (2026-07-26, FD-certified): the likelihood
        runs under ``jax.jit`` and is differentiable — the host-called qhull
        ``pure_callback`` returns only integer connectivity tables (frozen
        under differentiation via ``stop_gradient``; their true derivative is
        zero between re-wiring events), while point location, barycentric
        weights, dual areas and split points are computed in-graph from the
        traced vertices, so ``jax.grad`` returns the exact almost-everywhere
        derivative.

        In what sense "autodifferentiable"? The same sense as a ReLU network:
        the likelihood is piecewise-smooth — perfectly smooth within each
        triangulation topology, with measure-zero jump discontinuities at the
        triangle-flip (re-wiring) boundaries, where the barycentric interpolant
        jumps and has no gradient. A sampler almost surely never lands on a seam, and on
        either side autodiff returns the exact gradient of the branch the
        evaluation point is on (FD comparisons show the seams, autodiff does
        not — individual finite-difference steps can straddle a flip). This
        contrasts the adaptive rectangular (kernel-CDF) meshes, which are
        C-infinity by construction with no seams at all. ``DelaunayNN`` is the
        local, linearly precise alternative when continuity through Delaunay
        flips is required: its Sibson natural-neighbour weights approach the
        same value from either triangulation.

        Caveat for batched samplers: the callback is
        ``vmap_method="sequential"`` (one host qhull call per vmap lane) —
        the ``KNearestNeighbor`` / ``KNNBarycentric`` subclasses avoid the
        callback entirely and remain the batched-throughput option.

        Zeroed pixels
        -------------
        The `zeroed_pixels` parameter specifies a number of mesh vertices that are
        **excluded from the inversion**. These pixels are intended to correspond to
        *edge or boundary vertices* of the Delaunay mesh.

        Zeroing edge pixels helps to:
          - stabilize the linear inversion,
          - prevent poorly constrained boundary vertices from absorbing flux,
          - reduce edge artefacts in the reconstructed source.

        Zeroed pixels are always placed at the **end of the mesh parameter vector**
        and are not solved for; their values are fixed to zero. Internally, the
        inversion accounts for these excluded parameters when constructing and
        solving the linear system.

        Parameters
        ----------
        pixels : int
            The number of active mesh vertices (linear parameters) used to represent
            the source reconstruction.
        areas_factor : float, optional
            The barycentric area of Delaunay triangles is used to weight the regularization matrix.
            This factor scales these areas, allowing for tuning of the regularization strength
            based on triangle size.
        zeroed_pixels : int, optional
            The number of edge mesh vertices to exclude from the inversion. These
            are appended to the end of the mesh and fixed to zero.
        """

        pixels = int(pixels) + zeroed_pixels

        super().__init__()
        self.pixels = pixels
        self.areas_factor = areas_factor
        self._zeroed_pixels = zeroed_pixels

    @property
    def zeroed_pixels(self):
        """
        Return the **positive** mesh-local pixel indices to zero for a Delaunay mesh.

        For Delaunay meshes, `self.zeroed_pixels` is interpreted as a *count* of pixels
        to be zeroed at the end of the pixel block. For example:
            self.pixels = 780, self.zeroed_pixels = 30
        returns indices 750..779.

        Returns
        -------
        np.ndarray
            1D array of positive pixel indices to zero.
        """
        if self._zeroed_pixels <= 0:
            return np.array([], dtype=int)

        start = self.pixels - self._zeroed_pixels
        return np.arange(start, self.pixels, dtype=int)

    @property
    def skip_areas(self):
        """
        Whether to skip barycentric  area calculations and split point computations during Delaunay triangulation.
        When True, the Delaunay interface returns only the minimal set of outputs (points, simplices, mappings)
        without computing split_points or splitted_mappings. This optimization is useful for regularization
        schemes like Matérn kernels that don't require area-based calculations. Default is False.
        """
        return False

    @property
    def interpolator_cls(self):

        from autoarray.inversion.mesh.interpolator.delaunay import (
            InterpolatorDelaunay,
        )

        return InterpolatorDelaunay

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """
        Mapper objects describe the mappings between pixels in the masked 2D data and the pixels in a mesh,
        in both the `data` and `source` frames.

        This function returns a `MapperDelaunay` as follows:

        1) Before this routine is called, a sparse grid of (y,x) coordinates are computed from the 2D masked data,
           the `image_plane_mesh_grid`, which acts as the Delaunay triangle vertexes of the mesh and mapper.

        2) Before this routine is called, operations are performed on this `image_plane_mesh_grid` that transform it
           from a 2D grid which overlaps with the 2D mask of the data in the `data` frame to an irregular grid in
           the `source` frame, the `source_plane_mesh_grid`.

        3) If the border relocator is input, the border of the input `source_plane_data_grid` is used to relocate all of the
           grid's (y,x) coordinates beyond the border to the edge of the border.

        4) If the border relocatiro is input, the border of the input `source_plane_data_grid` is used to relocate all of the
           transformed `source_plane_mesh_grid`'s (y,x) coordinates beyond the border to the edge of the border.

        5) Use the transformed `source_plane_mesh_grid`'s (y,x) coordinates as the Vertex of the Delaunay mesh.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The vertex of every Delaunay triangle pixel in the `source` frame, which are initially derived by
            computing a sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame and
            applying a transformation to this.
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_plane_mesh_grid`.
        adapt_data
            Not used for a rectangular mesh.
        """
        relocated_grid = self.relocated_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            xp=xp,
        )

        relocated_mesh_grid = self.relocated_mesh_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            xp=xp,
        )

        return self.interpolator_cls(
            mesh=self,
            data_grid=relocated_grid,
            mesh_grid=relocated_mesh_grid,
            adapt_data=adapt_data,
            xp=xp,
        )
