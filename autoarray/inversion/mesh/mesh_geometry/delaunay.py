import numpy as np
from typing import Tuple

from autonerves import cached_property

from autoarray.geometry.geometry_2d_irregular import Geometry2DIrregular
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.mesh.mesh_geometry.abstract import AbstractMeshGeometry

from autoarray import exc


def voronoi_areas_numpy(points, qhull_options="Qbb Qc Qx Qm Q12 Pp"):
    """
    Compute Voronoi cell areas with a fully optimized pure-NumPy pipeline.
    Exact match to the per-cell SciPy Voronoi loop but much faster.
    """
    import scipy.spatial

    vor = scipy.spatial.Voronoi(points, qhull_options=qhull_options)

    vertices = vor.vertices
    point_region = vor.point_region
    regions = vor.regions
    N = len(point_region)

    # ------------------------------------------------------------
    # 1) Collect all region lists in one go (list comprehension is fast)
    # ------------------------------------------------------------
    region_lists = [regions[r] for r in point_region]

    # Precompute which regions are unbounded (vectorized test)
    unbounded = np.array([(-1 in r) for r in region_lists], dtype=bool)

    # Filter only bounded region vertex indices
    clean_regions = [
        np.asarray([v for v in r if v != -1], dtype=int) for r in region_lists
    ]

    # Compute lengths once
    lengths = np.array([len(r) for r in clean_regions], dtype=int)
    max_len = lengths.max()

    # ------------------------------------------------------------
    # 2) Build padded idx + mask in a vectorized-like way
    #
    # Instead of doing Python work inside the loop, we pre-pack
    # the flattened data and then reshape.
    # ------------------------------------------------------------
    idx = np.full((N, max_len), -1, dtype=int)
    mask = np.zeros((N, max_len), dtype=bool)

    # Single loop remaining: extremely cheap
    for i, (r, L) in enumerate(zip(clean_regions, lengths)):
        if L:
            idx[i, :L] = r
            mask[i, :L] = True

    # ------------------------------------------------------------
    # 3) Gather polygon vertices (vectorized)
    # ------------------------------------------------------------
    safe_idx = idx.clip(min=0)
    verts = vertices[safe_idx]  # (N, max_len, 2)

    # Extract x, y with masked invalid entries zeroed
    x = np.where(mask, verts[..., 1], 0.0)
    y = np.where(mask, verts[..., 0], 0.0)

    # ------------------------------------------------------------
    # 4) Vectorized "previous index" per polygon
    # ------------------------------------------------------------
    safe_lengths = np.where(lengths == 0, 1, lengths)
    j = np.arange(max_len)
    prev = (j[None, :] - 1) % safe_lengths[:, None]

    # Efficient take-along-axis
    x_prev = np.take_along_axis(x, prev, axis=1)
    y_prev = np.take_along_axis(y, prev, axis=1)

    # ------------------------------------------------------------
    # 5) Shoelace vectorized
    # ------------------------------------------------------------
    cross = x * y_prev - y * x_prev
    areas = 0.5 * np.abs(cross.sum(axis=1))

    # ------------------------------------------------------------
    # 6) Mark unbounded regions
    # ------------------------------------------------------------
    areas[unbounded] = -1.0

    return areas


class MeshGeometryDelaunay(AbstractMeshGeometry):

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        dual_areas=None,
        xp=np,
        **kwargs,
    ):
        """
        The geometry of a Delaunay triangulation / Voronoi mesh.

        Parameters
        ----------
        dual_areas
            The barycentric dual area of every mesh vertex, as computed by the
            interpolator that built this geometry (see
            `autoarray.inversion.mesh.interpolator.delaunay`). These are the
            quadrature weights `areas_for_magnification` returns. When the
            geometry is built standalone (e.g. in the tests) this is `None` and
            the dual areas are computed host-side on demand instead.
        """

        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            xp=xp,
            **kwargs,
        )

        self.dual_areas = dual_areas

    @cached_property
    def mesh_grid_xy(self):
        """
        The default convention in `scipy.spatial` is to represent 2D coordinates as (x,y) pairs, whereas PyAutoArray
        represents 2D coordinates as (y,x) pairs.

        Therefore, this property simply converts the (y,x) grid of irregular coordinates into an (x,y) grid.
        """
        return self._xp.stack(
            [self.mesh_grid.array[:, 0], self.mesh_grid.array[:, 1]]
        ).T

    @property
    def origin(self) -> Tuple[float, float]:
        """
        The (y,x) origin of the Voronoi grid, which is fixed to (0.0, 0.0) for simplicity.
        """
        return 0.0, 0.0

    @property
    def geometry(self):
        shape_native_scaled = (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

        scaled_maxima = (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

        scaled_minima = (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

        return Geometry2DIrregular(
            shape_native_scaled=shape_native_scaled,
            scaled_maxima=scaled_maxima,
            scaled_minima=scaled_minima,
        )

    @cached_property
    def neighbors(self) -> Neighbors:
        """
        Returns a ndarray describing the neighbors of every pixel in a Delaunay triangulation, where a neighbor is
        defined as two Delaunay triangles which are directly connected to one another in the triangulation.

        see `Neighbors` for a complete description of the neighboring scheme.

        The neighbors of a Voronoi mesh are computed using the `ridge_points` attribute of the scipy `Voronoi`
        object, as described in the method `voronoi_neighbors_from`.
        """
        import scipy.spatial

        delaunay = scipy.spatial.Delaunay(self.mesh_grid_xy)

        indptr, indices = delaunay.vertex_neighbor_vertices

        sizes = indptr[1:] - indptr[:-1]

        neighbors = -1 * np.ones(
            shape=(self.mesh_grid_xy.shape[0], int(np.max(sizes))), dtype="int"
        )

        for k in range(self.mesh_grid_xy.shape[0]):
            neighbors[k][0 : sizes[k]] = indices[indptr[k] : indptr[k + 1]]

        return Neighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @cached_property
    def voronoi(self) -> "scipy.spatial.Voronoi":
        """
        Returns a `scipy.spatial.Voronoi` object from the 2D (y,x) grid of irregular coordinates, which correspond to
        the centre of every Voronoi pixel.

        This object contains numerous attributes describing a Voronoi mesh. PyAutoArray uses
        the `vertex_neighbor_vertices` attribute to determine the neighbors of every Delaunay triangle.

        There are numerous exceptions that `scipy.spatial.Delaunay` may raise when the input grid of coordinates used
        to compute the Delaunay triangulation are ill posed. These exceptions are caught and combined into a single
        `MeshException`, which helps exception handling in the `inversion` package.
        """
        import scipy.spatial
        from scipy.spatial import QhullError

        try:
            return scipy.spatial.Voronoi(
                self.mesh_grid_xy,
                qhull_options="Qbb Qc Qx Qm",
            )
        except (ValueError, OverflowError, QhullError) as e:
            raise exc.MeshException() from e

    @property
    def voronoi_areas(self):
        return voronoi_areas_numpy(points=self.mesh_grid_xy)

    @property
    def areas_for_magnification(self) -> np.ndarray:
        """
        Returns the **barycentric dual area** of every pixel in the mesh: the sum of `triangle_area / 3` over the
        Delaunay triangles that touch that vertex.

        These are the exact quadrature weights of the mesh's piecewise-linear (barycentric) interpolant. The
        Delaunay mapper reconstructs the source as the linear interpolant through the vertex values `s_i`, and for
        that interpolant

            integral over the hull of f  =  sum_i s_i * dual_i

        holds exactly, not approximately -- the dual areas tile the convex hull of the mesh exactly (they sum to the
        hull area), so `sum(reconstruction * areas_for_magnification)` is the integral of the reconstructed source
        over the mesh.

        The value is taken from the interpolator, which computes it **in-graph** as part of the same Delaunay
        construction that builds the mappings (`scipy_delaunay` / `jax_delaunay` and their Matérn variants). It is
        therefore free here, and on the JAX path it is a traced value -- so this property is safe to evaluate inside
        a `jax.jit` (e.g. PyAutoLens' per-sample latent evaluation). When the geometry is constructed standalone,
        without an interpolator (`dual_areas is None`), the dual areas are computed host-side from a
        `scipy.spatial.Delaunay` of the mesh grid, which gives the identical answer.

        This is **not** `voronoi_areas`, which sums the shoelace area of each point's `scipy.spatial.Voronoi` cell.
        Those are correct arithmetic but the wrong quantity for this use: a Voronoi cell that is bounded but sits at
        the edge of the mesh extends out to the circumcentres of the outermost triangles instead of being clipped to
        the hull, so it can be orders of magnitude larger than that vertex's dual area. Weighting the reconstruction
        by them biased magnification by -13% to -99% across the audited configurations (PyAutoArray#522); the fix is
        PyAutoArray#524. `voronoi_areas` itself is unchanged and remains available for the geometric uses that
        genuinely want a Voronoi cell.
        """
        if self.dual_areas is not None:
            return self.dual_areas

        import scipy.spatial

        from autoarray.inversion.mesh.interpolator.delaunay import (
            barycentric_dual_area_from,
        )

        mesh_grid_xy = np.asarray(self.mesh_grid_xy)

        return barycentric_dual_area_from(
            mesh_grid_xy,
            scipy.spatial.Delaunay(mesh_grid_xy).simplices,
            xp=np,
        )
