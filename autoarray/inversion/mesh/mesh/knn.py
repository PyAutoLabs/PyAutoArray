from typing import Optional

from autoarray.inversion.mesh.mesh.delaunay import Delaunay


class KNearestNeighbor(Delaunay):

    def __init__(
        self,
        pixels: int,
        zeroed_pixels: Optional[int] = 0,
        k_neighbors=10,
        radius_scale=1.5,
        areas_factor=0.5,
        split_neighbor_division=2,
    ):
        """
        A mesh that defines pixel connectivity using a k-nearest-neighbour
        scheme rather than explicit triangle adjacency.

        This mesh inherits the Delaunay geometry but does not use its interpolation scheme
        but instead interpolates  by connecting each mesh vertex to its
        `k_neighbors` nearest neighbouring vertices. These neighbour relationships are
        used to impose smoothness constraints on the reconstructed source.

        Neighbour connections may be further restricted using a distance-based criterion,
        and optionally subdivided to improve stability for highly irregular meshes.

        **JAX & gradient support** (2026-07, FD-certified): the kNN
        interpolation is pure JAX (blocked brute-force ``lax.top_k`` +
        Wendland weights — no scipy callback), so the full likelihood is
        differentiable, with gradients flowing through both the traced query
        points and the traced mesh vertices. (The parent ``Delaunay`` mesh
        is also differentiable as of 2026-07-26 via its frozen integer
        tables; this mesh's remaining edge is batched throughput — no
        per-vmap-lane host callback.) Pair it with a split-family
        regularization (``ConstantSplit`` / ``AdaptSplit``) or a kernel
        scheme — the neighbor-based schemes (``Constant`` / ``Adapt``) call
        scipy on the traced mesh grid and cannot differentiate.

        Parameters
        ----------
        pixels : int
            The number of active mesh vertices (linear parameters) used to represent
            the source reconstruction.
        zeroed_pixels : int, optional
            The number of edge mesh vertices to exclude from the inversion. These
            boundary pixels are appended to the end of the parameter vector and
            fixed to zero to reduce edge artefacts.
        k_neighbors : int, optional
            The number of nearest neighbours used to define connectivity for each
            mesh vertex when constructing the regularization matrix.
        radius_scale : float, optional
            A multiplicative factor applied to the characteristic neighbour distance
            that limits which neighbours are included. This prevents distant vertices
            from contributing to regularization in sparsely sampled regions.
        areas_factor : float, optional
            The barycentric area of Delaunay triangles is used to weight the
            regularization matrix. This factor scales these areas, allowing the
            regularization strength to be tuned based on local triangle size.
        split_neighbor_division : int, optional
            Controls how neighbour connections are subdivided when forming the
            regularization operator, improving numerical stability for irregular
            point distributions.
        """

        self.k_neighbors = k_neighbors
        self.radius_scale = radius_scale
        self.areas_factor = areas_factor
        self.split_neighbor_division = split_neighbor_division

        super().__init__(pixels=pixels, zeroed_pixels=zeroed_pixels)

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

        from autoarray.inversion.mesh.interpolator.knn import (
            InterpolatorKNearestNeighbor,
        )

        return InterpolatorKNearestNeighbor


class KNNBarycentric(KNearestNeighbor):
    """
    A mesh that inherits k-nearest-neighbour connectivity from
    :class:`KNearestNeighbor` but uses :class:`InterpolatorKNNBarycentric` to
    compute interpolation weights as locally-exact barycentric coordinates on
    the triangle formed by the 3 nearest source-plane mesh vertices, rather
    than a Wendland kernel.

    This is a JAX-native approximation to Delaunay barycentric interpolation
    that avoids the scipy.spatial.Delaunay callback. The kNN connectivity knobs
    (``k_neighbors``, ``radius_scale``, ``split_neighbor_division``) are
    inherited and still control the regularization-spacing computation, but the
    *interpolation* weights always use k=3 + barycentric and ignore them.
    Gradients are FD-certified like the parent (2026-07), but this mesh
    FAILED its science gate as a Delaunay replacement (PyAutoArray#317 —
    ~5% of vertices are never any query's nearest-3, drifting the
    log-evidence by ~2%): use it for gradient experiments, not production
    science.
    """

    @property
    def interpolator_cls(self):

        from autoarray.inversion.mesh.interpolator.knn import (
            InterpolatorKNNBarycentric,
        )

        return InterpolatorKNNBarycentric
