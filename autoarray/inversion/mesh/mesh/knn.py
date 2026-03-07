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
