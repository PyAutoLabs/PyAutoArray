"""Sibson natural-neighbour interpolation on a Delaunay mesh."""

from autoarray.inversion.mesh.interpolator.sibson import (
    SIBSON_MAX_CAVITY_TRIANGLES,
    SIBSON_MAX_NEIGHBORS,
    SIBSON_QUERY_CHUNK,
)
from autoarray.inversion.mesh.mesh.delaunay import Delaunay


class DelaunayNN(Delaunay):
    """An irregular source mesh with Sibson natural-neighbour interpolation.

    This mesh has the same vertices, Delaunay connectivity, area calculation,
    and split-regularization support as :class:`Delaunay`. Its interpolation
    weights differ: every coordinate uses all natural neighbours in the local
    circumcircle cavity instead of only the three vertices of its containing
    triangle. The resulting interpolant is local and linearly precise, while
    remaining continuous when the Delaunay diagonal flips as a mass model
    moves the source-plane mesh.

    Under JAX, qhull supplies only integer connectivity via ``pure_callback``.
    The callback is stopped under differentiation because that connectivity is
    piecewise constant. Circumcircles, natural-neighbour weights, split points,
    and the inversion remain in JAX and are autodifferentiable with respect to
    the mesh and query coordinates. At an exactly degenerate topology the
    connectivity itself still has no derivative, but the Sibson interpolant
    approaches the same value through an ordinary diagonal flip.

    The fixed cavity and neighbour caps make mapper shapes static for JIT
    compilation. If either cap is exceeded, the affected weights become NaN so
    the sample is rejected instead of using a truncated interpolation. The
    class constants have headroom over the observed production Hilbert meshes.

    Parameters
    ----------
    pixels
        Number of active mesh vertices used by the inversion.
    zeroed_pixels
        Number of final boundary vertices fixed to zero.
    areas_factor
        Scale applied to the area-derived split-cross offsets.
    """

    max_cavity_triangles = SIBSON_MAX_CAVITY_TRIANGLES
    max_neighbors = SIBSON_MAX_NEIGHBORS
    query_chunk = SIBSON_QUERY_CHUNK

    def __init__(
        self,
        pixels: int,
        zeroed_pixels: int | None = 0,
        areas_factor: float = 0.5,
    ):
        super().__init__(
            pixels=pixels,
            zeroed_pixels=zeroed_pixels,
            areas_factor=areas_factor,
        )

    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.sibson import (
            InterpolatorDelaunayNN,
        )

        return InterpolatorDelaunayNN
