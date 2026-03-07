from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    RectangularAdaptDensity,
)


class RectangularUniform(RectangularAdaptDensity):
    """
    A uniform rectangular mesh of pixels used to reconstruct a source on a
    regular grid, with no adaptive weighting.

    The mesh geometry is fixed and defined by a 2D shape
    `(total_y_pixels, total_x_pixels)`. Pixels are indexed in row-major order:

        - Index 0 corresponds to the top-left pixel.
        - Indices increase left-to-right across rows and top-to-bottom
          between rows.

    Each source-plane coordinate is associated with the rectangular pixel
    in which it lies. No interpolation is performed — every coordinate
    contributes fully to a single pixel.

    Uniform behaviour
    -----------------
    Unlike `RectangularAdaptDensity` and `RectangularAdaptImage`, this mesh
    applies no adaptive weighting based on data density or an adapt image.
    All pixels are treated equally in the reconstruction, and the effective
    resolution is determined solely by the fixed mesh geometry and the
    observational constraints.

    This provides a simple and stable baseline reconstruction method,
    particularly useful for controlled experiments or when adaptive
    refinement is not required.

    Edge handling
    -------------
    Boundary (edge) pixels are automatically identified via the mesh
    neighbour structure and may be internally excluded (zeroed) during
    inversion to improve numerical stability and reduce edge artefacts.
    """

    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
            InterpolatorRectangularUniform,
        )

        return InterpolatorRectangularUniform
