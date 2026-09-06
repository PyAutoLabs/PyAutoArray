from typing import Tuple

from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
    RectangularRTUAdaptDensity,
)


class RectangularBilinearAdaptDensity(RectangularRTUAdaptDensity):

    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        respect_small_datasets: bool = True,
    ):
        """
        A rectangular mesh of pixels used to reconstruct a source on a regular
        grid, whose pixels adapt to the density of the traced source-plane
        coordinates through the empirical rank-CDF transform.

        The mesh is defined by a 2D shape `(total_y_pixels, total_x_pixels)` and
        is indexed in row-major order:

            - Index 0 corresponds to the top-left pixel.
            - Indices increase from left to right across each row,
              and from top to bottom across rows.

        Adaptive behaviour
        ------------------
        The mesh adapts to the spatial density of the traced points through the
        per-axis empirical (rank) CDF of the traced coordinates — a sort plus a
        cumulative sum, linearly interpolated between points: mesh pixels
        shrink where many coordinates land (e.g. regions of high magnification
        in gravitational lensing) and grow where sampling is sparse. The
        inversion therefore achieves higher effective resolution in these
        regions without changing the fixed rectangular topology.

        The transform is conceptually simple (no kernel hyperparameters) and
        costs O(N log N) per likelihood evaluation, making this the fast
        default rectangular mesh for CPU fitting.

        When to use Bilinear vs RTU
        ---------------------------
        The rank CDF depends on the traced-point positions through their
        ranks, so with pixelization over-sampling 1 the likelihood is exactly
        piecewise-constant in mass/shear — gradients are identically zero.
        Gradient-based (JAX) samplers therefore need
        `over_sample_size_pixelization >= 4` on imaging data, or the
        `RectangularRTUAdaptDensity` / `RectangularRTUAdaptImage` meshes
        (smooth kernel-CDF transform, correct gradients in every
        configuration). The interferometer sparse path has no over-sampling,
        so gradient work there must use the RTU meshes (or
        `RectangularUniform`). The RTU meshes are also the recommended option
        on GPU, where their kernel-CDF cost is not the bottleneck.

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

        Raises
        ------
        MeshException
            If either dimension is less than 3, as a minimum of 3×3 pixels
            is required to define interior and boundary structure.
        """
        super().__init__(
            shape=shape, respect_small_datasets=respect_small_datasets
        )

    @property
    def interpolator_kwargs(self) -> dict:
        """
        Extra keyword arguments `interpolator_from` forwards to
        `interpolator_cls` — the rank-CDF transform selector; the empirical
        rank CDF has no kernel hyperparameters.
        """
        return {"transform": "rank"}
