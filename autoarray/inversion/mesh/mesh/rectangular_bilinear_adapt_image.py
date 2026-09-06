from typing import Tuple

from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_image import (
    RectangularRTUAdaptImage,
)


class RectangularBilinearAdaptImage(RectangularRTUAdaptImage):

    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        weight_power: float = 1.0,
        weight_floor: float = 0.0,
        respect_small_datasets: bool = True,
    ):
        """
        A rectangular mesh of pixels used to reconstruct a source on a regular
        grid, with adaptive weighting driven by an external adapt image and
        the empirical rank-CDF transform.

        The mesh geometry is fixed and defined by a 2D shape
        `(total_y_pixels, total_x_pixels)`. Pixels are indexed in row-major order:

            - Index 0 corresponds to the top-left pixel.
            - Indices increase left-to-right across rows and top-to-bottom
              between rows.

        Adaptive behaviour (adapt image)
        --------------------------------
        Like `RectangularRTUAdaptImage`, this mesh adapts using an *adapt
        image*: weights that emphasise specific regions of the source plane,
        typically bright regions of a previously estimated reconstruction.
        Pixels corresponding to higher adapt-image intensity receive increased
        weighting, controlled by `weight_power` and `weight_floor` (see
        `RectangularRTUAdaptImage` for the full description).

        The weighted lattice transform is the per-axis empirical rank CDF of
        the traced coordinates (a sort plus a weighted cumulative sum) — no
        kernel hyperparameters and O(N log N) per likelihood evaluation,
        making this the fast default adaptive rectangular mesh for CPU
        fitting.

        When to use Bilinear vs RTU
        ---------------------------
        See `RectangularBilinearAdaptDensity`: gradient-based (JAX) samplers
        need `over_sample_size_pixelization >= 4` on imaging data, or the RTU
        meshes; interferometer gradient work must use the RTU meshes, which
        are also the recommended option on GPU.

        Edge handling
        -------------
        Boundary (edge) pixels are automatically identified via the mesh
        neighbour structure and may be internally excluded (zeroed) during
        inversion to improve numerical stability and reduce edge artefacts.

        Parameters
        ----------
        shape : Tuple[int, int]
            The 2D dimensions of the rectangular pixel grid
            `(total_y_pixels, total_x_pixels)`.
        weight_power : float, optional
            Exponent applied to the adapt-image weights to control the strength
            of adaptivity.
        weight_floor : float, optional
            Minimum weight applied to ensure numerical stability in low-intensity
            regions.
        """
        super().__init__(
            shape=shape,
            weight_power=weight_power,
            weight_floor=weight_floor,
            respect_small_datasets=respect_small_datasets,
        )

    @property
    def interpolator_kwargs(self) -> dict:
        """
        Extra keyword arguments `interpolator_from` forwards to
        `interpolator_cls` — the rank-CDF transform selector; the empirical
        rank CDF has no kernel hyperparameters.
        """
        return {"transform": "rank"}
