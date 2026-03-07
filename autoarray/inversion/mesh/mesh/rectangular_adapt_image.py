import numpy as np
from typing import Tuple

from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    RectangularAdaptDensity,
)


class RectangularAdaptImage(RectangularAdaptDensity):

    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        weight_power: float = 1.0,
        weight_floor: float = 0.0,
    ):
        """
        A uniform rectangular mesh of pixels used to reconstruct a source on a
        regular grid, with adaptive weighting driven by an external adapt image.

        The mesh geometry is fixed and defined by a 2D shape
        `(total_y_pixels, total_x_pixels)`. Pixels are indexed in row-major order:

            - Index 0 corresponds to the top-left pixel.
            - Indices increase left-to-right across rows and top-to-bottom
              between rows.

        Each source-plane coordinate is associated with the rectangular pixel
        in which it lies. No interpolation is performed — every coordinate
        contributes fully to a single pixel.

        Adaptive behaviour (adapt image)
        --------------------------------
        Unlike a purely density-based rectangular mesh, this class adapts the
        effective reconstruction using an *adapt image*. The adapt image provides
        weights that emphasise specific regions of the source plane, typically
        bright regions of a previously estimated reconstruction.

        Pixels corresponding to higher adapt-image intensity receive increased
        weighting, allowing the inversion to prioritise reconstructing structure
        in bright regions of the source. This leads to:

          - improved resolution in high-signal regions,
          - smoother behaviour in faint regions,
          - reduced overfitting of noise in low-signal areas.

        The weighting applied to each pixel is controlled by:

          - `weight_power`: raises the adapt-image values to a power, increasing
            or decreasing contrast between bright and faint regions.
          - `weight_floor`: sets a minimum weight to prevent pixels in very faint
            regions from becoming unconstrained.

        This approach is particularly effective in strong gravitational lensing,
        where the adapt image typically traces the intrinsic brightness
        distribution of the source.

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

        super().__init__(shape=shape)

        self.weight_power = weight_power
        self.weight_floor = weight_floor

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
        mesh_weight_map = adapt_data.array
        mesh_weight_map = xp.clip(mesh_weight_map, 1e-12, None)
        mesh_weight_map = mesh_weight_map**self.weight_power

        # Apply floor using xp.where (safe for NumPy and JAX)
        mesh_weight_map = xp.where(
            mesh_weight_map < self.weight_floor,
            self.weight_floor,
            mesh_weight_map,
        )

        # Normalize
        mesh_weight_map = mesh_weight_map / xp.sum(mesh_weight_map)

        return mesh_weight_map
