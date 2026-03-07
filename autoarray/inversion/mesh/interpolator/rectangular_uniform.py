from typing import Tuple
import numpy as np

from autoconf import cached_property

from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator
from autoarray.geometry.geometry_2d import Geometry2D


def rectangular_mappings_weights_via_interpolation_from(
    shape_native: Tuple[int, int],
    data_grid: np.ndarray,
    mesh_grid: np.ndarray,
    xp=np,
):
    """
    Compute bilinear interpolation weights and corresponding rectangular mesh indices for an irregular grid.

    Given a flattened regular rectangular mesh grid and an irregular grid of data points, this function
    determines for each irregular point:
    - the indices of the 4 nearest rectangular mesh pixels (top-left, top-right, bottom-left, bottom-right), and
    - the bilinear interpolation weights with respect to those pixels.

    The function supports JAX and is compatible with JIT compilation.

    Parameters
    ----------
    shape_native
        The shape (Ny, Nx) of the original rectangular mesh grid before flattening.
    data_grid
        The irregular grid of (y, x) points to interpolate.
    mesh_grid
        The flattened regular rectangular mesh grid of (y, x) coordinates.

    Returns
    -------
    mappings : np.ndarray of shape (N, 4)
        Indices of the four nearest rectangular mesh pixels in the flattened mesh grid.
        Order is: top-left, top-right, bottom-left, bottom-right.
    weights : np.ndarray of shape (N, 4)
        Bilinear interpolation weights corresponding to the four nearest mesh pixels.

    Notes
    -----
    - Assumes the mesh grid is uniformly spaced.
    - The weights sum to 1 for each irregular point.
    - Uses bilinear interpolation in the (y, x) coordinate system.
    """
    mesh_grid = mesh_grid.reshape(*shape_native, 2)

    # Assume mesh is shaped (Ny, Nx, 2)
    Ny, Nx = mesh_grid.shape[:2]

    # Get mesh spacings and lower corner
    y_coords = mesh_grid[:, 0, 0]  # shape (Ny,)
    x_coords = mesh_grid[0, :, 1]  # shape (Nx,)

    dy = y_coords[1] - y_coords[0]
    dx = x_coords[1] - x_coords[0]

    y_min = y_coords[0]
    x_min = x_coords[0]

    # shape (N_irregular, 2)
    irregular = data_grid

    # Compute normalized mesh coordinates (floating indices)
    fy = (irregular[:, 0] - y_min) / dy
    fx = (irregular[:, 1] - x_min) / dx

    # Integer indices of top-left corners
    ix = xp.floor(fx).astype(xp.int32)
    iy = xp.floor(fy).astype(xp.int32)

    # Clip to stay within bounds
    ix = xp.clip(ix, 0, Nx - 2)
    iy = xp.clip(iy, 0, Ny - 2)

    # Local coordinates inside the cell (0 <= tx, ty <= 1)
    tx = fx - ix
    ty = fy - iy

    # Bilinear weights
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    weights = xp.stack([w00, w10, w01, w11], axis=1)  # shape (N_irregular, 4)

    # Compute indices of 4 surrounding pixels in the flattened mesh
    i00 = iy * Nx + ix
    i10 = iy * Nx + (ix + 1)
    i01 = (iy + 1) * Nx + ix
    i11 = (iy + 1) * Nx + (ix + 1)

    mappings = xp.stack([i00, i10, i01, i11], axis=1)  # shape (N_irregular, 4)

    return mappings, weights


class InterpolatorRectangularUniform(AbstractInterpolator):

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        adapt_data: np.ndarray = None,
        mesh_weight_map: np.ndarray = None,
        xp=np,
    ):
        """
        A grid of (y,x) coordinates which represent a uniform rectangular pixelization.

        A `InterpolatorRectangular` is ordered such pixels begin from the top-row and go rightwards and then downwards.
        It is an ndarray of shape [total_pixels, 2], where the first dimension of the ndarray corresponds to the
        pixelization's pixel index and second element whether it is a y or x arc-second coordinate.

        For example:

        - grid[3,0] = the y-coordinate of the 4th pixel in the rectangular pixelization.
        - grid[6,1] = the x-coordinate of the 7th pixel in the rectangular pixelization.

        This class is used in conjuction with the `inversion/pixelizations` package to create rectangular pixelizations
        and mappers that perform an `Inversion`.

        Parameters
        ----------
        values
            The grid of (y,x) coordinates corresponding to the centres of each pixel in the rectangular pixelization.
        shape_native
            The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) origin of the pixelization.
        """
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            adapt_data=adapt_data,
            xp=xp,
        )
        self.mesh_weight_map = mesh_weight_map

    @cached_property
    def mesh_geometry(self):

        from autoarray.inversion.mesh.mesh_geometry.rectangular import (
            MeshGeometryRectangular,
        )

        return MeshGeometryRectangular(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=self.data_grid,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )

    @cached_property
    def _mappings_sizes_weights(self):

        mappings, weights = rectangular_mappings_weights_via_interpolation_from(
            shape_native=self.mesh.shape,
            mesh_grid=self.mesh_grid.array,
            data_grid=self.data_grid.over_sampled.array,
            xp=self._xp,
        )

        sizes = 4 * self._xp.ones(len(mappings), dtype="int")

        return mappings, sizes, weights
