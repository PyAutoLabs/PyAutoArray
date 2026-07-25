from typing import Optional

import numpy as np

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.structures.grids import grid_2d_util


def append_with_circle_edge_points(image_plane_mesh_grid, centre, radius, n_points):
    """
    Generate N uniformly spaced (y, x) coordinates around a circle.

    Parameters
    ----------
    centre : (float, float)
        The (y, x) centre of the circle.
    radius : float
        Circle radius.
    n_points : int
        Number of points around the circle.
    xp : array namespace (np or jnp)
        Function will use NumPy or JAX depending on what is passed.

    Returns
    -------
    coords : (n_points, 2) xp.ndarray
        Array of (y, x) coordinates sampled uniformly around the circle.
    """
    y0, x0 = centre

    # angles from 0 to 2π
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # parametric circle
    ys = y0 + radius * np.sin(theta)
    xs = x0 + radius * np.cos(theta)

    circle_edge_points = np.stack([ys, xs], axis=-1)

    return Grid2DIrregular(np.vstack([image_plane_mesh_grid, circle_edge_points]))


class AbstractImageMesh:
    def __init__(self):
        """
        An abstract image mesh, which is used by pixelizations to determine the (y,x) mesh coordinates from image
        data.

        """
        pass

    @property
    def uses_adapt_images(self) -> bool:
        raise NotImplementedError

    def image_plane_mesh_grid_from(
        self,
        mask: Mask2D,
        adapt_data: Optional[np.ndarray] = None,
    ) -> Grid2DIrregular:
        raise NotImplementedError

    def mesh_pixels_per_image_pixels_from(
        self, mask: Mask2D, mesh_grid: Grid2DIrregular
    ) -> Array2D:
        """
        Returns an array containing the number of mesh pixels in every pixel of the data's mask.

        For example, image-mesh adaption may be performed on a 3.0" circular mask of data. The high weight pixels
        may have 3 or more mesh pixels per image pixel, whereas low weight regions may have zero pixels. The array
        returned by this function gives the integer number of pixels in each data pixel.

        Parameters
        ----------
        mask
            The mask of the dataset being analysed, which the pixelization grid maps too. The number of
            mesh pixels mapped inside each of this mask's image-pixels is returned.
        mesh_grid
            The image mesh-grid computed by the class which adapts to the data's mask. The number of image mesh pixels
            that fall within each of the data's mask pixels is returned.

        Returns
        -------
        An array containing the integer number of image-mesh pixels that fall without each of the data's mask.
        """

        mesh_pixels_per_image_pixels = grid_2d_util.grid_pixels_in_mask_pixels_from(
            grid=np.array(mesh_grid),
            shape_native=mask.shape_native,
            pixel_scales=mask.pixel_scales,
            origin=mask.origin,
        )

        return Array2D(values=mesh_pixels_per_image_pixels, mask=mask)
