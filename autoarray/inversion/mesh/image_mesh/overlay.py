import numpy as np
from typing import Optional

from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.mesh.image_mesh.abstract import AbstractImageMesh
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.settings import Settings

from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.util.dataset_util import cap_mesh_shape_for_small_datasets
from autoarray import numba_util


@numba_util.jit()
def total_pixels_2d_from(mask_2d: np.ndarray, overlaid_centres: np.ndarray) -> int:
    """
    Returns the total number of pixels on the overlaid grid which are within the mask.

    Parameters
    ----------
    mask_2d
        The mask within which pixelization pixels must be inside
    overlaid_centres
        The centres of the unmasked pixelization grid pixels.
    """

    total_pixels = 0

    for overlaid_pixel_index in range(overlaid_centres.shape[0]):
        y = overlaid_centres[overlaid_pixel_index, 0]
        x = overlaid_centres[overlaid_pixel_index, 1]

        if not mask_2d[y, x]:
            total_pixels += 1

    return total_pixels


@numba_util.jit()
def overlay_for_mask_from(
    total_pixels: int,
    mask: np.ndarray,
    overlaid_centres: np.ndarray,
) -> np.ndarray:
    """
    Returns the mapping between every masked pixel on a grid and a set of pixel centres corresponding to a grid
    which is laid over the masked grid.

    The overlaid grid is unmasked and therefore has coordinates extending beyond the masked grid. Only pixels
    within the masked grid are returned.

    This is performed by checking whether each pixel's centre is within the masks and then mapping their indexes.

    Parameters
    ----------
    total_pixels
        The total number of pixels in the overlaid grid which fall within the masked grid's mask.
    mask
        The masks within which the overlaid pixels must be inside
    overlaid_centres
        The centres of the overlaid grid pixels.
    """

    overlay_for_mask = np.zeros(total_pixels)

    pixel_index = 0

    for full_pixel_index in range(overlaid_centres.shape[0]):
        y = overlaid_centres[full_pixel_index, 0]
        x = overlaid_centres[full_pixel_index, 1]

        if not mask[y, x]:
            overlay_for_mask[pixel_index] = full_pixel_index
            pixel_index += 1

    return overlay_for_mask


@numba_util.jit()
def mask_for_overlay_from(
    mask: np.ndarray,
    overlaid_centres: np.ndarray,
    total_pixels: int,
) -> np.ndarray:
    """
    Returns the mapping between every overlaid grid pixel and masked pixel on a grid.

    The overlaid grid is unmasked and therefore has coordinates extending beyond the masked grid. Only pixels
    within the masked grid are returned.

    This is performed by checking whether each pixel's centre is within the masks and then mapping their indexes.

    Pixels are paired with the next masked pixel index. This may mean that a pixel is not paired with a
    pixel near it, if the next pixel is on the next row of the grid. This is not a problem, as it is only
    unmasked pixels that are referenced when perform certain mapping where this information is not required.

    Parameters
    ----------
    mask
        The masks within which pixelization pixels must be inside
    overlaid_centres
        The centres of the unmasked pixelization grid pixels.
    total_pixels
        The total number of pixels in the overlaid grid which fall within the masks.
    """

    total_overlaid_pixels = overlaid_centres.shape[0]

    mask_for_overlay = np.zeros(total_overlaid_pixels)
    pixel_index = 0

    for unmasked_sparse_pixel_index in range(total_overlaid_pixels):
        y = overlaid_centres[unmasked_sparse_pixel_index, 0]
        x = overlaid_centres[unmasked_sparse_pixel_index, 1]

        mask_for_overlay[unmasked_sparse_pixel_index] = pixel_index

        if not mask[y, x]:
            if pixel_index < total_pixels - 1:
                pixel_index += 1

    return mask_for_overlay


@numba_util.jit()
def overlay_via_unmasked_overlaid_from(
    unmasked_overlay_grid: np.ndarray, overlay_for_mask: np.ndarray
) -> np.ndarray:
    """
    Use the unmasked overlaid grid of (y,x) coordinates and the already computed mapping between these grid pixels to
    the 1D overlaid grid indexes to compute the masked overlaid grid of (y,x) coordinates.

    Parameters
    ----------
    unmasked_overlay_grid
        The (y,x) coordinate grid of every unmasked sparse grid pixel.
    overlay_for_mask
        The index mapping between every unmasked sparse 1D index and masked sparse 1D index.

    Returns
    -------
    np.ndarray
        The masked sparse grid of (y,x) Cartesian coordinates.
    """
    total_pix_pixels = overlay_for_mask.shape[0]

    pix_grid = np.zeros((total_pix_pixels, 2))

    for pixel_index in range(total_pix_pixels):
        pix_grid[pixel_index, 0] = unmasked_overlay_grid[
            overlay_for_mask[pixel_index], 0
        ]
        pix_grid[pixel_index, 1] = unmasked_overlay_grid[
            overlay_for_mask[pixel_index], 1
        ]

    return pix_grid


class Overlay(AbstractImageMesh):
    def __init__(self, shape=(3, 3), respect_small_datasets: bool = True):
        """
        Computes an image-mesh by overlaying a uniform grid of (y,x) coordinates over the masked image that the
        pixelization is fitting.

        For example, the masked image data may consist of a 2D annular region. This image mesh class determines
        the image-mesh grid as follows:

        1) Overlay a uniform grid of dimensions `shape` over the annular mask, covering its four extreme (y,x)
        coordinates (e.g. max(x), min(x), max(y), min(y)).

        2) Find all pixels in the overlaid uniform grid which are contained within the annular mask, discarding
        all other pixels.

        Parameters
        ----------
        shape
            The 2D shape of the grid which is overlaid over the grid to determine the image mesh.
        respect_small_datasets
            When `PYAUTO_SMALL_DATASETS=1` is set, `shape` is capped per axis to the small-datasets
            cap, matching the cap `Grid2D.uniform` and `Mask2D.circular` apply to the data. Pass
            `False` to opt out for an image mesh whose resolution is load-bearing for the script.
        """

        super().__init__()

        shape = cap_mesh_shape_for_small_datasets(
            shape, respect_small_datasets=respect_small_datasets
        )

        self.shape = (int(shape[0]), int(shape[1]))

    def image_plane_mesh_grid_from(
        self,
        mask: Mask2D,
        adapt_data: Optional[np.ndarray] = None,
    ) -> Grid2DIrregular:
        """
        Returns an image-mesh by overlaying a uniform grid of (y,x) coordinates over the masked image that the
        pixelization is fitting.

        See the `__init__` docstring for a full description of how this is performed.

        Parameters
        ----------
        grid
            The grid of (y,x) coordinates of the image data the pixelization fits, which the overlay grid is laid
            over.
        adapt_data
            Not used by this image mesh.
        """
        pixel_scales = mask.pixel_scales

        grid = mask.derive_grid.unmasked

        pixel_scales = (
            (grid.shape_native_scaled_interior[0] + pixel_scales[0]) / (self.shape[0]),
            (grid.shape_native_scaled_interior[1] + pixel_scales[1]) / (self.shape[1]),
        )

        origin = mask.mask_centre

        unmasked_overlay_grid = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=self.shape,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        overlaid_centres = np.array(
            geometry_util.grid_pixel_centres_2d_slim_from(
                grid_scaled_2d_slim=unmasked_overlay_grid,
                shape_native=mask.shape_native,
                pixel_scales=mask.pixel_scales,
            )
        ).astype("int")

        total_pixels = total_pixels_2d_from(
            mask_2d=mask.array,
            overlaid_centres=overlaid_centres,
        )

        overlay_for_mask = overlay_for_mask_from(
            total_pixels=total_pixels,
            mask=mask.array,
            overlaid_centres=overlaid_centres,
        ).astype("int")

        mesh_grid = overlay_via_unmasked_overlaid_from(
            unmasked_overlay_grid=unmasked_overlay_grid,
            overlay_for_mask=overlay_for_mask,
        )

        return Grid2DIrregular(values=mesh_grid)

    @property
    def uses_adapt_images(self) -> bool:
        return False
