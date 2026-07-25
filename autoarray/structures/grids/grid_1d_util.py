from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING, List, Union, Tuple

if TYPE_CHECKING:
    from autoarray.mask.mask_1d import Mask1D

from autoarray.structures.arrays import array_1d_util
from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray import type as ty


def convert_grid_1d(
    grid_1d: Union[np.ndarray, List], mask_1d: Mask1D, store_native: bool = False, xp=np
) -> np.ndarray:
    """
    The ``manual`` classmethods in the ``Grid1D`` object take as input a list or ndarray which is returned as a ``Grid1D``.

    This function performs the following checks and conversions on the input:

    1. If the input is a list, convert it to an ndarray.
    2. Check that the number of pixels in the array is identical to that of the mask.
    3. Map the input ndarray to its ``slim`` representation.

    For a ``Grid1D``, ``slim`` refers to a 1D NumPy array of shape [total_unmasked_pixels] and ``native`` a 1D
    NumPy array of shape [total_pixels].

    Parameters
    ----------
    grid_1d
        The input (x,) grid of coordinates which is converted to an ndarray if it is a list.
    mask_1d
        The mask of the output ``Grid1D``.
    store_native
        If True, the ndarray is stored in its native format [total_pixels]. This avoids mapping large data arrays
        to and from the slim / native formats, which can be a computational bottleneck.
    xp
        The array module to use (default ``numpy``; pass ``jax.numpy`` for JAX support).
    """

    grid_1d = grid_2d_util.convert_grid(grid=grid_1d)

    is_native = grid_1d.shape[0] == mask_1d.shape_native[0]

    if is_native == store_native:
        return grid_1d
    elif not store_native:
        return grid_1d_slim_from(
            grid_1d_native=grid_1d,
            mask_1d=mask_1d,
        )
    return grid_1d_native_from(grid_1d_slim=grid_1d, mask_1d=mask_1d, xp=xp)


def grid_1d_slim_via_shape_slim_from(
    shape_slim: Tuple[int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float] = (0.0,),
    xp=np,
) -> np.ndarray:
    """
    This routine computes the (x) scaled coordinates at the centre of every pixel defined by a 1D shape of the
    overall grid.

    Grid2D are defined from left to right, where the first unmasked pixel corresponds to index 0.

    The grid is returned on an array of shape (total_unmasked_pixels, ).

    Parameters
    ----------
    shape_slim
        The (x) shape of the 1D array the grid of coordinates is computed for.
    pixel_scales
        The (x) scaled units to pixel units conversion factor of the 1D mask array.
    origin
        The (x) origin of the 1D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid array has dimensions (total_unmasked_pixels, ).

    Examples
    --------
    grid_1d = grid_1d_via_shape_slim_from(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    return grid_1d_slim_via_mask_from(
        mask_1d=np.full(fill_value=False, shape=shape_slim),
        pixel_scales=pixel_scales,
        origin=origin,
        xp=xp,
    )


def grid_1d_slim_via_mask_from(
    mask_1d: np.ndarray,
    pixel_scales: ty.PixelScales,
    origin: Tuple[float] = (0.0,),
    xp=np,
) -> np.ndarray:
    """
    For a grid, every unmasked pixel of its 1D mask with shape (total_pixels,) is divided into a finer uniform
    grid of shape (total_pixels, ). This routine computes the (x) scaled coordinates at the centre of every
    pixel defined by this 1D mask array.

    Grid2D are defined from left to right, where the first unmasked pixel corresponds to index 0.

    The grid is returned on an array of shape (total_unmasked_pixels, ).

    Parameters
    ----------
    mask_1d
        A 1D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        grid.
    pixel_scales
        The (x,) scaled units to pixel units conversion factor of the 1D mask array.
    origin
        The (x,) origin of the 1D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (x) scaled coordinates at the centre of every unmasked pixel in the 1D mask
        array. The grid array has dimensions (total_unmasked_pixels, ).

    Examples
    --------
    mask = np.array([True, False, True, False, False, False])
    grid_slim = grid_1d_slim_via_mask_from(mask_1d=mask, pixel_scales=(0.5,), origin=(0.0,))
    """
    centres_scaled = geometry_util.central_scaled_coordinate_1d_from(
        shape_slim=mask_1d.shape, pixel_scales=pixel_scales, origin=origin
    )
    indices = xp.arange(mask_1d.shape[0])
    unmasked = xp.logical_not(mask_1d)
    coords = (indices - centres_scaled[0]) * pixel_scales[0]
    return coords[unmasked]


def grid_1d_slim_from(
    grid_1d_native: np.ndarray,
    mask_1d: np.ndarray,
) -> np.ndarray:
    """
    For a native 1D grid and mask of shape [total_pixels], map the values of all unmasked pixels to a slimmed grid of
    shape [total_unmasked_pixels].

    The pixel coordinate origin is at the left of the native grid and goes right-wards, such
    that for a grid of shape (4,) where pixels 0, 1 and 3 are unmasked:

    - pixel [0] of the 1D native grid will correspond to index 0 of the 1D grid.
    - pixel [1] of the 1D native grid will correspond to index 1 of the 1D grid.
    - pixel [3] of the 1D native grid will correspond to index 2 of the 1D grid.

    Parameters
    ----------
    grid_1d_native : ndarray
        The native grid of (x) values which are mapped to the slimmed grid.
    mask_1d
        A 1D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        A 1D slim grid of values mapped from the 1D native grid with dimensions (total_unmasked_pixels).
    """

    return array_1d_util.array_1d_slim_from(
        array_1d_native=grid_1d_native,
        mask_1d=mask_1d,
    )


def grid_1d_native_from(
    grid_1d_slim: np.ndarray,
    mask_1d: np.ndarray,
    xp=np,
) -> np.ndarray:
    """
    For a slimmed 1D grid of shape [total_unmasked_pixels], that was computed by extracting the unmasked values
    from a native 1D grid of shape [total_pixels], map the slimmed grid's coordinates back to the native 1D
    grid where masked values are set to zero.

    This uses a 1D array 'slim_to_native' where each index gives the 1D pixel indexes of the grid's native unmasked
    pixels, for example:

    - If slim_to_native[0] = [0], the first value of the 1D slimmed grid maps to the pixel [0] of the native 1D grid.
    - If slim_to_native[1] = [2], the second value of the 1D slimmed grid maps to the pixel [2] of the native 1D grid.
    - If slim_to_native[4] = [9], the fifth value of the 1D slimmed grid maps to the pixels [9] of the native 1D grid.

    Parameters
    ----------
    grid_1d_slim
        The (x) values of the slimmed 1D grid which are mapped to the native 1D grid.
    mask_1d
        A 1D array of bools, where `False` values means unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        A NumPy array of shape [total_pixels] corresponding to the (x) values of the native 1D
        mapped from the slimmed grid.
    """
    return array_1d_util.array_1d_native_from(
        array_1d_slim=grid_1d_slim, mask_1d=mask_1d, xp=xp
    )
