from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from autoarray.mask.mask_1d import Mask1D

from autoarray.mask import mask_1d_util
from autoarray.structures.arrays import array_2d_util


def convert_array_1d(
    array_1d: Union[np.ndarray, List],
    mask_1d: Mask1D,
    store_native: bool = False,
    xp=np,
) -> np.ndarray:
    """
    The `manual` classmethods in the `Array2D` object take as input a list or ndarray which is returned as an
    Array2D.

    This function performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to an ndarray.
    2) Check that the number of pixels in the array is identical to that of the mask.
    3) Map the input ndarray to its `slim` representation.

    For an Array2D, `slim` refers to a 1D NumPy array of shape [total_values] and `native` a 2D NumPy array of shape
    [total_y_values, total_values].

    Parameters
    ----------
    array_1d
        The input structure which is converted to an ndarray if it is a list.
    mask_1d
        The mask of the output Array2D.
    store_native
        If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels]. This avoids
        mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
    """
    array_1d = array_2d_util.convert_array(array=array_1d)

    is_native = array_1d.shape[0] == mask_1d.shape_native[0]

    if is_native == store_native:
        return array_1d
    elif not store_native:
        return array_1d_slim_from(
            array_1d_native=array_1d,
            mask_1d=mask_1d,
        )
    return array_1d_native_from(array_1d_slim=array_1d, mask_1d=mask_1d, xp=xp)


def array_1d_slim_from(
    array_1d_native: np.ndarray,
    mask_1d: np.ndarray,
) -> np.ndarray:
    """
    For a 1D array and mask, map the values of all unmasked pixels to its slimmed 1D array.

    The 1D array has native dimensions corresponding to the array pixels (without masking), for example a native array
    may have shape [total_unmasked_pixels].

    The pixel coordinate origin is at the left of the 1D array and goes right, with pixels then going right in
    each pixel.

    For example, for an array of shape (3,) where all pixels are unmasked:

    - pixel[0] of the native 1D array will correspond to index 0 of the slim array (which is the first pixel in
    the array).
    - pixel[1] of the native 1D array will correspond to index 1 of the slim array (which is the second pixel in
    the array).

    If the native array is masked and the third pixel is masked (e.g. its mask_1d entry is `True`) then:

    - pixels [0] and [1] of the native 1D array will correspond to indexes 0, 1 of the slim array.
    - pixel [3] of the native 1D array do not map to the slim array (the pixels is masked).

    Parameters
    ----------
    array_1d_native
        A 1D array of values on the dimensions of the native array.
    mask_1d
        A 1D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        The slimmed 1D array of values mapped from the native 1d array with
        dimensions [total_unmasked_pixels].

    Examples
    --------

    array_1d_native = np.array([ 1.0,  2.0,  5.0,  6.0])

    mask = np.array([True, False, False, False]])

    array_1d_slim = array_1d_slim_from(array_1d_native, array_2d=array_2d)
    """
    unmasked_indices = ~mask_1d
    line_1d_slim = array_1d_native[unmasked_indices]
    return line_1d_slim


def array_1d_native_from(
    array_1d_slim: np.ndarray,
    mask_1d: np.ndarray,
    xp=np,
) -> np.ndarray:
    shape = mask_1d.shape[0]

    native_index_for_slim_index_1d = mask_1d_util.native_index_for_slim_index_1d_from(
        mask_1d=mask_1d,
        xp=xp,
    ).astype("int")

    return array_1d_via_indexes_1d_from(
        array_1d_slim=array_1d_slim,
        shape=shape,
        native_index_for_slim_index_1d=native_index_for_slim_index_1d,
        xp=xp,
    )


def array_1d_via_indexes_1d_from(
    array_1d_slim: np.ndarray,
    shape: int,
    native_index_for_slim_index_1d: np.ndarray,
    xp=np,
) -> np.ndarray:
    """
    For a slimmed 1D array with indexes mapping the slimmed array values to their native array indexes,
    return the native 1D array.

    The pixel coordinate origin is at the left of the 1D array and goes right, with pixels then going right in
    each pixel.

    For example, for an array of shape (3,3) where all pixels are unmasked:

    - pixel[0] of the native 1D array will correspond to index 0 of the slim array (which is the first pixel in
    the line).
    - pixel[1] of the native 1D array will correspond to index 1 of the slim array (which is the second pixel in
    the line).

    If the native line is masked and the third pixel is masked (e.g. its mask_1d entry is `True`) then:

    - pixels [0] and [1] of the native 1D array will correspond to indexes 0, 1 of the slim array.
    - pixels [3] of the native 1D array do not map to the slim array (these pixels are masked).

    Parameters
    ----------
    array_1d_slim
        The slimmed array of shape [total_x_pixels] which are mapped to the native array.
    shape
        The 1D dimensions of the native 1D line.
    native_index_for_slim_index_1d : np.narray
        An array of shape [total_x_pixelss] that maps from the slimmed array to the native array.

    Returns
    -------
    ndarray
        The native 1D array of values mapped from the slimmed array with dimensions (total_x_pixels).
    """
    array = xp.zeros(shape, dtype=array_1d_slim.dtype)

    if xp.__name__.startswith("jax"):
        array = array.at[native_index_for_slim_index_1d].set(array_1d_slim)
    else:
        array[native_index_for_slim_index_1d] = array_1d_slim

    return array
