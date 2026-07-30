from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray.mask import mask_2d_util

from autoarray import exc


def convert_array(array: Union[np.ndarray, List]) -> np.ndarray:
    """
    Convert the input array to a NumPy ``ndarray`` if it is a list, or unwrap it if it is an autoarray object.

    Parameters
    ----------
    array : list or ndarray
        The array which may be converted to an ndarray
    """

    try:
        array = array.array
    except AttributeError:
        pass

    if isinstance(array, list):
        array = np.asarray(array)

    return array


def check_array_2d(array_2d: np.ndarray):
    """
    Check that an array intended for use as an ``Array2D`` slim representation has exactly 1 dimension.

    Raises ``ArrayException`` if the array is not 1D.

    Parameters
    ----------
    array_2d
        The array to check, which must have shape [total_unmasked_pixels].
    """
    if len(array_2d.shape) != 1:
        raise exc.ArrayException(
            "An array input into the Array2D.__new__ method is not of shape 1."
        )


def check_array_2d_and_mask_2d(array_2d: np.ndarray, mask_2d: Mask2D):
    """
    The `manual` classmethods in the `Array2D` object take as input a list or ndarray which is returned as an
    Array2D.

    This function checks the dimensions of the input `array_2d` and maps it to its `slim` representation.

    For an Array2D, `slim` refers to a 1D NumPy array of shape [total_values].

    Parameters
    ----------
    array_2d
        The input structure which is converted to its slim representation.
    mask_2d
        The mask of the output Array2D.
    """
    if len(array_2d.shape) == 1:
        if array_2d.shape[0] != mask_2d.pixels_in_mask:
            raise exc.ArrayException(
                f"""
                The input array is a slim 1D array, but it does not have the same number of entries as pixels in
                the mask.

                This indicates that the number of unmaksed pixels in the mask  is different to the input slim array 
                shape.

                The shapes of the two arrays (which this exception is raised because they are different) are as follows:

                Input array_2d_slim.shape = {array_2d.shape[0]}
                Input mask_2d.pixels_in_mask = {mask_2d.pixels_in_mask}
                Input mask_2d.shape_native = {mask_2d.shape_native}
                """
            )

    if len(array_2d.shape) == 2:
        if array_2d.shape != mask_2d.shape_native:
            raise exc.ArrayException(
                f"""
                The input array is 2D but not the same dimensions as the mask.

                This indicates the mask's shape is different to the input array shape.

                The shapes of the two arrays (which this exception is raised because they are different) are as follows:

                Input array_2d shape = {array_2d.shape}
                Input mask_2d shape_native = {mask_2d.shape_native}
                """
            )


def convert_array_2d(
    array_2d: Union[np.ndarray, List],
    mask_2d: Mask2D,
    store_native: bool = False,
    skip_mask: bool = False,
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
    array_2d
        The input structure which is converted to an ndarray if it is a list.
    mask_2d
        The mask of the output Array2D.
    store_native
        If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels]. This avoids
        mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
    """
    array_2d = convert_array(array=array_2d)

    check_array_2d_and_mask_2d(array_2d=array_2d, mask_2d=mask_2d)

    is_native = len(array_2d.shape) == 2

    if is_native and not skip_mask:
        array_2d *= ~mask_2d

    if is_native == store_native:
        return array_2d
    elif not store_native:
        return array_2d_slim_from(
            array_2d_native=array_2d,
            mask_2d=mask_2d,
        )
    return array_2d_native_from(array_2d_slim=array_2d, mask_2d=mask_2d, xp=xp)


def convert_array_2d_to_slim(array_2d: np.ndarray, mask_2d: Mask2D) -> np.ndarray:
    """
    The `manual` classmethods in the `Array2D` object take as input a list or ndarray which is returned as an
    Array2D.

    This function checks the dimensions of the input `array_2d` and maps it to its `slim` representation.

    For an Array2D, `slim` refers to a 1D NumPy array of shape [total_values].

    Parameters
    ----------
    array_2d
        The input structure which is converted to its slim representation.
    mask_2d
        The mask of the output Array2D.
    """

    if len(array_2d.shape) == 1:
        array_2d_slim = array_2d

        return array_2d_slim

    return array_2d_slim_from(
        array_2d_native=array_2d,
        mask_2d=mask_2d,
    )


def convert_array_2d_to_native(
    array_2d: np.ndarray, mask_2d: Mask2D, xp=np
) -> np.ndarray:
    """
    The `manual` classmethods in the `Array2D` object take as input a list or ndarray which is returned as an
    Array2D.

    This function checks the dimensions of the input `array_2d` and maps it to its `native` representation.

    For an Array2D, `native` a 2D NumPy array of shape [total_y_values, total_values].

    Parameters
    ----------
    array_2d
        The input structure which is converted to an ndarray if it is a list.
    mask_2d
        The mask of the output Array2D.
    """

    if len(array_2d.shape) == 2:
        array_2d_native = array_2d * np.invert(mask_2d)

        if array_2d.shape != mask_2d.shape_native:
            raise exc.ArrayException(
                "The input array is 2D but not the same dimensions as the mask "
                "(e.g. the mask 2D shape.)"
            )

        return array_2d_native

    if array_2d.shape[0] != mask_2d.pixels_in_mask:
        raise exc.ArrayException(
            "The input 1D array does not have the same number of entries as pixels in"
            "the mask."
        )

    return array_2d_native_from(array_2d_slim=array_2d, mask_2d=mask_2d, xp=xp)


def extracted_array_2d_from(
    array_2d: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    """
    Resize an array to a new size by extracting a sub-set of the array.

    The extracted input coordinates use NumPy convention, such that the upper values should be specified as +1 the
    dimensions of the extracted array.

    In the example below, an array of size (5,5) is extracted using the coordinates y0=1, y1=4, x0=1, x1=4. This
    extracts an array of dimensions (3,3) and is equivalent to array_2d[1:4, 1:4].

    Parameters
    ----------
    array_2d
        The 2D array that an array is extracted from.
    y0
        The top row number (e.g. the higher y-coodinate) of the array that is extracted for the resize.
    y1
        The bottom row number (e.g. the lower y-coodinate) of the array that is extracted for the resize.
    x0
        The left column number (e.g. the lower x-coodinate) of the array that is extracted for the resize.
    x1
        The right column number (e.g. the higher x-coodinate) of the array that is extracted for the resize.

    Returns
    -------
    ndarray
        The extracted 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    extracted_array = extract_array_2d(array_2d=array_2d, y0=1, y1=4, x0=1, x1=4)
    """
    new_shape = (y1 - y0, x1 - x0)
    resized_array = np.zeros(new_shape, dtype=array_2d.dtype)

    # Compute valid slice ranges
    y_start = max(y0, 0)
    y_end = min(y1, array_2d.shape[0])
    x_start = max(x0, 0)
    x_end = min(x1, array_2d.shape[1])

    # Target insertion indices
    y_insert_start = y_start - y0
    y_insert_end = y_insert_start + (y_end - y_start)
    x_insert_start = x_start - x0
    x_insert_end = x_insert_start + (x_end - x_start)

    resized_array[y_insert_start:y_insert_end, x_insert_start:x_insert_end] = array_2d[
        y_start:y_end, x_start:x_end
    ]

    return resized_array


def resized_array_2d_from(
    array_2d: np.ndarray,
    resized_shape: Tuple[int, int],
    origin: Tuple[int, int] = None,
    pad_value: int = 0.0,
) -> np.ndarray:
    """
    Resize an array to a new size around a central pixel.

    If the origin (e.g. the central pixel) of the resized array is not specified, the central pixel of the array is
    calculated automatically. For example, a (5,5) array's central pixel is (2,2). For even dimensions the central
    pixel is assumed to be the lower indexed value, e.g. a (6,4) array's central pixel is calculated as (2,1).

    Parameters
    ----------
    array_2d
        The 2D array that is resized.
    resized_shape
        The (y,x) new pixel dimension of the trimmed array.
    origin
        The origin of the resized array, e.g. the central pixel around which the array is extracted.
    pad_value
        If the resized array is bigger in size than the input array, the value the padded edge values are filled in
        using.

    Returns
    -------
    ndarray
        The resized 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = resize_array_2d(array_2d=array_2d, new_shape=(2,2), origin=(2, 2))
    """

    if origin is None:
        y_centre = array_2d.shape[0] // 2
        x_centre = array_2d.shape[1] // 2
        origin = (y_centre, x_centre)

    # Define window edges so that length == resized_shape dimension exactly
    y_min = origin[0] - resized_shape[0] // 2
    y_max = y_min + resized_shape[0]

    x_min = origin[1] - resized_shape[1] // 2
    x_max = x_min + resized_shape[1]

    resized_array = np.full(resized_shape, pad_value, dtype=array_2d.dtype)

    # Calculate source indices clipped to array bounds
    src_y_start = max(y_min, 0)
    src_y_end = min(y_max, array_2d.shape[0])
    src_x_start = max(x_min, 0)
    src_x_end = min(x_max, array_2d.shape[1])

    # Calculate destination indices corresponding to source indices
    dst_y_start = max(0, -y_min)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, -x_min)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    # Copy overlapping region from source to destination
    resized_array[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = array_2d[
        src_y_start:src_y_end, src_x_start:src_x_end
    ]

    return resized_array


def index_2d_for_index_slim_from(indexes_slim: np.ndarray, shape_native) -> np.ndarray:
    """
    For pixels on a native 2D array of shape (total_y_pixels, total_x_pixels), this array maps the slimmed 1D pixel
    indexes to their corresponding 2D pixel indexes.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For a 2D array of shape (3,3), flattened 1D pixel indexes are converted as follows:

    - 1D Pixel index 0 maps -> 2D pixel index [0,0].
    - 1D Pixel index 1 maps -> 2D pixel index [0,1].
    - 1D Pixel index 4 maps -> 2D pixel index [1,0].
    - 1D Pixel index 8 maps -> 2D pixel index [2,2].

    Parameters
    ----------
    indexes_slim
        The slim 1D pixel indexes which are mapped to 2D indexes.
    shape_native
        The shape of the 2D array which the pixels are natively defined on.

    Returns
    -------
    ndarray
        An array of native 2d pixel indexes with dimensions (total_indexes, 2).

    Examples
    --------
    indexes_slim = np.array([0, 1, 2, 5])
    indexes_2d = index_2d_for_index_slim_from(indexes_slim=indexes_slim, shape=(3,3))
    """
    # Calculate row indices by integer division by number of columns
    rows = indexes_slim // shape_native[1]

    # Calculate column indices by modulo number of columns
    cols = indexes_slim % shape_native[1]

    # Stack rows and cols horizontally into shape (N, 2)
    index_2d_for_index_slim = np.vstack((rows, cols)).T

    return index_2d_for_index_slim


def index_slim_for_index_2d_from(indexes_2d: np.ndarray, shape_native) -> np.ndarray:
    """
    For pixels on a native 2D array of shape (total_y_pixels, total_x_pixels), this array maps the 2D pixel indexes to
    their corresponding slimmed 1D pixel indexes.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For a 2D array of shape (3,3), 2D pixel indexes are converted as follows:

    - 2D Pixel index [0,0] maps -> 1D pixel index 0.
    - 2D Pixel index [0,1] maps -> 2D pixel index 1.
    - 2D Pixel index [1,0] maps -> 2D pixel index 4.
    - 2D Pixel index [2,2] maps -> 2D pixel index 8.

    Parameters
    ----------
    indexes_2d
        The native 2D pixel indexes which are mapped to slimmed 1D indexes.
    shape_native
        The shape of the 2D array which the pixels are defined on.

    Returns
    -------
    ndarray
        An array of 1d pixel indexes with dimensions (total_indexes).

    Examples
    --------
    indexes_2d = np.array([[0,0], [1,0], [2,0], [2,2]])
    indexes_slim = index_slim_for_index_2d_from(indexes_2d=indexes_2d, shape_native=(3,3))
    """
    # Calculate 1D indexes as row_index * number_of_columns + col_index
    index_slim_for_index_native_2d = (
        indexes_2d[:, 0] * shape_native[1] + indexes_2d[:, 1]
    )

    return index_slim_for_index_native_2d


def array_2d_slim_from(
    array_2d_native: np.ndarray,
    mask_2d: np.ndarray,
) -> np.ndarray:
    """
    For a 2D array and mask, map the values of all unmasked pixels to its slimmed 1D array.

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards.

    For example, for an array of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D array will correspond to index 0 of the 1D array.
    - pixel [0,1] of the 2D array will correspond to index 1 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 3 of the 1D array.
    - pixel [2,0] of the 2D array will correspond to index 6 of the 1D array.

    Parameters
    ----------
    array_2d_native
        A 2D array of values on the dimensions of the grid, with shape [total_y_pixels, total_x_pixels].
    mask_2d
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        The slimmed 1D array of values mapped from the native 2D array with dimensions (total_unmasked_pixels).

    Examples
    --------

    array_2d = np.array([[ 1.0,  2.0,  5.0,  6.0],
                             [ 3.0,  4.0,  7.0,  8.0],
                             [ 9.0, 10.0, 13.0, 14.0],
                             [11.0, 12.0, 15.0, 16.0])

    mask = np.array([[True, False],
                     [False, False]])

    array_2d_slim = array_2d_slim_from(mask=mask, array_2d=array_2d)
    """
    return array_2d_native[~mask_2d.astype(bool)]


def array_2d_native_from(
    array_2d_slim: np.ndarray, mask_2d: np.ndarray, xp=np
) -> np.ndarray:
    """
    For a slimmed 2D array that was computed by mapping unmasked values from a native 2D array of shape
    [total_y_pixels, total_x_pixels], map its values back to the original 2D array where masked values are set to zero.

    This uses the array ``slim_to_native`` where each index gives the 2D pixel indexes of the 1D array's
    unmasked pixels, for example:

    - If ``slim_to_native[0] = [0,0]``, the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If ``slim_to_native[1] = [0,1]``, the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If ``slim_to_native[4] = [1,1]``, the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    This example uses an array `slim_to_native`.

    Parameters
    ----------
    array_2d_slim
        The slimmed array of values which are mapped to a 2D array.
    mask_2d
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        A 2D array of values mapped from the 1D array with dimensions shape.

    Examples
    --------
    slim_to_native = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

    array_1d = np.array([[2.0, 4.0, 5.0, 6.0, 8.0])

    array_2d = map_masked_1d_array_to_2d_array_from(
        array_1d=array_1d, shape=(3,3), slim_to_native=slim_to_native)
    """

    shape = (mask_2d.shape[0], mask_2d.shape[1])

    # Deliberately NumPy, not ``xp``: this index map is derived from the mask,
    # which is concrete geometry and never traced. Its computation is a
    # ``where(~mask)``, whose output shape depends on the *values* of the mask, so
    # under ``jnp`` inside a ``jax.jit`` trace it raises
    # ``ConcretizationTypeError``. Computing it in NumPy yields a concrete index
    # array, which is a valid static operand for the scatter below — that is
    # where ``xp`` genuinely belongs.
    native_index_for_slim_index_2d = mask_2d_util.native_index_for_slim_index_2d_from(
        mask_2d=mask_2d
    ).astype("int")

    return array_2d_via_indexes_from(
        array_2d_slim=array_2d_slim,
        shape=shape,
        native_index_for_slim_index_2d=native_index_for_slim_index_2d,
        xp=xp,
    )


def array_2d_via_indexes_from(
    array_2d_slim: np.ndarray,
    shape: Tuple[int, int],
    native_index_for_slim_index_2d: np.ndarray,
    xp=np,
) -> np.ndarray:
    """
    For a slimmed array with indexes mapping the slimmed array values to their native array, return the native 2D
    array.

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards.

    For example, for an array of shape (3,3) and where all pixels are unmasked:

    - pixel [0,0] of the native 2D array will correspond to index 0 of the slim array.
    - pixel [0,1] of the native 2D array will correspond to index 1 of the slim array.
    - pixel [1,0] of the native 2D array will correspond to index 3 of the slim array.
    - pixel [2,0] of the native 2D array will correspond to index 6 of the slim array.

    Parameters
    ----------
    array_2d_slim
        The slimmed array of shape [total_values] which are mapped to the native array..
    shape
        The 2D dimensions of the native 2D array.
    native_index_for_slim_index_2d : np.narray
        An array of shape [total_values] that maps from the slimmed array to the native array.

    Returns
    -------
    ndarray
        The native 2D array of values mapped from the slimmed array with dimensions (total_values, total_values).
    """
    array = xp.zeros(shape, dtype=array_2d_slim.dtype)

    if xp.__name__.startswith("jax"):
        array = array.at[
            native_index_for_slim_index_2d[:, 0],
            native_index_for_slim_index_2d[:, 1],
        ].set(array_2d_slim)
    else:
        array[tuple(native_index_for_slim_index_2d.T)] = array_2d_slim

    return array
