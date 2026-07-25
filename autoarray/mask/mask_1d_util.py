import numpy as np


def native_index_for_slim_index_1d_from(mask_1d: np.ndarray, xp=np) -> np.ndarray:
    """
    Returns an array of shape [total_unmasked_pixels] that maps every unmasked pixel to its
    corresponding native 2D pixel using its (y,x) pixel indexes.

    For example, for a grid size of 3x3, if pixel [0,1] corresponds to the first pixel in the masked slim array:

    - The first pixel in this pixel on the 1D array is native_index_for_slim_index_2d[0] = [0,1]

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked.

    Returns
    -------
    ndarray
        An array that maps pixels from a slimmed array of shape [total_unmasked_pixels] to its native array
        of shape [total_pixels, total_pixels].

    Examples
    --------
    mask_2d = np.array([[True, True, True],
                     [True, False, True]
                     [True, True, True]])

    native_index_for_slim_index_1d =  native_index_for_slim_index_1d_from(mask_2d=mask_2d)

    """
    return xp.flatnonzero(~mask_1d)
