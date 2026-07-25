import numpy as np
from typing import Tuple

from autoarray import numba_util


@numba_util.jit()
def data_slim_to_pixelization_unique_from(
    data_pixels,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_sizes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index,
    pix_pixels: int,
    sub_size: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an array describing the unique mappings between the sub-pixels of every slim data pixel and the pixelization
    pixels, which is used to perform efficiently linear algebra calculations.

    For example, assuming `sub_size=2`:

    - If 3 sub-pixels in image pixel 0 map to pixelization pixel 2 then `data_pix_to_unique[0, 0] = 2`.
    - If the fourth sub-pixel maps to pixelization pixel 4, then `data_to_pix_unique[0, 1] = 4`.

    The size of the second index depends on the number of unique sub-pixel to pixelization pixels mappings in a given
    data pixel. In the example above, there were only two unique sets of mapping, but for high levels of sub-gridding
    there could be many more unique mappings all of which must be stored.

    The array `data_to_pix_unique` does not describe how many sub-pixels uniquely map to each pixelization pixel for
    a given data pixel. This information is contained in the array `data_weights`. For the example above,
    where `sub_size=2` and therefore `sub_fraction=0.25`:

    - `data_weights[0, 0] = 0.75` (because 3 sub-pixels mapped to this pixelization pixel).
    - `data_weights[0, 1] = 0.25` (because 1 sub-pixel mapped to this pixelization pixel).

    The `sub_fractions` are stored as opposed to the number of sub-pixels, because these values are used directly
    when performing the linear algebra calculation.

    The array `pix_lengths` in a 1D array of dimensions [data_pixels] describing how many unique pixelization pixels
    each data pixel's set of sub-pixels maps too.

    Parameters
    ----------
    data_pixels
        The total number of data pixels in the dataset.
    pix_indexes_for_sub_slim_index
        Maps an unmasked data sub pixel to its corresponding pixelization pixel.
    sub_size
        The size of the sub-grid defining the number of sub-pixels in every data pixel.

    Returns
    -------
    ndarray
        The unique mappings between the sub-pixels of every data pixel and the pixelization pixels, alongside arrays
        that give the weights and total number of mappings.
    """

    sub_fraction = 1.0 / (sub_size**2.0)

    max_pix_mappings = int(np.max(pix_sizes_for_sub_slim_index))

    # TODO : Work out if we can reduce size from np.max(sub_size) using sub_size of max_pix_mappings.

    data_to_pix_unique = -1 * np.ones(
        (data_pixels, max_pix_mappings * np.max(sub_size) ** 2)
    )
    data_weights = np.zeros((data_pixels, max_pix_mappings * np.max(sub_size) ** 2))
    pix_lengths = np.zeros(data_pixels)
    pix_check = -1 * np.ones(shape=pix_pixels)

    ip_sub_start = 0

    for ip in range(data_pixels):
        pix_check[:] = -1

        pix_size = 0

        ip_sub_end = ip_sub_start + sub_size[ip] ** 2

        for ip_sub in range(ip_sub_start, ip_sub_end):
            for pix_interp_index in range(pix_sizes_for_sub_slim_index[ip_sub]):
                pix = pix_indexes_for_sub_slim_index[ip_sub, pix_interp_index]
                pixel_weight = pix_weights_for_sub_slim_index[ip_sub, pix_interp_index]

                if pix_check[pix] > -0.5:
                    data_weights[ip, int(pix_check[pix])] += (
                        sub_fraction[ip] * pixel_weight
                    )

                else:
                    data_to_pix_unique[ip, pix_size] = pix
                    data_weights[ip, pix_size] += sub_fraction[ip] * pixel_weight
                    pix_check[pix] = pix_size
                    pix_size += 1

        ip_sub_start = ip_sub_end

        pix_lengths[ip] = pix_size

    return data_to_pix_unique, data_weights, pix_lengths
