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

    data_to_pix_unique = np.full(
        (data_pixels, max_pix_mappings * np.max(sub_size) ** 2), -1.0
    )
    data_weights = np.zeros((data_pixels, max_pix_mappings * np.max(sub_size) ** 2))
    pix_lengths = np.zeros(data_pixels)
    pix_check = np.full(pix_pixels, -1.0)

    ip_sub_start = 0

    for ip in range(data_pixels):
        # Reset only the pixelization pixels the previous data pixel touched (its unique list), rather than
        # all `pix_pixels` entries: the full reset cost O(data_pixels * pix_pixels) per call (PyAutoArray#629).
        if ip > 0:
            for unique_index in range(int(pix_lengths[ip - 1])):
                pix_check[int(data_to_pix_unique[ip - 1, unique_index])] = -1.0

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


@numba_util.jit()
def adaptive_pixel_signal_sums_from(
    pixels: int,
    pixel_weights: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    adapt_data: np.ndarray,
    exclude_zero_weight_mappings: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The summed weighted adapt-data signal and the number of counted mapping entries in every pixelization pixel,
    visiting only the first `pix_size_for_sub_slim_index[i]` (live) entries of every row.

    This is the numba form of steps 1-6 of `mapper_util.adaptive_pixel_signals_from` for the NumPy path. Natural
    neighbour (`DelaunayNN`) mapping tables are `SIBSON_MAX_NEIGHBORS` (32) columns wide but hold at most ~15 live
    entries per row, so the padded scatter spent most of its time on padding (PyAutoArray#629).

    The result is bit-identical to the padded `np.add.at` scatter: the live entries are added into each pixel in
    the same row-major order, starting from the same `0.0`, as the same `adapt_data * weight` products, and every
    entry skipped here (padding, a `-1` index, and a zero weight when `exclude_zero_weight_mappings`) only ever
    reached the scatter's discarded out-of-bounds slot.

    Parameters
    ----------
    pixels
        The number of pixelization pixels the signals are summed into.
    pixel_weights
        The interpolation weight of every (sub-pixel, mapping) entry, shape [total_sub_pixels, max_mappings].
    pix_indexes_for_sub_slim_index
        The pixelization pixel of every (sub-pixel, mapping) entry, padded with -1.
    pix_size_for_sub_slim_index
        The number of live mapping entries of every sub-pixel.
    slim_index_for_sub_slim_index
        The slim data pixel of every sub-pixel.
    adapt_data
        The adapt image whose values are summed.
    exclude_zero_weight_mappings
        Whether entries with a zero weight are skipped (Delaunay interpolators, PyAutoArray#627).

    Returns
    -------
    The summed signal and the count of every pixel, each of shape [pixels].
    """
    pixel_signals = np.zeros(pixels)
    pixel_counts = np.zeros(pixels)

    for sub_index in range(pix_indexes_for_sub_slim_index.shape[0]):
        data_value = adapt_data[slim_index_for_sub_slim_index[sub_index]]

        for mapping_index in range(pix_size_for_sub_slim_index[sub_index]):
            pix = pix_indexes_for_sub_slim_index[sub_index, mapping_index]

            if pix < 0 or pix >= pixels:
                continue

            weight = pixel_weights[sub_index, mapping_index]

            if exclude_zero_weight_mappings and weight == 0.0:
                continue

            pixel_signals[pix] += data_value * weight
            pixel_counts[pix] += 1.0

    return pixel_signals, pixel_counts
