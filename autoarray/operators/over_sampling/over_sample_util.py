from __future__ import annotations
from functools import lru_cache
import numpy as np
from typing import TYPE_CHECKING, Union
from typing import List, Tuple

from autoarray.structures.arrays.uniform_2d import Array2D

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.mask.mask_2d import Mask2D


from autoarray import type as ty


def over_sample_size_convert_to_array_2d_from(
    over_sample_size: Union[int, np.ndarray], mask: Union[np.ndarray, Mask2D]
):
    """
    Returns the over sample size as an `Array2D` object, for example converting it from a single integer.

    The interface allows a user to specify the `over_sample_size` as either:

    - A single integer, whereby over sampling is performed to this degree for every pixel.
    - An ndarray with the same number of entries as the mask, to enable adaptive over sampling.

    This function converts these input structures to an `Array2D` which is used internally in the source code
    to perform computations.

    Parameters
    ----------
    over_sample_size
        The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
        values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
        into each pixel.

    Returns
    -------

    """
    if isinstance(over_sample_size, int):
        over_sample_size = np.full(
            fill_value=over_sample_size, shape=mask.pixels_in_mask
        )

    return Array2D(values=np.array(over_sample_size).astype("int"), mask=mask)


def mask_2d_upscaled_from(mask_2d: Mask2D, over_sample_size: int) -> Mask2D:
    """
    Returns the input mask upscaled by an integer over sample size, where every unmasked pixel becomes an
    unmasked block of over_sample_size x over_sample_size fine pixels.

    The upscaled mask has pixel scales divided by the over sample size and the same origin, such that the
    fine pixel centres coincide with the sub-pixel centres of a uniform over-sampled grid of the input mask.

    This is used to perform PSF convolution at a higher resolution than the image (see the `Convolver`
    object's `convolve_over_sample_size`), where the existing convolution machinery runs unchanged on the
    upscaled mask.

    Parameters
    ----------
    mask_2d
        The mask defining the image-resolution grid which is upscaled.
    over_sample_size
        The integer factor by which the mask is upscaled in each dimension.

    Returns
    -------
    The upscaled mask, of shape [total_y_pixels * over_sample_size, total_x_pixels * over_sample_size].
    """
    mask_fine = np.repeat(
        np.repeat(np.array(mask_2d), over_sample_size, axis=0),
        over_sample_size,
        axis=1,
    )

    pixel_scales = (
        mask_2d.pixel_scales[0] / over_sample_size,
        mask_2d.pixel_scales[1] / over_sample_size,
    )

    return Mask2D(mask=mask_fine, pixel_scales=pixel_scales, origin=mask_2d.origin)


def sub_slim_to_fine_slim_from(mask_2d: Mask2D, over_sample_size: int) -> np.ndarray:
    """
    Returns the permutation mapping every uniform over-sampled (sub-gridded) slim index of a mask to its slim
    index on the upscaled mask returned by `mask_2d_upscaled_from`.

    Over-sampled arrays are ordered as per-pixel sub-blocks: the s^2 sub-pixels of the first unmasked pixel
    come first (row-major within the block), then the s^2 sub-pixels of the second unmasked pixel, and so on
    (see `OverSampler`). The upscaled mask's slim ordering is instead row-major over the whole fine grid.
    This permutation converts between the two: for a sub-gridded array `values` in sub-block order,
    `fine_slim[perm] = values` scatters it into fine-mask slim order, and `fine_slim[perm]` gathers it back.

    Parameters
    ----------
    mask_2d
        The mask defining the image-resolution grid.
    over_sample_size
        The uniform integer over sample size of the sub-grid.

    Returns
    -------
    An integer array of shape [total_unmasked_pixels * over_sample_size**2] whose k-th entry is the fine-mask
    slim index of the k-th sub-pixel.
    """
    s = over_sample_size

    mask = np.array(mask_2d)
    ny, nx = mask.shape

    fine_slim_index_native = np.full((ny * s, nx * s), -1, dtype="int")
    mask_fine = np.repeat(np.repeat(mask, s, axis=0), s, axis=1)
    fine_slim_index_native[~mask_fine] = np.arange(np.sum(~mask_fine))

    ys, xs = np.where(~mask)

    block = np.arange(s)
    rows = ys[:, None, None] * s + block[None, :, None]
    cols = xs[:, None, None] * s + block[None, None, :]

    return fine_slim_index_native[rows, cols].reshape(-1)


def convolve_bin_segment_ids_from(
    sub_size: np.ndarray, convolve_over_sample_size: int
) -> np.ndarray:
    """
    Returns the segment ids mapping every over-sampled evaluation sample to its cell on the uniform
    fine grid used by oversampled PSF convolution.

    Evaluation over sampling and convolution over sampling compose via the k x s coupling: each pixel is
    evaluated on its own (possibly adaptive) sub-grid of size ``sub_size_i = k_i * s`` and the evaluated
    values are partially binned down to the uniform ``s x s`` block the oversampled Convolver requires
    (see ``binned_to_convolve_size_from``).

    Samples are in per-pixel sub-block order (row-major within each pixel's block). Within pixel ``p``'s
    block, the sample at (row, col) belongs to fine cell ``(row // k_p) * s + (col // k_p)``, and its
    global segment id is ``p * s**2 +`` that cell.

    Parameters
    ----------
    sub_size
        The per-pixel evaluation sub-grid sizes (slim, one entry per unmasked pixel). Every entry must be
        divisible by ``convolve_over_sample_size``.
    convolve_over_sample_size
        The uniform over sample size ``s`` of the PSF convolution.

    Returns
    -------
    An integer array of shape [total evaluation samples] of segment ids into the uniform fine grid
    (``total_unmasked_pixels * s**2`` segments).
    """
    s = int(convolve_over_sample_size)
    sub_size = np.asarray(sub_size).astype("int")

    return _convolve_bin_segment_ids_cached(sub_size.tobytes(), sub_size.shape[0], s)


@lru_cache(maxsize=16)
def _convolve_bin_segment_ids_cached(
    sub_size_bytes: bytes, n_pixels: int, s: int
) -> np.ndarray:
    """
    Memoized body of `convolve_bin_segment_ids_from`. The segment ids are static per
    (grid, s) pair but the partial bin runs once per likelihood evaluation in a fit,
    so the per-pixel construction loop is cached on the sizes' bytes.
    """
    from autoarray import exc

    sub_size = np.frombuffer(sub_size_bytes, dtype="int").reshape(n_pixels)

    if np.any(sub_size % s != 0):
        raise exc.GridException(
            f"Every over_sample_size entry must be divisible by "
            f"convolve_over_sample_size={s} for oversampled PSF convolution, but "
            f"sizes {np.unique(sub_size[sub_size % s != 0])} are not."
        )

    segment_ids = np.empty(int(np.sum(sub_size**2)), dtype="int64")

    offset = 0
    for p, n in enumerate(sub_size):
        k = n // s
        rows, cols = np.divmod(np.arange(n * n), n)
        segment_ids[offset : offset + n * n] = p * s**2 + (rows // k) * s + (cols // k)
        offset += n * n

    segment_ids.setflags(write=False)
    return segment_ids


def binned_to_convolve_size_from(
    values, sub_size: np.ndarray, convolve_over_sample_size: int, xp=np
):
    """
    Partially bin over-sampled evaluation values down to the uniform resolution required by oversampled
    PSF convolution (the k x s coupling).

    Each pixel's evaluated block (size ``k_i * s`` per side, per-pixel sub-block order) is reduced to an
    ``s x s`` block by the mean of each ``k_i x k_i`` group, producing values of length
    ``total_unmasked_pixels * s**2`` in the per-pixel sub-block order the oversampled Convolver expects.
    Trailing dimensions (e.g. the source axis of a mapping matrix) are supported.

    When every entry of ``sub_size`` equals ``convolve_over_sample_size`` the input is already at the
    convolution resolution and is returned unchanged (the existing equal-sizes behaviour, byte-identical).

    Parameters
    ----------
    values
        The evaluated values, one per over-sampled sample in per-pixel sub-block order, with optional
        trailing dimensions.
    sub_size
        The per-pixel evaluation sub-grid sizes (slim). Every entry must be divisible by
        ``convolve_over_sample_size``.
    convolve_over_sample_size
        The uniform over sample size ``s`` of the PSF convolution.
    """
    s = int(convolve_over_sample_size)
    sub_size = np.asarray(sub_size).astype("int")

    if np.all(sub_size == s):
        return values

    segment_ids = convolve_bin_segment_ids_from(
        sub_size=sub_size, convolve_over_sample_size=s
    )
    n_segments = sub_size.shape[0] * s**2

    counts = np.bincount(segment_ids, minlength=n_segments).astype("float")
    trailing_shape = values.shape[1:]
    counts = counts.reshape((n_segments,) + (1,) * len(trailing_shape))

    if xp.__name__.startswith("jax"):
        import jax

        sums = jax.ops.segment_sum(values, segment_ids, n_segments)
        return sums / counts

    sums = np.zeros((n_segments,) + trailing_shape)
    np.add.at(sums, segment_ids, np.asarray(values))
    return sums / counts


def total_sub_pixels_2d_from(sub_size: np.ndarray) -> int:
    """
    Returns the total number of sub-pixels in unmasked pixels in a mask.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and included when counting sub pixels.
    sub_size
        The size of the sub-grid that each pixel of the 2D mask array is divided into.

    Returns
    -------
    int
        The total number of sub pixels that are unmasked.

    Examples
    --------

    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])

    total_sub_pixels = total_sub_pixels_from(mask=mask, sub_size=2)
    """
    return int(np.sum(sub_size**2))


def slim_index_for_sub_slim_index_via_mask_2d_from(
    mask_2d: np.ndarray, sub_size: np.ndarray
) -> np.ndarray:
    """ "
    For pixels on a native 2D array of shape (total_y_pixels, total_x_pixels), compute a slimmed array which, for
    every unmasked pixel on the native 2D array, maps the slimmed sub-pixel indexes to their slimmed pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - slim_index_for_sub_slim_index[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the native 2D array.
    - slim_index_for_sub_slim_index[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the native 2D array.
    - slim_index_for_sub_slim_index[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the native 2D array.

    Parameters
    ----------
    mask_2d
        The mask whose indexes are mapped.
    sub_size
        The sub-size of the grid on the mask, so that the sub-mask indexes can be computed correctly.

    Returns
    -------
    np.ndarray
        The array of shape [total_unmasked_pixels] mapping every unmasked pixel on the native 2D mask array to its
        slimmed index on the sub-mask array.

    Examples
    --------
    mask = np.array([[True, False, True]])
    slim_index_for_sub_slim_index = slim_index_for_sub_slim_index_via_mask_2d_from(mask_2d=mask_2d, sub_size=2)
    """

    # Step 1: Identify unmasked (False) pixels
    unmasked_indices = np.argwhere(~mask_2d)
    n_unmasked = unmasked_indices.shape[0]

    # Step 2: Compute total number of sub-pixels
    sub_pixels_per_pixel = sub_size**2

    # Step 3: Repeat slim indices for each sub-pixel
    slim_indices = np.arange(n_unmasked)
    slim_index_for_sub_slim_index = np.repeat(slim_indices, sub_pixels_per_pixel)

    return slim_index_for_sub_slim_index


def sub_size_radial_bins_from(
    radial_grid: np.ndarray,
    sub_size_list: np.ndarray,
    radial_list: np.ndarray,
) -> np.ndarray:
    """
    Returns an adaptive sub-grid size based on the radial distance of every pixel from the centre of the mask.

    The adaptive sub-grid size is computed as follows:

    1) Compute the radial distance of every pixel in the mask from the centre of the mask.
    2) For every pixel, determine the sub-grid size based on the radial distance of that pixel. For example, if
    the first entry in `radial_list` is 0.5 and the first entry in `sub_size_list` 8, all pixels with a radial
    distance less than 0.5 will have a sub-grid size of 8x8.

    This scheme can produce high sub-size values towards the centre of the mask, where the galaxy is brightest and
    has the most rapidly changing light profile which requires a high sub-grid size to resolve accurately.

    Parameters
    ----------
    mask
        The mask defining the 2D region where the over-sampled grid is computed.
    radial_grid
        The radial distance of every pixel from the centre of the mask.
    sub_size_list
        The sub-grid size for every radial bin.
    radial_list
        The radial distance defining each bin, which are refeneced based on the previous entry. For example, if
        the first entry is 0.5, the second 1.0 and the third 1.5, the adaptive sub-grid size will be between 0.5
        and 1.0 for the first sub-grid size, between 1.0 and 1.5 for the second sub-grid size, etc.

    Returns
    -------
    A uniform over-sampling object with an adaptive sub-grid size based on the radial distance of every pixel from
    the centre of the mask.
    """

    # Use np.searchsorted to find the first index where radial_grid[i] < radial_list[j]
    bin_indices = np.searchsorted(radial_list, radial_grid, side="left")

    # Clip indices to stay within bounds of sub_size_list
    bin_indices = np.clip(bin_indices, 0, len(sub_size_list) - 1)

    return sub_size_list[bin_indices]


def grid_2d_slim_over_sampled_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: ty.PixelScales,
    sub_size: np.ndarray,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Compute a sub-sampled 2D grid of coordinates for every unmasked pixel,
    skipping pixels where sub_size == 0 (to avoid divide-by-zero).
    """

    H, W = mask_2d.shape
    sy, sx = pixel_scales
    oy, ox = origin

    # 1) Find unmasked pixel indices in row-major order
    rows, cols = np.nonzero(~mask_2d)
    Npix = rows.size

    if Npix == 0:
        return np.empty((0, 2), dtype=float)

    # 2) Broadcast or validate sub_size array
    sub_arr = np.asarray(sub_size)
    sub_arr = np.full(Npix, sub_arr, dtype=int) if sub_arr.size == 1 else sub_arr

    # 3) Mask out any pixels with invalid sub_size <= 0
    valid_mask = sub_arr > 0
    rows, cols, sub_arr = rows[valid_mask], cols[valid_mask], sub_arr[valid_mask]

    if sub_arr.size == 0:
        return np.empty((0, 2), dtype=float)

    # 4) Compute pixel centers
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y_pix = (cy - rows) * sy + oy
    x_pix = (cols - cx) * sx + ox

    # 5) For each valid pixel, generate its sub-pixel coords
    coords_list = []
    for i, s in enumerate(sub_arr):
        dy = sy / s
        dx = sx / s

        y_off = np.linspace(+sy / 2 - dy / 2, -sy / 2 + dy / 2, s)
        x_off = np.linspace(-sx / 2 + dx / 2, +sx / 2 - dx / 2, s)

        y_sub, x_sub = np.meshgrid(y_off, x_off, indexing="ij")

        coords = np.stack([y_pix[i] + y_sub.ravel(), x_pix[i] + x_sub.ravel()], axis=1)
        coords_list.append(coords)

    return np.vstack(coords_list)


def over_sample_size_via_radial_bins_from(
    grid: Grid2D,
    sub_size_list: List[int],
    radial_list: List[float],
    centre_list: List[Tuple] = None,
) -> Array2D:
    """
    Returns an adaptive sub-grid size based on the radial distance of every pixel from the centre of the mask.

    When ``PYAUTO_SMALL_DATASETS=1`` returns a uniform size-2 array
    immediately, skipping the expensive radial-bin computation and numba JIT.

    The adaptive sub-grid size is computed as follows:

    1) Compute the radial distance of every pixel in the mask from the centre of the mask (or input centres).
    2) For every pixel, determine the sub-grid size based on the radial distance of that pixel. For example, if
    the first entry in `radial_list` is 0.5 and the first entry in `sub_size_list` 8, all pixels with a radial
    distance less than 0.5 will have a sub-grid size of 8x8.

    This scheme can produce high sub-size values towards the centre of the mask, where the galaxy is brightest and
    has the most rapidly changing light profile which requires a high sub-grid size to resolve accurately.

    If the data has multiple galaxies, the `centre_list` can be used to define the centre of each galaxy
    and therefore increase the sub-grid size based on the light profile of each individual galaxy.

    Parameters
    ----------
    mask
        The mask defining the 2D region where the over-sampled grid is computed.
    sub_size_list
        The sub-grid size for every radial bin.
    radial_list
        The radial distance defining each bin, which are refeneced based on the previous entry. For example, if
        the first entry is 0.5, the second 1.0 and the third 1.5, the adaptive sub-grid size will be between 0.5
        and 1.0 for the first sub-grid size, between 1.0 and 1.5 for the second sub-grid size, etc.
    centre_list
        A list of centres for each galaxy whose centres require higher sub-grid sizes.

    Returns
    -------
    A uniform over-sampling object with an adaptive sub-grid size based on the radial distance of every pixel from
    the centre of the mask.
    """

    import os

    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
        return Array2D(values=np.full(grid.shape_slim, 2.0), mask=grid.mask)

    if centre_list is None:
        centre_list = [grid.mask.mask_centre]

    sub_size = np.zeros(grid.shape_slim)

    for centre in centre_list:
        radial_grid = grid.distances_to_coordinate_from(coordinate=centre)

        sub_size_of_centre = sub_size_radial_bins_from(
            radial_grid=np.array(radial_grid.array),
            sub_size_list=np.array(sub_size_list),
            radial_list=np.array(radial_list),
        )

        sub_size = np.where(sub_size_of_centre > sub_size, sub_size_of_centre, sub_size)

    return Array2D(values=sub_size, mask=grid.mask)


def over_sample_size_via_adapt_from(
    data: Array2D,
    noise_map: Array2D,
    signal_to_noise_cut: float = 5.0,
    sub_size_lower: int = 2,
    sub_size_upper: int = 4,
) -> Array2D:
    """
    Returns an adaptive sub-grid size based on the signal-to-noise of the data.

    The adaptive sub-grid size is computed as follows:

    1) The signal-to-noise of every pixel is computed as the data divided by the noise-map.
    2) For all pixels with signal-to-noise above the signal-to-noise cut, the sub-grid size is set to the upper
      value. For all other pixels, the sub-grid size is set to the lower value.

    This scheme can produce low sub-size values over entire datasets if the data has a low signal-to-noise. However,
    just because the data has a low signal-to-noise does not mean that the sub-grid size should be low.

    To mitigate this, the signal-to-noise cut is set to the maximum signal-to-noise of the data divided by 2.0 if
    it this value is below the signal-to-noise cut.

    Parameters
    ----------
    data
        The data which is to be fitted via a calculation using this over-sampling sub-grid.
    noise_map
        The noise-map of the data.
    signal_to_noise_cut
        The signal-to-noise cut which defines whether the sub-grid size is the upper or lower value.
    sub_size_lower
        The sub-grid size for pixels with signal-to-noise below the signal-to-noise cut.
    sub_size_upper
        The sub-grid size for pixels with signal-to-noise above the signal-to-noise cut.

    Returns
    -------
    The adaptive sub-grid sizes.
    """
    signal_to_noise = data / noise_map

    if np.max(signal_to_noise) < (2.0 * signal_to_noise_cut):
        signal_to_noise_cut = np.max(signal_to_noise) / 2.0

    sub_size = np.where(
        signal_to_noise > signal_to_noise_cut, sub_size_upper, sub_size_lower
    )

    return Array2D(values=sub_size, mask=data.mask)
