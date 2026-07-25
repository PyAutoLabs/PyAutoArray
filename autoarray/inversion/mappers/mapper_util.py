from functools import partial
import numpy as np
from typing import Tuple


def adaptive_pixel_signals_from(
    pixels: int,
    pixel_weights: np.ndarray,
    signal_scale: float,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    adapt_data: np.ndarray,
    xp=np,
) -> np.ndarray:
    """
    Returns the signal in each pixel, where the signal is the sum of its mapped data values.
    These pixel-signals are used to compute the effective regularization weight of each pixel.

    The pixel signals are computed as follows:

    1) Divide by the number of mappe data points in the pixel, to ensure all pixels have the same
       'relative' signal (i.e. a pixel with 10 pixels doesn't have x2 the signal of one with 5).

    2) Divided by the maximum pixel-signal, so that all signals vary between 0 and 1. This ensures that the
       regularization weight_list are defined identically for any data quantity or signal-to-noise_map ratio.

    3) Raised to the power of the parameter *signal_scale*, so the method can control the relative
       contribution regularization in different regions of pixelization.

    Parameters
    ----------
    pixels
        The total number of pixels in the pixelization the regularization scheme is applied to.
    signal_scale
        A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
        low signal regions.
    regular_to_pix
        A 1D array util every pixel on the grid to a pixel on the pixelization.
    adapt_data
        The image of the galaxy which is used to compute the weigghted pixel signals.
    """

    M_sub, B = pix_indexes_for_sub_slim_index.shape

    # 1) Flatten the per‐mapping tables:
    flat_pixidx = pix_indexes_for_sub_slim_index.reshape(-1)  # (M_sub*B,)
    flat_weights = pixel_weights.reshape(-1)  # (M_sub*B,)

    # 2) Build a matching “parent‐slim” index for each flattened entry:
    I_sub = xp.repeat(xp.arange(M_sub), B)  # (M_sub*B,)

    # 3) Mask out any k >= pix_size_for_sub_slim_index[i]
    valid = (xp.arange(B)[None, :] < pix_size_for_sub_slim_index[:, None]).reshape(-1)

    flat_weights = xp.where(valid, flat_weights, 0.0)
    flat_pixidx = xp.where(
        valid, flat_pixidx, pixels
    )  # send invalid indices to an out-of-bounds slot

    # 4) Look up data & multiply by mapping weights:
    flat_data_vals = xp.take(adapt_data[slim_index_for_sub_slim_index], I_sub, axis=0)
    flat_contrib = flat_data_vals * flat_weights  # (M_sub*B,)

    pixel_signals = xp.zeros((pixels + 1,))
    pixel_counts = xp.zeros((pixels + 1,))

    # 5) Scatter‐add into signal sums and counts:
    if xp.__name__.startswith("jax"):
        pixel_signals = pixel_signals.at[flat_pixidx].add(flat_contrib)
        pixel_counts = pixel_counts.at[flat_pixidx].add(valid.astype(float))
    else:
        xp.add.at(pixel_signals, flat_pixidx, flat_contrib)
        xp.add.at(pixel_counts, flat_pixidx, valid.astype(float))

    # 6) Drop the extra “out-of-bounds” slot:
    pixel_signals = pixel_signals[:pixels]
    pixel_counts = pixel_counts[:pixels]

    # 7) Normalize
    pixel_counts = xp.where(pixel_counts > 0, pixel_counts, 1.0)
    pixel_signals = pixel_signals / pixel_counts
    max_sig = xp.max(pixel_signals)
    pixel_signals = xp.where(max_sig > 0, pixel_signals / max_sig, pixel_signals)

    # 8) Exponentiate
    return pixel_signals**signal_scale


def sparse_triplets_from(
    pix_indexes_for_sub,  # (M_sub, P)
    pix_weights_for_sub,  # (M_sub, P)
    slim_index_for_sub,  # (M_sub,)
    fft_index_for_masked_pixel,  # (N_unmasked,)
    sub_fraction_slim,  # (N_unmasked,)
    *,
    return_rows_slim: bool = True,
    xp=np,
):
    """
    Build sparse source→image mapping triplets (rows, cols, vals)
    for a fixed-size interpolation stencil.

    This supports both:
      - NumPy (xp=np)
      - JAX  (xp=jax.numpy)

    Parameters
    ----------
    pix_indexes_for_sub
        Source pixel indices for each subpixel (M_sub, P)
    pix_weights_for_sub
        Interpolation weights for each subpixel (M_sub, P)
    slim_index_for_sub
        Mapping subpixel -> slim image pixel index (M_sub,)
    fft_index_for_masked_pixel
        Mapping slim pixel -> rectangular FFT-grid pixel index (N_unmasked,)
    sub_fraction_slim
        Oversampling normalization per slim pixel (N_unmasked,)
    xp
        Backend module (np or jnp)

    Returns
    -------
    rows : (nnz,) int32
        Rectangular FFT grid row index per mapping entry
    cols : (nnz,) int32
        Source pixel index per mapping entry
    vals : (nnz,) float64
        Mapping weight per entry including sub_fraction normalization
    """
    # ----------------------------
    # NumPy path (HOST)
    # ----------------------------
    if xp is np:
        pix_indexes_for_sub = np.asarray(pix_indexes_for_sub, dtype=np.int32)
        pix_weights_for_sub = np.asarray(pix_weights_for_sub, dtype=np.float64)
        slim_index_for_sub = np.asarray(slim_index_for_sub, dtype=np.int32)
        fft_index_for_masked_pixel = np.asarray(
            fft_index_for_masked_pixel, dtype=np.int32
        )
        sub_fraction_slim = np.asarray(sub_fraction_slim, dtype=np.float64)

        M_sub, P = pix_indexes_for_sub.shape

        sub_ids = np.repeat(np.arange(M_sub, dtype=np.int32), P)  # (M_sub*P,)

        cols = pix_indexes_for_sub.reshape(-1)  # int32
        vals = pix_weights_for_sub.reshape(-1)  # float64

        slim_rows = slim_index_for_sub[sub_ids]  # int32
        vals = vals * sub_fraction_slim[slim_rows]  # float64

        if return_rows_slim:
            return slim_rows, cols, vals

        rows = fft_index_for_masked_pixel[slim_rows]
        return rows, cols, vals

    # ----------------------------
    # JAX path (DEVICE)
    # ----------------------------
    # We intentionally avoid np.asarray anywhere here.
    # Assume xp is jax.numpy (or a compatible array module).
    pix_indexes_for_sub = xp.asarray(pix_indexes_for_sub, dtype=xp.int32)
    pix_weights_for_sub = xp.asarray(pix_weights_for_sub, dtype=xp.float64)
    slim_index_for_sub = xp.asarray(slim_index_for_sub, dtype=xp.int32)
    fft_index_for_masked_pixel = xp.asarray(fft_index_for_masked_pixel, dtype=xp.int32)
    sub_fraction_slim = xp.asarray(sub_fraction_slim, dtype=xp.float64)

    M_sub, P = pix_indexes_for_sub.shape

    sub_ids = xp.repeat(xp.arange(M_sub, dtype=xp.int32), P)

    cols = pix_indexes_for_sub.reshape(-1)
    vals = pix_weights_for_sub.reshape(-1)

    slim_rows = slim_index_for_sub[sub_ids]
    vals = vals * sub_fraction_slim[slim_rows]

    if return_rows_slim:
        return slim_rows, cols, vals

    rows = fft_index_for_masked_pixel[slim_rows]
    return rows, cols, vals


def mapping_matrix_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pixels: int,
    total_mask_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    sub_fraction: np.ndarray,
    use_mixed_precision: bool = False,
    xp=np,
) -> np.ndarray:
    """
    Returns the mapping matrix, which is a matrix representing the mapping between every unmasked sub-pixel of the data
    and the pixels of a pixelization. Non-zero entries signify a mapping, whereas zeros signify no mapping.

    For example, if the data has 5 unmasked pixels (with `sub_size=1` so there are not sub-pixels) and the pixelization
    3 pixels, with the following mappings:

    data pixel 0 -> pixelization pixel 0
    data pixel 1 -> pixelization pixel 0
    data pixel 2 -> pixelization pixel 1
    data pixel 3 -> pixelization pixel 1
    data pixel 4 -> pixelization pixel 2

    The mapping matrix (which is of dimensions [data_pixels, pixelization_pixels]) would appear as follows:

    [1, 0, 0] [0->0]
    [1, 0, 0] [1->0]
    [0, 1, 0] [2->1]
    [0, 1, 0] [3->1]
    [0, 0, 1] [4->2]

    The mapping matrix is actually built using the sub-grid of the grid, whereby each pixel is divided into a grid of
    sub-pixels which are all paired to pixels in the pixelization. The entries in the mapping matrix now become
    fractional values dependent on the sub-pixel sizes.

    For example, for a 2x2 sub-pixels in each pixel means the fractional value is 1.0/(2.0^2) = 0.25, if we have the
    following mappings:

    data pixel 0 -> data sub pixel 0 -> pixelization pixel 0
    data pixel 0 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 0 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 2 -> data sub pixel 0 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 1 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 2 -> pixelization pixel 3
    data pixel 2 -> data sub pixel 3 -> pixelization pixel 3

    The mapping matrix (which is still of dimensions [data_pixels, pixelization_pixels]) appears as follows:

    [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
    [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
    [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]

    For certain pixelizations each data sub-pixel maps to multiple pixelization pixels in a weighted fashion, for
    example a Delaunay pixelization where there are 3 mappings per sub-pixel whose weights are determined via a
    nearest neighbor interpolation scheme.

    In this case, each mapping value is multiplied by this interpolation weight (which are in the array
    `pix_weights_for_sub_slim_index`) when the mapping matrix is constructed.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    pixels
        The number of pixels in the pixelization.
    total_mask_pixels
        The number of datas pixels in the observed datas and thus on the grid.
    slim_index_for_sub_slim_index
        The mappings between the data's sub slimmed indexes and the slimmed indexes on the non sub-sized indexes.
    sub_fraction
        The fractional area each sub-pixel takes up in an pixel.
    """

    M_sub, B = pix_indexes_for_sub_slim_index.shape
    M = int(total_mask_pixels)
    S = int(pixels)

    # Indices always int32
    pix_idx = xp.asarray(pix_indexes_for_sub_slim_index, dtype=xp.int32)
    pix_size = xp.asarray(pix_size_for_sub_slim_index, dtype=xp.int32)
    slim_parent = xp.asarray(slim_index_for_sub_slim_index, dtype=xp.int32)

    # Everything else computed in float64
    w64 = xp.asarray(pix_weights_for_sub_slim_index, dtype=xp.float64)
    frac64 = xp.asarray(sub_fraction, dtype=xp.float64)

    # Output dtype only (big allocation)
    out_dtype = xp.float32 if use_mixed_precision else xp.float64

    # 1) Flatten
    flat_pixidx = pix_idx.reshape(-1)  # (M_sub*B,)
    flat_w = w64.reshape(-1)  # float64
    flat_parent = xp.repeat(slim_parent, B)  # int32
    flat_count = xp.repeat(pix_size, B)  # int32

    # 2) valid mask: k < pix_size[i]
    k = xp.tile(xp.arange(B, dtype=xp.int32), M_sub)
    valid = k < flat_count

    # 3) Zero out invalid weights (float64)
    flat_w = flat_w * valid.astype(xp.float64)

    # 4) Redirect -1 indices to extra bin S
    OUT = S
    flat_pixidx = xp.where(flat_pixidx < 0, OUT, flat_pixidx)

    # 5) Multiply by sub_fraction of the slim row (float64)
    flat_frac = xp.take(frac64, flat_parent, axis=0)
    flat_contrib64 = flat_w * flat_frac

    # 6) Scatter into (M × (S+1)) (destination float32 or float64)
    mat = xp.zeros((M, S + 1), dtype=out_dtype)

    # Cast only at the write (keeps upstream math float64)
    flat_contrib_out = flat_contrib64.astype(out_dtype)

    if xp.__name__.startswith("jax"):
        mat = mat.at[flat_parent, flat_pixidx].add(flat_contrib_out)
    else:
        xp.add.at(mat, (flat_parent, flat_pixidx), flat_contrib_out)

    # 7) Drop extra column
    return mat[:, :S]


def mapped_to_source_via_mapping_matrix_from(
    mapping_matrix: np.ndarray, array_slim: np.ndarray
) -> np.ndarray:
    """
    Map a masked 2D image (in slim form) into the source plane by summing and averaging
    each image-pixel's contribution to its mapped source-pixels.

    Each row i of `mapping_matrix` describes how image-pixel i is distributed (with
    weights) across the source-pixels j.  `array_slim[i]` is then multiplied by those
    weights and summed over i to give each source-pixel’s total mapped value; finally,
    we divide by the number of nonzero contributions to form an average.

    Parameters
    ----------
    mapping_matrix : ndarray of shape (M, N)
        mapping_matrix[i, j] ≥ 0 is the weight by which image-pixel i contributes to
        source-pixel j.  Zero means “no contribution.”
    array_slim : ndarray of shape (M,)
        The slimmed image values for each image-pixel i.

    Returns
    -------
    mapped_to_source : ndarray of shape (N,)
        The averaged, mapped values on each of the N source-pixels.
    """
    # weighted sums: sum over i of array_slim[i] * mapping_matrix[i, j]
    # ==> vector‐matrix multiply: (1×M) dot (M×N) → (N,)
    mapped_to_source = array_slim @ mapping_matrix

    # count how many nonzero contributions each source-pixel j received
    counts = np.count_nonzero(mapping_matrix > 0.0, axis=0)

    # avoid division by zero: only divide where counts > 0
    nonzero = counts > 0
    mapped_to_source[nonzero] /= counts[nonzero]

    return mapped_to_source


def data_weight_total_for_pix_from(
    pix_indexes_for_sub_slim_index: np.ndarray,  # shape (M, B)
    pix_weights_for_sub_slim_index: np.ndarray,  # shape (M, B)
    pixels: int,
) -> np.ndarray:
    """
    Returns the total weight of every pixelization pixel, which is the sum of
    the weights of all data‐points (sub‐pixels) that map to that pixel.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index : np.ndarray, shape (M, B), int
        For each of M sub‐slim indexes, the B pixelization‐pixel indices it maps to.
    pix_weights_for_sub_slim_index : np.ndarray, shape (M, B), float
        For each of those mappings, the corresponding interpolation weight.
    pixels : int
        The total number of pixelization pixels N.

    Returns
    -------
    np.ndarray, shape (N,)
        The per‐pixel total weight: for each j in [0..N-1], the sum of all
        pix_weights_for_sub_slim_index[i,k] such that pix_indexes_for_sub_slim_index[i,k] == j.
    """
    # Flatten arrays
    flat_idxs = pix_indexes_for_sub_slim_index.ravel()
    flat_weights = pix_weights_for_sub_slim_index.ravel()

    # Filter out -1 (invalid mappings)
    valid_mask = flat_idxs >= 0
    flat_idxs = flat_idxs[valid_mask]
    flat_weights = flat_weights[valid_mask]

    # Sum weights by pixel index
    return np.bincount(flat_idxs, weights=flat_weights, minlength=pixels)
