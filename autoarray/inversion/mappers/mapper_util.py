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
    exclude_zero_weight_mappings: bool = False,
    xp=np,
) -> np.ndarray:
    """
    Returns the signal in each pixel, where the signal is the sum of its mapped data values.
    These pixel-signals are used to compute the effective regularization weight of each pixel.

    The pixel signals are computed as follows:

    1) Divide by the number of mapped data points in the pixel, to ensure all pixels have the same
       'relative' signal (i.e. a pixel with 10 pixels doesn't have x2 the signal of one with 5).
       By default every mapping entry within a data point's size counts, including entries whose
       weight is zero. With `exclude_zero_weight_mappings=True` (set by Delaunay interpolators),
       zero-weight entries are not counted -- see that parameter.

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
    exclude_zero_weight_mappings
        If `True`, mapping entries whose weight is exactly zero are not counted towards a pixel's number of
        mapped data points. Delaunay interpolators set this (via their `exclude_zero_weight_mappings` class
        attribute): a data point lying exactly on a Delaunay vertex has barycentric weights (0, 0, 1) in *any*
        triangle around that vertex (a point on an edge has a zero weight in either neighbouring triangle), and
        scipy's `find_simplex` and the JAX Delaunay walk legitimately choose different triangles. Counting those
        zero-weight duplicates made the pixel signals, and through them the adaptive regularization matrix,
        differ between the NumPy and JAX backends from identical inputs (PyAutoArray#627). The set of non-zero
        (index, weight) pairs is unique, so excluding zero weights makes the result independent of that choice.

        It is `False` for every other interpolator. In particular `KNNBarycentric` clips negative barycentric
        weights to exactly zero for every data point outside the triangle formed by its 3 nearest vertices --
        whole regions, not rare coincidences -- so excluding them there would change pixel signals broadly
        and make the count (and hence the likelihood) jump wherever a clipped weight reaches zero.
    """

    if xp.__name__.startswith("jax"):
        pixel_signals, pixel_counts = _pixel_signal_sums_jax(
            pixels=pixels,
            pixel_weights=pixel_weights,
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=pix_size_for_sub_slim_index,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            adapt_data=adapt_data,
            exclude_zero_weight_mappings=exclude_zero_weight_mappings,
            xp=xp,
        )
    elif _use_numba():
        from autoarray.inversion.mappers import mapper_numba_util

        pix_indexes = np.asarray(pix_indexes_for_sub_slim_index)

        pixel_signals, pixel_counts = mapper_numba_util.adaptive_pixel_signal_sums_from(
            pixels=pixels,
            pixel_weights=np.asarray(pixel_weights).reshape(pix_indexes.shape),
            pix_indexes_for_sub_slim_index=pix_indexes,
            pix_size_for_sub_slim_index=np.asarray(pix_size_for_sub_slim_index),
            slim_index_for_sub_slim_index=np.asarray(slim_index_for_sub_slim_index),
            adapt_data=np.asarray(adapt_data),
            exclude_zero_weight_mappings=bool(exclude_zero_weight_mappings),
        )
    else:
        pixel_signals, pixel_counts = _pixel_signal_sums_numpy(
            pixels=pixels,
            pixel_weights=pixel_weights,
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=pix_size_for_sub_slim_index,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            adapt_data=adapt_data,
            exclude_zero_weight_mappings=exclude_zero_weight_mappings,
        )

    # 7) Normalize
    #
    # Both divisions here use a *safe denominator* rather than a `where` around
    # the quotient. `xp.where(cond, a / b, a)` guards only the selection: the
    # division is still evaluated for every element, so a zero denominator
    # produces a NaN that NumPy discards along with the unselected branch but
    # that JAX propagates (the standard `where`-inside-`grad` trap). A
    # zero-signal adapt image — one whose emission lands nowhere near the
    # pixels being adapted — makes `max_sig` exactly 0, and the two backends
    # then disagree on the same input: NumPy returns a finite likelihood with a
    # `RuntimeWarning`, JAX returns NaN.
    pixel_counts = xp.where(pixel_counts > 0, pixel_counts, 1.0)
    pixel_signals = pixel_signals / pixel_counts
    max_sig = xp.max(pixel_signals)
    max_sig = xp.where(max_sig > 0, max_sig, 1.0)
    pixel_signals = pixel_signals / max_sig

    # 8) Exponentiate
    #
    # Same trap, one step further on. `0.0 ** signal_scale` is finite forwards
    # but its derivative, `signal_scale * 0.0 ** (signal_scale - 1)`, is
    # infinite for `signal_scale < 1` and NaN under `jax.grad`; and a negative
    # signal raised to a fractional power is NaN on both backends. Zero-signal
    # pixels are the common case (a pixel the adapt image does not reach), so
    # exponentiate a safe base and select the result for those pixels
    # afterwards, keeping the whole line finite and differentiable.
    #
    # The signals are documented above as varying between 0 and 1, so a
    # non-positive signal contributes nothing: it is given 0, not the NaN (or,
    # for an even integer `signal_scale`, the spurious positive weight) that a
    # direct exponentiation would return.
    zero_signal = pixel_signals <= 0.0
    safe_signals = xp.where(zero_signal, 1.0, pixel_signals)
    return xp.where(zero_signal, 0.0, safe_signals**signal_scale)


def _use_numba() -> bool:
    """
    Whether the NumPy branch of `adaptive_pixel_signals_from` uses the numba kernel
    `mapper_numba_util.adaptive_pixel_signal_sums_from`.

    True when numba is importable and `general.yaml` `numba: use_numba` is true (the default when the config
    cannot be read); otherwise the vectorised `np.bincount` form `_pixel_signal_sums_numpy` is used. Both are
    bit-identical to the padded scatter.
    """
    try:
        from autonerves import conf

        use_numba = conf.instance["general"]["numba"]["use_numba"]
    except Exception:
        use_numba = True

    if not use_numba:
        return False

    try:
        import numba  # noqa: F401
    except ImportError:
        return False

    return True


def _pixel_signal_sums_jax(
    pixels,
    pixel_weights,
    pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index,
    slim_index_for_sub_slim_index,
    adapt_data,
    exclude_zero_weight_mappings,
    xp,
):
    """
    The JAX branch of `adaptive_pixel_signals_from` steps 1-6: the summed weighted adapt-data signal and the
    number of counted mapping entries in every pixel, built with fixed shapes over the full padded tables so it
    traces under `jit` and stays differentiable.
    """
    M_sub, B = pix_indexes_for_sub_slim_index.shape

    # 1) Flatten the per-mapping tables:
    flat_pixidx = pix_indexes_for_sub_slim_index.reshape(-1)  # (M_sub*B,)
    flat_weights = pixel_weights.reshape(-1)  # (M_sub*B,)

    # 2) Build a matching "parent-slim" index for each flattened entry:
    I_sub = xp.repeat(xp.arange(M_sub), B)  # (M_sub*B,)

    # 3) Mask out any k >= pix_size_for_sub_slim_index[i] and, for Delaunay tables, any entry whose
    #    weight is zero (see `exclude_zero_weight_mappings` above, PyAutoArray#627). The flag is a
    #    static Python bool, so this is jit-safe; the mask is not differentiated, so the function
    #    stays differentiable under JAX.
    valid = (xp.arange(B)[None, :] < pix_size_for_sub_slim_index[:, None]).reshape(-1)

    if exclude_zero_weight_mappings:
        valid = valid & (flat_weights != 0)

    flat_weights = xp.where(valid, flat_weights, 0.0)
    flat_pixidx = xp.where(
        valid, flat_pixidx, pixels
    )  # send invalid indices to an out-of-bounds slot

    # 4) Look up data & multiply by mapping weights:
    flat_data_vals = xp.take(adapt_data[slim_index_for_sub_slim_index], I_sub, axis=0)
    flat_contrib = flat_data_vals * flat_weights  # (M_sub*B,)

    pixel_signals = xp.zeros((pixels + 1,))
    pixel_counts = xp.zeros((pixels + 1,))

    # 5) Scatter-add into signal sums and counts:
    pixel_signals = pixel_signals.at[flat_pixidx].add(flat_contrib)
    pixel_counts = pixel_counts.at[flat_pixidx].add(valid.astype(float))

    # 6) Drop the extra "out-of-bounds" slot:
    return pixel_signals[:pixels], pixel_counts[:pixels]


def _pixel_signal_sums_numpy(
    pixels,
    pixel_weights,
    pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index,
    slim_index_for_sub_slim_index,
    adapt_data,
    exclude_zero_weight_mappings,
):
    """
    The NumPy branch of `adaptive_pixel_signals_from` steps 1-6, computed over the live mapping entries only.

    Natural-neighbour (`DelaunayNN`) tables are `SIBSON_MAX_NEIGHBORS` (32) columns wide but hold at most ~15 live
    entries per row, so scattering over every padded slot (the JAX branch's fixed-shape form) spent most of its
    time on padding (PyAutoArray#629). Here the tables are first trimmed to their widest live row, the live
    entries compacted in row-major order and summed with `np.bincount`.

    The result is bit-identical to the padded scatter: `np.bincount` adds the compacted entries into each bin
    in the same row-major order that `np.add.at` visited them, starting from the same `0.0`, and every entry it
    drops (padding, `-1` indexes, and zero weights when `exclude_zero_weight_mappings`) only ever reached the
    padded scatter's discarded out-of-bounds slot. Counts are integers, so summing them as integers is exact.
    """
    pix_indexes = np.asarray(pix_indexes_for_sub_slim_index)
    weights = np.asarray(pixel_weights).reshape(pix_indexes.shape)
    sizes = np.asarray(pix_size_for_sub_slim_index)

    # Trim every table to its widest live row (a view; padding beyond it is never read).
    width = int(np.max(sizes)) if sizes.size else 0
    width = min(max(width, 0), pix_indexes.shape[1])
    pix_indexes = pix_indexes[:, :width]
    weights = weights[:, :width]

    valid = np.arange(width)[None, :] < sizes[:, None]

    if exclude_zero_weight_mappings:
        valid &= weights != 0

    # A `-1` index inside a row's size only ever landed in the padded scatter's out-of-bounds slot.
    valid &= pix_indexes >= 0

    rows = np.nonzero(valid)[0]
    live_indexes = pix_indexes[valid]
    live_weights = weights[valid]

    data_values = np.asarray(adapt_data)[
        np.asarray(slim_index_for_sub_slim_index)[rows]
    ]

    pixel_signals = np.bincount(
        live_indexes, weights=data_values * live_weights, minlength=pixels
    )[:pixels]
    pixel_counts = np.bincount(live_indexes, minlength=pixels)[:pixels].astype(float)

    return pixel_signals, pixel_counts


def sparse_triplets_from(
    pix_indexes_for_sub,  # (M_sub, P)
    pix_weights_for_sub,  # (M_sub, P)
    slim_index_for_sub,  # (M_sub,)
    fft_index_for_masked_pixel,  # (N_unmasked,)
    sub_fraction_slim,  # (N_unmasked,)
    *,
    return_rows_slim: bool = True,
    slim_weights=None,
    pix_sizes_for_sub=None,
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
    slim_weights
        Optional per-slim-pixel weights (N_unmasked,) multiplied into `vals`, i.e. the triplets
        of `diag(w) A` rather than `A`. The interferometer sparse inversion passes the primary
        beam here so the curvature and data vector use the beamed response `P A`. `None`
        (default) applies no weighting.
    pix_sizes_for_sub
        Optional number of used stencil slots per sub-pixel (M_sub,), i.e. the mapper's
        `pix_sizes_for_sub_slim_index`. When given, every entry that `mapping_matrix_from` drops
        (slot index `>= pix_sizes_for_sub`, or a negative source pixel index such as the `-1`
        padding Delaunay meshes write into unused slots) is given value 0 and column 0, so it
        contributes nothing to any scatter-add and never indexes out of range. `None` (default)
        returns the raw stencil entries unmasked.
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

        if slim_weights is not None:
            vals = vals * np.asarray(slim_weights, dtype=np.float64)[slim_rows]

        if pix_sizes_for_sub is not None:
            slot = np.tile(np.arange(P, dtype=np.int32), M_sub)
            pix_sizes = np.asarray(pix_sizes_for_sub, dtype=np.int32)
            valid = (slot < np.repeat(pix_sizes, P)) & (cols >= 0)
            vals = np.where(valid, vals, 0.0)
            cols = np.where(valid, cols, 0).astype(np.int32)

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

    if slim_weights is not None:
        vals = vals * xp.asarray(slim_weights, dtype=xp.float64)[slim_rows]

    if pix_sizes_for_sub is not None:
        slot = xp.tile(xp.arange(P, dtype=xp.int32), M_sub)
        pix_sizes = xp.asarray(pix_sizes_for_sub, dtype=xp.int32)
        valid = (slot < xp.repeat(pix_sizes, P)) & (cols >= 0)
        vals = xp.where(valid, vals, 0.0)
        cols = xp.where(valid, cols, 0)

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

    # Output dtype only (big allocation). fp32 is a JAX-only (GPU throughput)
    # choice: the NumPy backend is the fp64 parity oracle of the workspace smoke
    # scripts and always allocates float64, as the `Settings` docstring promises
    # (PyAutoArray#552).
    out_dtype = (
        xp.float32
        if (use_mixed_precision and xp.__name__.startswith("jax"))
        else xp.float64
    )

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
