import numpy as np

from autoarray import numba_util


@numba_util.jit()
def curvature_direct_conv(
    preload,
    iy,
    ix,
    flat,
    indptr,
    col,
    val,
    cscptr,
    csc_row,
    csc_val,
    ny,
    nx,
    pix_pixels,
):
    """
    `F = Aᵀ W~ A` by convolving each source column of `A` over the extent rectangle.

    For source column `s`:

    1. `u = W~ A[:, s]` on the `(ny, nx)` extent grid -- each of the `nnz_s` non-zeros of
       the column scatters a shifted copy of the `W~` kernel onto `u`. Splitting the row
       into the two contiguous halves of the wrapped `preload` row makes the inner loop a
       pure contiguous AXPY.
    2. `F[s, :] = Aᵀ u` -- one gather per non-zero of the whole mapping operator.

    Cost `O(nnz·M + S·nnz)` with `M = ny·nx`, versus the historic pair loop's `O(N² P²)`:
    the convolution replaces the `N²` pixel-pair space with the `M`-cell extent rectangle,
    which is the whole point of the extent-grid form. It beats the FFT route (the JAX and
    NumPy `InterferometerSparseOperator` paths) while the source columns stay sparse; where
    the two cross depends on the extent as well as the column density, and the factory
    decides with :func:`curvature_route_seconds_from` (PyAutoArray#628).

    The result is a complete, symmetric `[pix_pixels, pix_pixels]` matrix: the kernel
    loops the full column x row space rather than halving on symmetry, so no mirroring
    pass is required afterwards.

    Parameters
    ----------
    preload
        The real `(2 * ny, 2 * nx)` `W~` operator as a function of pixel offsets, i.e.
        `InterferometerSparseOperator.nufft_precision_operator`. Indexed with wrapped
        (negative) offsets, exactly as the FFT paths convolve with it.
    iy, ix
        The extent-grid row / column of every masked (sub-slim) pixel.
    flat
        `iy * nx + ix`, the extent-flat index of every masked pixel.
    indptr, col, val
        The mapping operator `A` in CSR form (rows = masked pixels, columns = source
        pixels).
    cscptr, csc_row, csc_val
        The same triplets in CSC form (source-pixel major), which the convolution sweeps.
    ny, nx
        The extent rectangle's shape.
    pix_pixels
        The number of source pixels, i.e. `mapper.params`.

    Returns
    -------
    The curvature matrix `F`, of shape `[pix_pixels, pix_pixels]`.
    """
    n_pix = indptr.shape[0] - 1
    m_cells = ny * nx
    nx2 = 2 * nx

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))
    u = np.zeros(m_cells)

    for sp in range(pix_pixels):
        u[:] = 0.0

        for t in range(cscptr[sp], cscptr[sp + 1]):
            i0 = csc_row[t]
            wi = csc_val[t]
            i_y = iy[i0]
            i_x = ix[i0]
            off = nx2 - i_x

            for jy in range(ny):
                dy = jy - i_y
                base = jy * nx

                for jx in range(i_x):
                    u[base + jx] += wi * preload[dy, off + jx]

                for jx in range(i_x, nx):
                    u[base + jx] += wi * preload[dy, jx - i_x]

        row = curvature_matrix[sp]

        for i1 in range(n_pix):
            ui = u[flat[i1]]

            for t in range(indptr[i1], indptr[i1 + 1]):
                row[col[t]] += val[t] * ui

    return curvature_matrix


_PARALLEL_CACHE: dict = {}


def direct_conv_parallel_kernel():
    """
    Compile (once) :func:`curvature_direct_conv` with `prange` over source columns.

    Built lazily rather than decorated at import time for three reasons: `numba.prange`
    has to be resolvable in the function's own scope under `nopython`;
    `numba_util.jit` cannot express `parallel=True` (it takes the shared library-wide
    `general.yaml` numba options); and the thread count numba bakes in is read from
    `NUMBA_NUM_THREADS` at *its* import, so a thread-scaling arm must set that variable
    before this is first called.

    Each source column owns its accumulator `u` and writes only its own row of `F`, so
    the parallel loop needs neither a reduction nor a lock -- the parallel kernel returns
    exactly the serial kernel's matrix, which the tests pin.

    Returns
    -------
    The compiled parallel kernel, with the same signature as
    :func:`curvature_direct_conv`.
    """
    if "kernel" in _PARALLEL_CACHE:
        return _PARALLEL_CACHE["kernel"]

    import numba
    from numba import prange

    @numba.njit(cache=True, parallel=True, nogil=True)
    def _curvature_direct_conv_parallel(
        preload,
        iy,
        ix,
        flat,
        indptr,
        col,
        val,
        cscptr,
        csc_row,
        csc_val,
        ny,
        nx,
        pix_pixels,
    ):
        n_pix = indptr.shape[0] - 1
        m_cells = ny * nx
        nx2 = 2 * nx

        curvature_matrix = np.zeros((pix_pixels, pix_pixels))

        for sp in prange(pix_pixels):
            u = np.zeros(m_cells)

            for t in range(cscptr[sp], cscptr[sp + 1]):
                i0 = csc_row[t]
                wi = csc_val[t]
                i_y = iy[i0]
                i_x = ix[i0]
                off = nx2 - i_x

                for jy in range(ny):
                    dy = jy - i_y
                    base = jy * nx

                    for jx in range(i_x):
                        u[base + jx] += wi * preload[dy, off + jx]

                    for jx in range(i_x, nx):
                        u[base + jx] += wi * preload[dy, jx - i_x]

            row = curvature_matrix[sp]

            for i1 in range(n_pix):
                ui = u[flat[i1]]

                for t in range(indptr[i1], indptr[i1 + 1]):
                    row[col[t]] += val[t] * ui

        return curvature_matrix

    _PARALLEL_CACHE["kernel"] = _curvature_direct_conv_parallel
    return _curvature_direct_conv_parallel


def kernel_inputs_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_sizes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    extent_index_for_masked_pixel: np.ndarray,
    extent_shape,
    pix_pixels: int,
) -> dict:
    """
    Flatten a mapper's `[N_pix, P_max]` triplet arrays into the flat CSR / CSC / extent
    layout :func:`curvature_direct_conv` consumes.

    The rows are indexed on the *unmasked extent* rectangle, which is the grid the `W~`
    operator lives on -- the same grid `InversionInterferometerSparse` builds its COO
    triplets against. The extent row / column of each masked pixel is recovered from the
    mask's own `extent_index_for_masked_pixel` (`iy = flat // nx`, `ix = flat % nx`)
    rather than from a re-origined `native_index_for_slim_index`, so nothing new has to
    be stored on the dataset and the two paths cannot drift.

    A real `Mapper` pads its `[N_pix, P_max]` triplet rows to the longest row, so the
    valid entries are the first `pix_sizes_for_sub_slim_index[i]` of each. Selecting them
    with a boolean mask keeps C (row-major) order, which is exactly CSR order, and keeps
    this out of Python: the marshalling runs once per likelihood evaluation and a
    per-pixel loop here would show up as kernel cost.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index, pix_sizes_for_sub_slim_index, pix_weights_for_sub_slim_index
        The mapper's dense triplet arrays. Weights are used as-is, which is only correct
        without over-sampling (`sub_fraction == 1`) -- the caller enforces that.
    extent_index_for_masked_pixel
        The mask's flat extent-grid index of every masked pixel, ordered to match the
        triplet rows.
    extent_shape
        The `(ny, nx)` shape of the unmasked extent, i.e.
        `mask.shape_native_masked_pixels`.
    pix_pixels
        The number of source pixels, i.e. `mapper.params`.

    Returns
    -------
    A dict of the flat arrays and scalars the kernel takes, keyed by its argument names
    (`preload` excepted, which the caller supplies).
    """
    flat = np.asarray(extent_index_for_masked_pixel, dtype=np.int64)
    sizes = np.asarray(pix_sizes_for_sub_slim_index, dtype=np.int64)
    indexes = np.asarray(pix_indexes_for_sub_slim_index, dtype=np.int64)
    weights = np.asarray(pix_weights_for_sub_slim_index, dtype=np.float64)

    ny, nx = int(extent_shape[0]), int(extent_shape[1])

    n_pix = flat.shape[0]

    iy = flat // nx
    ix = flat % nx

    indptr = np.zeros(n_pix + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    nnz = int(indptr[-1])

    valid = np.arange(indexes.shape[1], dtype=np.int64)[None, :] < sizes[:, None]

    col = np.ascontiguousarray(indexes[valid])
    val = np.ascontiguousarray(weights[valid])

    # Source-major (CSC) view of the same triplets.
    row_of_nnz = np.repeat(np.arange(n_pix, dtype=np.int64), sizes)
    order = np.argsort(col, kind="stable")
    csc_row = row_of_nnz[order]
    csc_val = val[order]
    counts = np.bincount(col, minlength=int(pix_pixels)).astype(np.int64)
    cscptr = np.zeros(int(pix_pixels) + 1, dtype=np.int64)
    np.cumsum(counts, out=cscptr[1:])

    return {
        "iy": np.ascontiguousarray(iy),
        "ix": np.ascontiguousarray(ix),
        "flat": np.ascontiguousarray(flat),
        "indptr": indptr,
        "col": col,
        "val": val,
        "cscptr": cscptr,
        "csc_row": np.ascontiguousarray(csc_row),
        "csc_val": np.ascontiguousarray(csc_val),
        "n_pix": n_pix,
        "nnz": nnz,
        "ny": ny,
        "nx": nx,
        "pix_pixels": int(pix_pixels),
    }


def nnz_per_source_column_from(mapper) -> float:
    """
    The mean number of non-zero mapping weights per source column of `A`.

    The kernel's convolution step costs `O(nnz · M)` while the FFT route's cost is set by
    the number of source *columns* rather than their density, so at a fixed extent the two
    cross at a fixed non-zeros-per-column value (~60 on Delaunay meshes, ~77 on
    rectangular ones at the autolens_profiling#226 geometries). That value moves with the
    extent, so the default routing uses :func:`curvature_route_seconds_from`; this ratio is
    what an explicit `Settings.interferometer_numba_nnz_per_source_max` hard gate compares
    against.

    `nnz = pix_sizes_for_sub_slim_index.sum()` is the total number of valid triplets and
    `mapper.params` the number of source pixels, so this is simply their ratio.
    """
    nnz = float(np.asarray(mapper.pix_sizes_for_sub_slim_index).sum())
    params = int(mapper.params)

    if params == 0:
        return 0.0

    return nnz / params


def _fft_length_cost(n: int) -> float:
    """
    The work of one length-`n` complex FFT, as `scipy.fft`'s pocketfft backend estimates it.

    This mirrors pocketfft's own planner heuristic (`util::cost_guess`): `n` times the sum
    of its prime factors, with a 1.1 penalty on factors above 5, so a smooth length such as
    `560 = 2^4 * 5 * 7` costs `560 * 20` while an awkward one such as
    `682 = 2 * 11 * 31` costs `682 * 48.2` -- 2.4x more per element although
    `log2(682)` and `log2(560)` differ by 3 %. That factorisation penalty is what makes the
    FFT route relatively slower on the Aris masks (extents 83 and 341, so FFT lengths
    `166 = 2 * 83` and `682`) than `n log n` predicts, and is the reason the route gate is
    a cost model rather than a fixed non-zeros-per-column constant.

    For a length with a large prime factor (`p^2 > n`, `n >= 50`) pocketfft switches to
    Bluestein's algorithm when that is cheaper -- two FFTs of the smallest 11-smooth length
    `>= 2n - 1`, with pocketfft's 1.5 fudge factor -- and so does this estimate.
    """
    n = int(n)

    def cost_guess(m: int) -> float:
        m_in = m
        result = 0.0

        while m % 4 == 0:
            result += 2.0
            m //= 4

        while m % 2 == 0:
            result += 2.0
            m //= 2

        x = 3
        while x * x <= m:
            while m % x == 0:
                result += float(x) if x <= 5 else 1.1 * float(x)
                m //= x
            x += 2

        if m > 1:
            result += float(m) if m <= 5 else 1.1 * float(m)

        return result * float(m_in)

    def largest_prime_factor(m: int) -> int:
        largest = 1
        x = 2
        while x * x <= m:
            while m % x == 0:
                largest = x
                m //= x
            x += 1
        return m if m > 1 else largest

    def good_size(m: int) -> int:
        """The smallest `2^a 3^b 5^c 7^d 11^e >= m` (pocketfft's `good_size_cmplx`)."""
        best = 2 * m
        f11 = 1
        while f11 < best:
            f7 = f11
            while f7 < best:
                f5 = f7
                while f5 < best:
                    f3 = f5
                    while f3 < best:
                        f2 = f3
                        while f2 < m:
                            f2 *= 2
                        best = min(best, f2)
                        f3 *= 3
                    f5 *= 5
                f7 *= 7
            f11 *= 11
        return best

    direct = cost_guess(n)

    if n < 50:
        return direct

    p = largest_prime_factor(n)

    if p * p <= n:
        return direct

    bluestein = 1.5 * 2.0 * cost_guess(good_size(2 * n - 1))

    return min(direct, bluestein)


def fft_route_work_from(source_pixels: int, extent_shape) -> float:
    """
    The FFT route's curvature work, in pocketfft cost units.

    `InterferometerSparseOperator.curvature_matrix_diag_from` (NumPy branch) applies `W~`
    to every source column once: an `rfft2` and an `irfft2` of the `(2ny, 2nx)` padded
    extent, in batches of `batch_size` columns. Batching changes the memory traffic, not
    the number of transforms, so the work is `S` times one forward plus one inverse 2D
    real transform. A 2D real transform of shape `(a, b)` is `a` real transforms of length
    `b` (half the cost of a complex one) followed by `b // 2 + 1` complex transforms of
    length `a`.

    The sparse `Aᵀ G` projection (`S · nnz` multiply-adds) is left out: the `direct_conv`
    kernel does the identical gather, so it cancels in the comparison.
    """
    ny, nx = int(extent_shape[0]), int(extent_shape[1])
    a, b = 2 * ny, 2 * nx

    per_transform = 0.5 * a * _fft_length_cost(b) + (b // 2 + 1) * _fft_length_cost(a)

    return float(source_pixels) * 2.0 * per_transform


def direct_conv_work_from(nnz: int, extent_shape) -> float:
    """
    The `direct_conv` kernel's curvature work, in multiply-adds.

    :func:`curvature_direct_conv` scatters a shifted copy of the `W~` kernel over the whole
    `(ny, nx)` extent rectangle for every non-zero of the mapping operator, so the
    convolution step is `nnz · ny · nx` multiply-adds whatever the number of source pixels
    (its `S · nnz` gather is shared with the FFT route and left out of both).
    """
    ny, nx = int(extent_shape[0]), int(extent_shape[1])

    return float(nnz) * float(ny) * float(nx)


# Seconds per unit of `fft_route_work_from` and of `direct_conv_work_from`, fitted by least
# squares on log(seconds) to 18 single-thread timings (`curvature_route_seconds_from`):
#
# - Aris 512x512 mask (91,507 masked pixels, extent 341x341, Delaunay, nnz 273,661), whole
#   NumPy likelihood call (the curvature build is 96-98 % of it), PyAutoArray#628:
#   FFT 28.3 / 44.1 / 97.3 s and direct_conv 38.4 / 34.2 / 41.2 s at S = 1,459 / 2,577 / 5,746.
# - autolens_profiling#226 synthetic bake-off (`bakeoff_v2026.8.17.1.json`, curvature build
#   only, `rfft2_numpy` = the same algorithm as the NumPy FFT route): sma / alma / alma_high
#   extents 70^2 / 140^2 / 280^2, Delaunay S = 1,500 (P = 3) and rectangular S = 784 (P = 4).
#
# Only the ratio of the two constants (0.430) decides a route. Measured / predicted
# FFT-to-direct_conv time ratio over the nine paired points lies in 0.68-1.47 (log-rms 0.26),
# and the predicted faster route matches the measured one at all nine:
#
#   point                    nnz/col  measured t_fft/t_dc  predicted
#   bake-off sma Delaunay       7.7          5.93            6.62
#   bake-off sma rect          19.7          2.83            2.59
#   bake-off alma Delaunay     30.8          1.54            1.88
#   bake-off alma rect         78.5          0.99            0.74
#   bake-off alma_high Del.   123.1          0.63            0.47
#   bake-off alma_high rect   314.1          0.27            0.18
#   Aris 512, S = 1,459       187.6          0.74            0.89
#   Aris 512, S = 2,577       106.2          1.29            1.57
#   Aris 512, S = 5,746        47.6          2.36            3.49
#
# The absolute residuals are larger (log-rms 0.55 FFT, 0.52 direct_conv): the two campaigns ran
# on the same i9-10885H ~2.5x apart in clock state (the bake-off documents the drift), which
# moves both routes together and cancels in the ratio. Implied crossovers in non-zeros per
# source column: ~51 / 58 / 58 at the 70^2 / 140^2 / 280^2 bake-off extents (measured ~60 Delaunay,
# ~77 rectangular), ~166 at the Aris 512 extent 341^2 (measured between 106 and 188) and ~309 at
# the Aris 125 extent 83^2, whose FFT length 166 = 2 * 83 is pocketfft's Bluestein case.
#
# The FFT work form was checked (not fitted) with a single-thread `rfft2` + `irfft2`
# micro-benchmark at padded lengths 140 / 166 / 240 / 280 / 560 / 662 / 682: seconds per
# `fft_route_work_from` unit varied 1.8x across them, against 5.8x for seconds per `N log2 N`
# -- the awkward lengths 166, 662 and 682 cost 2-5x more per element than smooth ones.
FFT_ROUTE_SECONDS_PER_UNIT = 7.62e-10
DIRECT_CONV_SECONDS_PER_UNIT = 1.77e-09


def curvature_route_seconds_from(source_pixels: int, nnz: int, extent_shape):
    """
    Predicted single-thread seconds of the FFT route and of the `direct_conv` route.

    Both are computed from quantities known before any curvature work is done -- the
    number of source pixels `S = mapper.params`, the mapping operator's non-zeros
    `nnz = mapper.pix_sizes_for_sub_slim_index.sum()` and the extent rectangle
    `(ny, nx) = mask.shape_native_masked_pixels` (the sparse operator's
    `(y_shape, x_shape)`) -- so the factory can route on them.

    The model is

        t_fft = c_fft · S · [rfft2 + irfft2 work on (2ny, 2nx)]
        t_dc  = c_dc  · nnz · ny · nx

    The FFT route scales with the number of source *columns* and with the padded extent's
    transform cost, including its factorisation; `direct_conv` scales with the number of
    non-zeros times the extent area and is flat in `S` at fixed `nnz` (34-41 s from 2,577
    to 5,746 vertices on the Aris 512x512 mask). A fixed non-zeros-per-column gate is the
    special case of this model at one extent, which is why it could be right for the
    125x125 reference geometry (crossover ~60) and wrong for the 512x512 one (crossover
    ~130-150).

    Returns
    -------
    `(t_fft, t_direct_conv)` in seconds on the calibration machine.
    """
    t_fft = FFT_ROUTE_SECONDS_PER_UNIT * fft_route_work_from(
        source_pixels=source_pixels, extent_shape=extent_shape
    )
    t_direct_conv = DIRECT_CONV_SECONDS_PER_UNIT * direct_conv_work_from(
        nnz=nnz, extent_shape=extent_shape
    )

    return t_fft, t_direct_conv
