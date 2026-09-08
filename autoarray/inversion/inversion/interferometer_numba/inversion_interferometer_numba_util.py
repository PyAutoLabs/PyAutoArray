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
    NumPy `InterferometerSparseOperator` paths) while the source columns stay sparse --
    measured crossover ~60 non-zeros per source column on Delaunay meshes, ~77 on
    rectangular ones (autolens_profiling issue #226).

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

    This is the geometry the `direct_conv` kernel's cost scales with, and therefore the
    quantity the factory gates on: the kernel's convolution step costs `O(nnz · M)`,
    while the FFT route's cost is set by the number of source *columns* rather than their
    density, so the two cross at a roughly fixed non-zeros-per-column value (~60 on
    Delaunay meshes, ~77 on rectangular ones).

    `nnz = pix_sizes_for_sub_slim_index.sum()` is the total number of valid triplets and
    `mapper.params` the number of source pixels, so this is simply their ratio.
    """
    nnz = float(np.asarray(mapper.pix_sizes_for_sub_slim_index).sum())
    params = int(mapper.params)

    if params == 0:
        return 0.0

    return nnz / params
