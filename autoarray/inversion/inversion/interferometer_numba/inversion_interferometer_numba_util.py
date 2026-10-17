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


@numba_util.jit()
def curvature_direct_conv_symmetric(
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
    order,
    rank,
    ylo_suffix,
    xlo_row,
    xhi_row,
):
    """
    `F = Aᵀ W~ A` by the :func:`curvature_direct_conv` convolution, skipping the work
    that `F`'s symmetry and the mask's footprint make redundant.

    :func:`curvature_direct_conv` convolves every source column over the whole `(ny, nx)`
    extent rectangle and gathers every entry of its row of `F`. Two cuts remove work
    that cannot change the result:

    1. **Symmetry, via y-ranked columns.** The source columns are processed in the order
       `order` (by the top extent row `ymin` of their non-zeros, see
       :func:`symmetric_restriction_inputs_from`); `rank` is the inverse permutation.
       Column `sp` of rank `r` only computes the entries `F[sp, c]` with
       `rank[c] >= r`, and the others are filled by a mirror pass at the end
       (`F[c, sp] = F[sp, c]`). Every pixel `A[:, c]` touches lies in an extent row
       `>= ymin(c)`, so column `sp` only needs `u = W~ A[:, sp]` on the rows
       `>= ylo_suffix[r] = min_{rank(c) >= r} ymin(c)` -- the rows above are neither
       convolved nor gathered.
    2. **The mask's x-range per row.** `u` is only ever read at masked pixels, so each
       extent row `jy` is convolved only over `[xlo_row[jy], xhi_row[jy])`, the x-range
       of the masked pixels in that row (rows with no masked pixel are skipped).

    Both cuts are exact for *any* column ordering: whatever `order` is, each computed
    entry `F[sp, c]` receives exactly the contributions :func:`curvature_direct_conv`
    gives it (the restricted rows and columns hold every masked pixel `A[:, c]` touches),
    and every entry is either computed or mirrored. The ordering only decides how much
    work is skipped. Ranking by each column's top row `ymin` (stable; empty columns last)
    makes `ylo_suffix[r] = ymin(order[r])`, the largest the suffix minimum can be for
    every rank, so column `r` convolves only from its own top row down: about half the
    rows on average for compact columns, more for tall ones (`work_fraction` measures it
    exactly). On Aris's real 512x512 Delaunay mesh, where multiple imaging makes columns
    tall (median span 90 of 341 rows), `ymin` order gives f = 0.51 against 0.59 for
    centroid order (PyAutoArray#635).

    The mirror relies on `F` being symmetric, i.e. on `W~` being even in the pixel offset
    (`preload[-dy, -dx] = preload[dy, dx]` with wrapped indices), which the
    interferometer `nufft_precision_operator` is by construction.

    The gather is column-major and branch-free: for rank `r` it computes, for each later-
    ranked column `c = order[k]`, `k >= r`, the dot product
    `F[sp, c] = Σ_{t in column c} val_t · u[flat[row_t]]` over that column's own non-zeros
    -- `Σ_{k >= r} nnz_k` multiply-adds, about half of the full kernel's `nnz` per column,
    with no per-entry rank test. The CSC non-zeros are first copied into rank order
    (`ranked_flat` / `ranked_val`, `O(nnz)`), so the sweep for rank `r` is one contiguous
    pass over the tail of those arrays. The pixels column `c` touches lie in the rows
    `>= ylo_suffix[r]` and inside the mask x-range by construction, where `u` has just
    been computed.

    Each computed entry is accumulated in the same order as in
    :func:`curvature_direct_conv`: the same `t` loop per extent cell for `u`, and, within
    a column, the CSC non-zeros in increasing pixel index (:func:`kernel_inputs_from`
    sorts them stably), which is the order that kernel's CSR sweep adds them in. So every
    computed entry is bit-identical to that kernel's entry, whatever the pixel ordering.
    A mirrored entry is the transposed computed one, so it equals the full kernel's entry
    to round-off: the full kernel's own `F` is symmetric only to round-off.

    The kernel keeps a single `u` buffer and is serial; the parallel path uses
    :func:`direct_conv_parallel_kernel`, whose columns cost the same whatever their order.

    Parameters
    ----------
    preload, iy, ix, flat, indptr, col, val, cscptr, csc_row, csc_val, ny, nx, pix_pixels
        As :func:`curvature_direct_conv`.
    order
        The source columns in processing order (`[pix_pixels]`).
    rank
        The inverse permutation of `order`: `rank[order[r]] = r`.
    ylo_suffix
        The first extent row column `order[r]` must convolve (`[pix_pixels]`,
        non-decreasing; `ny` for none).
    xlo_row, xhi_row
        Per extent row, the half-open x-range `[xlo, xhi)` of its masked pixels
        (`xhi <= xlo` for a row with none).

    Returns
    -------
    The curvature matrix `F`, of shape `[pix_pixels, pix_pixels]`.
    """
    nx2 = 2 * nx

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))
    u = np.zeros(ny * nx)

    # The CSC non-zeros in rank order, with the pixel's extent-flat index resolved.
    ranked_ptr = np.zeros(pix_pixels + 1, dtype=np.int64)

    for k in range(pix_pixels):
        c = order[k]
        ranked_ptr[k + 1] = ranked_ptr[k] + cscptr[c + 1] - cscptr[c]

    ranked_flat = np.empty(ranked_ptr[pix_pixels], dtype=np.int64)
    ranked_val = np.empty(ranked_ptr[pix_pixels])

    for k in range(pix_pixels):
        c = order[k]
        q = ranked_ptr[k]

        for t in range(cscptr[c], cscptr[c + 1]):
            ranked_flat[q] = flat[csc_row[t]]
            ranked_val[q] = csc_val[t]
            q += 1

    for r in range(pix_pixels):
        sp = order[r]
        y0 = ylo_suffix[r]

        u[y0 * nx :] = 0.0

        for t in range(cscptr[sp], cscptr[sp + 1]):
            i0 = csc_row[t]
            wi = csc_val[t]
            i_y = iy[i0]
            i_x = ix[i0]
            off = nx2 - i_x

            for jy in range(y0, ny):
                x0 = xlo_row[jy]
                x1 = xhi_row[jy]

                if x1 <= x0:
                    continue

                dy = jy - i_y
                base = jy * nx

                for jx in range(x0, min(i_x, x1)):
                    u[base + jx] += wi * preload[dy, off + jx]

                for jx in range(max(i_x, x0), x1):
                    u[base + jx] += wi * preload[dy, jx - i_x]

        row = curvature_matrix[sp]

        for k in range(r, pix_pixels):
            value = 0.0

            for t in range(ranked_ptr[k], ranked_ptr[k + 1]):
                value += ranked_val[t] * u[ranked_flat[t]]

            row[order[k]] = value

    for a in range(pix_pixels):
        for b in range(pix_pixels):
            if rank[b] > rank[a]:
                curvature_matrix[b, a] = curvature_matrix[a, b]

    return curvature_matrix


# The `symmetric_restriction_inputs_from` keys `curvature_direct_conv_symmetric` takes after
# the `curvature_direct_conv` arguments, in argument order.
SYMMETRIC_KERNEL_KEYS = ("order", "rank", "ylo_suffix", "xlo_row", "xhi_row")


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


def _symmetric_restriction_from(
    column_counts, column_ymin, iy, ix, ny, nx, pix_pixels
) -> dict:
    """
    The column ordering, row restriction and per-row mask x-range of
    :func:`curvature_direct_conv_symmetric`, plus its `work_fraction`, from per-source-
    column statistics of the mapping operator and the masked pixels' extent positions.

    Shared by :func:`symmetric_restriction_inputs_from` (from the kernel's CSC arrays) and
    :func:`symmetric_work_fraction_from` (from a mapper's triplets, for the factory's cost
    model), so the gate and the kernel see the same work fraction: `column_ymin` is an
    integer minimum, identical however it was computed, so both get the same order.
    """
    pix_pixels = int(pix_pixels)
    ny, nx = int(ny), int(nx)

    column_counts = np.asarray(column_counts, dtype=np.int64)
    column_ymin = np.asarray(column_ymin, dtype=np.int64)
    non_empty = column_counts > 0

    # Rank by each column's top extent row, stably. An empty column has `ymin = ny`, which
    # exceeds every real row, so it is ranked last and restricted to nothing.
    column_ymin = np.where(non_empty, column_ymin, ny)

    order = np.argsort(column_ymin, kind="stable").astype(np.int64)
    rank = np.empty(pix_pixels, dtype=np.int64)
    rank[order] = np.arange(pix_pixels, dtype=np.int64)

    # Column of rank r needs the rows spanned by every column of rank >= r. In `ymin` order
    # this is just ymin(order[r]); the suffix minimum keeps it right for any order.
    ylo_suffix = np.ascontiguousarray(
        np.minimum.accumulate(column_ymin[order][::-1])[::-1], dtype=np.int64
    )

    iy = np.asarray(iy, dtype=np.int64)
    ix = np.asarray(ix, dtype=np.int64)

    pix_by_row = np.argsort(iy, kind="stable").astype(np.int64)
    pix_by_row_ptr = np.searchsorted(
        iy[pix_by_row], np.arange(ny + 1, dtype=np.int64)
    ).astype(np.int64)

    # Half-open masked x-range of each extent row; a row with no masked pixel keeps the
    # empty range [nx, 0), which the kernel skips.
    xlo_row = np.full(ny, nx, dtype=np.int64)
    xhi_row = np.zeros(ny, dtype=np.int64)

    row_non_empty = np.diff(pix_by_row_ptr) > 0

    if np.any(row_non_empty):
        x_by_row = ix[pix_by_row]
        starts = pix_by_row_ptr[:-1][row_non_empty]
        xlo_row[row_non_empty] = np.minimum.reduceat(x_by_row, starts)
        xhi_row[row_non_empty] = np.maximum.reduceat(x_by_row, starts) + 1

    width = np.maximum(xhi_row - xlo_row, 0)
    width_from_row = np.zeros(ny + 1, dtype=np.int64)
    width_from_row[:ny] = np.cumsum(width[::-1])[::-1]

    nnz = int(column_counts.sum())

    if nnz == 0 or ny * nx == 0:
        work_fraction = 0.0
    else:
        work = np.sum(
            column_counts[order].astype(np.float64)
            * width_from_row[ylo_suffix].astype(np.float64)
        )
        work_fraction = float(work / (float(nnz) * float(ny) * float(nx)))

    return {
        "order": order,
        "rank": rank,
        "ylo_suffix": ylo_suffix,
        "xlo_row": xlo_row,
        "xhi_row": xhi_row,
        "work_fraction": work_fraction,
    }


def symmetric_restriction_inputs_from(
    iy, ix, cscptr, csc_row, ny, nx, pix_pixels
) -> dict:
    """
    The extra inputs of :func:`curvature_direct_conv_symmetric` and the fraction of
    :func:`curvature_direct_conv`'s convolution work it performs.

    - `order` / `rank`: the source columns sorted (stably) by their top extent row
      `ymin` (the lowest row of their non-zeros), and its inverse permutation. Empty
      columns are ranked last.
    - `ylo_suffix[r]`: the suffix minimum, over the columns of rank `>= r`, of each
      column's lowest extent row (`ny` when they are all empty). Column `order[r]` only
      convolves the rows `>= ylo_suffix[r]`.
    - `xlo_row` / `xhi_row`: per extent row the half-open x-range of its masked pixels
      (`[nx, 0)`, i.e. empty, for a row with none).
    - `work_fraction`: the convolution work of the symmetric kernel relative to the full
      one,

          Σ_cols nnz_col · Σ_{y >= ylo_suffix(col)} width(y)  /  (nnz · ny · nx),

      with `width(y) = xhi_row[y] - xlo_row[y]`. It is a property of the geometry and the
      mesh only (PyAutoArray#635): f ~ 0.41 on the synthetic Aris-512 geometry (compact
      3-NN columns), f ~ 0.51 on Aris's real 512x512 Delaunay mesh (multiply imaged,
      tall columns).

    Everything is vectorised NumPy (`reduceat` over the CSC and row-sorted segments), as
    it runs once per likelihood evaluation. The keys the kernel takes, in its argument
    order, are :data:`SYMMETRIC_KERNEL_KEYS`.

    Parameters
    ----------
    iy, ix
        The extent-grid row / column of every masked pixel (:func:`kernel_inputs_from`).
    cscptr, csc_row
        The mapping operator's CSC column pointers and row (masked pixel) indices.
    ny, nx
        The extent rectangle's shape.
    pix_pixels
        The number of source pixels.

    Returns
    -------
    A dict keyed by the kernel's extra argument names (:data:`SYMMETRIC_KERNEL_KEYS`),
    plus `work_fraction`.
    """
    iy = np.asarray(iy, dtype=np.int64)
    cscptr = np.asarray(cscptr, dtype=np.int64)
    csc_row = np.asarray(csc_row, dtype=np.int64)

    pix_pixels = int(pix_pixels)
    ny = int(ny)

    column_counts = np.diff(cscptr)
    non_empty = column_counts > 0

    column_ymin = np.full(pix_pixels, ny, dtype=np.int64)

    if np.any(non_empty):
        column_ymin[non_empty] = np.minimum.reduceat(
            iy[csc_row], cscptr[:-1][non_empty]
        )

    return _symmetric_restriction_from(
        column_counts=column_counts,
        column_ymin=column_ymin,
        iy=iy,
        ix=ix,
        ny=ny,
        nx=nx,
        pix_pixels=pix_pixels,
    )


def symmetric_work_fraction_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_sizes_for_sub_slim_index: np.ndarray,
    extent_index_for_masked_pixel: np.ndarray,
    extent_shape,
    pix_pixels: int,
) -> float:
    """
    The `work_fraction` of :func:`symmetric_restriction_inputs_from`, straight from a
    mapper's triplets, for the factory's route decision.

    It skips the CSC sort :func:`kernel_inputs_from` performs (`bincount` and
    `np.minimum.at` over the valid triplets instead), and returns exactly the value the
    inversion's own restriction inputs will give.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index, pix_sizes_for_sub_slim_index
        The mapper's dense triplet arrays (the weights do not enter).
    extent_index_for_masked_pixel
        The mask's flat extent-grid index of every masked pixel, ordered to match the
        triplet rows.
    extent_shape
        The `(ny, nx)` extent rectangle.
    pix_pixels
        The number of source pixels, i.e. `mapper.params`.
    """
    flat = np.asarray(extent_index_for_masked_pixel, dtype=np.int64)
    sizes = np.asarray(pix_sizes_for_sub_slim_index, dtype=np.int64)
    indexes = np.asarray(pix_indexes_for_sub_slim_index, dtype=np.int64)

    ny, nx = int(extent_shape[0]), int(extent_shape[1])
    pix_pixels = int(pix_pixels)

    iy = flat // nx
    ix = flat % nx

    valid = np.arange(indexes.shape[1], dtype=np.int64)[None, :] < sizes[:, None]
    col = indexes[valid]
    y_of_nnz = np.repeat(iy, sizes)

    column_counts = np.bincount(col, minlength=pix_pixels).astype(np.int64)
    column_ymin = np.full(pix_pixels, ny, dtype=np.int64)
    np.minimum.at(column_ymin, col, y_of_nnz)

    return _symmetric_restriction_from(
        column_counts=column_counts,
        column_ymin=column_ymin,
        iy=iy,
        ix=ix,
        ny=ny,
        nx=nx,
        pix_pixels=pix_pixels,
    )["work_fraction"]


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
# The absolute residuals are larger (log-rms 0.56 FFT, 0.52 direct_conv): the two campaigns ran
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

# Time of the serial symmetric kernel (`curvature_direct_conv_symmetric`) relative to the full
# `curvature_direct_conv`, per unit of its `work_fraction` f: `t_dc` is multiplied by
# `min(1, kappa * f)` when the symmetric kernel runs. Fitted by least squares on
# log(t_sym / t_full) to quiet-core single-thread timings of this library kernel on RAL (job
# 411835, euclid-ral-compute-1, load < 1, A-C-C-A order, spreads <= 1.004; PyAutoPulse#62;
# PyAutoArray#635), synthetic Aris-512 geometry (extent 341x341, circular mask, 91k masked
# pixels, 3-NN weights, nnz 274,083), f from `symmetric_restriction_inputs_from` on that
# geometry. These rows were measured with the columns in centroid-row order, the kernel's
# order at the time; on this geometry of compact columns the `ymin` order it now uses gives
# f = 0.414 / 0.408 / 0.403, so the ordering barely matters here:
#
#   S        f       t_full   t_sym   measured ratio   predicted (kappa f)
#   1,428  0.416    22.04 s  12.19 s      0.553            0.548
#   2,540  0.411    22.52 s  12.17 s      0.540            0.540
#   5,712  0.404    23.88 s  12.59 s      0.527            0.532
#
# Measured / predicted lies in 0.99-1.01 (log-rms 0.008). kappa > 1 because the x-range splits
# and the `S^2` mirror pass cost a little beyond the convolution work f counts; the ranked CSC
# gather (half the non-zeros per column, branch-free) keeps the ratio flat in S. With it the
# Aris 512 S = 1,459 point (measured FFT 28.3 s vs direct_conv 38.4 s on the full kernel) is
# predicted direct_conv-faster (t_fft / t_dc 0.89 -> ~1.6 at f = 0.42).
#
# Check, not part of the fit: Aris's real 512x512 data with its step-8 Delaunay mesh (S = 1,459,
# nnz 273,753, extent 341^2), laptop single-thread timings under falling load. Multiple imaging
# feeds one source vertex from 2-4 lensed images, so the columns are tall (median span 90 of
# 341 rows, against 15 on the synthetic geometry) and the ordering matters:
#
#   order          f      t_full   t_sym   measured speedup   predicted 1 / (kappa f)
#   centroid row  0.587   40.4 s   30.5 s       1.32x              1.29x
#   ymin          0.511   40.4 s   27.6 s       1.46x              1.48x
#
# Re-timed in ymin order on RAL (jobs 412490 / 412492, euclid-ral-compute-1, exclusive, load
# <= 1, compile excluded), same data, step 8 / 6 / 4 meshes:
#
#   vertices   f      t_full   t_sym   measured speedup   predicted 1 / (kappa f)
#   1,459    0.511   22.43 s  15.08 s       1.49x              1.48x
#   2,577    0.504   22.83 s  14.95 s       1.53x              1.50x
#   5,746    0.495   24.22 s  15.35 s       1.58x              1.53x
#
# The kernel wins while kappa f < 1, i.e. f < 0.76; the packaged gate
# (`interferometer_direct_conv_symmetric_max_work_fraction: 0.7`) keeps a predicted gain of at
# least 1.08x.
DIRECT_CONV_SYMMETRIC_WORK_FACTOR = 1.32


def curvature_route_seconds_from(
    source_pixels: int, nnz: int, extent_shape, work_fraction=None
):
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
    non-zeros times the extent area, so at a large extent it is ~flat in `S` at fixed
    `nnz` (34-41 s from 2,577 to 5,746 vertices on the Aris 512x512 mask). Both routes
    also pay `S`-dependent costs the model leaves out because they are common to both
    (the `S · nnz` gather, the dense solve). A fixed non-zeros-per-column gate is the
    special case of this model at one extent, which is why it could be right at the
    autolens_profiling#226 extents (70^2-280^2, crossover ~60) and wrong on the Aris
    512x512 mask (model crossover ~166, measured between 106 and 188).

    When the serial symmetric kernel (:func:`curvature_direct_conv_symmetric`) will run,
    the caller passes its `work_fraction` and

        t_dc *= min(1, DIRECT_CONV_SYMMETRIC_WORK_FACTOR · work_fraction)

    `work_fraction=None` (the default) predicts the full kernel, unchanged.

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

    if work_fraction is not None:
        t_direct_conv *= min(
            1.0, DIRECT_CONV_SYMMETRIC_WORK_FACTOR * float(work_fraction)
        )

    return t_fft, t_direct_conv
