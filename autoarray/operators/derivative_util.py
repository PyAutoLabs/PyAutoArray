"""
Sparse finite-difference derivative operators on masked 2D grids.

Each operator is a sparse matrix of shape [n_unmasked_pixels, n_unmasked_pixels]
which, acting on a 1D vector of values on the unmasked pixels of a 2D grid,
returns the finite-difference derivative of those values on the same pixels.
They are the building blocks of the gravitational-imaging (potential
correction) technique, where derivatives of the lensing potential and its
pixelized corrections are taken on masked image grids.

The implementations in this module are ported from the ``potential_correction``
package of Cao et al. 2025 (https://github.com/caoxiaoyue/lensing_potential_correction).
If you use this functionality in your research, please cite Cao et al. 2025;
citation materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.

Conventions (matching the rest of autoarray):

- ``mask`` is a bool 2D array where ``True`` means *masked* (excluded).
- Grid coordinates are (y, x), with y *decreasing* as the array index along
  axis 0 increases, and x increasing with the index along axis 1. The y-step
  of ``-pixel_scale`` used below encodes this.
- ``diff_types`` is an int array of shape [total_pixels, total_pixels, 2]
  whose last dimension is (y, x) and whose values select the finite-difference
  scheme available at each unmasked pixel: 0 = backward, 1 = central,
  2 = forward, -1 = none (pixel cannot support a derivative in the direction).
"""

import numpy as np
from scipy.sparse import csr_matrix

from autoarray import exc
from autoarray import numba_util


@numba_util.jit()
def clean_mask_iteration_from(mask):
    """
    One iteration of mask cleaning: masks every unmasked pixel which cannot
    support a first-derivative finite-difference scheme (backward, central or
    forward) in *both* the y and x directions, and records the scheme
    available at each surviving pixel.

    Parameters
    ----------
    mask
        The 2D bool mask (``True`` = masked). Must have shape of at least
        (3, 3) so a difference scheme can exist.

    Returns
    -------
    The cleaned mask and the ``diff_types`` array (see module docstring).
    """
    in_mask = np.copy(mask)
    in_mask = in_mask.astype("bool")
    out_mask = np.ones_like(in_mask, dtype="bool")
    n1, n2 = in_mask.shape
    diff_types = np.full((n1, n2, 2), -1, dtype="int")

    for i in range(n1):
        for j in range(n2):
            if not in_mask[i, j]:
                if_remove_y = True
                if_remove_x = True

                if i == 0:
                    if ~in_mask[i + 1, j] and ~in_mask[i + 2, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 2
                elif i == 1:
                    if ~in_mask[i - 1, j] and ~in_mask[i + 1, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 1
                    elif ~in_mask[i + 1, j] and ~in_mask[i + 2, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 2
                elif i == n1 - 1:
                    if ~in_mask[i - 1, j] and ~in_mask[i - 2, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 0
                elif i == n1 - 2:
                    if ~in_mask[i - 1, j] and ~in_mask[i + 1, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 1
                    elif ~in_mask[i - 1, j] and ~in_mask[i - 2, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 0
                else:
                    if ~in_mask[i - 1, j] and ~in_mask[i + 1, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 1
                    elif ~in_mask[i - 1, j] and ~in_mask[i - 2, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 0
                    elif ~in_mask[i + 1, j] and ~in_mask[i + 2, j]:
                        if_remove_y = False
                        diff_types[i, j, 0] = 2

                if j == 0:
                    if ~in_mask[i, j + 1] and ~in_mask[i, j + 2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 2
                elif j == 1:
                    if ~in_mask[i, j - 1] and ~in_mask[i, j + 1]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 1
                    elif ~in_mask[i, j + 1] and ~in_mask[i, j + 2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 2
                elif j == n2 - 1:
                    if ~in_mask[i, j - 1] and ~in_mask[i, j - 2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 0
                elif j == n2 - 2:
                    if ~in_mask[i, j - 1] and ~in_mask[i, j + 1]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 1
                    elif ~in_mask[i, j - 1] and ~in_mask[i, j - 2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 0
                else:
                    if ~in_mask[i, j - 1] and ~in_mask[i, j + 1]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 1
                    elif ~in_mask[i, j - 1] and ~in_mask[i, j - 2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 0
                    elif ~in_mask[i, j + 1] and ~in_mask[i, j + 2]:
                        if_remove_x = False
                        diff_types[i, j, 1] = 2

                if not (if_remove_y or if_remove_x):
                    out_mask[i, j] = False

    return out_mask, diff_types


def cleaned_mask_from(mask, max_iter: int = 50):
    """
    Iteratively cleans a mask until every unmasked pixel supports a
    first-derivative finite-difference scheme in both directions (see
    ``clean_mask_iteration_from``), returning the cleaned mask and the
    per-pixel scheme types.

    Parameters
    ----------
    mask
        The 2D bool mask (``True`` = masked).
    max_iter
        The maximum number of cleaning iterations before raising.

    Returns
    -------
    The cleaned mask and the ``diff_types`` array (see module docstring).
    """
    old_mask = np.copy(np.asarray(mask))
    clean_success = False

    for _ in range(max_iter):
        new_mask, diff_types = clean_mask_iteration_from(old_mask)
        if (new_mask == old_mask).all():
            clean_success = True
            break
        old_mask = new_mask

    if not clean_success:
        raise exc.MaskException(
            f"The mask is not fully cleaned after {max_iter} iterations"
        )

    return new_mask, diff_types


def _diff_types_of_cleaned_mask_from(mask):
    """
    Returns the ``diff_types`` of a mask, raising if the mask is not already
    cleaned (derivative operators are only defined on cleaned masks).
    """
    mask = np.asarray(mask).astype(bool)
    new_mask, diff_types = cleaned_mask_from(mask)
    if not (new_mask == mask).all():
        raise exc.MaskException(
            "The mask has not been fully cleaned: one or more unmasked pixels do "
            "not support a finite-difference scheme. Clean it first via "
            "cleaned_mask_from(mask)."
        )
    return mask, diff_types


@numba_util.jit()
def derivative_1st_triplets_from(mask, diff_types, dpix=1.0):
    """
    The (rows, cols, values) sparse triplets of the first-derivative operators
    Hx and Hy of a cleaned mask (see ``derivative_1st_operators_from``).
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked)

    rows_hx = np.full(n_unmasked_pixels * 2, -1, dtype=np.int64)
    cols_hx = np.full(n_unmasked_pixels * 2, -1, dtype=np.int64)
    data_hx = np.full(n_unmasked_pixels * 2, 0.0, dtype=np.float64)
    rows_hy = np.full(n_unmasked_pixels * 2, -1, dtype=np.int64)
    cols_hy = np.full(n_unmasked_pixels * 2, -1, dtype=np.int64)
    data_hy = np.full(n_unmasked_pixels * 2, 0.0, dtype=np.float64)

    # y decreases as the index along axis 0 increases; x increases with axis 1.
    step_y = -1.0 * dpix
    step_x = 1.0 * dpix

    index_dict = {}
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict[(i, j)] = count

    count_sparse_hy = 0
    count_sparse_hx = 0
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]

        if diff_types[i, j, 0] == 0:
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i - 1, j)]
            data_hy[count_sparse_hy] = -1.0 / step_y
            count_sparse_hy += 1
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i, j)]
            data_hy[count_sparse_hy] = 1.0 / step_y
            count_sparse_hy += 1
        elif diff_types[i, j, 0] == 1:
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i - 1, j)]
            data_hy[count_sparse_hy] = -1.0 / (2 * step_y)
            count_sparse_hy += 1
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i + 1, j)]
            data_hy[count_sparse_hy] = 1.0 / (2 * step_y)
            count_sparse_hy += 1
        elif diff_types[i, j, 0] == 2:
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i, j)]
            data_hy[count_sparse_hy] = -1.0 / step_y
            count_sparse_hy += 1
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i + 1, j)]
            data_hy[count_sparse_hy] = 1.0 / step_y
            count_sparse_hy += 1

        if diff_types[i, j, 1] == 0:
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i, j - 1)]
            data_hx[count_sparse_hx] = -1.0 / step_x
            count_sparse_hx += 1
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i, j)]
            data_hx[count_sparse_hx] = 1.0 / step_x
            count_sparse_hx += 1
        elif diff_types[i, j, 1] == 1:
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i, j - 1)]
            data_hx[count_sparse_hx] = -1.0 / (2 * step_x)
            count_sparse_hx += 1
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i, j + 1)]
            data_hx[count_sparse_hx] = 1.0 / (2 * step_x)
            count_sparse_hx += 1
        elif diff_types[i, j, 1] == 2:
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i, j)]
            data_hx[count_sparse_hx] = -1.0 / step_x
            count_sparse_hx += 1
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i, j + 1)]
            data_hx[count_sparse_hx] = 1.0 / step_x
            count_sparse_hx += 1

    return rows_hx, cols_hx, data_hx, rows_hy, cols_hy, data_hy


def derivative_1st_operators_from(mask, pixel_scale: float = 1.0):
    """
    The sparse first-derivative operators (Hy, Hx) of a cleaned mask.

    Hy (Hx) has shape [n_unmasked_pixels, n_unmasked_pixels]; acting on a 1D
    vector of values on the unmasked pixels it returns their first
    y (x) derivative, using the central / backward / forward scheme each
    pixel's neighbours permit.

    Parameters
    ----------
    mask
        The cleaned 2D bool mask (``True`` = masked); raises if not cleaned.
    pixel_scale
        The pixel size (e.g. in arcsec) setting the finite-difference step.

    Returns
    -------
    The (Hy, Hx) operators as ``scipy.sparse.csr_matrix``.
    """
    mask, diff_types = _diff_types_of_cleaned_mask_from(mask)
    rows_hx, cols_hx, data_hx, rows_hy, cols_hy, data_hy = (
        derivative_1st_triplets_from(mask, diff_types, dpix=pixel_scale)
    )

    n_unmasked = np.count_nonzero(~mask)
    Hx = csr_matrix((data_hx, (rows_hx, cols_hx)), shape=(n_unmasked, n_unmasked))
    Hy = csr_matrix((data_hy, (rows_hy, cols_hy)), shape=(n_unmasked, n_unmasked))
    return Hy, Hx


@numba_util.jit()
def derivative_2nd_triplets_from(mask, diff_types, dpix=1.0):
    """
    The (rows, cols, values) sparse triplets of the second-derivative
    operators Hxx and Hyy of a cleaned mask (see
    ``derivative_2nd_operators_from``).
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked)

    rows_hxx = np.full(n_unmasked_pixels * 3, -1, dtype=np.int64)
    cols_hxx = np.full(n_unmasked_pixels * 3, -1, dtype=np.int64)
    data_hxx = np.full(n_unmasked_pixels * 3, 0.0, dtype=np.float64)
    rows_hyy = np.full(n_unmasked_pixels * 3, -1, dtype=np.int64)
    cols_hyy = np.full(n_unmasked_pixels * 3, -1, dtype=np.int64)
    data_hyy = np.full(n_unmasked_pixels * 3, 0.0, dtype=np.float64)

    step_y = -1.0 * dpix
    step_x = 1.0 * dpix

    index_dict = {}
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict[(i, j)] = count

    count_sparse_hyy = 0
    count_sparse_hxx = 0
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]

        if diff_types[i, j, 0] == 0:
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i - 2, j)]
            data_hyy[count_sparse_hyy] = 1.0 / step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i - 1, j)]
            data_hyy[count_sparse_hyy] = -2.0 / step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i, j)]
            data_hyy[count_sparse_hyy] = 1.0 / step_y**2
            count_sparse_hyy += 1
        elif diff_types[i, j, 0] == 1:
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i - 1, j)]
            data_hyy[count_sparse_hyy] = 1.0 / step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i, j)]
            data_hyy[count_sparse_hyy] = -2.0 / step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i + 1, j)]
            data_hyy[count_sparse_hyy] = 1.0 / step_y**2
            count_sparse_hyy += 1
        elif diff_types[i, j, 0] == 2:
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i, j)]
            data_hyy[count_sparse_hyy] = 1.0 / step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i + 1, j)]
            data_hyy[count_sparse_hyy] = -2.0 / step_y**2
            count_sparse_hyy += 1
            rows_hyy[count_sparse_hyy] = count
            cols_hyy[count_sparse_hyy] = index_dict[(i + 2, j)]
            data_hyy[count_sparse_hyy] = 1.0 / step_y**2
            count_sparse_hyy += 1

        if diff_types[i, j, 1] == 0:
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j - 2)]
            data_hxx[count_sparse_hxx] = 1.0 / step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j - 1)]
            data_hxx[count_sparse_hxx] = -2.0 / step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j)]
            data_hxx[count_sparse_hxx] = 1.0 / step_x**2
            count_sparse_hxx += 1
        elif diff_types[i, j, 1] == 1:
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j - 1)]
            data_hxx[count_sparse_hxx] = 1.0 / step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j)]
            data_hxx[count_sparse_hxx] = -2.0 / step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j + 1)]
            data_hxx[count_sparse_hxx] = 1.0 / step_x**2
            count_sparse_hxx += 1
        elif diff_types[i, j, 1] == 2:
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j)]
            data_hxx[count_sparse_hxx] = 1.0 / step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j + 1)]
            data_hxx[count_sparse_hxx] = -2.0 / step_x**2
            count_sparse_hxx += 1
            rows_hxx[count_sparse_hxx] = count
            cols_hxx[count_sparse_hxx] = index_dict[(i, j + 2)]
            data_hxx[count_sparse_hxx] = 1.0 / step_x**2
            count_sparse_hxx += 1

    return rows_hxx, cols_hxx, data_hxx, rows_hyy, cols_hyy, data_hyy


def derivative_2nd_operators_from(mask, pixel_scale: float = 1.0):
    """
    The sparse second-derivative operators (Hyy, Hxx) of a cleaned mask.

    Hyy (Hxx) has shape [n_unmasked_pixels, n_unmasked_pixels]; acting on a
    1D vector of values on the unmasked pixels it returns their second
    y (x) derivative. The sum ``Hyy + Hxx`` is the Laplacian (Hamiltonian)
    operator of the masked grid, which converts a lensing potential to a
    convergence (times two).

    Parameters
    ----------
    mask
        The cleaned 2D bool mask (``True`` = masked); raises if not cleaned.
    pixel_scale
        The pixel size (e.g. in arcsec) setting the finite-difference step.

    Returns
    -------
    The (Hyy, Hxx) operators as ``scipy.sparse.csr_matrix``.
    """
    mask, diff_types = _diff_types_of_cleaned_mask_from(mask)
    rows_hxx, cols_hxx, data_hxx, rows_hyy, cols_hyy, data_hyy = (
        derivative_2nd_triplets_from(mask, diff_types, dpix=pixel_scale)
    )

    n_unmasked = np.count_nonzero(~mask)
    Hxx = csr_matrix((data_hxx, (rows_hxx, cols_hxx)), shape=(n_unmasked, n_unmasked))
    Hyy = csr_matrix((data_hyy, (rows_hyy, cols_hyy)), shape=(n_unmasked, n_unmasked))
    return Hyy, Hxx


@numba_util.jit()
def forward_difference_triplets_from(mask, dpix=1.0, max_order=2):
    """
    The (rows, cols, values) sparse triplets of the order-capped forward
    finite-difference operators used to build regularization matrices (see
    ``forward_difference_operators_from``).

    For every unmasked pixel a forward-difference stencil is emitted in each
    direction, of the highest order the run of consecutive unmasked pixels
    ahead of it permits, capped at ``max_order``: order L uses the L+1 point
    stencil with coefficients (-1)^(L-k) C(L, k) / step^L (k = 0..L), and
    order 0 degrades to the identity (a zeroth-order regularization of the
    pixel itself).
    """
    unmask = ~mask
    n1, n2 = unmask.shape
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked)

    # Rows L = 0..4 hold the forward-difference stencil coefficients of order L.
    coeffs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, -2.0, 1.0, 0.0, 0.0],
            [-1.0, 3.0, -3.0, 1.0, 0.0],
            [1.0, -4.0, 6.0, -4.0, 1.0],
        ]
    )

    max_entries = n_unmasked_pixels * (max_order + 1)
    rows_hx = np.full(max_entries, -1, dtype=np.int64)
    cols_hx = np.full(max_entries, -1, dtype=np.int64)
    data_hx = np.full(max_entries, 0.0, dtype=np.float64)
    rows_hy = np.full(max_entries, -1, dtype=np.int64)
    cols_hy = np.full(max_entries, -1, dtype=np.int64)
    data_hy = np.full(max_entries, 0.0, dtype=np.float64)

    step_y = -1.0 * dpix
    step_x = 1.0 * dpix

    index_dict = {}
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict[(i, j)] = count

    count_sparse_hy = 0
    count_sparse_hx = 0
    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]

        order_x = 0
        for k in range(1, max_order + 1):
            if j + k > n2 - 1:
                break
            if not unmask[i, j + k]:
                break
            order_x = k

        norm_x = step_x**order_x
        for k in range(order_x + 1):
            rows_hx[count_sparse_hx] = count
            cols_hx[count_sparse_hx] = index_dict[(i, j + k)]
            data_hx[count_sparse_hx] = coeffs[order_x, k] / norm_x
            count_sparse_hx += 1

        order_y = 0
        for k in range(1, max_order + 1):
            if i + k > n1 - 1:
                break
            if not unmask[i + k, j]:
                break
            order_y = k

        norm_y = step_y**order_y
        for k in range(order_y + 1):
            rows_hy[count_sparse_hy] = count
            cols_hy[count_sparse_hy] = index_dict[(i + k, j)]
            data_hy[count_sparse_hy] = coeffs[order_y, k] / norm_y
            count_sparse_hy += 1

    return (
        rows_hx[:count_sparse_hx],
        cols_hx[:count_sparse_hx],
        data_hx[:count_sparse_hx],
        rows_hy[:count_sparse_hy],
        cols_hy[:count_sparse_hy],
        data_hy[:count_sparse_hy],
    )


def forward_difference_operators_from(
    mask, pixel_scale: float = 1.0, max_order: int = 2
):
    """
    The sparse order-capped forward finite-difference operators (Hy, Hx) of a
    mask, used to build derivative-penalising regularization matrices.

    Unlike the pure derivative operators above, every unmasked pixel receives
    a row: the stencil order degrades gracefully (``max_order`` → ... → 1 → 0)
    with the run of consecutive unmasked pixels ahead of it, so pixels at the
    mask edge are regularized at lower order rather than dropped. With
    ``max_order=2`` this reproduces the curvature regularization scheme of
    potential-correction methods; ``max_order=4`` the fourth-order scheme.

    Parameters
    ----------
    mask
        The 2D bool mask (``True`` = masked).
    pixel_scale
        The pixel size setting the finite-difference step (regularization
        matrices are conventionally built with the default of 1.0, the
        regularization coefficient absorbing the scale).
    max_order
        The highest stencil order to emit (2 or 4).

    Returns
    -------
    The (Hy, Hx) operators as ``scipy.sparse.csr_matrix``.
    """
    if max_order not in (1, 2, 3, 4):
        raise ValueError(f"max_order must be in 1..4, got {max_order}")

    mask = np.asarray(mask).astype(bool)
    rows_hx, cols_hx, data_hx, rows_hy, cols_hy, data_hy = (
        forward_difference_triplets_from(mask, dpix=pixel_scale, max_order=max_order)
    )

    n_unmasked = np.count_nonzero(~mask)
    Hx = csr_matrix((data_hx, (rows_hx, cols_hx)), shape=(n_unmasked, n_unmasked))
    Hy = csr_matrix((data_hy, (rows_hy, cols_hy)), shape=(n_unmasked, n_unmasked))
    return Hy, Hx


def forward_difference_reg_matrix_from(
    mask, pixel_scale: float = 1.0, max_order: int = 2
):
    """
    The regularization matrix H = Hx^T Hx + Hy^T Hy built from the
    order-capped forward finite-difference operators of a mask (see
    ``forward_difference_operators_from``).

    Parameters
    ----------
    mask
        The 2D bool mask (``True`` = masked).
    pixel_scale
        The finite-difference step (default 1.0 for regularization use).
    max_order
        The highest stencil order (2 = curvature, 4 = fourth-order).

    Returns
    -------
    The [n_unmasked, n_unmasked] regularization matrix as a
    ``scipy.sparse.csr_matrix``.
    """
    Hy, Hx = forward_difference_operators_from(
        mask, pixel_scale=pixel_scale, max_order=max_order
    )
    return (Hx.T @ Hx + Hy.T @ Hy).tocsr()
