"""
Sparse bilinear interpolation from a coarse regular mesh to the unmasked
pixels of a finer 2D grid.

Used by the gravitational-imaging (potential correction) technique, where
pixelized corrections to the lensing potential are defined on a mesh a factor
coarser than the data grid and interpolated onto it. The interpolation is a
sparse matrix of shape [n_unmasked_fine_pixels, n_unmasked_coarse_pixels],
each row holding the bilinear weights of the four coarse-mesh corners of the
box enclosing (or nearest to) the fine pixel.

The implementations in this module are ported from the ``potential_correction``
package of Cao et al. 2025 (https://github.com/caoxiaoyue/lensing_potential_correction).
If you use this functionality in your research, please cite Cao et al. 2025;
citation materials are provided at
https://github.com/caoxiaoyue/potential_correction_paper.
"""

import numpy as np
from scipy.sparse import csr_matrix

from autoarray import exc
from autoarray import numba_util


@numba_util.jit()
def binned_image_from(arr, bin_factor=1):
    """
    Bins a 2D array by averaging over ``bin_factor`` x ``bin_factor`` blocks.
    """
    n0 = arr.shape[0] // bin_factor
    n1 = arr.shape[1] // bin_factor
    binned_arr = np.zeros((n0, n1), dtype=np.float64)
    n_per_bin = bin_factor**2
    for i in range(n0):
        for j in range(n1):
            for m in range(i * bin_factor, (i + 1) * bin_factor):
                for n in range(j * bin_factor, (j + 1) * bin_factor):
                    binned_arr[i, j] += arr[m, n]
            binned_arr[i, j] = binned_arr[i, j] / n_per_bin
    return binned_arr


@numba_util.jit()
def binned_mask_from(mask, bin_factor):
    """
    The mask of a grid binned coarser by ``bin_factor``: a coarse pixel is
    unmasked only if *every* fine pixel inside it is unmasked.

    The fine mask's shape must be divisible by ``bin_factor``.
    """
    unmask = (~mask).astype(np.float64)
    binned_unmask = binned_image_from(unmask, bin_factor)
    return ~np.isclose(binned_unmask, 1.0)


@numba_util.jit()
def interp_box_mask_from(mask):
    """
    The mask of the interpolation boxes of a coarse mesh: box (i, j) — whose
    corners are mesh pixels (i, j), (i, j+1), (i+1, j), (i+1, j+1) — is
    unmasked only if all four corners are unmasked.

    Returns a bool array of shape [n0 - 1, n1 - 1] for a mask of shape
    [n0, n1].
    """
    itp_box_mask = np.ones((mask.shape[0] - 1, mask.shape[1] - 1), dtype="bool")
    for i in range(mask.shape[0] - 1):
        for j in range(mask.shape[1] - 1):
            if (
                (~mask[i, j])
                and (~mask[i + 1, j])
                and (~mask[i, j + 1])
                and (~mask[i + 1, j + 1])
            ):
                itp_box_mask[i, j] = False
    return itp_box_mask


@numba_util.jit()
def bilinear_weights_from_box(box_x, box_y, position=(0.0, 0.0)):
    """
    The bilinear interpolation (or extrapolation) weights of ``position``
    inside the square box with corner coordinates ``box_x`` / ``box_y``.

    Parameters
    ----------
    box_x
        The 4 corner x coordinates, in [top-left, top-right, bottom-left,
        bottom-right] order.
    box_y
        The 4 corner y coordinates, in the same order.
    position
        The (y, x) coordinates at which the weights are evaluated.

    Returns
    -------
    An array of shape [4,] holding the weights in the same corner order.
    """
    y, x = position
    box_size = box_x[1] - box_x[0]
    wx = (x - box_x[0]) / box_size
    wy = (y - box_y[2]) / box_size

    weight_top_left = (1 - wx) * wy
    weight_top_right = wx * wy
    weight_bottom_left = (1 - wx) * (1 - wy)
    weight_bottom_right = wx * (1 - wy)

    return np.array(
        [weight_top_left, weight_top_right, weight_bottom_left, weight_bottom_right]
    )


@numba_util.jit()
def coarse_interp_triplets_from(
    mask_itp_box,
    xc_itp_box,
    yc_itp_box,
    xgrid_fine_1d,
    ygrid_fine_1d,
    xgrid_coarse,
    ygrid_coarse,
    mask_coarse,
):
    """
    The (rows, cols, values) sparse triplets of the coarse-to-fine bilinear
    interpolation matrix (see ``coarse_interp_matrix_from``).

    Each unmasked fine pixel is assigned the unmasked interpolation box whose
    centre is nearest to it (searching outward if the naively nearest box is
    masked), and receives the bilinear weights of that box's four corners.

    Parameters
    ----------
    mask_itp_box
        The interpolation-box mask (see ``interp_box_mask_from``).
    xc_itp_box, yc_itp_box
        The 2D x / y coordinates of the interpolation-box centres.
    xgrid_fine_1d, ygrid_fine_1d
        The 1D x / y coordinates of the unmasked fine-grid pixels.
    xgrid_coarse, ygrid_coarse
        The 2D x / y coordinates of the coarse mesh.
    mask_coarse
        The coarse-mesh mask.
    """
    unmask_coarse = ~mask_coarse
    i_indices_unmasked, j_indices_unmasked = np.where(unmask_coarse)
    n_unmasked_coarse_pixels = len(i_indices_unmasked)
    index_dict_coarse = {}
    for count in range(n_unmasked_coarse_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        index_dict_coarse[(i, j)] = count

    n_unmasked_fine_pixels = len(xgrid_fine_1d)
    rows_itp_mat = np.full(n_unmasked_fine_pixels * 4, -1, dtype=np.int64)
    cols_itp_mat = np.full(n_unmasked_fine_pixels * 4, -1, dtype=np.int64)
    data_itp_mat = np.full(n_unmasked_fine_pixels * 4, 0.0, dtype=np.float64)
    count_itp_mat = 0
    for count in range(n_unmasked_fine_pixels):
        this_x_fine = xgrid_fine_1d[count]
        this_y_fine = ygrid_fine_1d[count]

        j_min = np.argmin(np.abs(xc_itp_box[0, :] - this_x_fine))
        i_min = np.argmin(np.abs(yc_itp_box[:, 0] - this_y_fine))

        if ~mask_itp_box[i_min, j_min]:
            i = i_min
            j = j_min
        else:
            search_width = 2
            i = -1
            j = -1
            dist_tmp = 1e8
            while i == -1:
                m_lo = max(0, i_min - search_width)
                m_hi = min(mask_itp_box.shape[0], i_min + search_width + 1)
                n_lo = max(0, j_min - search_width)
                n_hi = min(mask_itp_box.shape[1], j_min + search_width + 1)
                for m in range(m_lo, m_hi):
                    for n in range(n_lo, n_hi):
                        if ~mask_itp_box[m, n]:
                            this_dist = np.sqrt(
                                (xc_itp_box[m, n] - this_x_fine) ** 2
                                + (yc_itp_box[m, n] - this_y_fine) ** 2
                            )
                            if this_dist < dist_tmp:
                                dist_tmp = this_dist
                                i = m
                                j = n
                search_width += 1

        itp_box_corners_x = (
            xgrid_coarse[i, j],
            xgrid_coarse[i, j + 1],
            xgrid_coarse[i + 1, j],
            xgrid_coarse[i + 1, j + 1],
        )
        itp_box_corners_y = (
            ygrid_coarse[i, j],
            ygrid_coarse[i, j + 1],
            ygrid_coarse[i + 1, j],
            ygrid_coarse[i + 1, j + 1],
        )
        itp_weights = bilinear_weights_from_box(
            itp_box_corners_x, itp_box_corners_y, position=(this_y_fine, this_x_fine)
        )
        itp_idx = (
            index_dict_coarse[(i, j)],
            index_dict_coarse[(i, j + 1)],
            index_dict_coarse[(i + 1, j)],
            index_dict_coarse[(i + 1, j + 1)],
        )

        for k in range(4):
            rows_itp_mat[count_itp_mat] = count
            cols_itp_mat[count_itp_mat] = itp_idx[k]
            data_itp_mat[count_itp_mat] = itp_weights[k]
            count_itp_mat += 1

    return rows_itp_mat, cols_itp_mat, data_itp_mat


def coarse_interp_matrix_from(
    mask_itp_box,
    xc_itp_box,
    yc_itp_box,
    xgrid_fine_1d,
    ygrid_fine_1d,
    xgrid_coarse,
    ygrid_coarse,
    mask_coarse,
):
    """
    The sparse bilinear interpolation matrix mapping a vector on the unmasked
    pixels of a coarse regular mesh to the unmasked pixels of a finer grid.

    See ``coarse_interp_triplets_from`` for the parameters.

    Returns
    -------
    A ``scipy.sparse.csr_matrix`` of shape
    [n_unmasked_fine_pixels, n_unmasked_coarse_pixels].
    """
    mask_itp_box = np.asarray(mask_itp_box)
    if np.count_nonzero(~mask_itp_box) == 0:
        raise exc.MeshException(
            "The coarse mesh has no unmasked interpolation box (no 2x2 block of "
            "unmasked coarse pixels) — it is too sparse for the fine grid. "
            "Decrease the coarsening factor."
        )
    rows, cols, data = coarse_interp_triplets_from(
        mask_itp_box,
        np.asarray(xc_itp_box),
        np.asarray(yc_itp_box),
        np.asarray(xgrid_fine_1d),
        np.asarray(ygrid_fine_1d),
        np.asarray(xgrid_coarse),
        np.asarray(ygrid_coarse),
        mask_coarse,
    )
    n_unmasked_coarse = np.count_nonzero(~np.asarray(mask_coarse))
    return csr_matrix(
        (data, (rows, cols)), shape=(len(xgrid_fine_1d), n_unmasked_coarse)
    )
