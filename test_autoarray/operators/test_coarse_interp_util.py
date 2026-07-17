import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray.operators import coarse_interp_util
from autoarray.operators import derivative_util


def test__binned_mask_from__coarse_pixel_unmasked_only_if_all_fine_unmasked():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = True

    binned_mask = coarse_interp_util.binned_mask_from(mask, 2)

    assert binned_mask.shape == (2, 2)
    assert binned_mask[0, 0]
    assert not binned_mask[0, 1]
    assert not binned_mask[1, 0]
    assert not binned_mask[1, 1]


def test__interp_box_mask_from__box_unmasked_only_if_all_four_corners_unmasked():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True

    box_mask = coarse_interp_util.interp_box_mask_from(mask)

    assert box_mask.shape == (2, 2)
    assert box_mask[0, 0]
    assert not box_mask[0, 1]
    assert not box_mask[1, 0]
    assert not box_mask[1, 1]


def test__bilinear_weights_from_box__corners_and_centre():
    box_x = np.array([0.0, 1.0, 0.0, 1.0])
    box_y = np.array([1.0, 1.0, 0.0, 0.0])

    weights = coarse_interp_util.bilinear_weights_from_box(
        box_x, box_y, position=(1.0, 0.0)
    )
    assert weights == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1.0e-10)

    weights = coarse_interp_util.bilinear_weights_from_box(
        box_x, box_y, position=(0.5, 0.5)
    )
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=1.0e-10)


def coarse_fine_setup(fine_shape=(20, 20), fine_pixel_scale=0.5, factor=2):
    """
    A circular fine mask with its coarse binned counterpart and all the grids
    the interpolation matrix needs, mirroring how the potential-correction
    mesh pairs a coarse dpsi mesh to the data grid.
    """
    cy = (fine_shape[0] - 1) / 2.0
    cx = (fine_shape[1] - 1) / 2.0
    mask_fine = np.ones(fine_shape, dtype=bool)
    for i in range(fine_shape[0]):
        for j in range(fine_shape[1]):
            r = np.sqrt(
                ((i - cy) * fine_pixel_scale) ** 2 + ((j - cx) * fine_pixel_scale) ** 2
            )
            if r <= 3.6:
                mask_fine[i, j] = False
    mask_fine, _ = derivative_util.cleaned_mask_from(mask_fine)

    grid_fine = aa.Grid2D.uniform(shape_native=fine_shape, pixel_scales=fine_pixel_scale)
    xgrid_fine_1d = np.array(grid_fine.native[:, :, 1])[~mask_fine]
    ygrid_fine_1d = np.array(grid_fine.native[:, :, 0])[~mask_fine]

    coarse_shape = (fine_shape[0] // factor, fine_shape[1] // factor)
    coarse_pixel_scale = fine_pixel_scale * factor
    mask_coarse = coarse_interp_util.binned_mask_from(mask_fine, factor)

    grid_coarse = aa.Grid2D.uniform(
        shape_native=coarse_shape, pixel_scales=coarse_pixel_scale
    )
    xgrid_coarse = np.array(grid_coarse.native[:, :, 1])
    ygrid_coarse = np.array(grid_coarse.native[:, :, 0])

    box_centres = aa.Grid2D.uniform(
        shape_native=(coarse_shape[0] - 1, coarse_shape[1] - 1),
        pixel_scales=coarse_pixel_scale,
    )
    xc_itp_box = np.array(box_centres.native[:, :, 1])
    yc_itp_box = np.array(box_centres.native[:, :, 0])
    mask_itp_box = coarse_interp_util.interp_box_mask_from(mask_coarse)

    return (
        mask_itp_box,
        xc_itp_box,
        yc_itp_box,
        xgrid_fine_1d,
        ygrid_fine_1d,
        xgrid_coarse,
        ygrid_coarse,
        mask_coarse,
    )


def test__coarse_interp_matrix_from__rows_are_partition_of_unity():
    setup = coarse_fine_setup()

    itp_mat = coarse_interp_util.coarse_interp_matrix_from(*setup)

    row_sums = np.asarray(itp_mat.sum(axis=1)).ravel()
    assert row_sums == pytest.approx(np.ones_like(row_sums), abs=1.0e-10)


def test__coarse_interp_matrix_from__raises_when_no_unmasked_interp_box():
    setup = list(coarse_fine_setup())
    setup[0] = np.ones_like(setup[0])  # mask every interpolation box

    with pytest.raises(exc.MeshException):
        coarse_interp_util.coarse_interp_matrix_from(*setup)


def test__coarse_interp_matrix_from__box_search_stays_in_bounds_at_mesh_edge():
    # A coarse mesh whose only unmasked box sits in one corner: fine pixels
    # far from it force the nearest-box search to widen well beyond the mesh
    # bounds, which must clamp (the original implementation wrapped around
    # via negative indexing and crashed).
    mask_coarse = np.ones((5, 5), dtype=bool)
    mask_coarse[0:2, 0:2] = False

    coarse_pixel_scale = 1.0
    grid_coarse = aa.Grid2D.uniform(shape_native=(5, 5), pixel_scales=coarse_pixel_scale)
    xgrid_coarse = np.array(grid_coarse.native[:, :, 1])
    ygrid_coarse = np.array(grid_coarse.native[:, :, 0])

    box_centres = aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=coarse_pixel_scale)
    xc_itp_box = np.array(box_centres.native[:, :, 1])
    yc_itp_box = np.array(box_centres.native[:, :, 0])
    mask_itp_box = coarse_interp_util.interp_box_mask_from(mask_coarse)

    # fine positions in the far opposite corner of the mesh
    xgrid_fine_1d = np.array([2.0, 1.5])
    ygrid_fine_1d = np.array([-2.0, -1.5])

    itp_mat = coarse_interp_util.coarse_interp_matrix_from(
        mask_itp_box,
        xc_itp_box,
        yc_itp_box,
        xgrid_fine_1d,
        ygrid_fine_1d,
        xgrid_coarse,
        ygrid_coarse,
        mask_coarse,
    )

    row_sums = np.asarray(itp_mat.sum(axis=1)).ravel()
    assert row_sums == pytest.approx(np.ones_like(row_sums), abs=1.0e-10)


def test__coarse_interp_matrix_from__exact_on_bilinear_function():
    setup = coarse_fine_setup()
    (
        mask_itp_box,
        xc_itp_box,
        yc_itp_box,
        xgrid_fine_1d,
        ygrid_fine_1d,
        xgrid_coarse,
        ygrid_coarse,
        mask_coarse,
    ) = setup

    def f(y, x):
        return 1.5 + 2.0 * x - 3.0 * y + 0.5 * x * y

    values_coarse = f(ygrid_coarse[~mask_coarse], xgrid_coarse[~mask_coarse])

    itp_mat = coarse_interp_util.coarse_interp_matrix_from(*setup)

    values_fine = itp_mat @ values_coarse

    assert values_fine == pytest.approx(
        f(ygrid_fine_1d, xgrid_fine_1d), abs=1.0e-10
    )
