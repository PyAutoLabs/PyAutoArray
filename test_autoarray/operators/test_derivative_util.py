import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray.operators import derivative_util


def circular_mask_from(shape, radius, pixel_scale=1.0):
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    mask = np.ones(shape, dtype=bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            r = np.sqrt(((i - cy) * pixel_scale) ** 2 + ((j - cx) * pixel_scale) ** 2)
            if r <= radius:
                mask[i, j] = False
    return mask


def grids_of_unmasked_pixels_from(mask, pixel_scale=1.0):
    grid = aa.Grid2D.uniform(shape_native=mask.shape, pixel_scales=pixel_scale)
    ygrid = np.array(grid.native[:, :, 0])[~mask]
    xgrid = np.array(grid.native[:, :, 1])[~mask]
    return ygrid, xgrid


def test__cleaned_mask_from__removes_pixels_without_difference_scheme():
    mask = np.ones((7, 7), dtype=bool)
    mask[2:5, 2:5] = False
    mask[0, 0] = False  # isolated unmasked pixel, no neighbours

    cleaned, diff_types = derivative_util.cleaned_mask_from(mask)

    assert cleaned[0, 0]
    assert not cleaned[2:5, 2:5].any()
    assert (diff_types[~cleaned] >= 0).all()


def test__cleaned_mask_from__already_clean_mask_is_unchanged():
    mask = circular_mask_from(shape=(11, 11), radius=4.0)
    cleaned, _ = derivative_util.cleaned_mask_from(mask)

    cleaned_again, _ = derivative_util.cleaned_mask_from(cleaned)

    assert (cleaned_again == cleaned).all()


def test__derivative_1st_operators_from__exact_on_linear_function():
    pixel_scale = 0.5
    mask = circular_mask_from(shape=(12, 12), radius=2.4, pixel_scale=pixel_scale)
    mask, _ = derivative_util.cleaned_mask_from(mask)

    ygrid, xgrid = grids_of_unmasked_pixels_from(mask, pixel_scale=pixel_scale)
    values = 2.0 * ygrid + 3.0 * xgrid

    Hy, Hx = derivative_util.derivative_1st_operators_from(
        mask, pixel_scale=pixel_scale
    )

    assert Hy @ values == pytest.approx(2.0 * np.ones_like(values), abs=1.0e-10)
    assert Hx @ values == pytest.approx(3.0 * np.ones_like(values), abs=1.0e-10)


def test__derivative_1st_operators_from__raises_on_unclean_mask():
    mask = np.ones((7, 7), dtype=bool)
    mask[2:5, 2:5] = False
    mask[0, 0] = False

    with pytest.raises(exc.MaskException):
        derivative_util.derivative_1st_operators_from(mask)


def test__derivative_2nd_operators_from__exact_on_quadratic_function():
    pixel_scale = 0.5
    mask = circular_mask_from(shape=(12, 12), radius=2.4, pixel_scale=pixel_scale)
    mask, _ = derivative_util.cleaned_mask_from(mask)

    ygrid, xgrid = grids_of_unmasked_pixels_from(mask, pixel_scale=pixel_scale)
    values = ygrid**2 + 0.5 * xgrid**2

    Hyy, Hxx = derivative_util.derivative_2nd_operators_from(
        mask, pixel_scale=pixel_scale
    )

    assert Hyy @ values == pytest.approx(2.0 * np.ones_like(values), abs=1.0e-9)
    assert Hxx @ values == pytest.approx(1.0 * np.ones_like(values), abs=1.0e-9)


def test__forward_difference_operators_from__order_degrades_at_grid_edge():
    mask = np.zeros((4, 4), dtype=bool)

    Hy, Hx = derivative_util.forward_difference_operators_from(mask, max_order=2)

    values = np.ones(16)

    # Differences of any order >= 1 annihilate a constant; only the
    # zeroth-order rows (pixels with no unmasked pixel ahead) return it.
    result_x = Hx @ values
    result_y = Hy @ values

    for count, (i, j) in enumerate(np.argwhere(np.ones((4, 4), dtype=bool))):
        assert result_x[count] == pytest.approx(1.0 if j == 3 else 0.0, abs=1.0e-10)
        assert result_y[count] == pytest.approx(1.0 if i == 3 else 0.0, abs=1.0e-10)


def test__forward_difference_operators_from__fourth_order_annihilates_cubic():
    mask = np.zeros((10, 10), dtype=bool)
    pixel_scale = 1.0

    ygrid, xgrid = grids_of_unmasked_pixels_from(mask, pixel_scale=pixel_scale)
    values = xgrid**3

    Hy, Hx = derivative_util.forward_difference_operators_from(
        mask, pixel_scale=pixel_scale, max_order=4
    )

    result_x = Hx @ values

    # Rows with the full 4-point run ahead of them carry the fourth-order
    # stencil, which annihilates a cubic exactly.
    for count, (i, j) in enumerate(np.argwhere(np.ones((10, 10), dtype=bool))):
        if j <= 5:
            assert result_x[count] == pytest.approx(0.0, abs=1.0e-9)


def test__forward_difference_operators_from__masked_run_truncates_stencil_order():
    mask = np.zeros((1, 5), dtype=bool)
    mask[0, 3] = True

    Hy, Hx = derivative_util.forward_difference_operators_from(mask, max_order=2)

    # Unmasked pixels are columns j = 0, 1, 2, 4. Runs ahead: j=0 has j=1, 2
    # (order 2); j=1 has j=2 only (j=3 masked, order 1); j=2 has none
    # (order 0); j=4 is at the edge (order 0).
    expected_hx = np.array(
        [
            [1.0, -2.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert Hx.toarray() == pytest.approx(expected_hx, abs=1.0e-10)


def test__forward_difference_reg_matrix_from__symmetric_positive_semi_definite():
    mask = circular_mask_from(shape=(10, 10), radius=3.5)
    mask, _ = derivative_util.cleaned_mask_from(mask)

    for max_order in (2, 4):
        reg_matrix = derivative_util.forward_difference_reg_matrix_from(
            mask, max_order=max_order
        ).toarray()

        assert reg_matrix == pytest.approx(reg_matrix.T, abs=1.0e-12)
        eigenvalues = np.linalg.eigvalsh(reg_matrix)
        assert eigenvalues.min() > -1.0e-10
