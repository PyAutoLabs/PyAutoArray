import numpy as np
import pytest

import autoarray as aa


def test__all_false():
    grid_2d_util = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(4, 7),
        pixel_scales=(0.56, 0.56),
    )

    grid_1d_util = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(4, 7),
        pixel_scales=(0.56, 0.56),
    )

    mask = aa.Mask2D.all_false(shape_native=(4, 7), pixel_scales=(0.56, 0.56))
    mask[0, 0] = True

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert derive_grid.all_false.slim == pytest.approx(grid_1d_util, 1e-4)
    assert derive_grid.all_false.native == pytest.approx(grid_2d_util, 1e-4)
    assert (derive_grid.all_false.mask == np.full(fill_value=False, shape=(4, 7))).all()

    mask = aa.Mask2D.all_false(shape_native=(3, 3), pixel_scales=(1.0, 1.0))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.all_false.native
        == np.array(
            [
                [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]],
            ]
        )
    ).all()

    grid_2d_util = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(4, 7),
        pixel_scales=(0.8, 0.56),
    )

    grid_1d_util = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(4, 7),
        pixel_scales=(0.8, 0.56),
    )

    mask = aa.Mask2D.all_false(shape_native=(4, 7), pixel_scales=(0.8, 0.56))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert derive_grid.all_false.slim == pytest.approx(grid_1d_util, 1e-4)
    assert derive_grid.all_false.native == pytest.approx(grid_2d_util, 1e-4)

    mask = aa.Mask2D.all_false(shape_native=(3, 3), pixel_scales=(1.0, 2.0))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.all_false.native
        == np.array(
            [
                [[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]],
                [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
                [[-1.0, -2.0], [-1.0, 0.0], [-1.0, 2.0]],
            ]
        )
    ).all()


def test__unmasked():
    mask = aa.Mask2D.all_false(shape_native=(3, 3), pixel_scales=(1.0, 1.0))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.unmasked.slim
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    mask = aa.Mask2D.all_false(shape_native=(3, 3), pixel_scales=(1.0, 1.0))
    mask[1, 1] = True

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.unmasked.slim
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    mask = aa.Mask2D(
        mask=np.array([[False, True], [True, False], [True, False]]),
        pixel_scales=(1.0, 1.0),
        origin=(3.0, -2.0),
    )

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.unmasked.slim == np.array([[4.0, -2.5], [3.0, -1.5], [2.0, -1.5]])
    ).all()


def test__edge():
    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert derive_grid.edge.slim[0:11] == pytest.approx(
        np.array(
            [
                [3.0, -3.0],
                [3.0, -2.0],
                [3.0, -1.0],
                [3.0, -0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, -3.0],
                [2.0, 3.0],
                [1.0, -3.0],
                [1.0, -1.0],
            ]
        ),
        1e-4,
    )


def test__border_sub_1():
    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert derive_grid.border.slim[0:11] == pytest.approx(
        np.array(
            [
                [3.0, -3.0],
                [3.0, -2.0],
                [3.0, -1.0],
                [3.0, -0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, -3.0],
                [2.0, 3.0],
                [1.0, -3.0],
                [1.0, 3.0],
            ]
        ),
        1e-4,
    )


def test__masked_grid():
    mask = aa.Mask2D.all_false(shape_native=(3, 3), pixel_scales=(1.0, 1.0))
    mask[1, 1] = True

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.unmasked
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    mask = aa.Mask2D(
        mask=np.array([[False, True], [True, False], [True, False]]),
        pixel_scales=(1.0, 1.0),
        origin=(3.0, -2.0),
    )

    derive_grid = aa.DeriveGrid2D(mask=mask)

    masked_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask,
        pixel_scales=(1.0, 1.0),
        origin=(3.0, -2.0),
    )

    assert (derive_grid.unmasked == masked_grid_util).all()


def test__border_1d_grid():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, False, False, True, True, True, True],
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.border == np.array([[1.0, -2.0], [1.0, -1.0], [-0.0, 1.0]])
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    derive_grid = aa.DeriveGrid2D(mask=mask)

    assert (
        derive_grid.border
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()
