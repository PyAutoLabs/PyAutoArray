import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.mesh.border_relocator import (
    ellipse_params_via_border_pca_from,
    sub_border_pixel_slim_indexes_from,
)


def test__sub_border_pixel_slim_indexes_from():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask, sub_size=np.array([1])
    )

    assert (sub_border_pixels == np.array([0])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask, sub_size=np.array([1, 1, 1])
    )

    assert (sub_border_pixels == np.array([0, 1, 2])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask, sub_size=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    )

    assert (sub_border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask, sub_size=np.array([2])
    )

    assert (sub_border_pixels == np.array([3])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask, sub_size=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
    )

    assert (sub_border_pixels == np.array([0, 5, 9, 14, 23, 26, 31, 35])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask, sub_size=np.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
    )

    assert (sub_border_pixels == np.array([0, 11, 20, 33, 53, 60, 71, 80])).all()

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
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask,
        sub_size=np.array(
            [
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        ),
    )

    assert (
        sub_border_pixels
        == np.array(
            [
                0,
                4,
                8,
                13,
                17,
                21,
                25,
                28,
                33,
                36,
                53,
                58,
                71,
                74,
                91,
                94,
                99,
                102,
                106,
                110,
                115,
                119,
                123,
                127,
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True, True],
            [True, False, True, True, True, True, True, False, True, True],
            [True, False, True, False, False, False, True, False, True, True],
            [True, False, True, False, True, False, True, False, True, True],
            [True, False, True, False, False, False, True, False, True, True],
            [True, False, True, True, True, True, True, False, True, True],
            [True, False, False, False, False, False, False, False, True, True],
            [True, True, True, True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(
        mask_2d=mask,
        sub_size=np.array(
            [
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        ),
    )

    assert (
        sub_border_pixels
        == np.array(
            [
                0,
                4,
                8,
                13,
                17,
                21,
                25,
                28,
                33,
                36,
                53,
                58,
                71,
                74,
                91,
                94,
                99,
                102,
                106,
                110,
                115,
                119,
                123,
                127,
            ]
        )
    ).all()


def test__sub_border_slim():
    mask = np.array(
        [
            [False, False, False, False, False, False, False, True],
            [False, True, True, True, True, True, False, True],
            [False, True, False, False, False, True, False, True],
            [False, True, False, True, False, True, False, True],
            [False, True, False, False, False, True, False, True],
            [False, True, True, True, True, True, False, True],
            [False, False, False, False, False, False, False, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0))

    sub_border_flat_indexes_util = sub_border_pixel_slim_indexes_from(
        mask_2d=np.array(mask),
        sub_size=np.array(mask.pixels_in_mask * [2]),
    )

    border_relocator = aa.BorderRelocator(
        mask=mask, sub_size=np.array(mask.pixels_in_mask * [2])
    )

    assert border_relocator.sub_border_slim == pytest.approx(
        sub_border_flat_indexes_util, 1e-4
    )


def test__relocated_grid_from__inside_border_no_relocations():
    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1)
    )

    grid = aa.Grid2D.from_mask(
        mask=mask, over_sample_size=np.array(mask.pixels_in_mask * [2])
    )
    grid.over_sampled[1, :] = [0.1, 0.1]

    border_relocator = aa.BorderRelocator(
        mask=mask, sub_size=np.array(mask.pixels_in_mask * [2])
    )

    relocated_grid = border_relocator.relocated_grid_from(grid=grid)

    assert (relocated_grid.over_sampled[1] == np.array([0.1, 0.1])).all()


def test__relocated_grid_from__outside_border_includes_relocations():
    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1)
    )

    grid = aa.Grid2D.from_mask(
        mask=mask, over_sample_size=np.array(mask.pixels_in_mask * [2])
    )
    grid.over_sampled[1, :] = [10.1, 0.1]

    border_relocator = aa.BorderRelocator(
        mask=mask, sub_size=np.array(mask.pixels_in_mask * [2])
    )

    relocated_grid = border_relocator.relocated_grid_from(grid=grid)

    assert relocated_grid.over_sampled[1] == pytest.approx(
        [0.949953439, 0.009405479597], 1e-4
    )


def test__relocated_grid_from__positive_origin_included_in_relocate():
    mask = aa.Mask2D.circular(
        shape_native=(60, 60),
        radius=1.0,
        pixel_scales=(0.1, 0.1),
        centre=(1.0, 1.0),
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)
    grid.over_sampled[1, :] = [11.1, 1.0]

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=grid.over_sample_size)

    relocated_grid = border_relocator.relocated_grid_from(grid=grid)

    assert relocated_grid.over_sampled[1] == pytest.approx([1.95, 1.0], 1e-4)

def test__ellipse_params__near_isotropic_border_uses_deterministic_axis():
    border_grid = np.array(
        [
            [-0.0960155108, 0.0960155108],
            [-0.55, 0.0],
            [-0.0960155108, -0.0960155108],
            [0.0, 0.55],
            [0.0, -0.55],
            [0.0960155108, 0.0960155108],
            [0.55, 0.0],
            [0.0960155108, -0.0960155108],
        ]
    )

    _, a, b, phi = ellipse_params_via_border_pca_from(border_grid=border_grid)

    assert float(phi) == 0.0
    assert float(a) == pytest.approx(0.55 + 1.0e-12)
    assert float(b) == pytest.approx(0.55 + 1.0e-12)


def test__ellipse_params__anisotropic_border_retains_pca_major_axis():
    border_grid = np.array(
        [
            [-2.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, -1.0],
        ]
    )

    _, a, b, phi = ellipse_params_via_border_pca_from(border_grid=border_grid)

    assert abs(float(phi)) == pytest.approx(np.pi / 2.0)
    assert float(a) > float(b)

