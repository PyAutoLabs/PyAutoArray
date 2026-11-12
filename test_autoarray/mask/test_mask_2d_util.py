from autoarray import exc
from autoarray import util

import numpy as np
import pytest


def test__native_index_for_slim_index_2d_from():
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    sub_mask_index_for_sub_mask_1d_index = (
        util.mask_2d.native_index_for_slim_index_2d_from(mask_2d=mask)
    )

    assert (sub_mask_index_for_sub_mask_1d_index == np.array([[1, 1]])).all()

    mask = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
            [True, True, False],
        ]
    )

    sub_mask_index_for_sub_mask_1d_index = (
        util.mask_2d.native_index_for_slim_index_2d_from(mask_2d=mask)
    )

    assert (
        sub_mask_index_for_sub_mask_1d_index
        == np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [3, 2]])
    ).all()


def test__mask_2d_circular_from():
    mask = util.mask_2d.mask_2d_circular_from(
        shape_native=(3, 3), pixel_scales=(1.0, 1.0), radius=0.5
    )

    assert (
        mask == np.array([[True, True, True], [True, False, True], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_circular_from(
        shape_native=(3, 3), pixel_scales=(1.0, 1.0), radius=3.0
    )

    assert (
        mask
        == np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )
    ).all()

    mask = util.mask_2d.mask_2d_circular_from(
        shape_native=(4, 3), pixel_scales=(1.0, 1.0), radius=1.5001
    )

    assert (
        mask
        == np.array(
            [
                [True, False, True],
                [False, False, False],
                [False, False, False],
                [True, False, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_circular_from(
        shape_native=(4, 4), pixel_scales=(1.0, 1.0), radius=0.72
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_circular_from(
        shape_native=(3, 3), pixel_scales=(3.0, 3.0), radius=0.5, centre=(3, 3)
    )

    assert (
        mask == np.array([[True, True, False], [True, True, True], [True, True, True]])
    ).all()


def test__mask_2d_circular_annular_from():
    mask = util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        inner_radius=0.0,
        outer_radius=0.5,
    )

    assert (
        mask == np.array([[True, True, True], [True, False, True], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(4, 4),
        pixel_scales=(1.0, 1.0),
        inner_radius=0.81,
        outer_radius=2.0,
    )

    assert (
        mask
        == np.array(
            [
                [True, False, False, True],
                [False, True, True, False],
                [False, True, True, False],
                [True, False, False, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        inner_radius=0.5,
        outer_radius=3.0,
    )

    assert (
        mask
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()

    mask = util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(4, 3),
        pixel_scales=(1.0, 1.0),
        inner_radius=0.51,
        outer_radius=1.51,
    )

    assert (
        mask
        == np.array(
            [
                [True, False, True],
                [False, True, False],
                [False, True, False],
                [True, False, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(4, 3),
        pixel_scales=(1.0, 1.0),
        inner_radius=1.51,
        outer_radius=3.0,
    )

    assert (
        mask
        == np.array(
            [
                [False, True, False],
                [True, True, True],
                [True, True, True],
                [False, True, False],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        inner_radius=0.5,
        outer_radius=9.0,
        centre=(-3.0, 3.0),
    )

    assert mask.shape == (3, 3)
    assert (
        mask
        == np.array(
            [[False, False, False], [False, False, False], [False, False, True]]
        )
    ).all()


def test__mask_2d_elliptical_from():
    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        major_axis_radius=0.5,
        axis_ratio=1.0,
        angle=0.0,
    )

    assert (
        mask == np.array([[True, True, True], [True, False, True], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        major_axis_radius=1.3,
        axis_ratio=0.1,
        angle=0.0,
    )

    assert (
        mask
        == np.array([[True, True, True], [False, False, False], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        major_axis_radius=1.3,
        axis_ratio=0.1,
        angle=180.0,
    )

    assert (
        mask
        == np.array([[True, True, True], [False, False, False], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        major_axis_radius=1.3,
        axis_ratio=0.1,
        angle=360.0,
    )

    assert (
        mask
        == np.array([[True, True, True], [False, False, False], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(4, 3),
        pixel_scales=(1.0, 1.0),
        major_axis_radius=1.5,
        axis_ratio=0.9,
        angle=90.0,
    )

    assert (
        mask
        == np.array(
            [
                [True, False, True],
                [False, False, False],
                [False, False, False],
                [True, False, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(4, 3),
        pixel_scales=(1.0, 1.0),
        major_axis_radius=1.5,
        axis_ratio=0.1,
        angle=270.0,
    )

    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(3, 4),
        pixel_scales=(1.0, 1.0),
        major_axis_radius=1.5,
        axis_ratio=0.9,
        angle=0.0,
    )

    assert (
        mask
        == np.array(
            [
                [True, False, False, True],
                [False, False, False, False],
                [True, False, False, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_from(
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        major_axis_radius=4.8,
        axis_ratio=0.1,
        angle=45.0,
        centre=(-3.0, 3.0),
    )

    assert (
        mask == np.array([[True, True, True], [True, True, True], [True, True, False]])
    ).all()


def test__mask_2d_elliptical_annular_from():
    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=0.0,
        inner_axis_ratio=1.0,
        inner_phi=0.0,
        outer_major_axis_radius=0.5,
        outer_axis_ratio=1.0,
        outer_phi=0.0,
    )

    assert (
        mask == np.array([[True, True, True], [True, False, True], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=0.5,
        inner_axis_ratio=1.0,
        inner_phi=0.0,
        outer_major_axis_radius=3.0,
        outer_axis_ratio=1.0,
        outer_phi=0.0,
    )

    assert (
        mask
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=0.0,
        inner_axis_ratio=1.0,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.1,
        outer_phi=0.0,
    )

    assert (
        mask
        == np.array([[True, True, True], [False, False, False], [True, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=0.0,
        inner_axis_ratio=1.0,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.1,
        outer_phi=90.0,
    )

    assert (
        mask
        == np.array([[True, False, True], [True, False, True], [True, False, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(3, 3),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=0.0,
        inner_axis_ratio=1.0,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.1,
        outer_phi=45.0,
    )

    assert (
        mask
        == np.array([[True, True, False], [True, False, True], [False, True, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(3, 3),
        pixel_scales=(0.1, 1.0),
        inner_major_axis_radius=0.0,
        inner_axis_ratio=1.0,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.1,
        outer_phi=45.0,
    )

    assert (
        mask
        == np.array([[True, False, True], [True, False, True], [True, False, True]])
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(7, 5),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=1.0,
        inner_axis_ratio=0.1,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.5,
        outer_phi=90.0,
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, False, True, True],
                [True, False, True, False, True],
                [True, True, False, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(7, 5),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=2.0,
        inner_axis_ratio=0.1,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.5,
        outer_phi=90.0,
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(7, 5),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=1.0,
        inner_axis_ratio=0.1,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.8,
        outer_phi=90.0,
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
            ]
        )
    ).all()

    mask = util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(7, 5),
        pixel_scales=(1.0, 1.0),
        inner_major_axis_radius=1.0,
        inner_axis_ratio=0.1,
        inner_phi=0.0,
        outer_major_axis_radius=2.0,
        outer_axis_ratio=0.1,
        outer_phi=90.0,
        centre=(-1.0, 1.0),
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, False, True],
            ]
        )
    ).all()


def test__mask_2d_via_pixel_coordinates_from():
    mask = util.mask_2d.mask_2d_via_pixel_coordinates_from(
        shape_native=(2, 3), pixel_coordinates=[[0, 1], [1, 1], [1, 2]]
    )

    assert (mask == np.array([[True, False, True], [True, False, False]])).all()

    mask = util.mask_2d.mask_2d_via_pixel_coordinates_from(
        shape_native=(7, 7), pixel_coordinates=[[2, 2], [5, 5]], buffer=1
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True, True, True],
                [True, False, False, False, True, True, True],
                [True, False, False, False, True, True, True],
                [True, False, False, False, True, True, True],
                [True, True, True, True, False, False, False],
                [True, True, True, True, False, False, False],
                [True, True, True, True, False, False, False],
            ]
        )
    ).all()


def test__blurring_mask_2d_from():
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    blurring_mask = util.mask_2d.blurring_mask_2d_from(mask, kernel_shape_native=(3, 3))

    assert (
        blurring_mask
        == np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True],
            [True, True, True, True, True, True, True],
        ]
    )

    blurring_mask = util.mask_2d.blurring_mask_2d_from(mask, kernel_shape_native=(3, 3))

    assert (
        blurring_mask
        == np.array(
            [
                [False, False, False, True, False, False, False],
                [False, True, False, True, False, True, False],
                [False, False, False, True, False, False, False],
                [True, True, True, True, True, True, True],
                [False, False, False, True, False, False, False],
                [False, True, False, True, False, True, False],
                [False, False, False, True, False, False, False],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, False, True, True, True, False, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, False, True, True, True, False, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    blurring_mask = util.mask_2d.blurring_mask_2d_from(mask, kernel_shape_native=(5, 3))

    assert (
        blurring_mask
        == np.rot90(
            np.array(
                [
                    [True, True, True, True, True, True, True, True, True],
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, True, False, False, False, True, False, False],
                    [False, False, False, False, False, False, False, False, False],
                    [True, True, True, True, True, True, True, True, True],
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, True, False, False, False, True, False, False],
                    [False, False, False, False, False, False, False, False, False],
                    [True, True, True, True, True, True, True, True, True],
                ]
            )
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    blurring_mask = util.mask_2d.blurring_mask_2d_from(mask, kernel_shape_native=(3, 3))

    assert (
        blurring_mask
        == np.array(
            [
                [False, False, False, True, False, False, False, True, True],
                [False, True, False, True, False, True, False, True, True],
                [False, False, False, True, False, False, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [False, False, False, True, False, False, False, True, True],
                [False, True, False, True, False, True, False, True, True],
                [False, False, False, True, False, False, False, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
        ]
    )

    blurring_mask = util.mask_2d.blurring_mask_2d_from(mask, kernel_shape_native=(3, 3))

    assert (
        blurring_mask
        == np.array(
            [
                [False, False, False, True, False, False, False, True],
                [False, True, False, True, False, True, False, True],
                [False, False, False, True, False, False, False, True],
                [True, True, True, True, True, True, True, True],
                [False, False, False, True, False, False, False, True],
                [False, True, False, True, False, True, False, True],
                [False, False, False, True, False, False, False, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        )
    ).all()


def test__blurring_mask_2d_from__mask_extends_beyond_edge_so_raises_mask_exception():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True],
            [True, True, True, True, True, True, True],
        ]
    )

    with pytest.raises(exc.MaskException):
        util.mask_2d.blurring_mask_2d_from(mask, kernel_shape_native=(5, 5))


def test__mask_1d_indexes_from():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, False, True],
        ]
    )

    unmasked_slim = util.mask_2d.mask_slim_indexes_from(
        mask_2d=mask, return_masked_indexes=False
    )

    assert (unmasked_slim == np.array([23, 24, 25, 47])).all()

    masked_slim = util.mask_2d.mask_slim_indexes_from(
        mask_2d=mask, return_masked_indexes=True
    )

    assert masked_slim[0] == 0
    assert masked_slim[-1] == 48


def test__edge_1d_indexes_from():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    edge_pixels = util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

    assert (edge_pixels == np.array([0])).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, False, False, False, False, False, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    edge_pixels = util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

    assert (
        edge_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, False, False, False, False, False, True],
            [True, False, False, False, False, False, True],
            [True, False, False, False, False, False, True],
            [True, False, False, False, False, False, True],
            [True, False, False, False, False, False, True],
            [True, False, False, False, False, False, True],
            [True, True, True, True, True, True, True],
        ]
    )

    edge_pixels = util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

    assert (
        edge_pixels
        == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29])
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True, True],
            [True, True, False, False, False, True, True, True],
            [True, True, False, False, False, True, True, True],
            [True, False, False, False, False, False, True, True],
            [True, True, False, False, False, True, True, True],
            [True, True, True, True, True, True, True, True],
        ]
    )

    edge_pixels = util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

    assert (edge_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, True, True],
            [True, False, False, False, False, False, True, True],
            [True, False, False, False, False, False, True, True],
            [True, False, False, False, False, False, True, True],
            [True, False, False, False, False, False, True, True],
            [True, True, True, True, True, True, True, True],
        ]
    )

    edge_pixels = util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

    assert (
        edge_pixels
        == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])
    ).all()


def test__border_slim_indexes_from():
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

    border_pixels = util.mask_2d.border_slim_indexes_from(mask_2d=mask)

    assert (border_pixels == np.array([0])).all()

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

    border_pixels = util.mask_2d.border_slim_indexes_from(mask_2d=mask)

    assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

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

    border_pixels = util.mask_2d.border_slim_indexes_from(mask_2d=mask)

    assert (
        border_pixels
        == np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                13,
                14,
                17,
                18,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
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

    border_pixels = util.mask_2d.border_slim_indexes_from(mask_2d=mask)

    assert (
        border_pixels
        == np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                13,
                14,
                17,
                18,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
            ]
        )
    ).all()


def test__rescaled_mask_2d_from():
    mask = np.array(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    )

    rescaled_mask = util.mask_2d.rescaled_mask_2d_from(mask_2d=mask, rescale_factor=1.0)

    assert (
        rescaled_mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )
    ).all()


def test__buffed_mask_2d_from():
    mask = np.array(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    )

    buffed_mask = util.mask_2d.buffed_mask_2d_from(mask_2d=mask, buffer=1)

    assert (
        buffed_mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, False, True],
            [True, True, True, True, True],
        ]
    )

    buffed_mask = util.mask_2d.buffed_mask_2d_from(mask_2d=mask, buffer=1)

    assert (
        buffed_mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    )

    buffed_mask = util.mask_2d.buffed_mask_2d_from(mask_2d=mask, buffer=1)

    assert (
        buffed_mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, False, False, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
        ]
    )

    buffed_mask = util.mask_2d.buffed_mask_2d_from(mask_2d=mask, buffer=1)

    assert (
        buffed_mask
        == np.array(
            [
                [True, True, True, True, True, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, True, True, True, True, True],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, False, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    buffed_mask = util.mask_2d.buffed_mask_2d_from(mask_2d=mask, buffer=2)

    assert (
        buffed_mask
        == np.array(
            [
                [True, True, True, False, False, False, False],
                [True, True, True, False, False, False, False],
                [True, True, True, False, False, False, False],
                [True, True, True, False, False, False, False],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )
    ).all()


def test__trimmed_shape_for_kernel__centred_square_region__shape_is_region_plus_half_kernel_plus_buffer():
    mask = np.full((9, 9), True)
    mask[3:6, 3:6] = False

    # 3x3 region + half-kernel on each side + the default one-pixel buffer on each side.
    assert util.mask_2d.trimmed_shape_for_kernel(mask, kernel_shape_native=(3, 3)) == (
        7,
        7,
    )
    assert util.mask_2d.trimmed_shape_for_kernel(mask, kernel_shape_native=(1, 1)) == (
        5,
        5,
    )
    assert util.mask_2d.trimmed_shape_for_kernel(mask, kernel_shape_native=(5, 3)) == (
        9,
        7,
    )

    # buffer=0 is the exact minimum: region + half-kernel.
    assert util.mask_2d.trimmed_shape_for_kernel(
        mask, kernel_shape_native=(3, 3), buffer=0
    ) == (5, 5)
    assert util.mask_2d.trimmed_shape_for_kernel(
        mask, kernel_shape_native=(1, 1), buffer=0
    ) == (3, 3)
    assert util.mask_2d.trimmed_shape_for_kernel(
        mask, kernel_shape_native=(3, 3), buffer=2
    ) == (9, 9)


def test__trimmed_shape_for_kernel__parity_preserved_and_centre_kept():
    # Even frame: the central pixel is the lower of the central pair (3 of 0..7), and the
    # unmasked 2x2 block sits on pixels 3 and 4, so it is symmetric about the pixel boundary.
    mask = np.full((8, 8), True)
    mask[3:5, 3:5] = False

    shape = util.mask_2d.trimmed_shape_for_kernel(mask, kernel_shape_native=(3, 3))

    # 2 unmasked + 1 half-kernel + 1 buffer on each side.
    assert shape == (6, 6)

    # Cutting to that shape about the centre keeps every unmasked pixel a half-width plus the
    # buffer from the edge.
    trimmed = util.array_2d.resized_array_2d_from(
        array_2d=mask.astype(float), resized_shape=shape, pad_value=1.0
    ).astype(bool)
    assert trimmed.sum() == mask.sum() - (64 - 36)
    assert util.mask_2d.min_false_distance_to_edge(trimmed) == (2, 2)

    assert util.mask_2d.trimmed_shape_for_kernel(
        mask, kernel_shape_native=(3, 3), buffer=0
    ) == (4, 4)

    # Odd frame, region offset from the centre: the frame grows symmetrically so the
    # central pixel stays put, rather than re-centring on the region.
    mask = np.full((9, 9), True)
    mask[1:3, 5:8] = False

    shape = util.mask_2d.trimmed_shape_for_kernel(mask, kernel_shape_native=(3, 3))

    assert shape == (11, 11)
    assert shape[0] % 2 == 1 and shape[1] % 2 == 1
    assert util.mask_2d.trimmed_shape_for_kernel(
        mask, kernel_shape_native=(3, 3), buffer=0
    ) == (9, 9)


def test__trimmed_shape_for_kernel__region_at_edge__frame_pads_rather_than_cuts():
    mask = np.full((5, 5), True)
    mask[0, 2] = False

    shape = util.mask_2d.trimmed_shape_for_kernel(mask, kernel_shape_native=(5, 5))

    # The pixel at row 0 needs two rows above it plus the buffer: 2 before the centre + 2 pad
    # + 1 buffer, so 11 rows; the columns need the half-width plus buffer, 7.
    assert shape == (11, 7)
    assert util.mask_2d.trimmed_shape_for_kernel(
        mask, kernel_shape_native=(5, 5), buffer=0
    ) == (9, 5)


def test__trimmed_shape_for_kernel__invalid_inputs_raise():
    mask = np.full((5, 5), True)
    mask[2, 2] = False

    with pytest.raises(ValueError):
        util.mask_2d.trimmed_shape_for_kernel(mask, kernel_shape_native=(0, 3))

    with pytest.raises(ValueError):
        util.mask_2d.trimmed_shape_for_kernel(
            mask, kernel_shape_native=(3, 3), buffer=-1
        )

    with pytest.raises(ValueError):
        util.mask_2d.trimmed_shape_for_kernel(
            np.full((5, 5), True), kernel_shape_native=(3, 3)
        )
