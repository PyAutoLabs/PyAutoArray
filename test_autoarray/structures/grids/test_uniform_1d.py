from pathlib import Path
import numpy as np
import pytest

import autoarray as aa


def test__constructor__all_false_mask__native_slim_pixel_scales_origin_correct():
    mask = aa.Mask1D.all_false(shape_slim=(4,), pixel_scales=1.0)
    grid = aa.Grid1D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)

    assert type(grid) == aa.Grid1D
    assert (grid.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (grid.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert grid.pixel_scales == (1.0,)
    assert grid.origin == (0.0,)


def test__constructor__partial_mask__masked_pixel_zero_in_native_and_slim_roundtrip_correct():
    mask = aa.Mask1D(mask=[True, False, False], pixel_scales=1.0)
    grid = aa.Grid1D(values=[1.0, 2.0, 3.0], mask=mask)

    assert type(grid) == aa.Grid1D
    assert (grid.native == np.array([0.0, 2.0, 3.0])).all()
    assert (grid.slim == np.array([2.0, 3.0])).all()
    assert grid.pixel_scales == (1.0,)
    assert grid.origin == (0.0,)

    assert (grid.slim.native == np.array([0.0, 2.0, 3.0])).all()
    assert (grid.native.slim == np.array([2.0, 3.0])).all()


def test__no_mask__4_element_grid__native_slim_pixel_scales_origin_correct():
    grid_1d = aa.Grid1D.no_mask(
        values=[1.0, 2.0, 3.0, 4.0],
        pixel_scales=1.0,
    )

    assert type(grid_1d) == aa.Grid1D
    assert (grid_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (grid_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert grid_1d.pixel_scales == (1.0,)
    assert grid_1d.origin == (0.0,)


def test__no_mask__4_element_grid_with_origin__origin_stored_and_accessible():
    grid_1d = aa.Grid1D.no_mask(
        values=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0, origin=(1.0,)
    )

    assert type(grid_1d) == aa.Grid1D
    assert (grid_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (grid_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert grid_1d.pixel_scales == (1.0,)
    assert grid_1d.origin == (1.0,)


def test__from_mask__all_false_4_element__native_coordinates_are_pixel_centers():
    mask = aa.Mask1D.all_false(shape_slim=(4,), pixel_scales=1.0)
    grid = aa.Grid1D.from_mask(mask=mask)

    assert type(grid) == aa.Grid1D
    assert (grid.native == np.array([-1.5, -0.5, 0.5, 1.5])).all()
    assert (grid.slim == np.array([-1.5, -0.5, 0.5, 1.5])).all()
    assert grid.pixel_scales == (1.0,)
    assert grid.origin == (0.0,)


def test__from_mask__partial_mask__masked_pixel_zero_in_native_slim_has_unmasked_coordinates():
    mask = aa.Mask1D(mask=[True, False, False, False], pixel_scales=1.0)
    grid = aa.Grid1D.from_mask(mask=mask)

    assert type(grid) == aa.Grid1D
    assert (grid.native == np.array([0.0, -0.5, 0.5, 1.5])).all()
    assert (grid.slim == np.array([-0.5, 0.5, 1.5])).all()
    assert grid.pixel_scales == (1.0,)
    assert grid.origin == (0.0,)


def test__uniform__2_element_pixel_scale_1_origin_0__native_at_half_pixel_centers():
    grid_1d = aa.Grid1D.uniform(shape_native=(2,), pixel_scales=1.0, origin=(0.0,))

    assert type(grid_1d) == aa.Grid1D
    assert (grid_1d.native == np.array([-0.5, 0.5])).all()
    assert (grid_1d.slim == np.array([-0.5, 0.5])).all()
    assert grid_1d.pixel_scales == (1.0,)
    assert grid_1d.origin == (0.0,)


def test__uniform__2_element_pixel_scale_1_origin_1__native_offset_by_origin():
    grid_1d = aa.Grid1D.uniform(shape_native=(2,), pixel_scales=1.0, origin=(1.0,))

    assert type(grid_1d) == aa.Grid1D
    assert (grid_1d.native == np.array([0.5, 1.5])).all()
    assert (grid_1d.slim == np.array([0.5, 1.5])).all()
    assert grid_1d.pixel_scales == (1.0,)
    assert grid_1d.origin == (1.0,)


def test__uniform_from_zero__2_element_pixel_scale_1__starts_at_zero():
    grid_1d = aa.Grid1D.uniform_from_zero(
        shape_native=(2,),
        pixel_scales=1.0,
    )

    assert type(grid_1d) == aa.Grid1D
    assert (grid_1d.native == np.array([0.0, 1.0])).all()
    assert (grid_1d.slim == np.array([0.0, 1.0])).all()
    assert grid_1d.pixel_scales == (1.0,)
    assert grid_1d.origin == (0.0,)


def test__uniform_from_zero__3_element_pixel_scale_1p5__starts_at_zero_with_correct_spacing():
    grid_1d = aa.Grid1D.uniform_from_zero(
        shape_native=(3,),
        pixel_scales=1.5,
    )

    assert type(grid_1d) == aa.Grid1D
    assert (grid_1d.native == np.array([0.0, 1.5, 3.0])).all()
    assert (grid_1d.slim == np.array([0.0, 1.5, 3.0])).all()
    assert grid_1d.pixel_scales == (1.5,)
    assert grid_1d.origin == (0.0,)


def test__grid_2d_radial_projected_from__angle_0__projects_along_x_axis():
    grid_1d = aa.Grid1D.no_mask(
        values=[1.0, 2.0, 3.0, 4.0],
        pixel_scales=1.0,
    )

    grid_2d = grid_1d.grid_2d_radial_projected_from(angle=0.0)

    assert type(grid_2d) == aa.Grid2DIrregular
    assert grid_2d.slim == pytest.approx(
        np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]), abs=1.0e-4
    )


def test__grid_2d_radial_projected_from__angle_90__projects_along_negative_y_axis():
    grid_1d = aa.Grid1D.no_mask(
        values=[1.0, 2.0, 3.0, 4.0],
        pixel_scales=1.0,
    )

    grid_2d = grid_1d.grid_2d_radial_projected_from(angle=90.0)

    assert grid_2d.slim == pytest.approx(
        np.array([[-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0], [-4.0, 0.0]]), abs=1.0e-4
    )


def test__grid_2d_radial_projected_from__angle_45__projects_along_diagonal():
    grid_1d = aa.Grid1D.no_mask(
        values=[1.0, 2.0, 3.0, 4.0],
        pixel_scales=1.0,
    )

    grid_2d = grid_1d.grid_2d_radial_projected_from(angle=45.0)

    assert grid_2d.slim == pytest.approx(
        np.array(
            [
                [-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                [-1.0 * np.sqrt(2), 1.0 * np.sqrt(2)],
                [-1.5 * np.sqrt(2), 1.5 * np.sqrt(2)],
                [-2.0 * np.sqrt(2), 2.0 * np.sqrt(2)],
            ]
        ),
        1.0e-4,
    )


def test__recursive_shape_storage__no_mask__slim_native_slim_roundtrip_correct():
    mask = aa.Mask1D.all_false(shape_slim=(4,), pixel_scales=1.0)
    grid = aa.Grid1D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)

    assert (grid.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (grid.native.slim.native == np.array([1.0, 2.0, 3.0, 4.0])).all()


def test__recursive_shape_storage__partial_mask__slim_native_slim_roundtrip_correct():
    mask = aa.Mask1D(mask=[True, False, False], pixel_scales=1.0)
    grid = aa.Grid1D(values=[1.0, 2.0, 3.0], mask=mask)

    assert (grid.slim.native.slim == np.array([2.0, 3.0])).all()
    assert (grid.native.slim.native == np.array([0.0, 2.0, 3.0])).all()
