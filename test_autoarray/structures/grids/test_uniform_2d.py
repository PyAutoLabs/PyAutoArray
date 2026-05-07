from pathlib import Path
import numpy as np
import pytest

from autoconf import conf
import autoarray as aa
from autoarray import exc

test_grid_dir = Path(__file__).resolve().parent / "files"


def test__constructor__2x2_all_false_mask__native_slim_and_pixel_scales_correct():
    mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0)
    grid_2d = aa.Grid2D(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], mask=mask
    )

    assert type(grid_2d) == aa.Grid2D
    assert (
        grid_2d.native == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ).all()
    assert (
        grid_2d.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()
    assert grid_2d.pixel_scales == (1.0, 1.0)
    assert grid_2d.origin == (0.0, 0.0)


def test__constructor__exception_raised_if_input_grid_is_2d_and_not_shape_of_mask():
    with pytest.raises(exc.GridException):
        mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0)
        aa.Grid2D(values=[[[1.0, 1.0], [3.0, 3.0]]], mask=mask)


def test__constructor__1d_values_too_many_for_masked_pixels__raises_grid_exception():
    with pytest.raises(exc.GridException):
        mask = aa.Mask2D(mask=[[False, False], [True, False]], pixel_scales=1.0)
        aa.Grid2D(values=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], mask=mask)


def test__constructor__1d_values_too_few_for_masked_pixels__raises_grid_exception():
    with pytest.raises(exc.GridException):
        mask = aa.Mask2D(mask=[[False, False], [True, False]], pixel_scales=1.0)
        aa.Grid2D(values=[[1.0, 1.0], [2.0, 2.0]], mask=mask)


def test__no_mask__2x2_native_values__native_slim_pixel_scales_and_origin_correct():
    grid_2d = aa.Grid2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        pixel_scales=1.0,
    )

    assert type(grid_2d) == aa.Grid2D
    assert (grid_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
    assert (
        grid_2d.native == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ).all()
    assert (
        grid_2d.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()

    assert grid_2d.pixel_scales == (1.0, 1.0)
    assert grid_2d.origin == (0.0, 0.0)


def test__no_mask__with_origin__origin_stored_and_accessible():
    grid_2d = aa.Grid2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        shape_native=(4, 2),
        pixel_scales=1.0,
        origin=(0.0, 1.0),
    )

    assert type(grid_2d) == aa.Grid2D
    assert (grid_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
    assert (
        grid_2d.native == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ).all()
    assert (
        grid_2d.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()
    assert grid_2d.pixel_scales == (1.0, 1.0)
    assert grid_2d.origin == (0.0, 1.0)


def test__from_yx_2d__2x1_grid__correct_native_slim_and_pixel_scales():
    grid_2d = aa.Grid2D.from_yx_2d(
        y=[[1.0], [3.0]], x=[[2.0], [4.0]], pixel_scales=(2.0, 3.0)
    )

    assert type(grid_2d) == aa.Grid2D
    assert (grid_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (grid_2d.native == np.array([[[1.0, 2.0]], [[3.0, 4.0]]])).all()
    assert (grid_2d.slim == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert grid_2d.pixel_scales == (2.0, 3.0)
    assert grid_2d.origin == (0.0, 0.0)

    grid_2d = aa.Grid2D.from_yx_1d(
        y=[1.0, 3.0, 5.0, 7.0],
        x=[2.0, 4.0, 6.0, 8.0],
        shape_native=(2, 2),
        pixel_scales=1.0,
        origin=(0.0, 1.0),
    )

    assert type(grid_2d) == aa.Grid2D
    assert (grid_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
    assert (
        grid_2d.native == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ).all()
    assert (
        grid_2d.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()
    assert grid_2d.pixel_scales == (1.0, 1.0)
    assert grid_2d.origin == (0.0, 1.0)


def test__from_extent():
    grid_2d = aa.Grid2D.from_extent(
        extent=(-1.0, 1.0, 2.0, 3.0),
        shape_native=(2, 3),
    )

    assert type(grid_2d) == aa.Grid2D
    assert (
        grid_2d
        == np.array(
            [
                [3.0, -1.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [2.0, -1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
    ).all()
    assert (
        grid_2d.native
        == np.array(
            [
                [[3.0, -1.0], [3.0, 0.0], [3.0, 1.0]],
                [[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]],
            ]
        )
    ).all()
    assert (
        grid_2d.slim
        == np.array(
            [
                [3.0, -1.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [2.0, -1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
    ).all()
    assert grid_2d.pixel_scales == (1.0, 1.0)
    assert grid_2d.origin == (0.0, 0.0)


def test__uniform__2x2_pixel_scale_2__native_coordinates_correct():
    grid_2d = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

    assert type(grid_2d) == aa.Grid2D
    assert (
        grid_2d.native
        == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
    ).all()
    assert (
        grid_2d.slim == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
    ).all()
    assert grid_2d.pixel_scales == (2.0, 2.0)
    assert grid_2d.origin == (0.0, 0.0)


def test__uniform__2x2_with_origin__coordinates_offset_by_origin():
    grid_2d = aa.Grid2D.uniform(
        shape_native=(2, 2), pixel_scales=2.0, origin=(1.0, 1.0)
    )

    assert type(grid_2d) == aa.Grid2D
    assert (grid_2d == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])).all()
    assert (
        grid_2d.native == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
    ).all()
    assert (
        grid_2d.slim == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
    ).all()
    assert grid_2d.pixel_scales == (2.0, 2.0)
    assert grid_2d.origin == (1.0, 1.0)


def test__uniform__4x2_non_square__all_native_coordinates_correct():
    grid_2d = aa.Grid2D.uniform(shape_native=(4, 2), pixel_scales=0.5)

    assert type(grid_2d) == aa.Grid2D
    assert (
        grid_2d.native
        == np.array(
            [
                [[0.75, -0.25], [0.75, 0.25]],
                [[0.25, -0.25], [0.25, 0.25]],
                [[-0.25, -0.25], [-0.25, 0.25]],
                [[-0.75, -0.25], [-0.75, 0.25]],
            ]
        )
    ).all()
    assert (
        grid_2d.slim
        == np.array(
            [
                [0.75, -0.25],
                [0.75, 0.25],
                [0.25, -0.25],
                [0.25, 0.25],
                [-0.25, -0.25],
                [-0.25, 0.25],
                [-0.75, -0.25],
                [-0.75, 0.25],
            ]
        )
    ).all()
    assert grid_2d.pixel_scales == (0.5, 0.5)
    assert grid_2d.origin == (0.0, 0.0)


def test__bounding_box__square_bounding_box_3x3__slim_coordinates_and_pixel_scales_correct():
    grid_2d = aa.Grid2D.bounding_box(
        bounding_box=[-2.0, 2.0, -2.0, 2.0],
        shape_native=(3, 3),
        buffer_around_corners=False,
    )

    assert grid_2d.slim == pytest.approx(
        np.array(
            [
                [1.3333, -1.3333],
                [1.3333, 0.0],
                [1.3333, 1.3333],
                [0.0, -1.3333],
                [0.0, 0.0],
                [0.0, 1.3333],
                [-1.3333, -1.3333],
                [-1.3333, 0.0],
                [-1.3333, 1.3333],
            ]
        ),
        1.0e-4,
    )

    assert grid_2d.pixel_scales == pytest.approx((1.33333, 1.3333), 1.0e-4)
    assert grid_2d.origin == (0.0, 0.0)


def test__bounding_box__rectangular_bounding_box_2x3__slim_coordinates_and_pixel_scales_correct():
    grid_2d = aa.Grid2D.bounding_box(
        bounding_box=[-2.0, 2.0, -2.0, 2.0],
        shape_native=(2, 3),
        buffer_around_corners=False,
    )

    assert grid_2d.slim == pytest.approx(
        np.array(
            [
                [1.0, -1.3333],
                [1.0, 0.0],
                [1.0, 1.3333],
                [-1.0, -1.3333],
                [-1.0, 0.0],
                [-1.0, 1.3333],
            ]
        ),
        1.0e-4,
    )
    assert grid_2d.pixel_scales == pytest.approx((2.0, 1.33333), 1.0e4)
    assert grid_2d.origin == (0.0, 0.0)


def test__bounding_box__buffer_around_corners_2x3__coordinates_at_exact_corners():
    grid_2d = aa.Grid2D.bounding_box(
        bounding_box=[-2.0, 2.0, -2.0, 2.0],
        shape_native=(2, 3),
        buffer_around_corners=True,
    )

    assert (
        grid_2d.slim
        == np.array(
            [
                [2.0, -2.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [-2.0, -2.0],
                [-2.0, 0.0],
                [-2.0, 2.0],
            ]
        )
    ).all()
    assert grid_2d.pixel_scales == (4.0, 2.0)
    assert grid_2d.origin == (0.0, 0.0)


def test__bounding_box__buffer_around_corners_3x3_offset_box__coordinates_and_origin_correct():
    grid_2d = aa.Grid2D.bounding_box(
        bounding_box=[8.0, 10.0, -2.0, 3.0],
        shape_native=(3, 3),
        buffer_around_corners=True,
    )

    assert grid_2d == pytest.approx(
        np.array(
            [
                [10.0, -2.0],
                [10.0, 0.5],
                [10.0, 3.0],
                [9.0, -2.0],
                [9.0, 0.5],
                [9.0, 3.0],
                [8.0, -2.0],
                [8.0, 0.5],
                [8.0, 3.0],
            ]
        ),
        1.0e-4,
    )
    assert grid_2d.slim == pytest.approx(
        np.array(
            [
                [10.0, -2.0],
                [10.0, 0.5],
                [10.0, 3.0],
                [9.0, -2.0],
                [9.0, 0.5],
                [9.0, 3.0],
                [8.0, -2.0],
                [8.0, 0.5],
                [8.0, 3.0],
            ]
        ),
        1.0e-4,
    )
    assert grid_2d.pixel_scales == (1.0, 2.5)
    assert grid_2d.origin == (9.0, 0.5)


def test__grid_2d_via_deflection_grid_from__grid_deflected_by_itself__all_zero_coordinates():
    grid_2d = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

    grid_deflected = grid_2d.grid_2d_via_deflection_grid_from(deflection_grid=grid_2d)

    assert type(grid_deflected) == aa.Grid2D
    assert (
        grid_deflected == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        grid_deflected.native
        == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    ).all()
    assert (
        grid_deflected.slim
        == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (grid_deflected.mask == grid_2d.mask).all()
    assert grid_deflected.pixel_scales == (2.0, 2.0)
    assert grid_deflected.origin == (0.0, 0.0)


def test__blurring_grid_from():
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

    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0))

    blurring_mask_util = aa.util.mask_2d.blurring_mask_2d_from(
        mask_2d=np.array(mask), kernel_shape_native=(3, 5)
    )

    blurring_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=blurring_mask_util,
        pixel_scales=(2.0, 2.0),
    )

    mask = aa.Mask2D(mask=np.array(mask), pixel_scales=(2.0, 2.0))

    blurring_grid = aa.Grid2D.blurring_grid_from(mask=mask, kernel_shape_native=(3, 5))

    assert isinstance(blurring_grid, aa.Grid2D)
    assert len(blurring_grid.shape) == 2
    assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
    assert blurring_grid.pixel_scales == (2.0, 2.0)


def test__blurring_grid_via_kernel_shape_from():
    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, False, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0))

    blurring_mask_util = aa.util.mask_2d.blurring_mask_2d_from(
        mask_2d=np.array(mask), kernel_shape_native=(3, 5)
    )

    blurring_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=blurring_mask_util,
        pixel_scales=(2.0, 2.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    blurring_grid = grid_2d.blurring_grid_via_kernel_shape_from(
        kernel_shape_native=(3, 5)
    )

    assert isinstance(blurring_grid, aa.Grid2D)
    assert len(blurring_grid.shape) == 2
    assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
    assert blurring_grid.pixel_scales == (2.0, 2.0)


def test__from_mask():
    mask = np.array(
        [
            [True, True, False, False],
            [True, False, True, True],
            [True, True, False, False],
        ]
    )
    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0))

    grid_via_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(2.0, 2.0)
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    assert type(grid_2d) == aa.Grid2D
    assert grid_2d == pytest.approx(grid_via_util, 1e-4)
    assert grid_2d.pixel_scales == (2.0, 2.0)

    grid_2d_native = aa.util.grid_2d.grid_2d_native_from(
        grid_2d_slim=grid_2d.array,
        mask_2d=mask,
    )

    assert (grid_2d_native == grid_2d.native).all()


def test__to_and_from_fits_methods():
    grid_2d = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

    file_path = test_grid_dir / "grid_2d.fits"

    from autoconf.fitsable import output_to_fits
    output_to_fits(values=grid_2d.native.array.astype("float"), file_path=file_path, overwrite=True, header_dict=grid_2d.mask.header_dict)

    grid_from_fits = aa.Grid2D.from_fits(file_path=file_path, pixel_scales=2.0)

    assert type(grid_2d) == aa.Grid2D
    assert (
        grid_from_fits.native
        == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
    ).all()
    assert (
        grid_from_fits.slim
        == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
    ).all()
    assert grid_from_fits.pixel_scales == (2.0, 2.0)
    assert grid_from_fits.origin == (0.0, 0.0)


def test__shape_native_scaled():
    mask = aa.Mask2D.circular(shape_native=(3, 3), radius=1.1, pixel_scales=(1.0, 1.0))

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    assert grid_2d.shape_native_scaled_interior == (2.0, 2.0)

    mask = aa.Mask2D.elliptical(
        shape_native=(11, 11),
        major_axis_radius=3.0,
        axis_ratio=0.5,
        angle=30.0,
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)
    assert grid_2d.shape_native_scaled_interior == (2.0, 4.0)


def test__flipped():
    grid_2d = aa.Grid2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], pixel_scales=1.0
    )

    assert (
        grid_2d.flipped.slim
        == np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])
    ).all()
    assert (
        grid_2d.flipped.native
        == np.array([[[2.0, 1.0], [4.0, 3.0]], [[6.0, 5.0], [8.0, 7.0]]])
    ).all()

    grid_2d = aa.Grid2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], pixel_scales=1.0
    )

    assert (
        grid_2d.flipped.slim == np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
    ).all()
    assert (
        grid_2d.flipped.native == np.array([[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]])
    ).all()


def test__pixel_area():
    grid_2d = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

    assert grid_2d.pixel_area == 4.0

    grid_2d = aa.Grid2D.uniform(shape_native=(4, 3), pixel_scales=3.0)

    assert grid_2d.pixel_area == 9.0


def test__total_area():
    grid_2d = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

    assert grid_2d.total_area == 16.0

    grid_2d = aa.Grid2D.uniform(shape_native=(4, 3), pixel_scales=3.0)

    assert grid_2d.total_area == 108


def test__grid_2d_radial_projected_shape_slim_from():
    grid_2d = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=(1.0, 2.0))

    grid_radii = grid_2d.grid_2d_radial_projected_from(centre=(0.0, 0.0))
    grid_radial_shape_slim = grid_2d.grid_2d_radial_projected_shape_slim_from(
        centre=(0.0, 0.0)
    )

    grid_radii_util = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=grid_2d.geometry.extent,
        centre=(0.0, 0.0),
        pixel_scales=grid_2d.pixel_scales,
    )

    assert grid_radii == pytest.approx(grid_radii_util, 1.0e-4)
    assert grid_radial_shape_slim == grid_radii_util.shape[0]

    grid_2d = aa.Grid2D.uniform(shape_native=(3, 4), pixel_scales=(3.0, 2.0))

    grid_radii = grid_2d.grid_2d_radial_projected_from(centre=(0.3, 0.1))
    grid_radial_shape_slim = grid_2d.grid_2d_radial_projected_shape_slim_from(
        centre=(0.3, 0.1)
    )

    grid_radii_util = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=grid_2d.geometry.extent,
        centre=(0.3, 0.1),
        pixel_scales=grid_2d.pixel_scales,
    )

    assert grid_radii == pytest.approx(grid_radii_util, 1.0e-4)
    assert grid_radial_shape_slim == grid_radii_util.shape[0]

    grid_radii = grid_2d.grid_2d_radial_projected_from(centre=(0.3, 0.1), angle=60.0)
    grid_radial_shape_slim = grid_2d.grid_2d_radial_projected_shape_slim_from(
        centre=(0.3, 0.1)
    )

    grid_radii_util_angle = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_radii_util, centre=(0.3, 0.1), angle=60.0
    )

    grid_radii_util_angle = aa.util.geometry.transform_grid_2d_from_reference_frame(
        grid_2d=grid_radii_util_angle, centre=(0.3, 0.1), angle=0.0
    )

    assert (grid_radii == grid_radii_util_angle).all()
    assert grid_radial_shape_slim == grid_radii_util.shape[0]

    grid_radii = grid_2d.grid_2d_radial_projected_from(
        centre=(0.0, 0.0), remove_projected_centre=True
    )
    grid_radial_shape_slim = grid_2d.grid_2d_radial_projected_shape_slim_from(
        centre=(0.0, 0.0),
    )

    grid_radii_util = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=grid_2d.geometry.extent,
        centre=(0.0, 0.0),
        pixel_scales=grid_2d.pixel_scales,
    )

    assert (grid_radii == grid_radii_util[1:, :]).all()
    assert grid_radial_shape_slim == grid_radii_util.shape[0]


def test__in_radians():
    mask = np.array(
        [
            [True, True, False, False],
            [True, False, True, True],
            [True, True, False, False],
        ]
    )
    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0))

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    assert grid_2d.in_radians[0, 0] == pytest.approx(0.00000969627362, 1.0e-8)
    assert grid_2d.in_radians[0, 1] == pytest.approx(0.00000484813681, 1.0e-8)

    assert grid_2d.in_radians[0, 0] == pytest.approx(2.0 * np.pi / (180 * 3600), 1.0e-8)
    assert grid_2d.in_radians[0, 1] == pytest.approx(1.0 * np.pi / (180 * 3600), 1.0e-8)


def test__padded_grid_from():
    grid_2d = aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=3.0)

    padded_grid = grid_2d.padded_grid_from(kernel_shape_native=(3, 3))

    padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=np.full((6, 6), False), pixel_scales=(3.0, 3.0)
    )

    assert isinstance(padded_grid, aa.Grid2D)
    assert padded_grid.shape == (36, 2)
    assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
    assert (padded_grid == padded_grid_util).all()

    grid_2d = aa.Grid2D.uniform(shape_native=(4, 5), pixel_scales=2.0)

    padded_grid = grid_2d.padded_grid_from(kernel_shape_native=(3, 3))

    padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=np.full((6, 7), False), pixel_scales=(2.0, 2.0)
    )

    assert padded_grid.shape == (42, 2)
    assert (padded_grid == padded_grid_util).all()

    grid_2d = aa.Grid2D.uniform(shape_native=(5, 4), pixel_scales=1.0)

    padded_grid = grid_2d.padded_grid_from(kernel_shape_native=(3, 3))

    padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=np.full((7, 6), False), pixel_scales=(1.0, 1.0)
    )

    assert padded_grid.shape == (42, 2)
    assert (padded_grid == padded_grid_util).all()


def test__squared_distances_to_coordinate_from():
    mask = aa.Mask2D(
        [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
    )
    grid_2d = aa.Grid2D(values=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask)

    square_distances = grid_2d.squared_distances_to_coordinate_from(
        coordinate=(0.0, 0.0)
    )

    assert isinstance(square_distances, aa.Array2D)
    assert (square_distances.slim == np.array([2.0, 13.0, 5.0])).all()
    assert (square_distances.mask == mask).all()

    square_distances = grid_2d.squared_distances_to_coordinate_from(
        coordinate=(0.0, 1.0)
    )

    assert isinstance(square_distances, aa.Array2D)
    assert (square_distances.slim == np.array([1.0, 8.0, 2.0])).all()
    assert (square_distances.mask == mask).all()


def test__distance_from_coordinate_array():
    mask = aa.Mask2D(
        [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
    )
    grid_2d = aa.Grid2D(values=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask)

    square_distances = grid_2d.distances_to_coordinate_from(coordinate=(0.0, 0.0))

    assert (
        square_distances.slim == np.array([np.sqrt(2.0), np.sqrt(13.0), np.sqrt(5.0)])
    ).all()
    assert (square_distances.mask == mask).all()

    square_distances = grid_2d.distances_to_coordinate_from(coordinate=(0.0, 1.0))

    assert (square_distances.slim == np.array([1.0, np.sqrt(8.0), np.sqrt(2.0)])).all()
    assert (square_distances.mask == mask).all()


def test__grid_with_coordinates_within_distance_removed_from():
    grid_2d = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    grid_2d = grid_2d.grid_with_coordinates_within_distance_removed_from(
        coordinates=(0.0, 0.0), distance=0.05
    )

    assert (
        grid_2d
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

    grid_2d = aa.Grid2D.uniform(
        shape_native=(3, 3), pixel_scales=1.0, origin=(1.0, 1.0)
    )

    grid_2d = grid_2d.grid_with_coordinates_within_distance_removed_from(
        coordinates=(0.0, 0.0), distance=0.05
    )

    assert (
        grid_2d
        == np.array(
            [
                [2.0, 0.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ]
        )
    ).all()

    grid_2d = aa.Grid2D.uniform(
        shape_native=(3, 3), pixel_scales=1.0, origin=(1.0, 1.0)
    )

    grid_2d = grid_2d.grid_with_coordinates_within_distance_removed_from(
        coordinates=[(0.0, 0.0), (1.0, -1.0), (-1.0, -1.0), (2.0, 2.0)],
        distance=0.05,
    )

    assert (
        grid_2d
        == np.array(
            [
                [2.0, 0.0],
                [2.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ]
        )
    ).all()


def test__recursive_shape_storage():
    grid_2d = aa.Grid2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        pixel_scales=1.0,
    )

    assert type(grid_2d) == aa.Grid2D
    assert (
        grid_2d.native.slim.native
        == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ).all()
    assert (
        grid_2d.slim.native.slim
        == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()


def test__is_uniform():
    grid_2d = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    assert grid_2d.is_uniform == True

    grid_2d = aa.Grid2D.no_mask(
        values=[[[10.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        pixel_scales=1.0,
    )

    assert grid_2d.is_uniform == False


def test__apply_over_sampling():
    grid = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, over_sample_size=1)

    grid = grid.apply_over_sampling(over_sample_size=2)

    assert grid.over_sampled.shape[0] == 16
