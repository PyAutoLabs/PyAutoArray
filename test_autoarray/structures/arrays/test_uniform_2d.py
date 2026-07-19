from astropy.io import fits
import os
from pathlib import Path

import numpy as np
import pytest
import shutil

import autoarray as aa

test_data_path = Path(__file__).resolve().parent / "files"


def test__constructor__2x2_all_false_mask__native_matches_input_2d_values():
    mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0)
    array_2d = aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 0.0)


def test__constructor__partial_mask_with_1d_values__masked_pixels_zero_in_native():
    mask = aa.Mask2D(
        mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
    )
    array_2d = aa.Array2D(values=[1.0, 2.0, 4.0], mask=mask)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)


def test__constructor__partial_mask_with_2d_values__applies_mask_to_native():
    mask = aa.Mask2D(
        mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
    )
    array_2d = aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)


def test__constructor__store_native_true__stores_native_shape_including_masked_pixels():
    mask = aa.Mask2D(
        mask=[[False, False], [True, False]],
        pixel_scales=1.0,
        origin=(0.0, 1.0),
    )
    array_2d = aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0]], mask=mask, store_native=True)

    assert (array_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()


def test__no_mask__2x2_array__native_slim_and_pixel_scales_correct():
    array_2d = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]],
        pixel_scales=1.0,
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 0.0)


def test__no_mask__3x3_with_origin_and_anisotropic_scales__geometry_extents_correct():
    array_2d = aa.Array2D.no_mask(
        values=np.ones((9,)),
        shape_native=(3, 3),
        pixel_scales=(2.0, 1.0),
        origin=(-1.0, -2.0),
    )

    assert array_2d == pytest.approx(np.ones((9,)), 1e-4)
    assert array_2d.slim == pytest.approx(np.ones((9,)), 1e-4)
    assert array_2d.native == pytest.approx(np.ones((3, 3)), 1e-4)
    assert array_2d.pixel_scales == (2.0, 1.0)
    assert array_2d.geometry.central_pixel_coordinates == (1.0, 1.0)
    assert array_2d.geometry.shape_native_scaled == pytest.approx((6.0, 3.0))
    assert array_2d.origin == (-1.0, -2.0)
    assert array_2d.geometry.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
    assert array_2d.geometry.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)


def test__apply_mask__4x2_array_with_partial_mask__masked_rows_zero_in_native():
    mask = aa.Mask2D(
        mask=[[False, False], [False, False], [True, True], [True, True]],
        pixel_scales=2.0,
    )
    array_2d = aa.Array2D.no_mask(
        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        shape_native=(4, 2),
        pixel_scales=2.0,
    )
    array_2d = array_2d.apply_mask(mask=mask)

    assert (
        array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()


def test__full__fill_value_2_shape_2x2_with_origin__all_elements_equal_fill_value():
    array_2d = aa.Array2D.full(
        fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
    assert (array_2d.slim == np.array([2.0, 2.0, 2.0, 2.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)


def test__ones__2x2_shape__all_native_elements_are_one():
    array_2d = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (array_2d.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 0.0)


def test__zeros__2x2_shape__all_native_elements_are_zero():
    array_2d = aa.Array2D.zeros(shape_native=(2, 2), pixel_scales=1.0)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
    assert (array_2d.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()


def test__from_fits__4x3_ones_fits__native_is_ones_array():
    array_2d = aa.Array2D.from_fits(
        file_path=test_data_path / "4x3_ones.fits", hdu=0, pixel_scales=1.0
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.ones((4, 3))).all()
    assert (array_2d.slim == np.ones((12,))).all()


def test__from_fits__3x3_fits__header_bitpix_stored_correctly():
    array_2d = aa.Array2D.from_fits(
        file_path=test_data_path / "3x3_ones.fits", hdu=0, pixel_scales=1.0
    )

    assert array_2d.header.header_sci_obj["BITPIX"] == -64
    assert array_2d.header.header_hdu_obj["BITPIX"] == -64


def test__from_fits__4x3_fits__header_bitpix_stored_correctly():
    array_2d = aa.Array2D.from_fits(
        file_path=test_data_path / "4x3_ones.fits", hdu=0, pixel_scales=1.0
    )

    assert array_2d.header.header_sci_obj["BITPIX"] == -64
    assert array_2d.header.header_hdu_obj["BITPIX"] == -64


def test__from_yx_and_values__2x2_grid__native_matches_expected_pixel_layout():
    array_2d = aa.Array2D.from_yx_and_values(
        y=[0.5, 0.5, -0.5, -0.5],
        x=[-0.5, 0.5, -0.5, 0.5],
        values=[1.0, 2.0, 3.0, 4.0],
        shape_native=(2, 2),
        pixel_scales=1.0,
    )

    assert (array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()


def test__from_yx_and_values__3x2_grid__native_matches_expected_pixel_layout():
    array_2d = aa.Array2D.from_yx_and_values(
        y=[0.0, 1.0, -1.0, 0.0, -1.0, 1.0],
        x=[-0.5, 0.5, 0.5, 0.5, -0.5, -0.5],
        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    assert (array_2d.native == np.array([[3.0, 2.0], [6.0, 4.0], [5.0, 1.0]])).all()


def test__output_to_fits__3x3_ones__fits_file_has_ones_and_correct_pixel_scale_header():
    files_path = Path(__file__).resolve().parent / "files"

    array_2d = aa.Array2D.from_fits(
        file_path=files_path / "3x3_ones.fits", hdu=0, pixel_scales=1.0
    )

    output_test_path = files_path / "array" / "output_test"
    if output_test_path.exists():
        shutil.rmtree(output_test_path)

    os.makedirs(output_test_path)

    from autonerves.fitsable import output_to_fits
    output_to_fits(values=array_2d.native.array.astype("float"), file_path=output_test_path / "array.fits", header_dict=array_2d.mask.header_dict)

    array_from_fits = aa.Array2D.from_fits(
        file_path=output_test_path / "array.fits", hdu=0, pixel_scales=1.0
    )

    assert (array_from_fits.native == np.ones((3, 3))).all()
    assert array_from_fits.header.header_sci_obj["PIXSCAY"] == 1.0


def test__manual_native__exception_raised_if_input_array_is_2d_and_not_shape_of_mask():
    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0)
        aa.Array2D(values=[[1.0], [3.0]], mask=mask)


def test__constructor__1d_values_too_many_for_mask__raises_array_exception():
    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D(
            mask=[[False, False], [True, False]],
            pixel_scales=1.0,
        )
        aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)


def test__constructor__1d_values_too_few_for_mask__raises_array_exception():
    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D(
            mask=[[False, False], [True, False]],
            pixel_scales=1.0,
        )
        aa.Array2D(values=[1.0, 2.0], mask=mask)


@pytest.mark.parametrize(
    "new_shape,expected_native",
    [
        (
            (7, 7),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            (3, 3),
            np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]),
        ),
    ],
)
def test__resized_from__5x5_array_with_center_marked__resized_array_pads_or_crops_correctly(
    new_shape, expected_native
):
    array_2d = np.ones((5, 5))
    array_2d[2, 2] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, pixel_scales=(1.0, 1.0))

    array_2d = array_2d.resized_from(new_shape=new_shape)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == expected_native).all()
    assert array_2d.mask.pixel_scales == (1.0, 1.0)


@pytest.mark.parametrize(
    "kernel_shape,expected_shape",
    [
        ((3, 3), (7, 7)),
        ((5, 5), (9, 9)),
    ],
)
def test__padded_before_convolution_from__5x5_array__output_shape_padded_by_kernel_size(
    kernel_shape, expected_shape
):
    array_2d = np.ones((5, 5))
    array_2d[2, 2] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.padded_before_convolution_from(kernel_shape=kernel_shape)

    assert type(new_arr) == aa.Array2D
    assert new_arr.native[0, 0] == 0.0
    assert new_arr.shape_native == expected_shape
    assert new_arr.mask.pixel_scales == (1.0, 1.0)


def test__padded_before_convolution_from__9x9_array__output_shape_padded_by_7x7_kernel():
    array_2d = np.ones((9, 9))
    array_2d[4, 4] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.padded_before_convolution_from(kernel_shape=(7, 7))

    assert type(new_arr) == aa.Array2D
    assert new_arr.native[0, 0] == 0.0
    assert new_arr.shape_native == (15, 15)
    assert new_arr.mask.pixel_scales == (1.0, 1.0)


@pytest.mark.parametrize(
    "kernel_shape,expected_native",
    [
        ((3, 3), np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])),
        ((5, 5), np.array([[2.0]])),
    ],
)
def test__trimmed_after_convolution_from__5x5_array_with_center_marked__trims_to_non_padded_region(
    kernel_shape, expected_native
):
    array_2d = np.ones((5, 5))
    array_2d[2, 2] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.trimmed_after_convolution_from(kernel_shape=kernel_shape)

    assert type(new_arr) == aa.Array2D
    assert (new_arr.native == expected_native).all()
    assert new_arr.mask.pixel_scales == (1.0, 1.0)


def test__trimmed_after_convolution_from__9x9_array_with_center_marked__trims_to_non_padded_region():
    array_2d = np.ones((9, 9))
    array_2d[4, 4] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.trimmed_after_convolution_from(kernel_shape=(7, 7))

    assert type(new_arr) == aa.Array2D
    assert (
        new_arr.native == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()
    assert new_arr.mask.pixel_scales == (1.0, 1.0)


def test__binned_across_rows__4x3_ones_array__each_column_bin_is_one():
    array = aa.Array2D.no_mask(values=np.ones((4, 3)), pixel_scales=1.0)

    assert (array.binned_across_rows == np.array([1.0, 1.0, 1.0])).all()


def test__binned_across_rows__3x4_ones_array__each_column_bin_is_one():
    array = aa.Array2D.no_mask(values=np.ones((3, 4)), pixel_scales=1.0)

    assert (array.binned_across_rows == np.array([1.0, 1.0, 1.0, 1.0])).all()


def test__binned_across_rows__3x3_non_uniform_values__columns_averaged_correctly():
    array = aa.Array2D.no_mask(
        values=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
        pixel_scales=1.0,
    )

    assert (array.binned_across_rows == np.array([2.0, 6.0, 9.0])).all()


def test__binned_across_rows__3x3_with_partial_mask__masked_pixels_excluded_from_average():
    mask = aa.Mask2D(
        mask=[[False, False, False], [False, False, False], [True, False, False]],
        pixel_scales=1.0,
    )

    array = aa.Array2D(
        values=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
        mask=mask,
    )

    assert (array.binned_across_rows == np.array([1.5, 6.0, 9.0])).all()


def test__binned_across_columns__4x3_ones_array__each_row_bin_is_one():
    array = aa.Array2D.no_mask(values=np.ones((4, 3)), pixel_scales=1.0)

    assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0, 1.0])).all()


def test__binned_across_columns__3x4_ones_array__each_row_bin_is_one():
    array = aa.Array2D.no_mask(values=np.ones((3, 4)), pixel_scales=1.0)

    assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0])).all()


def test__binned_across_columns__3x3_non_uniform_values__rows_averaged_correctly():
    array = aa.Array2D.no_mask(
        values=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
        pixel_scales=1.0,
    )

    assert (array.binned_across_columns == np.array([2.0, 6.0, 9.0])).all()


def test__binned_across_columns__3x3_with_partial_mask__masked_pixels_excluded_from_average():
    mask = aa.Mask2D(
        mask=[[False, False, True], [False, False, False], [False, False, False]],
        pixel_scales=1.0,
    )

    array = aa.Array2D(
        values=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
        mask=mask,
    )

    assert (array.binned_across_columns == np.array([1.5, 6.0, 9.0])).all()


def test__brightest_coordinate_in_region_from__4x4_array__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(4, 4), pixel_scales=0.1)
    array_2d = aa.Array2D(
        values=[
            [1.0, 2.0, 3.0, 4.0],
            [6.0, 7.0, 8.0, 9.0],
            [11.0, 12.0, 13.0, 15.0],
            [16.0, 17.0, 18.0, 20.0],
        ],
        mask=mask,
    )

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(-0.26, 0.06, -0.06, 0.06)
    )

    assert brightest_coordinate == pytest.approx((-0.15, 0.05), 1.0e-4)


def test__brightest_coordinate_in_region_from__4x4_array_different_region__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(4, 4), pixel_scales=0.1)
    array_2d = aa.Array2D(
        values=[
            [1.0, 2.0, 3.0, 4.0],
            [6.0, 7.0, 8.0, 9.0],
            [11.0, 12.0, 13.0, 15.0],
            [16.0, 17.0, 18.0, 20.0],
        ],
        mask=mask,
    )

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(-0.051, 0.151, -0.151, 0.149)
    )

    assert brightest_coordinate == pytest.approx((-0.05, 0.05), 1.0e-4)


def test__brightest_coordinate_in_region_from__5x5_array__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.1)
    array_2d = aa.Array2D(
        values=[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ],
        mask=mask,
    )

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(-0.15, 0.15, -0.15, 0.15)
    )

    assert brightest_coordinate == pytest.approx((-0.1, 0.1), 1.0e-4)


def test__brightest_coordinate_in_region_from__5x5_array_asymmetric_region__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.1)
    array_2d = aa.Array2D(
        values=[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ],
        mask=mask,
    )

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(-0.25, 0.15, -0.15, 0.15)
    )

    assert brightest_coordinate == pytest.approx((-0.2, 0.1), 1.0e-4)

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(-0.15, 0.15, -0.15, 0.25)
    )

    assert brightest_coordinate == pytest.approx((-0.1, 0.2), 1.0e-4)


def test__brightest_coordinate_in_region_from__region_offset_from_origin__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(7, 7), pixel_scales=0.1)
    values = np.zeros((7, 7))
    values[5, 1] = 99.0
    array_2d = aa.Array2D(values=values, mask=mask)

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(-0.25, -0.05, -0.25, -0.05)
    )

    assert brightest_coordinate == pytest.approx((-0.2, -0.2), 1.0e-4)


def test__brightest_coordinate_in_region_from__region_fully_offset_negative_quadrant__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(11, 11), pixel_scales=0.1)
    values = np.zeros((11, 11))
    values[8, 2] = 77.0
    array_2d = aa.Array2D(values=values, mask=mask)

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(-0.45, -0.15, -0.45, -0.15)
    )

    assert brightest_coordinate == pytest.approx((-0.3, -0.3), 1.0e-4)


def test__brightest_coordinate_in_region_from__region_offset_positive_quadrant__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(11, 11), pixel_scales=0.1)
    values = np.zeros((11, 11))
    values[2, 8] = 55.0
    array_2d = aa.Array2D(values=values, mask=mask)

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(0.15, 0.45, 0.15, 0.45)
    )

    assert brightest_coordinate == pytest.approx((0.3, 0.3), 1.0e-4)


def test__brightest_coordinate_in_region_from__region_clipped_to_array_bounds__correct_peak_coordinate():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.1)
    values = np.zeros((5, 5))
    values[0, 0] = 42.0
    array_2d = aa.Array2D(values=values, mask=mask)

    brightest_coordinate = array_2d.brightest_coordinate_in_region_from(
        region=(0.15, 0.45, -0.45, -0.15)
    )

    assert brightest_coordinate == pytest.approx((0.2, -0.2), 1.0e-4)


def test__brightest_sub_pixel_coordinate_in_region_from__region_offset_from_origin__correct_sub_pixel_peak():
    mask = aa.Mask2D.all_false(shape_native=(7, 7), pixel_scales=0.1)
    values = np.zeros((7, 7))
    values[5, 1] = 100.0
    values[5, 2] = 50.0
    values[4, 1] = 50.0
    array_2d = aa.Array2D(values=values, mask=mask)

    brightest_coordinate = array_2d.brightest_sub_pixel_coordinate_in_region_from(
        region=(-0.25, -0.05, -0.25, -0.05), box_size=1
    )

    assert brightest_coordinate[0] < -0.15
    assert brightest_coordinate[1] > -0.2


def test__brightest_sub_pixel_coordinate_in_region_from__4x4_array__correct_sub_pixel_peak():
    mask = aa.Mask2D.all_false(shape_native=(4, 4), pixel_scales=0.1)
    array_2d = aa.Array2D(
        values=[
            [1.0, 2.0, 3.0, 4.0],
            [6.0, 7.0, 8.0, 9.0],
            [11.0, 12.0, 13.0, 15.0],
            [16.0, 17.0, 18.0, 20.0],
        ],
        mask=mask,
    )

    brightest_coordinate = array_2d.brightest_sub_pixel_coordinate_in_region_from(
        region=(-0.26, 0.06, -0.06, 0.06)
    )

    assert brightest_coordinate == pytest.approx((-0.1078947, 0.056315), 1.0e-4)


def test__brightest_sub_pixel_coordinate_in_region_from__5x5_array__correct_sub_pixel_peak():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.1)
    array_2d = aa.Array2D(
        values=[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ],
        mask=mask,
    )

    brightest_coordinate = array_2d.brightest_sub_pixel_coordinate_in_region_from(
        region=(-0.15, 0.15, -0.15, 0.15)
    )

    assert brightest_coordinate == pytest.approx((-0.11754, 0.103508), 1.0e-4)


def test__header__date_obs_and_time_obs__modified_julian_date_correct():
    header_sci_obj = {"DATE-OBS": "2000-01-01", "TIME-OBS": "00:00:00"}

    header = aa.Header(header_sci_obj=header_sci_obj, header_hdu_obj=None)

    assert header.modified_julian_date == 51544.0


def test__recursive_shape_storage__native_to_slim_to_native__roundtrip_correct():
    array_2d = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]],
        pixel_scales=1.0,
    )

    assert (array_2d.native.slim.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()


def test__recursive_shape_storage__slim_to_native_to_slim__roundtrip_correct():
    array_2d = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]],
        pixel_scales=1.0,
    )

    assert (array_2d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
