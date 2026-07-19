from astropy.io import fits
from pathlib import Path
import os
import numpy as np
import pytest
import shutil

import autoarray as aa

fits_path = Path(__file__).resolve().parent / "files" / "array_1d"

test_data_path = Path(__file__).resolve().parent / "files" / "array" / "output_test"


def create_fits(
    fits_path,
):
    if Path(fits_path).exists():
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(np.ones(3)))
    hdu_list[0].header.set("BITPIX", -64, "")
    hdu_list.writeto(Path(fits_path) / "3_ones.fits")

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(np.ones(4)))
    hdu_list[0].header.set("BITPIX", -64, "")
    hdu_list.writeto(Path(fits_path) / "4_ones.fits")


def clean_fits(fits_path):
    if Path(fits_path).exists():
        shutil.rmtree(fits_path)


def test__constructor__with_partial_mask__masked_pixels_set_to_zero_in_native():
    mask = aa.Mask1D(
        mask=[True, False, False, True, False, False],
        pixel_scales=1.0,
    )

    array_1d = aa.Array1D(values=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0], mask=mask)

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])).all()
    assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert array_1d.pixel_scale == 1.0
    assert array_1d.pixel_scales == (1.0,)
    assert array_1d.origin == (0.0,)


def test__no_mask__4_element_array__native_slim_and_grid_radial_correct():
    array_1d = aa.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.grid_radial == np.array(([0.0, 1.0, 2.0, 3.0]))).all()
    assert array_1d.pixel_scale == 1.0
    assert array_1d.pixel_scales == (1.0,)
    assert array_1d.origin == (0.0,)


def test__full__fill_value_1_shape_4__all_elements_equal_fill_value():
    array_1d = aa.Array1D.full(
        fill_value=1.0, shape_native=4, pixel_scales=1.0, origin=(4.0,)
    )

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert (array_1d.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert array_1d.pixel_scale == 1.0
    assert array_1d.pixel_scales == (1.0,)
    assert array_1d.origin == (4.0,)


def test__ones__shape_3_pixel_scale_3__all_native_elements_are_one():
    array_1d = aa.Array1D.ones(shape_native=3, pixel_scales=3.0, origin=(4.0,))

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([1.0, 1.0, 1.0])).all()
    assert (array_1d.slim == np.array([1.0, 1.0, 1.0])).all()
    assert array_1d.pixel_scale == 3.0
    assert array_1d.pixel_scales == (3.0,)
    assert array_1d.origin == (4.0,)


def test__zeros__shape_3_pixel_scale_3__all_native_elements_are_zero():
    array_1d = aa.Array1D.zeros(shape_native=3, pixel_scales=3.0, origin=(4.0,))

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([0.0, 0.0, 0.0])).all()
    assert (array_1d.slim == np.array([0.0, 0.0, 0.0])).all()
    assert array_1d.pixel_scale == 3.0
    assert array_1d.pixel_scales == (3.0,)
    assert array_1d.origin == (4.0,)


def test__from_fits__3_element_fits__native_and_slim_are_ones():
    create_fits(fits_path=fits_path)

    arr = aa.Array1D.from_fits(
        file_path=fits_path / "3_ones.fits", hdu=0, pixel_scales=1.0
    )

    assert type(arr) == aa.Array1D
    assert (arr.native == np.ones((3,))).all()
    assert (arr.slim == np.ones(3)).all()

    clean_fits(fits_path=fits_path)


def test__from_fits__4_element_fits__native_slim_and_array_are_ones():
    create_fits(fits_path=fits_path)

    arr = aa.Array1D.from_fits(
        file_path=fits_path / "4_ones.fits", hdu=0, pixel_scales=1.0
    )

    assert type(arr) == aa.Array1D
    assert (arr == np.ones((4,))).all()
    assert (arr.native == np.ones((4,))).all()
    assert (arr.slim == np.ones((4,))).all()

    clean_fits(fits_path=fits_path)


def test__from_fits__3_element_fits__header_bitpix_is_minus_64():
    create_fits(fits_path=fits_path)

    arr = aa.Array1D.from_fits(
        file_path=fits_path / "3_ones.fits", hdu=0, pixel_scales=1.0
    )

    assert arr.header.header_sci_obj["BITPIX"] == -64
    assert arr.header.header_hdu_obj["BITPIX"] == -64

    clean_fits(fits_path=fits_path)


def test__from_fits__4_element_fits__header_bitpix_is_minus_64():
    create_fits(fits_path=fits_path)

    arr = aa.Array1D.from_fits(
        file_path=fits_path / "4_ones.fits", hdu=0, pixel_scales=1.0
    )

    assert arr.header.header_sci_obj["BITPIX"] == -64
    assert arr.header.header_hdu_obj["BITPIX"] == -64

    clean_fits(fits_path=fits_path)


def test__output_to_fits__ones_array__fits_file_has_correct_values_and_header():
    arr = aa.Array1D.ones(shape_native=(3,), pixel_scales=1.0)

    if test_data_path.exists():
        shutil.rmtree(test_data_path)

    os.makedirs(test_data_path)

    from autonerves.fitsable import output_to_fits
    output_to_fits(values=arr.native.array.astype("float"), file_path=test_data_path / "array.fits", header_dict=arr.mask.header_dict)

    array_from_out = aa.Array1D.from_fits(
        file_path=test_data_path / "array.fits", hdu=0, pixel_scales=1.0
    )

    assert (array_from_out.native == np.ones((3,))).all()

    header_load = aa.header_obj_from(
        file_path=test_data_path / "array.fits", hdu=0
    )

    assert header_load["PIXSCA"] == 1.0
    assert header_load["ORIGIN"] == 0.0


def test__recursive_shape_storage__no_mask__native_slim_native_roundtrip_correct():
    array_1d = aa.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

    assert (array_1d.native.slim.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()


def test__recursive_shape_storage__with_partial_mask__native_slim_native_roundtrip_correct():
    mask = aa.Mask1D(
        mask=[True, False, False, True, False, False],
        pixel_scales=1.0,
    )

    array_1d = aa.Array1D(values=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0], mask=mask)

    assert (
        array_1d.native.slim.native == np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
    ).all()
    assert (array_1d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
