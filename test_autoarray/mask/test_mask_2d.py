from astropy.io import fits
import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from pathlib import Path

test_data_path = Path(Path(__file__).resolve().parent) / "files" / "mask"


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test__constructor__2x2_mask_with_scalar_pixel_scale__stored_as_tuple():
    mask = aa.Mask2D(mask=[[False, False], [True, True]], pixel_scales=1.0)

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[False, False], [True, True]])).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (0.0, 0.0)
    assert (mask.geometry.extent == np.array([-1.0, 1.0, -1.0, 1.0])).all()


def test__constructor__2x2_mask_with_anisotropic_pixel_scales_and_origin__stored_correctly():
    mask = aa.Mask2D(
        mask=[[False, False], [True, True]],
        pixel_scales=(2.0, 3.0),
        origin=(0.0, 1.0),
    )

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[False, False], [True, True]])).all()
    assert mask.pixel_scales == (2.0, 3.0)
    assert mask.origin == (0.0, 1.0)


def test__constructor__invert_true__boolean_values_inverted():
    mask = aa.Mask2D(
        mask=[[False, False, True], [True, True, False]],
        pixel_scales=1.0,
        invert=True,
    )

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[True, True, False], [False, False, True]])).all()


# ---------------------------------------------------------------------------
# all_false
# ---------------------------------------------------------------------------


def test__all_false__5x5_shape__all_pixels_unmasked():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0, invert=False)

    assert mask.shape == (5, 5)
    assert (
        mask
        == np.array(
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )
    ).all()


def test__all_false__3x3_with_invert_true__all_pixels_masked_with_correct_scales_and_origin():
    mask = aa.Mask2D.all_false(
        shape_native=(3, 3),
        pixel_scales=(2.0, 2.5),
        invert=True,
        origin=(1.0, 2.0),
    )

    assert mask.shape == (3, 3)
    assert (
        mask == np.array([[True, True, True], [True, True, True], [True, True, True]])
    ).all()
    assert mask.pixel_scales == (2.0, 2.5)
    assert mask.origin == (1.0, 2.0)


# ---------------------------------------------------------------------------
# circular
# ---------------------------------------------------------------------------


def test__circular__5x4_shape_radius_3p5__matches_util_output_and_mask_centre_at_origin():
    mask_via_util = aa.util.mask_2d.mask_2d_circular_from(
        shape_native=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
    )

    mask = aa.Mask2D.circular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        radius=3.5,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)


def test__circular__invert_true__output_is_inverted_util_result():
    mask_via_util = aa.util.mask_2d.mask_2d_circular_from(
        shape_native=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
    )

    mask = aa.Mask2D.circular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        radius=3.5,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


def test__circular__small_datasets_env__oversized_radius_clamped_to_disc(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    mask = aa.Mask2D.circular(shape_native=(100, 100), pixel_scales=0.1, radius=6.0)

    assert mask.shape_native == (16, 16)
    assert mask.pixel_scales == (0.6, 0.6)
    assert mask.is_circular
    n_unmasked = int((~mask.array).sum())
    assert 100 < n_unmasked < 256


def test__circular__small_datasets_env__in_bounds_radius_unchanged(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    capped = aa.Mask2D.circular(shape_native=(15, 15), pixel_scales=0.6, radius=2.0)
    reference = aa.util.mask_2d.mask_2d_circular_from(
        shape_native=(15, 15), pixel_scales=(0.6, 0.6), radius=2.0, centre=(0.0, 0.0)
    )

    assert (capped == reference).all()


def test__circular__small_datasets_env__tuple_pixel_scales_oversized_radius_clamped(
    monkeypatch,
):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    mask = aa.Mask2D.circular(
        shape_native=(200, 200), pixel_scales=(0.1, 0.1), radius=7.5
    )

    assert mask.shape_native == (16, 16)
    assert mask.is_circular


# ---------------------------------------------------------------------------
# circular_annular
# ---------------------------------------------------------------------------


def test__circular_annular__5x4_shape__matches_util_output_and_mask_centre_at_origin():
    mask_via_util = aa.util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        inner_radius=0.8,
        outer_radius=3.5,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.circular_annular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        inner_radius=0.8,
        outer_radius=3.5,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)


def test__circular_annular__invert_true__output_is_inverted_util_result():
    mask_via_util = aa.util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        inner_radius=0.8,
        outer_radius=3.5,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.circular_annular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        inner_radius=0.8,
        outer_radius=3.5,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


# ---------------------------------------------------------------------------
# elliptical
# ---------------------------------------------------------------------------


def test__elliptical__8x5_shape__matches_util_output_and_mask_centre_at_origin():
    mask_via_util = aa.util.mask_2d.mask_2d_elliptical_from(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        major_axis_radius=5.7,
        axis_ratio=0.4,
        angle=40.0,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.elliptical(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        major_axis_radius=5.7,
        axis_ratio=0.4,
        angle=40.0,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)


def test__elliptical__invert_true__output_is_inverted_util_result():
    mask_via_util = aa.util.mask_2d.mask_2d_elliptical_from(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        major_axis_radius=5.7,
        axis_ratio=0.4,
        angle=40.0,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.elliptical(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        major_axis_radius=5.7,
        axis_ratio=0.4,
        angle=40.0,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


# ---------------------------------------------------------------------------
# elliptical_annular
# ---------------------------------------------------------------------------


def test__elliptical_annular__8x5_shape__matches_util_output_and_mask_centre_at_origin():
    mask_via_util = aa.util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        inner_major_axis_radius=2.1,
        inner_axis_ratio=0.6,
        inner_phi=20.0,
        outer_major_axis_radius=5.7,
        outer_axis_ratio=0.4,
        outer_phi=40.0,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.elliptical_annular(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        inner_major_axis_radius=2.1,
        inner_axis_ratio=0.6,
        inner_phi=20.0,
        outer_major_axis_radius=5.7,
        outer_axis_ratio=0.4,
        outer_phi=40.0,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)


def test__elliptical_annular__invert_true__output_is_inverted_util_result():
    mask_via_util = aa.util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        inner_major_axis_radius=2.1,
        inner_axis_ratio=0.6,
        inner_phi=20.0,
        outer_major_axis_radius=5.7,
        outer_axis_ratio=0.4,
        outer_phi=40.0,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.elliptical_annular(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        inner_major_axis_radius=2.1,
        inner_axis_ratio=0.6,
        inner_phi=20.0,
        outer_major_axis_radius=5.7,
        outer_axis_ratio=0.4,
        outer_phi=40.0,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


# ---------------------------------------------------------------------------
# from_pixel_coordinates
# ---------------------------------------------------------------------------


def test__from_pixel_coordinates__single_coordinate_no_buffer__one_unmasked_pixel():
    mask = aa.Mask2D.from_pixel_coordinates(
        shape_native=(5, 5), pixel_coordinates=[[2, 2]], pixel_scales=1.0, buffer=0
    )

    assert (
        mask
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


def test__from_pixel_coordinates__single_coordinate_buffer_1__3x3_unmasked_region():
    mask = aa.Mask2D.from_pixel_coordinates(
        shape_native=(5, 5), pixel_coordinates=[[2, 2]], pixel_scales=1.0, buffer=1
    )

    assert (
        mask
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


def test__from_pixel_coordinates__two_coordinates_buffer_1__two_separate_unmasked_regions():
    mask = aa.Mask2D.from_pixel_coordinates(
        shape_native=(7, 7),
        pixel_coordinates=[[2, 2], [5, 5]],
        pixel_scales=1.0,
        buffer=1,
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


# ---------------------------------------------------------------------------
# from_fits / output_to_fits
# ---------------------------------------------------------------------------


def test__from_fits__output_to_fits__roundtrip_preserves_values_pixel_scales_and_header(
    tmp_path,
):
    mask = aa.Mask2D.from_fits(
        file_path=Path(test_data_path) / "3x3_ones.fits",
        hdu=0,
        pixel_scales=(1.0, 1.0),
    )

    from autonerves.fitsable import output_to_fits
    output_to_fits(
        values=mask.astype("float"),
        file_path=tmp_path / "mask.fits",
        header_dict=mask.header_dict,
        ext_name="mask",
    )

    mask = aa.Mask2D.from_fits(
        file_path=tmp_path / "mask.fits",
        hdu=0,
        pixel_scales=(1.0, 1.0),
        origin=(2.0, 2.0),
    )

    assert (mask == np.ones((3, 3))).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (2.0, 2.0)

    header = aa.header_obj_from(file_path=tmp_path / "mask.fits", hdu=0)

    assert header["PIXSCAY"] == 1.0
    assert header["PIXSCAX"] == 1.0
    assert header["ORIGINY"] == 0.0
    assert header["ORIGINX"] == 0.0


@pytest.mark.parametrize("resized_shape", [(1, 1), (5, 5)])
def test__from_fits__with_resized_mask_shape__output_shape_matches_requested_shape(
    resized_shape,
):
    mask = aa.Mask2D.from_fits(
        file_path=Path(test_data_path) / "3x3_ones.fits",
        hdu=0,
        pixel_scales=(1.0, 1.0),
        resized_mask_shape=resized_shape,
    )

    assert mask.shape_native == resized_shape


# ---------------------------------------------------------------------------
# exception: 1D mask without shape_native
# ---------------------------------------------------------------------------


def test__constructor__1d_mask_without_shape_native__raises_mask_exception():
    with pytest.raises(exc.MaskException):
        aa.Mask2D(mask=[False, False, True], pixel_scales=1.0)

    with pytest.raises(exc.MaskException):
        aa.Mask2D(mask=[False, False, True], pixel_scales=False)


# ---------------------------------------------------------------------------
# is_all_true / is_all_false — parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask_values,expected",
    [
        ([[False, False], [False, False]], False),
        ([[False, False]], False),
        ([[False, True], [False, False]], False),
        ([[True, True], [True, True]], True),
    ],
)
def test__is_all_true__various_masks__returns_correct_boolean(mask_values, expected):
    mask = aa.Mask2D(mask=mask_values, pixel_scales=1.0)

    assert mask.is_all_true == expected


@pytest.mark.parametrize(
    "mask_values,expected",
    [
        ([[False, False], [False, False]], True),
        ([[False, False]], True),
        ([[False, True], [False, False]], False),
        ([[True, True], [False, False]], False),
    ],
)
def test__is_all_false__various_masks__returns_correct_boolean(mask_values, expected):
    mask = aa.Mask2D(mask=mask_values, pixel_scales=1.0)

    assert mask.is_all_false == expected


# ---------------------------------------------------------------------------
# shape_native_masked_pixels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask_values,expected_shape",
    [
        (
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
            ],
            (7, 7),
        ),
        (
            [
                [True, True, True, True, True, True, True, True, False],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ],
            (8, 8),
        ),
        (
            [
                [True, True, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ],
            (8, 7),
        ),
    ],
)
def test__shape_native_masked_pixels__various_unmasked_regions__returns_bounding_box_shape(
    mask_values, expected_shape
):
    mask = aa.Mask2D(mask=mask_values, pixel_scales=1.0)

    assert mask.shape_native_masked_pixels == expected_shape


# ---------------------------------------------------------------------------
# rescaled_from / resized_from
# ---------------------------------------------------------------------------


def test__rescaled_from__5x5_mask_with_one_masked_pixel__rescaled_mask_matches_util():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    mask_rescaled = mask.rescaled_from(rescale_factor=2.0)

    mask_rescaled_manual = aa.util.mask_2d.rescaled_mask_2d_from(
        mask_2d=mask, rescale_factor=2.0
    )

    assert (mask_rescaled == mask_rescaled_manual).all()


@pytest.mark.parametrize(
    "new_shape,expected_masked_position",
    [
        ((7, 7), (3, 3)),
        ((3, 3), (1, 1)),
    ],
)
def test__resized_from__5x5_mask_with_center_masked__resized_mask_has_masked_pixel_at_center(
    new_shape, expected_masked_position
):
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    mask_resized = mask.resized_from(new_shape=new_shape)

    mask_resized_manual = np.full(fill_value=False, shape=new_shape)
    mask_resized_manual[expected_masked_position] = True

    assert (mask_resized == mask_resized_manual).all()


# ---------------------------------------------------------------------------
# mask_centre
# ---------------------------------------------------------------------------


def test__mask_centre__symmetric_horizontal_unmasked_pixels__centre_at_origin():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    assert mask.mask_centre == (0.0, 0.0)


def test__mask_centre__asymmetric_horizontal_unmasked_pixels__centre_offset_in_x():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, False],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    assert mask.mask_centre == (0.0, 0.5)


def test__mask_centre__asymmetric_vertical_unmasked_pixels__centre_offset_in_y():
    mask = aa.Mask2D(
        mask=[
            [True, True, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    assert mask.mask_centre == (0.5, 0.0)


def test__mask_centre__left_shifted_unmasked_pixels__centre_negative_x():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [False, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    assert mask.mask_centre == (0.0, -0.5)


def test__mask_centre__lower_offset_unmasked_pixels__centre_negative_y():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    assert mask.mask_centre == (-0.5, 0.0)


def test__mask_centre__lower_left_offset_unmasked_pixels__centre_negative_y_and_x():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [False, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    assert mask.mask_centre == (-0.5, -0.5)


# ---------------------------------------------------------------------------
# is_circular / circular_radius
# ---------------------------------------------------------------------------


def test__is_circular__non_circular_mask__returns_false():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    assert mask.is_circular == False


@pytest.mark.parametrize(
    "shape_native,radius",
    [
        ((5, 5), 1.0),
        ((10, 10), 3.0),
        ((10, 10), 4.0),
    ],
)
def test__is_circular__circular_mask__returns_true(shape_native, radius):
    mask = aa.Mask2D.circular(
        shape_native=shape_native, radius=radius, pixel_scales=(1.0, 1.0)
    )

    assert mask.is_circular == True


@pytest.mark.parametrize(
    "shape_native,radius,pixel_scales",
    [
        ((10, 10), 3.0, (1.0, 1.0)),
        ((30, 30), 5.5, (0.5, 0.5)),
    ],
)
def test__circular_radius__circular_mask__returns_radius_used_to_create_mask(
    shape_native, radius, pixel_scales
):
    mask = aa.Mask2D.circular(
        shape_native=shape_native, radius=radius, pixel_scales=pixel_scales
    )

    assert mask.circular_radius == pytest.approx(radius, 1e-4)


@pytest.mark.parametrize(
    "centre",
    [
        (0.25, 0.0),
        (0.0, 0.25),
        (0.27, 0.13),
        (-0.34, 0.41),
        (0.5, 0.5),
    ],
)
def test__is_circular__offset_centre_circular_mask__returns_true(centre):
    mask = aa.Mask2D.circular(
        shape_native=(20, 20),
        radius=0.4,
        pixel_scales=(0.1, 0.1),
        centre=centre,
    )

    assert mask.is_circular == True


def test__is_circular__annular_mask__returns_false():
    mask = aa.Mask2D.circular_annular(
        shape_native=(20, 20),
        inner_radius=0.1,
        outer_radius=0.4,
        pixel_scales=(0.1, 0.1),
    )

    assert mask.is_circular == False


def test__is_circular__square_unmasked_region__returns_false():
    mask_array = np.full((10, 10), True)
    mask_array[3:8, 3:8] = False

    mask = aa.Mask2D(mask=mask_array, pixel_scales=(1.0, 1.0))

    assert mask.is_circular == False


@pytest.mark.parametrize(
    "centre",
    [
        (0.25, 0.0),
        (0.0, 0.25),
        (-0.34, 0.41),
    ],
)
def test__circular_radius__offset_centre_circular_mask__returns_radius(centre):
    mask = aa.Mask2D.circular(
        shape_native=(40, 40),
        radius=0.8,
        pixel_scales=(0.1, 0.1),
        centre=centre,
    )

    assert mask.circular_radius == pytest.approx(0.8, abs=0.1)
