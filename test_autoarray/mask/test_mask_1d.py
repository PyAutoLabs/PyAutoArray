import numpy as np
from pathlib import Path
import pytest

import autoarray as aa
from autoarray import exc

test_data_path = Path(__file__).resolve().parent / "files" / "mask"


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test__constructor__2_element_mask__pixel_scale_and_extent_correct():
    mask = aa.Mask1D(mask=[False, True], pixel_scales=1.0)

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, True])).all()
    assert mask.pixel_scale == 1.0
    assert mask.pixel_scales == (1.0,)
    assert mask.origin == (0.0,)
    assert (mask.geometry.extent == np.array([-1.0, 1.0])).all()


def test__constructor__3_element_mask_with_origin__pixel_scale_origin_and_extent_correct():
    mask = aa.Mask1D(mask=[False, False, True], pixel_scales=3.0, origin=(1.0,))

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True])).all()
    assert mask.pixel_scale == 3.0
    assert mask.origin == (1.0,)
    assert (mask.geometry.extent == np.array([-3.5, 5.5])).all()


def test__constructor__4_element_mask_with_origin__pixel_scale_and_origin_correct():
    mask = aa.Mask1D(mask=[False, False, True, True], pixel_scales=3.0, origin=(1.0,))

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True, True])).all()
    assert mask.pixel_scale == 3.0
    assert mask.origin == (1.0,)


def test__constructor__invert_true__boolean_values_inverted():
    mask = aa.Mask1D(mask=[True, True, False], pixel_scales=1.0, invert=True)

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True])).all()


def test__constructor__input_is_2d_mask__raises_exception():
    with pytest.raises(exc.MaskException):
        aa.Mask1D(mask=[[False, False, True]], pixel_scales=1.0)


@pytest.mark.parametrize(
    "pixel_scales", [1, 1.0, np.float64(1.0), np.int32(1), np.float32(1.0)]
)
def test__constructor__widens_any_real_scalar_pixel_scales(pixel_scales):
    """
    `Mask1D` hand-rolled its own `type(pixel_scales) is float` check and never routed through
    `convert_pixel_scales_1d`, so an `int` or a NumPy scalar was stored bare. `Mask2D` already
    went through the chokepoint — this closed the 1D/2D divergence.
    """
    mask = aa.Mask1D(mask=[False, False, True], pixel_scales=pixel_scales)

    assert mask.pixel_scales == (1.0,)
    assert type(mask.pixel_scales[0]) is float


def test__constructor__scalar_pixel_scales__geometry_is_usable():
    """
    The bare scalar only surfaced later, when geometry subscripted it:
    `TypeError: 'int' object is not subscriptable`, naming nothing the caller passed.
    """
    mask = aa.Mask1D(mask=[False, False, True], pixel_scales=1)

    assert mask.geometry.scaled_maxima == (1.5,)


def test__constructor__tuple_pixel_scales_returned_unchanged():
    mask = aa.Mask1D(mask=[False, False, True], pixel_scales=(1.0,))

    assert mask.pixel_scales == (1.0,)


@pytest.mark.parametrize("pixel_scales", [0, 0.0, -1, -1.0, float("nan")])
def test__constructor__invalid_pixel_scales__raises_exception(pixel_scales):
    """
    Routing through `convert_pixel_scales_1d` brings `validate.validate_pixel_scales` with it,
    so `Mask1D` now rejects what `Mask2D` already rejected. A deliberate contract change.
    """
    with pytest.raises(ValueError):
        aa.Mask1D(mask=[False, False, True], pixel_scales=pixel_scales)


# ---------------------------------------------------------------------------
# is_all_true / is_all_false — parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask_values,expected",
    [
        ([False, False, False, False], False),
        ([False, False], False),
        ([False, True, False, False], False),
        ([True, True, True, True], True),
    ],
)
def test__is_all_true__various_masks__returns_correct_boolean(mask_values, expected):
    mask = aa.Mask1D(mask=mask_values, pixel_scales=1.0)

    assert mask.is_all_true == expected


@pytest.mark.parametrize(
    "mask_values,expected",
    [
        ([False, False, False, False], True),
        ([False, False], True),
        ([False, True, False, False], False),
        ([True, True, False, False], False),
    ],
)
def test__is_all_false__various_masks__returns_correct_boolean(mask_values, expected):
    mask = aa.Mask1D(mask=mask_values, pixel_scales=1.0)

    assert mask.is_all_false == expected
