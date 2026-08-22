import autoarray as aa
import numpy as np
import pytest


class _NotAConcreteScalar:
    """
    Stand-in for a JAX tracer: not a concrete Python/NumPy scalar, and raising if anything
    tries to resolve it to a bool, exactly as a tracer does inside `jax.jit`. Mirrors the
    stand-in in `test_autoarray/test_validate.py` — unit tests here are NumPy-only.
    """

    def __bool__(self):
        raise AssertionError("a tracer must never be resolved to a bool")


@pytest.mark.parametrize(
    "pixel_scales", [1, 1.0, np.float64(1.0), np.int32(1), np.float32(1.0)]
)
def test__convert_pixel_scales_1d__widens_any_real_scalar(pixel_scales):
    """
    Any concrete real scalar widens, not just an exact `float`. `int` is what a user types by
    hand; `np.floating` is what indexing an array or reading a FITS header returns.
    """
    assert aa.util.geometry.convert_pixel_scales_1d(pixel_scales=pixel_scales) == (1.0,)


@pytest.mark.parametrize(
    "pixel_scales", [1, 1.0, np.float64(1.0), np.int32(1), np.float32(1.0)]
)
def test__convert_pixel_scales_2d__widens_any_real_scalar(pixel_scales):
    assert aa.util.geometry.convert_pixel_scales_2d(pixel_scales=pixel_scales) == (
        1.0,
        1.0,
    )


@pytest.mark.parametrize("pixel_scales", [1, np.float64(1.0), np.int32(1)])
def test__convert_pixel_scales__widened_entries_are_python_floats(pixel_scales):
    """
    The widened value is cast, so an `int` or a NumPy scalar never reaches the geometry
    stored on a mask. `1 == 1.0` in Python, so the cast has to be asserted on the type.
    """
    (entry_1d,) = aa.util.geometry.convert_pixel_scales_1d(pixel_scales=pixel_scales)
    assert type(entry_1d) is float

    for entry in aa.util.geometry.convert_pixel_scales_2d(pixel_scales=pixel_scales):
        assert type(entry) is float


def test__convert_pixel_scales__tuple_input_is_returned_unchanged():
    pixel_scales_1d = (1.0,)
    assert (
        aa.util.geometry.convert_pixel_scales_1d(pixel_scales=pixel_scales_1d)
        is pixel_scales_1d
    )

    pixel_scales_2d = (1.0, 2.0)
    assert (
        aa.util.geometry.convert_pixel_scales_2d(pixel_scales=pixel_scales_2d)
        is pixel_scales_2d
    )


def test__convert_pixel_scales__a_tracer_passes_through_untouched():
    """Inside a `jax.jit` the value is traced; widening it would resolve it to a bool."""
    tracer_like = _NotAConcreteScalar()

    assert (
        aa.util.geometry.convert_pixel_scales_1d(pixel_scales=tracer_like)
        is tracer_like
    )
    assert (
        aa.util.geometry.convert_pixel_scales_2d(pixel_scales=tracer_like)
        is tracer_like
    )


def test__convert_pixel_scales__a_bool_is_not_treated_as_a_scalar():
    """
    `bool` is a subclass of `int`, but `True` reaching a pixel scale is a different mistake
    than the ones this widening serves — `validate.is_concrete_scalar` excludes it, so it is
    not silently accepted as a pixel scale of 1.0.
    """
    assert aa.util.geometry.convert_pixel_scales_1d(pixel_scales=True) is True
    assert aa.util.geometry.convert_pixel_scales_2d(pixel_scales=True) is True


def test__central_pixel_coordinates_1d_from():
    central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_1d_from(
        shape_slim=(3,)
    )

    assert central_pixel_coordinates == (1,)

    central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_1d_from(
        shape_slim=(4,)
    )

    assert central_pixel_coordinates == (1.5,)


def test__pixel_coordinates_1d_from():
    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(1.0,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(1.0,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(-1.0,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(0.0,), shape_slim=(3,), pixel_scales=(3.0,)
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(3.0,), shape_slim=(3,), pixel_scales=(3.0,)
    )

    assert pixel_coordinates == (2,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(-3.0,), shape_slim=(3,), pixel_scales=(3.0,)
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(3.0,), shape_slim=(3,), pixel_scales=(3.0,)
    )

    assert pixel_coordinates == (2,)

    # input coordinates are corners

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(-1.99,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(-0.01,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(0.01,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(-1.99,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(-0.01,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(0.01,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(1.99,), shape_slim=(2,), pixel_scales=(2.0,)
    )

    assert pixel_coordinates == (1,)

    # Input coordinates are centres

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(0.0,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(2.0,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(0.0,),
        shape_slim=(3,),
        pixel_scales=(3.0,),
        origins=(3.0,),
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(3.0,),
        shape_slim=(3,),
        pixel_scales=(3.0,),
        origins=(3.0,),
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(6.0,),
        shape_slim=(3,),
        pixel_scales=(3.0,),
        origins=(3.0,),
    )

    assert pixel_coordinates == (2,)

    # input coordinates are other corner

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(-0.99,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(0.99,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (0,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(1.01,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(2.99,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(1.01,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (1,)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
        scaled_coordinates_1d=(2.99,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert pixel_coordinates == (1,)


def test__scaled_coordinates_1d_from():
    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(0,), shape_slim=(3,), pixel_scales=(3.0,)
    )

    assert scaled_coordinates == (-3.0,)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(1,), shape_slim=(3,), pixel_scales=(3.0,)
    )

    assert scaled_coordinates == (0.0,)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(2,), shape_slim=(3,), pixel_scales=(3.0,)
    )

    assert scaled_coordinates == (3.0,)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(0,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert scaled_coordinates == (0.0,)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(1,),
        shape_slim=(2,),
        pixel_scales=(2.0,),
        origins=(1.0,),
    )

    assert scaled_coordinates == (2.0,)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(0,),
        shape_slim=(3,),
        pixel_scales=(3.0,),
        origins=(3.0,),
    )

    assert scaled_coordinates == (0.0,)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(1,),
        shape_slim=(3,),
        pixel_scales=(3.0,),
        origins=(3.0,),
    )

    assert scaled_coordinates == (3.0,)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
        pixel_coordinates_1d=(2,),
        shape_slim=(3,),
        pixel_scales=(3.0,),
        origins=(3.0,),
    )

    assert scaled_coordinates == (6.0,)


def test__central_pixel_coordinates_2d_from():
    central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(3, 3)
    )

    assert central_pixel_coordinates == (1, 1)

    central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(3, 3)
    )

    assert central_pixel_coordinates == (1, 1)

    central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(4, 4)
    )

    assert central_pixel_coordinates == (1.5, 1.5)

    central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(4, 4)
    )

    assert central_pixel_coordinates == (1.5, 1.5)


def test__pixel_coordinates_2d_from():
    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.0, -1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.0, 1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-1.0, -1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-1.0, 1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.0, -3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (0, 2)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, -3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (1, 2)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-3.0, -3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (2, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-3.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (2, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-3.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )

    assert pixel_coordinates == (2, 2)

    # Inputs are top-left corner

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.99, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.99, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.01, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.01, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.01, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.01, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.01, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.01, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.01, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.01, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 0)
    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-1.99, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-1.99, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.01, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.01, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-1.99, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-1.99, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )

    assert pixel_coordinates == (1, 1)

    # Inputs are centres

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.0, 0.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.0, 2.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, 0.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, 2.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(6.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(6.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(6.0, 6.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (0, 2)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.0, 6.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (1, 2)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (2, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (2, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.0, 6.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert pixel_coordinates == (2, 2)

    # Inputs are centres

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.99, -0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.99, 0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.01, -0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.01, 0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.01, 1.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(3.01, 2.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.01, 1.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(1.01, 2.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (0, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.99, -0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.99, 0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.99, -0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.99, 0.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 0)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.99, 1.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(0.99, 2.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.99, 1.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 1)

    pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(-0.99, 2.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert pixel_coordinates == (1, 1)

    # Inputs are centre

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 0), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (3.0, -3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 1), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (3.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 2), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (3.0, 3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 0), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (0.0, -3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 1), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (0.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 2), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (0.0, 3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(2, 0), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (-3.0, -3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(2, 1), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (-3.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(2, 2), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
    )

    assert scaled_coordinates == (-3.0, 3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert scaled_coordinates == (2.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 1),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert scaled_coordinates == (2.0, 2.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert scaled_coordinates == (0.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 1),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )

    assert scaled_coordinates == (0.0, 2.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (6.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 1),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (6.0, 3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(0, 2),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (6.0, 6.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (3.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 1),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (3.0, 3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(1, 2),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (3.0, 6.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(2, 0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (0.0, 0.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(2, 1),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (0.0, 3.0)

    scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(2, 2),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )

    assert scaled_coordinates == (0.0, 6.0)


def test__pixel_coordinates_wcs_2d_from():
    # -----------------------------
    # (2,2) grid: centre is (1.5, 1.5) in WCS pixels
    # pixel_scales = (2,2)
    # -----------------------------

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(1.0, -1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(1.0, 1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 2.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-1.0, -1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((2.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-1.0, 1.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((2.0, 2.0))

    # -----------------------------
    # (3,3) grid: centre is (2.0, 2.0) in WCS pixels
    # pixel_scales = (3,3)
    # -----------------------------

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(3.0, -3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(3.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 2.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(3.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 3.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.0, -3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((2.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((2.0, 2.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((2.0, 3.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-3.0, -3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((3.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-3.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((3.0, 2.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-3.0, 3.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((3.0, 3.0))

    # -----------------------------------------
    # Inputs near corners (continuous coordinates)
    # -----------------------------------------

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(1.99, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((0.505, 0.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(1.99, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((0.505, 1.495))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.01, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.495, 0.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.01, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.495, 1.495))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(2.01, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((0.495, 1.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(2.01, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((0.495, 2.495))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.01, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.495, 1.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.01, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.495, 2.495))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-0.01, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.505, 0.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-0.01, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.505, 1.495))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-1.99, -1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((2.495, 0.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-1.99, -0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((2.495, 1.495))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-0.01, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.505, 1.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-0.01, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((1.505, 2.495))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-1.99, 0.01),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((2.495, 1.505))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(-1.99, 1.99),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    assert pixel_coordinates == pytest.approx((2.495, 2.495))

    # -----------------------------------------
    # Inputs are centres (origins shift), still continuous outputs
    # -----------------------------------------

    # Inputs are centres (origins shift), continuous outputs

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(2.0, 0.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(2.0, 2.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 2.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.0, 0.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )
    assert pixel_coordinates == pytest.approx((2.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(0.0, 2.0),
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
        origins=(1.0, 1.0),
    )
    assert pixel_coordinates == pytest.approx((2.0, 2.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(6.0, 0.0),
        shape_native=(3, 3),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((1.0, 1.0))

    pixel_coordinates = aa.util.geometry.pixel_coordinates_wcs_2d_from(
        scaled_coordinates_2d=(6.0, 3.0),
        shape_native=(4, 4),
        pixel_scales=(3.0, 3.0),
        origins=(3.0, 3.0),
    )
    assert pixel_coordinates == pytest.approx((1.5, 2.5))


def test__transform_2d_grid_to_reference_frame():
    grid_2d = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_2d, centre=(0.0, 0.0), angle=0.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]), abs=1.0e-4
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_2d, centre=(0.0, 0.0), angle=45.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array(
            [
                [-np.sqrt(2) / 2.0, np.sqrt(2) / 2.0],
                [0.0, np.sqrt(2)],
                [np.sqrt(2) / 2.0, np.sqrt(2) / 2.0],
            ]
        ),
        abs=1.0e-4,
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_2d, centre=(0.0, 0.0), angle=90.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array([[-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]]), abs=1.0e-4
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_2d, centre=(0.0, 0.0), angle=180.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array([[0.0, -1.0], [-1.0, -1.0], [-1.0, 0.0]]), abs=1.0e-4
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_2d, centre=(5.0, 10.0), angle=0.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array([[-5.0, -9.0], [-4.0, -9.0], [-4.0, -10.0]])
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_2d, centre=(5.0, 10.0), angle=90.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array([[9.0, -5.0], [9.0, -4.0], [10.0, -4.0]])
    )


def test__transform_2d_grid_from_reference_frame():
    grid_2d = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
        grid_2d=grid_2d, centre=(0.0, 0.0), angle=0.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
        grid_2d=grid_2d, centre=(0.0, 0.0), angle=45.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array(
            [
                [np.sqrt(2) / 2.0, np.sqrt(2) / 2.0],
                [np.sqrt(2), 0.0],
                [np.sqrt(2) / 2.0, -np.sqrt(2) / 2.0],
            ]
        )
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
        grid_2d=grid_2d, centre=(2.0, 2.0), angle=90.0
    )

    assert transformed_grid_2d == pytest.approx(
        np.array([[3.0, 2.0], [3.0, 1.0], [2.0, 1.0]])
    )

    transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
        grid_2d=grid_2d, centre=(8.0, 5.0), angle=137.0
    )

    original_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
        grid_2d=transformed_grid_2d, centre=(8.0, 5.0), angle=137.0
    )

    assert grid_2d == pytest.approx(original_grid_2d, abs=1.0e-4)


def test__grid_pixels_2d_slim_from():
    # coordinates in centres_of_pixels

    grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

    grid_pixels = aa.util.geometry.grid_pixels_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (
        grid_pixels == np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
    ).all()

    # coordinates top-left of pixels

    grid_scaled = np.array([[2.0, -4], [2.0, 0.0], [0.0, -4], [0.0, 0.0]])

    grid_pixels = aa.util.geometry.grid_pixels_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

    # coordinates bottom-right of pixels

    grid_scaled = np.array([[0.0, 0.0], [0.0, 4.0], [-2.0, 0.0], [-2.0, 4.0]])

    grid_pixels = aa.util.geometry.grid_pixels_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([[1, 1], [1, 2], [2, 1], [2, 2]])).all()

    # -1.0 from all entries for a origin of (-1.0, -1.0)
    grid_scaled = np.array([[-1.0, -1.0], [-1.0, 3.0], [-3.0, -1.0], [-3.0, 3.0]])

    grid_pixels = aa.util.geometry.grid_pixels_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
        origin=(-1.0, -1.0),
    )

    assert (grid_pixels == np.array([[1, 1], [1, 2], [2, 1], [2, 2]])).all()


def test__grid_pixel_centres_2d_slim_from():
    # coordinates in centres of pixels

    grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

    # coordinates top-left of pixels

    grid_scaled = np.array([[1.99, -3.99], [1.99, 0.01], [-0.01, -3.99], [-0.01, 0.01]])

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

    # coordinates bottom-right of pixels

    grid_scaled = np.array([[0.01, -0.01], [0.01, 3.99], [-1.99, -0.01], [-1.99, 3.99]])

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

    # nonzero origin

    # +1.0 for all entries for a origin of (1.0, 1.0)
    grid_scaled = np.array([[2.0, -1.0], [2.0, 3.0], [0.0, -1.0], [0.0, 3.0]])

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
        origin=(1.0, 1.0),
    )

    assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()


def test__grid_pixel_indexes_2d_slim_from():
    # coordinates in centres of pixels

    grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

    grid_pixels = aa.util.geometry.grid_pixel_indexes_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([0, 1, 2, 3])).all()

    # coordinates top-left of pixels

    grid_scaled = np.array([[1.99, -3.99], [1.99, 0.01], [-0.01, -3.99], [-0.01, 0.01]])

    grid_pixels = aa.util.geometry.grid_pixel_indexes_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([0, 1, 2, 3])).all()

    # coordinates bottom-right of pixels

    grid_scaled = np.array([[0.01, -0.01], [0.01, 3.99], [-1.99, -0.01], [-1.99, 3.99]])

    grid_pixels = aa.util.geometry.grid_pixel_indexes_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (grid_pixels == np.array([0, 1, 2, 3])).all()

    # non-zero origin

    # +1.0 for all entries for a origin of (1.0, 1.0)
    grid_scaled = np.array([[2.0, -1.0], [2.0, 3.0], [0.0, -1.0], [0.0, 3.0]])

    grid_pixels = aa.util.geometry.grid_pixel_indexes_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
        origin=(1.0, 1.0),
    )

    assert (grid_pixels == np.array([0, 1, 2, 3])).all()


def test__grid_scaled_2d_slim_from():
    # coordinates in centres of pixels

    grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])

    grid_scaled = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (
        grid_scaled == np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])
    ).all()

    # coordinates top-left of pixels

    grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    grid_scaled = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (
        grid_scaled == np.array([[2.0, -4], [2.0, 0.0], [0.0, -4], [0.0, 0.0]])
    ).all()

    # coordinates bottom-right of pixels

    grid_pixels = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    grid_pixels = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    grid_scaled = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    assert (
        grid_scaled == np.array([[0.0, 0.0], [0.0, 4.0], [-2.0, 0.0], [-2.0, 4.0]])
    ).all()

    # non-zero origin

    grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])

    grid_scaled = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
        origin=(-1.0, -1.0),
    )

    # -1.0 from all entries for a origin of (-1.0, -1.0)
    assert (
        grid_scaled == np.array([[0.0, -3.0], [0.0, 1.0], [-2.0, -3.0], [-2.0, 1.0]])
    ).all()


def test__grid_pixel_centres_2d_from():
    # coordinates in centres of pixels

    grid_scaled = np.array([[[1.0, -2.0], [1.0, 2.0]], [[-1.0, -2.0], [-1.0, 2.0]]])

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_from(
        grid_scaled_2d=grid_scaled, shape_native=(2, 2), pixel_scales=(2.0, 4.0)
    )

    assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()

    # coordinates top-left of pixels

    grid_scaled = np.array(
        [[[1.99, -3.99], [1.99, 0.01]], [[-0.01, -3.99], [-0.01, 0.01]]]
    )

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_from(
        grid_scaled_2d=grid_scaled, shape_native=(2, 2), pixel_scales=(2.0, 4.0)
    )

    assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()

    # coordinates bottom-right of pixels

    grid_scaled = np.array(
        [[[0.01, -0.01], [0.01, 3.99]], [[-1.99, -0.01], [-1.99, 3.99]]]
    )

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_from(
        grid_scaled_2d=grid_scaled, shape_native=(2, 2), pixel_scales=(2.0, 4.0)
    )

    assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()

    # non-zero origin

    # +1.0 for all entries for a origin of (1.0, 1.0)
    grid_scaled = np.array([[[2.0, -1.0], [2.0, 3.0]], [[0.0, -1.0], [0.0, 3.0]]])

    grid_pixels = aa.util.geometry.grid_pixel_centres_2d_from(
        grid_scaled_2d=grid_scaled,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
        origin=(1.0, 1.0),
    )

    assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()
