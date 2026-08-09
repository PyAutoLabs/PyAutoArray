"""
Regression tests for PyAutoArray#333 — constructor input validation.

Each test in the "findings" section is built from @rhayes777's own snippet in the
issue body, and asserts the *failure*: that the input is rejected at construction
with a message naming the offending parameter, rather than accepted and surfaced as
a confusing traceback several calls later (or, worse, silently accepted).

Each finding is paired with a control asserting the valid input still works, so a
guard cannot pass by rejecting everything.

Tests are numpy-only, per phase 1 — the tracer-safety property is asserted against
the concreteness gate itself (`is_concrete_scalar`), which is what the guards
branch on, rather than by importing JAX into the library unit tests.
"""

import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray import validate


class _NotAConcreteScalar:
    """
    Stand-in for a JAX tracer: an object that is not a concrete Python/NumPy scalar
    and which raises if anything tries to resolve it to a bool, exactly as a tracer
    does inside `jax.jit`.
    """

    def __bool__(self):
        raise AssertionError(
            "a guard compared a non-concrete value — this is the TracerBoolConversionError path"
        )

    def __lt__(self, other):
        return self

    def __le__(self, other):
        return self

    def __ge__(self, other):
        return self


# ======================================================================================
# The concreteness gate
# ======================================================================================


def test__is_concrete_scalar__true_for_python_and_numpy_real_scalars():
    assert validate.is_concrete_scalar(1)
    assert validate.is_concrete_scalar(1.5)
    assert validate.is_concrete_scalar(-3)
    assert validate.is_concrete_scalar(np.float64(1.5))
    assert validate.is_concrete_scalar(np.int32(2))


def test__is_concrete_scalar__false_for_bool_none_arrays_and_tracer_likes():
    assert not validate.is_concrete_scalar(True)
    assert not validate.is_concrete_scalar(None)
    assert not validate.is_concrete_scalar("1.0")
    assert not validate.is_concrete_scalar(np.array([1.0, 2.0]))
    assert not validate.is_concrete_scalar(_NotAConcreteScalar())


def test__guards_pass_through_non_concrete_values__never_compare_them():
    """
    The binding JAX constraint: a guard must not evaluate a Python truth-test on a
    value that may be a tracer. `_NotAConcreteScalar.__bool__` raises, so these calls
    returning cleanly is the assertion.
    """
    tracer_like = _NotAConcreteScalar()

    validate.validate_positive_finite(value=tracer_like, name="pixel_scales")
    validate.validate_non_negative_finite(value=tracer_like, name="coefficient")
    validate.validate_pixel_scales(pixel_scales=tracer_like)
    validate.validate_pixel_scales(pixel_scales=(tracer_like, tracer_like))
    validate.validate_radii_ordered(inner_radius=tracer_like, outer_radius=tracer_like)
    validate.validate_shape_native(shape_native=(tracer_like, tracer_like))


def test__regularization_constructors__accept_a_tracer_like_coefficient():
    """A free model parameter arrives as a tracer under a traced fit; it must build."""
    tracer_like = _NotAConcreteScalar()

    assert aa.reg.Constant(coefficient=tracer_like).coefficient is tracer_like
    assert (
        aa.reg.Adapt(
            inner_coefficient=tracer_like, outer_coefficient=tracer_like
        ).inner_coefficient
        is tracer_like
    )


# ======================================================================================
# B6 — pixel_scales must be finite and positive
# ======================================================================================


@pytest.mark.parametrize("pixel_scales", [0.0, -0.1, float("nan"), float("inf")])
def test__b6__array_2d_rejects_non_positive_or_non_finite_pixel_scales(pixel_scales):
    with pytest.raises(ValueError, match="pixel_scales"):
        aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=pixel_scales)


@pytest.mark.parametrize("pixel_scales", [0.0, -0.1, float("nan")])
def test__b6__mask_2d_rejects_non_positive_or_non_finite_pixel_scales(pixel_scales):
    with pytest.raises(ValueError, match="pixel_scales"):
        aa.Mask2D.circular(shape_native=(5, 5), radius=1.0, pixel_scales=pixel_scales)


def test__b6__per_axis_pixel_scales_are_validated_and_the_message_names_the_axis():
    with pytest.raises(ValueError, match=r"pixel_scales\[1\]"):
        aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=(0.1, -0.1))


def test__b6__control__valid_pixel_scales_still_build():
    array = aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=0.1)
    assert array.pixel_scales == (0.1, 0.1)

    array = aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=(0.1, 0.2))
    assert array.pixel_scales == (0.1, 0.2)


# ======================================================================================
# B7 — annulus radii must be ordered
# ======================================================================================


def test__b7__circular_annular_rejects_inner_radius_above_outer_radius():
    with pytest.raises(exc.MaskException, match="inner_radius"):
        aa.Mask2D.circular_annular(
            shape_native=(10, 10),
            inner_radius=0.8,
            outer_radius=0.3,
            pixel_scales=0.1,
        )


def test__b7__circular_annular_rejects_equal_radii():
    with pytest.raises(exc.MaskException, match="inner_radius"):
        aa.Mask2D.circular_annular(
            shape_native=(10, 10),
            inner_radius=0.5,
            outer_radius=0.5,
            pixel_scales=0.1,
        )


def test__b7__elliptical_annular_rejects_inner_major_axis_above_outer():
    with pytest.raises(exc.MaskException, match="inner_major_axis_radius"):
        aa.Mask2D.elliptical_annular(
            shape_native=(10, 10),
            inner_major_axis_radius=0.8,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=0.3,
            outer_axis_ratio=1.0,
            outer_phi=0.0,
            pixel_scales=0.1,
        )


def test__b7__control__well_ordered_annulus_still_builds_and_is_not_empty():
    mask = aa.Mask2D.circular_annular(
        shape_native=(10, 10), inner_radius=0.3, outer_radius=0.8, pixel_scales=0.1
    )
    assert mask.pixels_in_mask == 68


# ======================================================================================
# B8 — shape_native must have no zero-length axis
# ======================================================================================


@pytest.mark.parametrize("shape_native", [(0, 0), (0, 5), (5, 0)])
def test__b8__grid_2d_uniform_rejects_a_zero_length_axis(shape_native):
    with pytest.raises(exc.MaskException, match="shape_native"):
        aa.Grid2D.uniform(shape_native=shape_native, pixel_scales=0.1)


def test__b8__mask_2d_all_false_rejects_a_zero_length_axis():
    with pytest.raises(exc.MaskException, match="shape_native"):
        aa.Mask2D.all_false(shape_native=(0, 5), pixel_scales=0.1)


def test__b8__control__a_non_degenerate_grid_still_builds():
    grid = aa.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.1)
    assert grid.shape_slim == 25


# ======================================================================================
# B5 — data and noise_map shapes must agree
# ======================================================================================


def test__b5__imaging_rejects_a_noise_map_whose_shape_differs_from_the_data():
    data = aa.Array2D.no_mask(values=np.ones((10, 10)), pixel_scales=0.1)
    noise_map = aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=0.1)

    with pytest.raises(exc.DatasetException, match="noise_map"):
        aa.Imaging(data=data, noise_map=noise_map)


def test__b5__the_message_reports_both_shapes():
    data = aa.Array2D.no_mask(values=np.ones((10, 10)), pixel_scales=0.1)
    noise_map = aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=0.1)

    with pytest.raises(exc.DatasetException) as error:
        aa.Imaging(data=data, noise_map=noise_map)

    assert "(10, 10)" in str(error.value)
    assert "(5, 5)" in str(error.value)


def test__b5__control__matched_shapes_still_build():
    data = aa.Array2D.no_mask(values=np.ones((10, 10)), pixel_scales=0.1)
    noise_map = aa.Array2D.no_mask(values=np.ones((10, 10)), pixel_scales=0.1)

    dataset = aa.Imaging(data=data, noise_map=noise_map)

    assert dataset.shape_native == (10, 10)


# ======================================================================================
# B13 — regularization coefficients must be non-negative
# ======================================================================================


def test__b13__constant_rejects_a_negative_coefficient():
    with pytest.raises(ValueError, match="coefficient"):
        aa.reg.Constant(coefficient=-1.0)


@pytest.mark.parametrize(
    "regularization_cls",
    [
        aa.reg.Constant,
        aa.reg.ConstantSplit,
        aa.reg.Zeroth,
        aa.reg.BrightnessZeroth,
        aa.reg.CurvatureMask,
        aa.reg.FourthOrderMask,
        aa.reg.ExponentialKernel,
        aa.reg.GaussianKernel,
        aa.reg.MaternKernel,
    ],
)
def test__b13__every_single_coefficient_scheme_rejects_a_negative_coefficient(
    regularization_cls,
):
    """The reporter named `Constant`; the same hole was open in every sibling scheme."""
    with pytest.raises(ValueError, match="coefficient"):
        regularization_cls(coefficient=-1.0)


@pytest.mark.parametrize(
    "regularization_cls", [aa.reg.Adapt, aa.reg.AdaptSplit, aa.reg.MaternAdaptKernel]
)
def test__b13__inner_and_outer_coefficient_schemes_reject_negatives(regularization_cls):
    with pytest.raises(ValueError, match="inner_coefficient"):
        regularization_cls(inner_coefficient=-1.0, outer_coefficient=1.0)

    with pytest.raises(ValueError, match="outer_coefficient"):
        regularization_cls(inner_coefficient=1.0, outer_coefficient=-1.0)


def test__b13__constant_zeroth_rejects_negatives_on_both_coefficients():
    with pytest.raises(ValueError, match="coefficient_neighbor"):
        aa.reg.ConstantZeroth(coefficient_neighbor=-1.0, coefficient_zeroth=1.0)

    with pytest.raises(ValueError, match="coefficient_zeroth"):
        aa.reg.ConstantZeroth(coefficient_neighbor=1.0, coefficient_zeroth=-1.0)


def test__b13__adapt_split_zeroth_rejects_a_negative_zeroth_coefficient():
    with pytest.raises(ValueError, match="zeroth_coefficient"):
        aa.reg.AdaptSplitZeroth(
            zeroth_coefficient=-1.0, inner_coefficient=1.0, outer_coefficient=1.0
        )


def test__b13__nan_and_inf_coefficients_are_rejected():
    with pytest.raises(ValueError, match="coefficient"):
        aa.reg.Constant(coefficient=float("nan"))

    with pytest.raises(ValueError, match="coefficient"):
        aa.reg.Constant(coefficient=float("inf"))


def test__b13__control__zero_is_permitted_as_a_request_for_no_regularization():
    """Zero is degenerate but meaningful, unlike a negative — it must not be rejected."""
    assert aa.reg.Constant(coefficient=0.0).coefficient == 0.0


def test__b13__control__a_positive_coefficient_still_builds():
    assert aa.reg.Constant(coefficient=1.0).coefficient == 1.0


class _MockLinearObj:
    params = 4


def test__b13__the_sharpened_finding__negative_weights_can_no_longer_leak():
    """
    `regularization_matrix_from` squares the coefficient, which hides the sign and led
    the reporter to read a negative value as inert. `regularization_weights_from`
    returns it *unsquared*, so a negative coefficient leaked negative regularization
    weights into every consumer of that method. Rejecting at construction closes it.
    """
    weights = aa.reg.Constant(coefficient=2.0).regularization_weights_from(
        linear_obj=_MockLinearObj()
    )
    assert np.all(np.asarray(weights) >= 0)

    with pytest.raises(ValueError, match="coefficient"):
        aa.reg.Constant(coefficient=-1.0).regularization_weights_from(
            linear_obj=_MockLinearObj()
        )
