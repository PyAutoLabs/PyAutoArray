"""
``AdaptSplitPower`` / ``AdaptSplitZerothPower``: the corrected siblings of the split adaptive schemes.

The split family shares ``pixel_splitted_regularization_matrix_from`` with ``ConstantSplit``, so it
never carried ``Adapt``'s factor-2 scatter asymmetry. The only difference is the coefficient
convention, and with ``power=1.0`` (the default) the two families coincide exactly.
"""

import numpy as np
import pytest

import autoarray as aa


@pytest.mark.parametrize("coefficient", [0.5, 2.0, 7.0])
def test__uniform_coefficients__matrix_equals_constant_split_exactly(
    delaunay_mapper_9_3x3, coefficient
):
    regularization_matrix = aa.reg.AdaptSplitPower(
        inner_coefficient=coefficient, outer_coefficient=coefficient
    ).regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    regularization_matrix_constant_split = aa.reg.ConstantSplit(
        coefficient=coefficient
    ).regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    assert (regularization_matrix == regularization_matrix_constant_split).all()


def test__power_2_reproduces_the_legacy_adapt_split_matrix(delaunay_mapper_9_3x3):
    regularization_matrix_legacy = aa.reg.AdaptSplit(
        inner_coefficient=1.0, outer_coefficient=2.0, signal_scale=1.0
    ).regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    regularization_matrix_power = aa.reg.AdaptSplitPower(
        inner_coefficient=1.0, outer_coefficient=2.0, signal_scale=1.0, power=2.0
    ).regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    assert (regularization_matrix_power == regularization_matrix_legacy).all()


def test__weights_are_unsquared_by_default(delaunay_mapper_9_3x3):
    weights_legacy = aa.reg.AdaptSplit(
        inner_coefficient=1.0, outer_coefficient=2.0
    ).regularization_weights_from(linear_obj=delaunay_mapper_9_3x3)

    weights_power = aa.reg.AdaptSplitPower(
        inner_coefficient=1.0, outer_coefficient=2.0
    ).regularization_weights_from(linear_obj=delaunay_mapper_9_3x3)

    assert weights_power**2.0 == pytest.approx(weights_legacy, 1.0e-12)


def test__zeroth__power_2_reproduces_the_legacy_matrix(delaunay_mapper_9_3x3):
    regularization_matrix_legacy = aa.reg.AdaptSplitZeroth(
        inner_coefficient=1.0,
        outer_coefficient=2.0,
        signal_scale=1.0,
        zeroth_coefficient=3.0,
        zeroth_signal_scale=2.0,
    ).regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    regularization_matrix_power = aa.reg.AdaptSplitZerothPower(
        inner_coefficient=1.0,
        outer_coefficient=2.0,
        signal_scale=1.0,
        zeroth_coefficient=3.0,
        zeroth_signal_scale=2.0,
        power=2.0,
    ).regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    assert (regularization_matrix_power == regularization_matrix_legacy).all()


def test__zeroth__split_leg_equals_constant_split_plus_the_unchanged_zeroth_leg(
    delaunay_mapper_9_3x3,
):
    """
    ``zeroth_coefficient`` is squared exactly once by ``BrightnessZeroth`` and is therefore unchanged
    between the two classes — only the split leg's convention moves.
    """
    regularization_matrix = aa.reg.AdaptSplitZerothPower(
        inner_coefficient=2.0,
        outer_coefficient=2.0,
        signal_scale=1.0,
        zeroth_coefficient=3.0,
        zeroth_signal_scale=2.0,
    ).regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    regularization_matrix_expected = aa.reg.ConstantSplit(
        coefficient=2.0
    ).regularization_matrix_from(
        linear_obj=delaunay_mapper_9_3x3
    ) + aa.reg.BrightnessZeroth(
        coefficient=3.0, signal_scale=2.0
    ).regularization_matrix_from(
        linear_obj=delaunay_mapper_9_3x3
    )

    assert regularization_matrix == pytest.approx(
        regularization_matrix_expected, 1.0e-12
    )


def test__matern__weights_are_unsquared_by_default_and_squared_at_power_2():
    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid,
        pixel_signals=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )

    kwargs = dict(
        scale=0.1,
        nu=0.5,
        inner_coefficient=0.1,
        outer_coefficient=0.2,
        signal_scale=0.1,
    )

    weights_legacy = aa.reg.MaternAdaptKernel(**kwargs).regularization_weights_from(
        linear_obj=mapper
    )

    weights_power_1 = aa.reg.MaternAdaptPowerKernel(
        **kwargs
    ).regularization_weights_from(linear_obj=mapper)

    weights_power_2 = aa.reg.MaternAdaptPowerKernel(
        power=2.0, **kwargs
    ).regularization_weights_from(linear_obj=mapper)

    assert weights_power_1**2.0 == pytest.approx(weights_legacy, 1.0e-12)
    assert (weights_power_2 == weights_legacy).all()
