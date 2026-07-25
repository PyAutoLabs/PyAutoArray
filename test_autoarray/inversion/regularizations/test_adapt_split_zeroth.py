import numpy as np
import pytest

import autoarray as aa


def test__regularization_matrix_from(delaunay_mapper_9_3x3):
    reg = aa.reg.AdaptSplit(
        inner_coefficient=1.0, outer_coefficient=2.0, signal_scale=1.0
    )

    regularization_matrix_adaptive = reg.regularization_matrix_from(
        linear_obj=delaunay_mapper_9_3x3
    )

    reg = aa.reg.BrightnessZeroth(coefficient=3.0, signal_scale=2.0)

    regularization_matrix_zeroth = reg.regularization_matrix_from(
        linear_obj=delaunay_mapper_9_3x3
    )

    regularization_matrix = (
        regularization_matrix_adaptive + regularization_matrix_zeroth
    )

    reg = aa.reg.AdaptSplitZeroth(
        inner_coefficient=1.0,
        outer_coefficient=2.0,
        signal_scale=1.0,
        zeroth_coefficient=3.0,
        zeroth_signal_scale=2.0,
    )

    regularization_matrix_both = reg.regularization_matrix_from(
        linear_obj=delaunay_mapper_9_3x3
    )

    assert (regularization_matrix_both == regularization_matrix).all()
