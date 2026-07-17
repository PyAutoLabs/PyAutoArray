import numpy as np
import pytest

import autoarray as aa


class MockMaskedLinearObj:
    def __init__(self, mask):
        self.mask = mask

    @property
    def params(self):
        return int(np.count_nonzero(~self.mask))


def test__regularization_matrix_from__matches_hand_computed_operators():
    mask = np.array([[False, False, False]])

    reg = aa.reg.CurvatureMask(coefficient=1.0)

    regularization_matrix = reg.regularization_matrix_from(
        linear_obj=MockMaskedLinearObj(mask=mask)
    )

    # Along x: pixel 0 carries the 2nd-order stencil [1, -2, 1], pixel 1 the
    # 1st-order [-1, 1] and pixel 2 the 0th-order [1]. Along y no pixel has a
    # neighbour below, so every row is 0th-order (the identity).
    hx = np.array([[1.0, -2.0, 1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
    hy = np.eye(3)
    expected = hx.T @ hx + hy.T @ hy

    assert regularization_matrix == pytest.approx(expected, abs=1.0e-10)


def test__regularization_matrix_from__symmetric_positive_semi_definite():
    mask = np.ones((8, 8), dtype=bool)
    mask[2:6, 2:6] = False

    reg = aa.reg.CurvatureMask(coefficient=1.0)

    regularization_matrix = reg.regularization_matrix_from(
        linear_obj=MockMaskedLinearObj(mask=mask)
    )

    assert regularization_matrix == pytest.approx(regularization_matrix.T, abs=1.0e-12)
    assert np.linalg.eigvalsh(regularization_matrix).min() > -1.0e-10


def test__regularization_matrix_from__scales_linearly_with_coefficient():
    mask = np.ones((8, 8), dtype=bool)
    mask[2:6, 2:6] = False
    linear_obj = MockMaskedLinearObj(mask=mask)

    matrix_coeff_1 = aa.reg.CurvatureMask(coefficient=1.0).regularization_matrix_from(
        linear_obj=linear_obj
    )
    matrix_coeff_3 = aa.reg.CurvatureMask(coefficient=3.0).regularization_matrix_from(
        linear_obj=linear_obj
    )

    assert matrix_coeff_3 == pytest.approx(3.0 * matrix_coeff_1, abs=1.0e-10)


def test__regularization_weights_from():
    mask = np.ones((8, 8), dtype=bool)
    mask[2:6, 2:6] = False

    reg = aa.reg.CurvatureMask(coefficient=2.0)

    weights = reg.regularization_weights_from(
        linear_obj=MockMaskedLinearObj(mask=mask)
    )

    assert weights == pytest.approx(2.0 * np.ones(16), abs=1.0e-10)
