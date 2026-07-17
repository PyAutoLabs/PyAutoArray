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
    mask = np.array([[False, False, False, False, False]])

    reg = aa.reg.FourthOrderMask(coefficient=1.0)

    regularization_matrix = reg.regularization_matrix_from(
        linear_obj=MockMaskedLinearObj(mask=mask)
    )

    # Along x the stencil order degrades with the run of pixels ahead:
    # pixel 0 carries the 4th-order stencil, pixel 1 the 3rd, pixel 2 the
    # 2nd, pixel 3 the 1st and pixel 4 the 0th. Along y every row is
    # 0th-order (the identity).
    hx = np.array(
        [
            [1.0, -4.0, 6.0, -4.0, 1.0],
            [0.0, -1.0, 3.0, -3.0, 1.0],
            [0.0, 0.0, 1.0, -2.0, 1.0],
            [0.0, 0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    hy = np.eye(5)
    expected = hx.T @ hx + hy.T @ hy

    assert regularization_matrix == pytest.approx(expected, abs=1.0e-10)


def test__regularization_matrix_from__symmetric_positive_semi_definite():
    mask = np.ones((10, 10), dtype=bool)
    mask[2:8, 2:8] = False

    reg = aa.reg.FourthOrderMask(coefficient=1.0)

    regularization_matrix = reg.regularization_matrix_from(
        linear_obj=MockMaskedLinearObj(mask=mask)
    )

    assert regularization_matrix == pytest.approx(regularization_matrix.T, abs=1.0e-12)
    assert np.linalg.eigvalsh(regularization_matrix).min() > -1.0e-10


def test__regularization_matrix_from__scales_linearly_with_coefficient():
    mask = np.ones((10, 10), dtype=bool)
    mask[2:8, 2:8] = False
    linear_obj = MockMaskedLinearObj(mask=mask)

    matrix_coeff_1 = aa.reg.FourthOrderMask(
        coefficient=1.0
    ).regularization_matrix_from(linear_obj=linear_obj)
    matrix_coeff_3 = aa.reg.FourthOrderMask(
        coefficient=3.0
    ).regularization_matrix_from(linear_obj=linear_obj)

    assert matrix_coeff_3 == pytest.approx(3.0 * matrix_coeff_1, abs=1.0e-10)
