import pytest

import autoarray as aa
import numpy as np


def _mapper_and_grid():
    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )
    mapper = aa.m.MockMapper(source_plane_mesh_grid=source_plane_mesh_grid)
    return mapper


@pytest.mark.parametrize(
    "reg, rtol",
    [
        (aa.reg.MaternKernel(coefficient=3.0, scale=2.0, nu=2.0), 1.0e-8),
        (aa.reg.ExponentialKernel(coefficient=3.0, scale=2.0), 1.0e-8),
        # GaussianKernel's formed matrix carries an extra trace-scaled diagonal
        # stabilisation jitter the analytic shortcut deliberately excludes, so the
        # agreement is at the jitter scale rather than machine precision.
        (aa.reg.GaussianKernel(coefficient=3.0, scale=0.5), 1.0e-6),
    ],
)
def test__kernel_log_det_shortcut__matches_slogdet_of_formed_matrix(reg, rtol):
    mapper = _mapper_and_grid()

    formed = reg.regularization_matrix_from(linear_obj=mapper)
    expected = np.linalg.slogdet(formed)[1]

    shortcut = reg.log_det_regularization_matrix_term_from(linear_obj=mapper)

    assert shortcut == pytest.approx(expected, rel=rtol)


def test__matern_adapt_kernel_log_det_shortcut__matches_slogdet_of_formed_matrix():
    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )
    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid,
        pixel_signals=np.array([0.1, 0.4, 0.2, 0.9, 0.5, 0.3]),
    )

    reg = aa.reg.MaternAdaptKernel(
        scale=2.0, nu=2.0, inner_coefficient=0.5, outer_coefficient=4.0
    )

    formed = reg.regularization_matrix_from(linear_obj=mapper)
    expected = np.linalg.slogdet(formed)[1]

    shortcut = reg.log_det_regularization_matrix_term_from(linear_obj=mapper)

    assert shortcut == pytest.approx(expected, rel=1.0e-8)


def test__kernel_jitter_kwarg__none_default_is_byte_identical_to_1e_minus_8():
    mapper = _mapper_and_grid()

    for reg_none, reg_explicit in [
        (
            aa.reg.MaternKernel(coefficient=1.0, scale=2.0, nu=2.0),
            aa.reg.MaternKernel(coefficient=1.0, scale=2.0, nu=2.0, jitter=1e-8),
        ),
        (
            aa.reg.ExponentialKernel(coefficient=1.0, scale=2.0),
            aa.reg.ExponentialKernel(coefficient=1.0, scale=2.0, jitter=1e-8),
        ),
        (
            aa.reg.GaussianKernel(coefficient=1.0, scale=0.5),
            aa.reg.GaussianKernel(coefficient=1.0, scale=0.5, jitter=1e-8),
        ),
    ]:
        matrix_none = reg_none.regularization_matrix_from(linear_obj=mapper)
        matrix_explicit = reg_explicit.regularization_matrix_from(linear_obj=mapper)

        assert np.array_equal(matrix_none, matrix_explicit)


def test__non_kernel_regularization__has_no_log_det_shortcut():
    mapper = _mapper_and_grid()

    assert (
        aa.reg.Constant(coefficient=1.0).log_det_regularization_matrix_term_from(
            linear_obj=mapper
        )
        is None
    )


def test__inversion_log_det_regularization__kernel_shortcut_only_under_slogdet():
    mapper = _mapper_and_grid()
    reg = aa.reg.MaternKernel(coefficient=3.0, scale=2.0, nu=2.0)
    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=mapper.source_plane_mesh_grid,
        regularization=reg,
    )

    formed = reg.regularization_matrix_from(linear_obj=mapper)

    # Default ("cholesky") path: byte-identical to factorizing the formed matrix —
    # the shortcut is never consulted.
    default_inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        regularization_matrix=formed,
    )
    expected_default = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(formed))))
    assert default_inversion.log_det_regularization_matrix_term == pytest.approx(
        expected_default, rel=1.0e-12
    )

    # Opt-in "slogdet" path: the analytically exact factorization shortcut.
    slogdet_inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        regularization_matrix=formed,
        settings=aa.Settings(log_det_method="slogdet"),
    )
    expected_shortcut = reg.log_det_regularization_matrix_term_from(linear_obj=mapper)
    assert slogdet_inversion.log_det_regularization_matrix_term == pytest.approx(
        expected_shortcut, rel=1.0e-12
    )

    # On a well-conditioned fixture the two agree to high precision.
    assert slogdet_inversion.log_det_regularization_matrix_term == pytest.approx(
        default_inversion.log_det_regularization_matrix_term, rel=1.0e-8
    )
