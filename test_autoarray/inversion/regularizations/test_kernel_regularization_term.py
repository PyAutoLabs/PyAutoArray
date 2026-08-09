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


RECONSTRUCTION = np.array([0.3, -1.2, 0.7, 2.1, -0.4, 1.5])


@pytest.mark.parametrize(
    "reg, rtol",
    [
        (aa.reg.MaternKernel(coefficient=3.0, scale=2.0, nu=2.0), 1.0e-8),
        (aa.reg.ExponentialKernel(coefficient=3.0, scale=2.0), 1.0e-8),
        # GaussianKernel's formed matrix carries an extra trace-scaled diagonal
        # stabilisation jitter (and a symmetrisation) which the analytic shortcut
        # deliberately excludes, exactly as its log-det shortcut does, so the
        # agreement is at the jitter scale rather than machine precision.
        (aa.reg.GaussianKernel(coefficient=3.0, scale=0.5), 1.0e-6),
    ],
)
def test__kernel_regularization_term_shortcut__matches_formed_matrix(reg, rtol):
    mapper = _mapper_and_grid()

    formed = reg.regularization_matrix_from(linear_obj=mapper)
    expected = RECONSTRUCTION @ (formed @ RECONSTRUCTION)

    shortcut = reg.regularization_term_from(
        linear_obj=mapper, reconstruction=RECONSTRUCTION
    )

    assert shortcut == pytest.approx(expected, rel=rtol)


def test__matern_adapt_kernel_regularization_term_shortcut__matches_formed_matrix():
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
    expected = RECONSTRUCTION @ (formed @ RECONSTRUCTION)

    shortcut = reg.regularization_term_from(
        linear_obj=mapper, reconstruction=RECONSTRUCTION
    )

    assert shortcut == pytest.approx(expected, rel=1.0e-8)


def test__matern_adapt_kernel__does_not_inherit_the_zero_coefficient_matern_term():
    """
    `MaternAdaptKernel` passes `coefficient=0.0` to `MaternKernel.__init__` (its adaptive
    weights live inside the covariance instead). Were it to inherit `MaternKernel`'s
    `regularization_term_from`, the `self.coefficient` factor would zero the term — this
    pins the override that prevents that.
    """
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

    assert reg.coefficient == 0.0

    term = reg.regularization_term_from(
        linear_obj=mapper, reconstruction=RECONSTRUCTION
    )

    assert term != 0.0
    assert np.isfinite(term)


def test__non_kernel_regularization__has_no_regularization_term_shortcut():
    mapper = _mapper_and_grid()

    assert (
        aa.reg.Constant(coefficient=1.0).regularization_term_from(
            linear_obj=mapper, reconstruction=RECONSTRUCTION
        )
        is None
    )


def test__inversion_regularization_term__kernel_shortcut_only_under_cho_solve():
    mapper = _mapper_and_grid()
    reg = aa.reg.MaternKernel(coefficient=3.0, scale=2.0, nu=2.0)
    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=mapper.source_plane_mesh_grid,
        regularization=reg,
    )

    formed = reg.regularization_matrix_from(linear_obj=mapper)

    # Default ("matmul") path: byte-identical to contracting the formed matrix — the
    # shortcut is never consulted.
    default_inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        regularization_matrix=formed,
        reconstruction=RECONSTRUCTION,
    )
    expected_default = RECONSTRUCTION @ (formed @ RECONSTRUCTION)
    assert default_inversion.regularization_term == pytest.approx(
        expected_default, rel=1.0e-12
    )

    # Opt-in "cho_solve" path: the Cholesky-solve shortcut.
    cho_solve_inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        regularization_matrix=formed,
        reconstruction=RECONSTRUCTION,
        settings=aa.Settings(regularization_term_method="cho_solve"),
    )
    expected_shortcut = reg.regularization_term_from(
        linear_obj=mapper, reconstruction=RECONSTRUCTION
    )
    assert cho_solve_inversion.regularization_term == pytest.approx(
        expected_shortcut, rel=1.0e-12
    )

    # On a well-conditioned fixture the two agree to high precision.
    assert cho_solve_inversion.regularization_term == pytest.approx(
        default_inversion.regularization_term, rel=1.0e-8
    )


def test__inversion_regularization_term__falls_back_when_a_scheme_has_no_shortcut():
    """
    A `Constant` scheme returns `None`, so even under "cho_solve" the whole computation
    must fall back to the formed matrix rather than dropping that object's contribution.
    """
    mapper = _mapper_and_grid()
    reg = aa.reg.Constant(coefficient=1.0)
    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=mapper.source_plane_mesh_grid,
        regularization=reg,
    )

    # Any SPD matrix serves as the formed `H` here — the point of the test is that the
    # `Constant` scheme has no shortcut, so the formed matrix is what must be used. It is
    # built from a kernel scheme only because `Constant`'s own matrix needs mesh
    # neighbours this mock fixture does not carry.
    formed = aa.reg.MaternKernel(
        coefficient=3.0, scale=2.0, nu=2.0
    ).regularization_matrix_from(linear_obj=mapper)

    inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        regularization_matrix=formed,
        reconstruction=RECONSTRUCTION,
        settings=aa.Settings(regularization_term_method="cho_solve"),
    )

    assert inversion.regularization_term == pytest.approx(
        RECONSTRUCTION @ (formed @ RECONSTRUCTION), rel=1.0e-12
    )


def test__cho_solve_beats_explicit_inverse_on_an_ill_conditioned_covariance():
    """
    The reason the shortcut exists. On clustered vertices `cond(C)` is large and the
    explicitly formed `C^-1` carries round-off amplified by it; the Cholesky solve does
    not.

    Graded against an exactly-known reference: choosing the right-hand side as
    `s = C v` makes `C^-1 s = v`, so the true quadratic form is `s^T v = v^T C v` — a
    well-conditioned expression needing no inverse at all.
    """
    from autoarray.inversion.regularization.matern_kernel import (
        matern_cov_matrix_from,
        inv_via_cholesky,
        quadratic_form_via_cholesky,
    )

    # Deliberately clustered points -> small minimum pairwise separation -> large cond(C),
    # mirroring the traced vertices of the kNN mesh families.
    rng = np.random.default_rng(42)
    points = np.concatenate(
        [
            rng.normal(scale=3.0e-3, size=(24, 2)),
            rng.normal(loc=1.0, scale=3.0e-3, size=(24, 2)),
        ]
    )

    covariance = matern_cov_matrix_from(scale=1.0, nu=2.5, pixel_points=points)

    assert np.linalg.cond(covariance) > 1.0e8

    v = rng.normal(size=points.shape[0])
    s = covariance @ v
    reference = s @ v

    explicit = s @ (inv_via_cholesky(covariance) @ s)
    implicit = quadratic_form_via_cholesky(covariance, s)

    explicit_error = abs(explicit - reference) / abs(reference)
    implicit_error = abs(implicit - reference) / abs(reference)

    assert implicit_error < explicit_error
