import pytest

import autoarray as aa
import numpy as np

from autoarray.inversion.regularization.adapt import adapt_regularization_weights_from
from autoarray.inversion.regularization.matern_kernel import (
    apply_jitter,
    matern_cov_matrix_from,
)


def _mapper():
    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )
    return aa.m.MockMapper(source_plane_mesh_grid=source_plane_mesh_grid)


def _adapt_mapper():
    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )
    return aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid,
        pixel_signals=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    )


@pytest.mark.parametrize(
    "reg_absolute, reg_relative",
    [
        (
            aa.reg.MaternKernel(coefficient=3.0, scale=2.0, nu=2.0),
            aa.reg.MaternKernel(
                coefficient=3.0, scale=2.0, nu=2.0, jitter_relative=True
            ),
        ),
        (
            aa.reg.ExponentialKernel(coefficient=3.0, scale=2.0),
            aa.reg.ExponentialKernel(coefficient=3.0, scale=2.0, jitter_relative=True),
        ),
        (
            aa.reg.GaussianKernel(coefficient=3.0, scale=0.5),
            aa.reg.GaussianKernel(coefficient=3.0, scale=0.5, jitter_relative=True),
        ),
    ],
)
def test__unweighted_kernels__relative_and_absolute_jitter_agree(
    reg_absolute, reg_relative
):
    """
    The unweighted kernels have ``K(0) == 1``, so their covariance diagonal is ~1 and the
    absolute and relative conventions coincide. This pins that switching the flag on is
    not a silent science change for these three schemes.
    """
    mapper = _mapper()

    absolute = reg_absolute.regularization_matrix_from(linear_obj=mapper)
    relative = reg_relative.regularization_matrix_from(linear_obj=mapper)

    assert relative == pytest.approx(absolute, rel=1.0e-7)


def test__default_is_byte_identical_to_absolute_jitter():
    """
    `jitter_relative` defaults to False on every scheme, so default-constructed
    regularization matrices must be bit-for-bit what they were before the flag existed.
    """
    mapper = _mapper()
    adapt_mapper = _adapt_mapper()

    for reg, linear_obj in [
        (aa.reg.MaternKernel(coefficient=3.0, scale=2.0, nu=2.0), mapper),
        (aa.reg.ExponentialKernel(coefficient=3.0, scale=2.0), mapper),
        (aa.reg.GaussianKernel(coefficient=3.0, scale=0.5), mapper),
        (
            aa.reg.MaternAdaptKernel(
                scale=2.0, nu=2.0, inner_coefficient=0.5, outer_coefficient=4.0
            ),
            adapt_mapper,
        ),
    ]:
        assert reg.jitter_relative is False

        covariance = matern_cov_matrix_from(
            scale=2.0,
            nu=2.0,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            jitter=0.0,
        )
        pixels = covariance.shape[0]

        assert np.array_equal(
            apply_jitter(covariance, jitter=1e-8, jitter_relative=False),
            covariance + 1e-8 * np.eye(pixels),
        )


def test__apply_jitter_relative__is_a_pure_rescaling_of_the_diagonal():
    covariance = matern_cov_matrix_from(
        scale=1.0,
        nu=2.0,
        pixel_points=np.random.default_rng(0).normal(size=(12, 2)),
        jitter=0.0,
        weights=np.linspace(0.1, 5.0, 12),
    )

    jittered = apply_jitter(covariance, jitter=1e-3, jitter_relative=True)

    # Diagonal scaled by exactly (1 + jitter); off-diagonal untouched.
    assert np.diag(jittered) == pytest.approx(np.diag(covariance) * (1.0 + 1e-3))

    off_diagonal = ~np.eye(covariance.shape[0], dtype=bool)
    assert np.array_equal(jittered[off_diagonal], covariance[off_diagonal])


def test__adaptive_weights__absolute_jitter_swamps_faint_pixels__relative_does_not():
    """
    The bug 2b fixes. `MaternAdaptKernel`'s covariance diagonal is `C_ii = w_i^2`, which
    spans the adaptive-weight dynamic range. At wide inner/outer coefficients the fixed
    absolute 1e-8 reaches 100% of the faintest pixel's variance and destroys its kernel
    structure; the relative convention perturbs every pixel by 1e-8 whatever its scale.
    """
    points = np.random.default_rng(0).normal(size=(40, 2))
    pixel_signals = np.linspace(0.0, 1.0, 40)

    kernel_weights = 1.0 / adapt_regularization_weights_from(
        inner_coefficient=0.1, outer_coefficient=100.0, pixel_signals=pixel_signals
    )

    covariance = matern_cov_matrix_from(
        scale=1.0, nu=2.0, pixel_points=points, weights=kernel_weights, jitter=0.0
    )

    faintest_variance = np.diag(covariance).min()

    # The regime the absolute convention breaks in: the faintest pixel's own variance has
    # fallen to the jitter's magnitude.
    assert faintest_variance == pytest.approx(1.0e-8, rel=0.5)

    absolute = apply_jitter(covariance, jitter=1e-8, jitter_relative=False)
    relative = apply_jitter(covariance, jitter=1e-8, jitter_relative=True)

    absolute_distortion = (
        np.diag(absolute) - np.diag(covariance)
    ).max() / faintest_variance
    relative_distortion = np.max(
        (np.diag(relative) - np.diag(covariance)) / np.diag(covariance)
    )

    # Absolute: order-unity corruption of the faint pixel. Relative: 1e-8, as intended.
    assert absolute_distortion > 0.5
    assert relative_distortion == pytest.approx(1.0e-8, rel=1.0e-6)


def test__relative_jitter__preserves_conditioning_on_clustered_vertices():
    """
    The jitter is load-bearing for the Cholesky on clustered (traced) mesh vertices. The
    relative convention must not weaken that protection — on an unweighted kernel, where
    the diagonal is ~1, the two conventions must give the same conditioning.
    """
    rng = np.random.default_rng(42)
    points = np.concatenate(
        [
            rng.normal(scale=3.0e-3, size=(24, 2)),
            rng.normal(loc=1.0, scale=3.0e-3, size=(24, 2)),
        ]
    )

    covariance = matern_cov_matrix_from(
        scale=1.0, nu=2.5, pixel_points=points, jitter=0.0
    )

    absolute = apply_jitter(covariance, jitter=1e-8, jitter_relative=False)
    relative = apply_jitter(covariance, jitter=1e-8, jitter_relative=True)

    # Both factorize, and the conditioning is materially the same.
    np.linalg.cholesky(absolute)
    np.linalg.cholesky(relative)

    assert np.linalg.cond(relative) == pytest.approx(
        np.linalg.cond(absolute), rel=1.0e-3
    )


def test__adapt_kernel_end_to_end__flag_reaches_the_regularization_matrix():
    """
    The flag must be threaded to every covariance call site, not just the constructor.
    """
    mapper = _adapt_mapper()

    absolute = aa.reg.MaternAdaptKernel(
        scale=2.0, nu=2.0, inner_coefficient=0.1, outer_coefficient=100.0
    )
    relative = aa.reg.MaternAdaptKernel(
        scale=2.0,
        nu=2.0,
        inner_coefficient=0.1,
        outer_coefficient=100.0,
        jitter_relative=True,
    )

    assert not np.array_equal(
        absolute.regularization_matrix_from(linear_obj=mapper),
        relative.regularization_matrix_from(linear_obj=mapper),
    )

    # The log-det and term shortcuts must see the same covariance the matrix did.
    assert absolute.log_det_regularization_matrix_term_from(
        linear_obj=mapper
    ) != relative.log_det_regularization_matrix_term_from(linear_obj=mapper)

    reconstruction = np.array([0.3, -1.2, 0.7, 2.1, -0.4, 1.5])
    assert absolute.regularization_term_from(
        linear_obj=mapper, reconstruction=reconstruction
    ) != relative.regularization_term_from(
        linear_obj=mapper, reconstruction=reconstruction
    )
