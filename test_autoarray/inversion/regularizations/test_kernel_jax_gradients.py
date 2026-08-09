"""
JAX leg of the gate for the kernel-scheme linear-algebra work: the
`quadratic_form_via_cholesky` term shortcut and the `apply_jitter` conventions must be
`jit`-safe, differentiable, and agree with the NumPy path.

These mirror, at library level, what
`autolens_workspace_test/scripts/imaging/jax_grad/regularization.py` certifies at
workspace level. Skipped when JAX is absent (it is an optional dependency).

The Matern kernel is deliberately not covered here: its JAX path needs the modified
Bessel `K_nu` from `tfp-nightly`, which is an optional-of-an-optional. The Gaussian and
Exponential kernels exercise the same shared code (`apply_jitter`,
`quadratic_form_via_cholesky`) without it.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from autoarray.inversion.regularization.matern_kernel import (  # noqa: E402
    apply_jitter,
    inv_via_cholesky,
    quadratic_form_via_cholesky,
)
from autoarray.inversion.regularization.gaussian_kernel import (  # noqa: E402
    gauss_cov_matrix_from,
)
from autoarray.inversion.regularization.exponential_kernel import (  # noqa: E402
    exp_cov_matrix_from,
)

POINTS = np.random.default_rng(0).normal(size=(10, 2))
VECTOR = np.random.default_rng(1).normal(size=10)


@pytest.mark.parametrize("jitter_relative", [False, True])
@pytest.mark.parametrize("cov_from", [gauss_cov_matrix_from, exp_cov_matrix_from])
def test__apply_jitter__is_jit_safe_in_both_conventions(jitter_relative, cov_from):
    """
    `jitter_relative` is a static Python bool, so the branch inside `apply_jitter` must
    resolve at trace time rather than on a tracer.
    """

    @jax.jit
    def build(points):
        covariance = cov_from(scale=1.0, pixel_points=points, jitter=0.0, xp=jnp)
        return apply_jitter(
            covariance, jitter=1e-8, jitter_relative=jitter_relative, xp=jnp
        )

    covariance = build(jnp.asarray(POINTS))

    assert covariance.shape == (POINTS.shape[0], POINTS.shape[0])
    assert bool(jnp.all(jnp.isfinite(covariance)))


@pytest.mark.parametrize("cov_from", [gauss_cov_matrix_from, exp_cov_matrix_from])
def test__quadratic_form_via_cholesky__gradient_is_finite_difference_certified(
    cov_from,
):
    """
    The certification the workspace `jax_grad/regularization.py` script exists to give:
    autodiff through the Cholesky solve must match a central finite difference.
    """

    def term(scale):
        covariance = cov_from(
            scale=scale, pixel_points=jnp.asarray(POINTS), jitter=1e-8, xp=jnp
        )
        return quadratic_form_via_cholesky(covariance, jnp.asarray(VECTOR), xp=jnp)

    autodiff = float(jax.grad(term)(1.3))

    step = 1.0e-6
    finite_difference = float((term(1.3 + step) - term(1.3 - step)) / (2.0 * step))

    assert autodiff == pytest.approx(finite_difference, rel=1.0e-6)


@pytest.mark.parametrize("cov_from", [gauss_cov_matrix_from, exp_cov_matrix_from])
def test__quadratic_form_via_cholesky__eager_and_jit_agree(cov_from):
    def term(scale):
        covariance = cov_from(
            scale=scale, pixel_points=jnp.asarray(POINTS), jitter=1e-8, xp=jnp
        )
        return quadratic_form_via_cholesky(covariance, jnp.asarray(VECTOR), xp=jnp)

    assert float(jax.jit(term)(1.3)) == pytest.approx(float(term(1.3)), rel=1.0e-12)


@pytest.mark.parametrize("cov_from", [gauss_cov_matrix_from, exp_cov_matrix_from])
def test__quadratic_form_via_cholesky__matches_explicit_inverse_and_numpy(cov_from):
    """
    The shortcut must be the same quantity as the explicit-inverse contraction it
    replaces, on both backends.
    """
    covariance_jax = cov_from(
        scale=1.3, pixel_points=jnp.asarray(POINTS), jitter=1e-8, xp=jnp
    )
    covariance_numpy = cov_from(scale=1.3, pixel_points=POINTS, jitter=1e-8)

    implicit = float(
        quadratic_form_via_cholesky(covariance_jax, jnp.asarray(VECTOR), xp=jnp)
    )
    explicit = float(
        jnp.asarray(VECTOR)
        @ (inv_via_cholesky(covariance_jax, xp=jnp) @ jnp.asarray(VECTOR))
    )
    numpy_implicit = float(quadratic_form_via_cholesky(covariance_numpy, VECTOR))

    assert implicit == pytest.approx(explicit, rel=1.0e-9)
    assert implicit == pytest.approx(numpy_implicit, rel=1.0e-9)


def test__regularization_term_from__is_differentiable_end_to_end_under_jax():
    """
    The scheme-level hook, not just the helper: gradients must reach the mesh grid the
    term is built from.
    """
    import autoarray as aa

    regularization = aa.reg.GaussianKernel(coefficient=3.0, scale=1.0)

    class _Obj:
        def __init__(self, array):
            self.source_plane_mesh_grid = type("_G", (), {"array": array})()

    def term(points):
        return regularization.regularization_term_from(
            linear_obj=_Obj(points), reconstruction=jnp.asarray(VECTOR), xp=jnp
        )

    value = float(term(jnp.asarray(POINTS)))
    gradient = jax.grad(term)(jnp.asarray(POINTS))

    assert np.isfinite(value)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


@pytest.mark.parametrize("jitter_relative", [False, True])
def test__relative_jitter__does_not_break_gradients(jitter_relative):
    def term(scale):
        covariance = gauss_cov_matrix_from(
            scale=scale,
            pixel_points=jnp.asarray(POINTS),
            jitter=1e-8,
            jitter_relative=jitter_relative,
            xp=jnp,
        )
        return quadratic_form_via_cholesky(covariance, jnp.asarray(VECTOR), xp=jnp)

    autodiff = float(jax.grad(term)(1.3))

    step = 1.0e-6
    finite_difference = float((term(1.3 + step) - term(1.3 - step)) / (2.0 * step))

    assert np.isfinite(autodiff)
    assert autodiff == pytest.approx(finite_difference, rel=1.0e-6)
