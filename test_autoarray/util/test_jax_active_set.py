"""
Tests of the certified active-set positive-only solver, `autoarray.util.jax_active_set`.

Tolerances were declared before the tests were first run (PyAutoArray#566):

- solution vs SciPy `nnls` (an independent active-set NNLS on the least-squares form): `atol 1e-10`;
- solution vs the library's PDIP solve (`solve_nnls_primal`): `rtol 1e-8` (amended after the first run: plus an
  absolute floor of `1e-8 * max|x|`, because PDIP never reaches the active set's exact zeros);
- gradient vs central finite differences: `rtol 1e-6` (the NNLS solution is piecewise linear in `q`, so a
  central difference inside one active-set piece is exact up to rounding);
- gradient vs `jax.grad` through PDIP's relaxed-KKT `custom_vjp`: `rtol 1e-4` (PDIP differentiates a relaxed
  central-path system, an approximation to the exact active-set derivative; amended after the first run: plus
  an absolute floor of `1e-4 * max|grad|`, because PDIP leaks ~1e-7 onto active coordinates whose exact
  derivative is zero).
"""

import importlib

import numpy as np
import pytest


# jax is an `[optional]` extra and is absent on the NumPy-only matrix env: every test in this module skips
# there. The no-module-level-jax-import guard lives in `test_jax_active_set_import.py` so it still runs.
if importlib.util.find_spec("jax") is None:
    pytestmark = pytest.mark.skip(reason="requires jax (the [optional] extras)")

    def test__placeholder_requires_jax():  # pragma: no cover
        pass

else:
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from scipy.optimize import nnls

    from autoarray.util import jax_active_set
    from autoarray.util.jax_nnls import solve_nnls_primal


TARGET_KAPPA = 1.0e-11


def _qp(n, seed, rows_factor=3):
    """
    A seeded NNLS problem in least-squares form (A, b) and its quadratic-program form (Q, q) = (A^T A, A^T b).

    A Gaussian `A` with a Gaussian `b` gives an unconstrained solution with roughly half its entries negative.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(rows_factor * n, n))
    b = rng.normal(size=rows_factor * n)
    return A, b, A.T @ A, A.T @ b


def _pdip(Q, q):
    return solve_nnls_primal(jnp.asarray(Q), jnp.asarray(q), target_kappa=TARGET_KAPPA)


def _reference_passes(Q, q, permanent=None, max_passes=40, tau_rel=1.0e-9):
    """
    NumPy reference of the free-all certified active-set scheme (a port of the autolens_profiling harness's
    `active_set_certified`), returning the number of restricted passes to certification (0 when the
    unconstrained solve is already non-negative) and the solution.
    """
    n = Q.shape[0]
    permanent = np.zeros(n, bool) if permanent is None else np.asarray(permanent)

    def restricted(fixed):
        x = np.zeros(n)
        free = ~fixed
        if free.any():
            idx = np.where(free)[0]
            x[idx] = np.linalg.solve(Q[np.ix_(idx, idx)], q[idx])
        return x

    x = restricted(permanent)
    if not (x < 0.0).any():
        return 0, x

    fixed = permanent | (x < 0.0)
    tau_g = tau_rel * np.max(np.abs(q))

    for p in range(1, max_passes + 1):
        x = restricted(fixed)
        g = Q @ x - q
        tau_x = tau_rel * np.max(np.abs(x))
        primal = ~fixed & (x < -tau_x)
        dual = fixed & ~permanent & (g < -tau_g)
        if not primal.any() and not dual.any():
            return p, x
        fixed = (fixed | primal) & ~dual

    return None, x


@pytest.mark.parametrize("n, seed", [(20, 0), (20, 1), (60, 2), (60, 3)])
def test__solve_certified__matches_scipy_nnls_and_pdip(n, seed):
    A, b, Q, q = _qp(n, seed)

    # The fixture must exercise the constraint: the unconstrained solution has negative entries.
    assert (np.linalg.solve(Q, q) < 0.0).any()

    x, certified, passes = jax_active_set.solve_certified(
        jnp.asarray(Q), jnp.asarray(q)
    )
    x = np.asarray(x)

    assert bool(certified)
    assert x.dtype == np.float64
    assert (x >= 0.0).all()

    x_scipy, _ = nnls(A, b)
    assert np.max(np.abs(x - x_scipy)) < 1.0e-10

    # Exactly zero (not merely small) on the active set.
    assert (x[x_scipy == 0.0] == 0.0).all()

    # PDIP is an interior-point method: it approaches the active set's zeros but never reaches them exactly
    # (measured up to 7.6e-12 on an active entry), so the absolute floor is relative to the solution's scale.
    x_pdip = np.asarray(_pdip(Q, q))
    assert x == pytest.approx(x_pdip, rel=1.0e-8, abs=1.0e-8 * np.max(np.abs(x)))


@pytest.mark.parametrize("n, seed", [(20, 0), (20, 1), (60, 2), (60, 3)])
def test__solve_certified__exits_early_at_the_reference_pass_count(n, seed):
    _, _, Q, q = _qp(n, seed)

    budget = 16
    _, certified, passes = jax_active_set.solve_certified(
        jnp.asarray(Q), jnp.asarray(q), pass_budget=budget
    )

    reference_passes, _ = _reference_passes(Q, q)

    assert bool(certified)
    assert int(passes) == reference_passes
    assert int(passes) < budget


def test__solve_certified__non_negative_unconstrained_solution_needs_no_pass():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    x_true = np.array([1.0, 2.0])
    q = Q @ x_true

    x, certified, passes = jax_active_set.solve_certified(
        jnp.asarray(Q), jnp.asarray(q)
    )

    assert bool(certified)
    assert int(passes) == 0
    assert np.asarray(x) == pytest.approx(x_true, rel=1.0e-12)


def test__solve_certified__permanent_fixed_set_is_never_released():
    A, b, Q, q = _qp(20, 4)

    # Hold at zero indices the free problem would make strictly positive, so releasing them would lower the
    # objective (a negative gradient) -- the scheme must hold them anyway.
    x_free, _ = nnls(A, b)
    positive = np.where(x_free > 0.0)[0]
    permanent = np.zeros(20, bool)
    permanent[positive[:3]] = True

    x, certified, _ = jax_active_set.solve_certified(
        jnp.asarray(Q), jnp.asarray(q), permanent=jnp.asarray(permanent)
    )
    x = np.asarray(x)

    assert bool(certified)
    assert (x[permanent] == 0.0).all()

    # It is the NNLS solution of the problem with those columns removed.
    keep = ~permanent
    x_reduced, _ = nnls(A[:, keep], b)
    assert np.max(np.abs(x[keep] - x_reduced)) < 1.0e-10

    # And the released-if-it-were-allowed indices really would want to move.
    g = Q @ x - q
    assert (g[permanent] < 0.0).any()


def _hard_system():
    """A system the default search certifies only after more than one restricted pass."""
    _, _, Q, q = _qp(60, 2)
    reference_passes, _ = _reference_passes(Q, q)
    assert reference_passes >= 2
    return jnp.asarray(Q), jnp.asarray(q)


def test__solve_certified_with_fallback__exhausted_budget_returns_pdip_bit_exactly():
    Q, q = _hard_system()

    def pdip_fn():
        return solve_nnls_primal(Q, q, target_kappa=TARGET_KAPPA)

    x, certified, passes = jax_active_set.solve_certified_with_fallback(
        Q, q, pdip_fn=pdip_fn, fallback=True, pass_budget=1
    )

    assert not bool(certified)
    assert int(passes) == 1
    assert np.array_equal(np.asarray(x), np.asarray(pdip_fn()))


def test__solve_certified_with_fallback__no_fallback_flags_the_uncertified_iterate():
    Q, q = _hard_system()

    def pdip_fn():  # pragma: no cover - must not be called
        raise AssertionError("the fallback must not run with fallback=False")

    x, certified, passes = jax_active_set.solve_certified_with_fallback(
        Q, q, pdip_fn=pdip_fn, fallback=False, pass_budget=1
    )

    assert not bool(certified)
    assert int(passes) == 1
    assert np.isfinite(np.asarray(x)).all()


def test__solve_certified_with_fallback__certified_returns_the_active_set_solution():
    Q, q = _hard_system()

    def pdip_fn():
        return solve_nnls_primal(Q, q, target_kappa=TARGET_KAPPA)

    x, certified, _ = jax_active_set.solve_certified_with_fallback(
        Q, q, pdip_fn=pdip_fn, fallback=True
    )
    x_direct, _, _ = jax_active_set.solve_certified(Q, q)

    assert bool(certified)
    assert np.array_equal(np.asarray(x), np.asarray(x_direct))


def test__solve_certified__jit_and_vmap_match_per_system_results():
    systems = [_qp(20, seed) for seed in range(4)]
    Qs = jnp.asarray(np.stack([s[2] for s in systems]))
    qs = jnp.asarray(np.stack([s[3] for s in systems]))

    scalar = [jax_active_set.solve_certified(Qs[i], qs[i]) for i in range(4)]

    jitted = jax.jit(jax_active_set.solve_certified)
    for i in range(4):
        x, certified, passes = jitted(Qs[i], qs[i])
        assert np.asarray(x) == pytest.approx(np.asarray(scalar[i][0]), abs=1.0e-12)
        assert bool(certified) == bool(scalar[i][1])
        assert int(passes) == int(scalar[i][2])

    x_b, certified_b, passes_b = jax.jit(jax.vmap(jax_active_set.solve_certified))(
        Qs, qs
    )

    for i in range(4):
        assert np.asarray(x_b[i]) == pytest.approx(
            np.asarray(scalar[i][0]), abs=1.0e-12
        )
        assert bool(certified_b[i]) == bool(scalar[i][1])
        # Per-lane pass counts survive the batched while_loop.
        assert int(passes_b[i]) == int(scalar[i][2])


def test__solve_certified_with_fallback__vmap():
    systems = [_qp(20, seed) for seed in range(4)]
    Qs = jnp.asarray(np.stack([s[2] for s in systems]))
    qs = jnp.asarray(np.stack([s[3] for s in systems]))

    def solve(Q, q):
        return jax_active_set.solve_certified_with_fallback(
            Q,
            q,
            pdip_fn=lambda: solve_nnls_primal(Q, q, target_kappa=TARGET_KAPPA),
            fallback=True,
        )[0]

    x_b = jax.jit(jax.vmap(solve))(Qs, qs)

    for i in range(4):
        x_scipy, _ = nnls(systems[i][0], systems[i][1])
        assert np.max(np.abs(np.asarray(x_b[i]) - x_scipy)) < 1.0e-10


def test__solve_certified__gradient_matches_finite_differences_and_pdip():
    _, _, Q, q = _qp(20, 5)
    rng = np.random.default_rng(11)
    w = jnp.asarray(rng.normal(size=20))
    Q = jnp.asarray(Q)
    q = jnp.asarray(q)

    def objective(q_):
        return w @ jax_active_set.solve_certified(Q, q_)[0]

    grad = np.asarray(jax.grad(objective)(q))
    assert np.isfinite(grad).all()

    # The derivative is exactly zero along the active set's coordinates of q... and non-trivial elsewhere.
    x = np.asarray(jax_active_set.solve_certified(Q, q)[0])
    assert (grad[x == 0.0] == 0.0).all()
    assert np.abs(grad[x > 0.0]).max() > 0.0

    h = 1.0e-6 * float(jnp.max(jnp.abs(q)))
    fd = np.zeros(20)
    for i in range(20):
        e = jnp.zeros(20).at[i].set(h)
        fd[i] = (float(objective(q + e)) - float(objective(q - e))) / (2.0 * h)

    assert grad == pytest.approx(fd, rel=1.0e-6, abs=1.0e-12)

    grad_pdip = np.asarray(
        jax.grad(lambda q_: w @ solve_nnls_primal(Q, q_, target_kappa=TARGET_KAPPA))(q)
    )
    # PDIP's relaxed-KKT gradient leaks a small value onto active coordinates, where the exact derivative is
    # zero (measured 1.8e-7 against max|grad| ~ 1e-2), so the absolute floor is relative to the gradient's scale.
    assert grad == pytest.approx(
        grad_pdip, rel=1.0e-4, abs=1.0e-4 * np.max(np.abs(grad))
    )


def test__solve_certified__gradient_wrt_matrix_matches_finite_differences():
    _, _, Q, q = _qp(20, 6)
    rng = np.random.default_rng(12)
    w = jnp.asarray(rng.normal(size=20))
    direction = rng.normal(size=(20, 20))
    direction = jnp.asarray(direction + direction.T)
    Q = jnp.asarray(Q)
    q = jnp.asarray(q)

    def objective(t):
        return w @ jax_active_set.solve_certified(Q + t * direction, q)[0]

    derivative = float(jax.grad(objective)(0.0))

    h = 1.0e-6
    fd = (float(objective(h)) - float(objective(-h))) / (2.0 * h)

    assert derivative == pytest.approx(fd, rel=1.0e-6)
