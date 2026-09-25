"""
Regression tests for the JAX positive-only (PDIP NNLS) solve on real SLaM MGE systems (PyAutoArray#571).

The fixture `files/mge_slam_nnls_systems.npz` holds 8 `(curvature_reg_matrix, data_vector)` systems captured
from the SLaM `source_lp[1]` MGE model (2 x 20 lens Gaussians with `sigma_min = pixel_scale / 10` plus 20
source Gaussians, 60 linear columns) exactly as the JAX likelihood hands them to
`reconstruction_positive_only_from` (see `files/README.md` for the generator and versions):

- keys 0-4: the Jacobi-preconditioned PDIP solve never converges, even with a 200-iteration cap;
- keys 5-6: it hits the production 50-iteration cap but converges by 200;
- key 7: a healthy system (19 iterations).

Mechanism (issue comment "Step 3 diagnosis"): Jacobi scaling turns the signal-free source-Gaussian columns,
whose diagonal is only the no-regularization floor, into degenerate coordinates on which the PDIP dual
diverges. The fix is the ``preconditioning="raw"`` mode, which the inversion dispatches for mapper-less
inversions: the forward solve runs on the raw system with a data-scaled tolerance. With it, every fixture
system must converge within the production cap and reach the NumPy `fnnls_cholesky` objective
`0.5 x^T Q x - q^T x`, single, end-to-end and under `vmap`, with finite gradients. The Jacobi mode must now
*report* its non-convergence (`stats["converged"] == 0`), and the control test pins today's Jacobi answer
bit-identically on well-conditioned random systems.

Tolerances were declared before the first run: objective within `1e-8 * |obj_fnnls| + 1e-8` (the objectives
are negative, ~ -9.7e5, so the bound is additive rather than the multiplicative `obj * (1 + 1e-8)` of the
issue plan, which would demand PDIP beat fnnls); control solution vs SciPy `nnls` within `1e-8 * max|x|`.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.inversion import inversion_util
from autoarray.util.fnnls import fnnls_cholesky


requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)

FIXTURE = Path(__file__).parent / "files" / "mge_slam_nnls_systems.npz"
PRODUCTION_MAX_ITER = 50
OBJECTIVE_RTOL = 1.0e-8
OBJECTIVE_ATOL = 1.0e-8


def _load_systems():
    with np.load(FIXTURE) as data:
        meta = json.loads(str(data["meta"]))
        systems = [
            (np.asarray(data[f"Q_{s['key']}"]), np.asarray(data[f"q_{s['key']}"]))
            for s in meta["systems"]
        ]
    return meta, systems


META, SYSTEMS = _load_systems()
KEYS = [s["key"] for s in META["systems"]]
IDS = [f"{s['key']}-{s['category']}" for s in META["systems"]]


def _objective(Q, q, x):
    return float(0.5 * x @ Q @ x - q @ x)


def _fnnls(Q, q):
    """The NumPy oracle, started exactly as `reconstruction_positive_only_from` starts it."""
    return np.asarray(
        fnnls_cholesky(Q, q, P_initial=np.linalg.solve(Q, q) > 0), dtype=float
    )


def _jacobi(Q, q):
    """The Jacobi scaling `reconstruction_positive_only_from` applies on the JAX path."""
    D = 1.0 / np.sqrt(np.diag(Q))
    return Q * D[:, None] * D[None, :], q * D, D


def _assert_objective_reaches_fnnls(Q, q, x):
    obj_fnnls = _objective(Q, q, _fnnls(Q, q))
    assert np.all(np.isfinite(x)), "PDIP returned a non-finite solution"
    obj = _objective(Q, q, x)
    assert obj <= obj_fnnls + OBJECTIVE_RTOL * abs(obj_fnnls) + OBJECTIVE_ATOL, (
        obj,
        obj_fnnls,
    )


@pytest.fixture(scope="module")
def jnp():
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jaxnnls.pdip import EPSILON

    # jaxnnls fixes its tolerance scale at import time from the default dtype; a float32-era import would
    # make every fixture "converge" at a 6e-4 KKT tolerance and hide the bug.
    assert EPSILON < 1.0e-10, EPSILON

    return jnp


def test__fixture_is_the_captured_slam_mge_set():
    assert FIXTURE.stat().st_size < 1_000_000
    assert [s["category"] for s in META["systems"]] == 5 * [
        "never_converges_cap200"
    ] + 2 * ["cap50_hit_converges_by_200"] + ["healthy"]
    for Q, q in SYSTEMS:
        assert Q.shape == (60, 60) and q.shape == (60,)
        np.testing.assert_allclose(Q, Q.T, rtol=0, atol=1e-8 * np.abs(Q).max())


def _raw(jnp, Q, q, stats=None, settings=None):
    return inversion_util.reconstruction_positive_only_from(
        data_vector=jnp.asarray(q),
        curvature_reg_matrix=jnp.asarray(Q),
        settings=settings or aa.Settings(),
        xp=jnp,
        stats=stats,
        preconditioning="raw",
    )


@requires_jax
@pytest.mark.parametrize("key", KEYS, ids=IDS)
def test__raw_pdip_converges_within_production_cap(jnp, key):
    """The "raw" solve (un-preconditioned, data-scaled tolerance) converges on every fixture system and reaches
    the fnnls objective to 1e-12 relative (measured <= 4e-13)."""
    from autoarray.util.jax_nnls import data_scaled_solver_tol, solve_nnls

    Q, q = SYSTEMS[key]
    Qj, qj = jnp.asarray(Q), jnp.asarray(q)

    x, _, _, converged, pdip_iter = solve_nnls(
        Qj, qj, solver_tol=data_scaled_solver_tol(qj), max_iter=PRODUCTION_MAX_ITER
    )

    assert int(converged) == 1, f"PDIP did not converge ({int(pdip_iter)} iterations)"
    assert int(pdip_iter) < PRODUCTION_MAX_ITER
    x = np.asarray(x)
    assert np.all(np.isfinite(x))
    obj_fnnls = _objective(Q, q, _fnnls(Q, q))
    assert abs(_objective(Q, q, x) - obj_fnnls) <= 1.0e-12 * abs(obj_fnnls)


@requires_jax
@pytest.mark.parametrize("key", KEYS, ids=IDS)
def test__reconstruction_positive_only_from__jax_matches_numpy_objective(jnp, key):
    Q, q = SYSTEMS[key]
    settings = aa.Settings()
    stats = {}

    x_jax = np.asarray(_raw(jnp, Q, q, stats=stats, settings=settings))
    x_np = inversion_util.reconstruction_positive_only_from(
        data_vector=q, curvature_reg_matrix=Q, settings=settings, xp=np
    )

    assert np.all(np.isfinite(x_jax)), "JAX reconstruction is non-finite"
    obj_jax = _objective(Q, q, x_jax)
    obj_np = _objective(Q, q, np.asarray(x_np))
    assert abs(obj_jax - obj_np) <= OBJECTIVE_RTOL * abs(obj_np), (obj_jax, obj_np)

    assert stats["solver"] == "pdip"
    assert stats["preconditioning"] == "raw"
    assert int(stats["converged"]) == 1
    assert int(stats["iterations"]) < PRODUCTION_MAX_ITER


@requires_jax
def test__jacobi_mode_reports_non_convergence_on_the_witness_systems(jnp):
    """F1: the Jacobi mode still fails on fixture keys 0-6 (the mechanism is unchanged), but the failure is no
    longer silent -- `stats["converged"]` is 0 and the iteration count is the cap. Key 7 converges.
    """
    for key, (Q, q) in enumerate(SYSTEMS):
        stats = {}
        inversion_util.reconstruction_positive_only_from(
            data_vector=jnp.asarray(q),
            curvature_reg_matrix=jnp.asarray(Q),
            settings=aa.Settings(),
            xp=jnp,
            stats=stats,
        )
        assert stats["preconditioning"] == "jacobi"
        expected = 1 if META["systems"][key]["category"] == "healthy" else 0
        assert int(stats["converged"]) == expected, key
        if expected == 0:
            assert int(stats["iterations"]) == PRODUCTION_MAX_ITER


@requires_jax
def test__raw_pdip_converges_under_vmap(jnp):
    import jax

    def solve(Q, q):
        stats = {}
        x = inversion_util.reconstruction_positive_only_from(
            data_vector=q,
            curvature_reg_matrix=Q,
            settings=aa.Settings(),
            xp=jnp,
            stats=stats,
            preconditioning="raw",
        )
        return x, stats["converged"], stats["iterations"]

    Qs = jnp.asarray(np.stack([Q for Q, _ in SYSTEMS]))
    qs = jnp.asarray(np.stack([q for _, q in SYSTEMS]))

    x, converged, iterations = jax.jit(jax.vmap(solve))(Qs, qs)

    converged = np.asarray(converged)
    iterations = np.asarray(iterations)
    failed = [
        k for k in KEYS if converged[k] != 1 or iterations[k] >= PRODUCTION_MAX_ITER
    ]
    assert not failed, (
        f"failed lanes {failed}; converged {converged.tolist()}; "
        f"iterations {iterations.tolist()}"
    )
    for k, (Q, q) in enumerate(SYSTEMS):
        _assert_objective_reaches_fnnls(Q, q, np.asarray(x[k]))


@requires_jax
@pytest.mark.parametrize("key", KEYS, ids=IDS)
def test__raw_mode_gradient_is_finite_and_non_zero(jnp, key):
    """The raw mode keeps the Jacobi-space relaxed-KKT backward pass: a relaxed-KKT pass on the raw Q gives NaN
    gradients on keys 2-4, and the Jacobi-mode gradient itself is NaN on key 1 (its forward solve diverged).
    """
    import jax

    Q, q = SYSTEMS[key]
    w = jnp.linspace(0.5, 1.5, q.shape[0])

    gQ, gq = jax.grad(lambda Q_, q_: w @ _raw(jnp, Q_, q_), argnums=(0, 1))(
        jnp.asarray(Q), jnp.asarray(q)
    )

    assert np.all(np.isfinite(np.asarray(gQ))) and np.all(np.isfinite(np.asarray(gq)))
    assert np.any(np.asarray(gq) != 0.0)


@requires_jax
def test__raw_mode_gradient_matches_jacobi_mode_on_the_healthy_system(jnp):
    """Where the Jacobi forward solve converges (key 7) the two modes differentiate the same relaxed system from
    nearly the same iterate: measured max relative difference 2.8e-3."""
    import jax

    key = [s["key"] for s in META["systems"] if s["category"] == "healthy"][0]
    Q, q = SYSTEMS[key]
    w = jnp.linspace(0.5, 1.5, q.shape[0])

    def jacobi(Q_, q_):
        return w @ inversion_util.reconstruction_positive_only_from(
            data_vector=q_, curvature_reg_matrix=Q_, settings=aa.Settings(), xp=jnp
        )

    g_raw = np.asarray(jax.grad(lambda q_: w @ _raw(jnp, Q, q_))(jnp.asarray(q)))
    g_jac = np.asarray(jax.grad(lambda q_: jacobi(jnp.asarray(Q), q_))(jnp.asarray(q)))

    assert np.abs(g_raw - g_jac).max() <= 1.0e-2 * np.abs(g_jac).max()


def _random_system(n, seed):
    """A seeded, well-conditioned NNLS problem (as `test_jax_active_set._qp`): roughly half the unconstrained
    solution is negative, so positivity binds."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3 * n, n))
    b = rng.normal(size=3 * n)
    return A, b, A.T @ A, A.T @ b


@requires_jax
@pytest.mark.parametrize("n, seed", [(12, 0), (30, 1), (60, 2)])
def test__control__well_conditioned_pdip_unchanged(jnp, n, seed):
    """
    Must pass before and after any fix, bit-identically: the library JAX path is compared against the
    upstream jaxnnls solve of the same Jacobi-scaled system, recomputed in-test (today's answer).
    """
    import jaxnnls
    from jaxnnls.pdip import solve_nnls as upstream_solve_nnls
    from scipy.optimize import nnls

    A, b, Q, q = _random_system(n, seed)
    Q_pc, q_pc, D = _jacobi(Q, q)

    _, _, _, converged, pdip_iter = upstream_solve_nnls(
        jnp.asarray(Q_pc), jnp.asarray(q_pc)
    )
    assert int(converged) == 1 and int(pdip_iter) < PRODUCTION_MAX_ITER

    expected = (
        np.asarray(
            jaxnnls.solve_nnls_primal(
                jnp.asarray(Q_pc), jnp.asarray(q_pc), target_kappa=1.0e-11
            )
        )
        * D
    )

    x = np.asarray(
        inversion_util.reconstruction_positive_only_from(
            data_vector=jnp.asarray(q),
            curvature_reg_matrix=jnp.asarray(Q),
            settings=aa.Settings(),
            xp=jnp,
        )
    )

    np.testing.assert_array_equal(x, expected)

    x_scipy, _ = nnls(A, b)
    np.testing.assert_allclose(x, x_scipy, rtol=0, atol=1e-8 * np.abs(x_scipy).max())

    stats = {}
    x_raw = np.asarray(_raw(jnp, Q, q, stats=stats))
    assert int(stats["converged"]) == 1
    np.testing.assert_allclose(
        x_raw, x_scipy, rtol=0, atol=1e-8 * np.abs(x_scipy).max()
    )


# ---------------------------------------------------------------------------------------------------------------
# PyAutoArray#573: NaN gradients of the "raw" mode.
#
# `files/mge_grad_nan_systems.npz` holds 4 (20 x 20) systems captured from the autolens_workspace_test
# `jax_grad/mge.py` model (MGE source, NFWSph + shear) at the PRNGKey perturbations 2, 10, 12 and 14 (see
# `files/README.md`). The raw forward solve stops at the data-scaled tolerance with s * z ~ 1e-10 .. 2.5e-9, far
# above `nnls_target_kappa = 1e-11`; the relaxed-KKT solve on the Jacobi system then has to push toward the
# boundary from z / s ~ 1e13 and hits its 50-iteration cap with NaN (or "converges" with s < 0), so the gradient
# is NaN. The fix polishes the mapped iterate with a few tight PDIP iterations on the Jacobi system first.
# ---------------------------------------------------------------------------------------------------------------

GRAD_NAN_FIXTURE = Path(__file__).parent / "files" / "mge_grad_nan_systems.npz"


def _load_grad_nan_systems():
    with np.load(GRAD_NAN_FIXTURE) as data:
        meta = json.loads(str(data["meta"]))
        systems = [
            (np.asarray(data[f"Q_{s['key']}"]), np.asarray(data[f"q_{s['key']}"]))
            for s in meta["systems"]
        ]
    return meta, systems


GRAD_NAN_META, GRAD_NAN_SYSTEMS = _load_grad_nan_systems()
GRAD_NAN_IDS = [f"prng{s['prng_key']}" for s in GRAD_NAN_META["systems"]]


def test__grad_nan_fixture_is_the_captured_jax_grad_mge_set():
    assert GRAD_NAN_FIXTURE.stat().st_size < 50_000
    assert [s["prng_key"] for s in GRAD_NAN_META["systems"]] == [2, 10, 12, 14]
    for Q, q in GRAD_NAN_SYSTEMS:
        assert Q.shape == (20, 20) and q.shape == (20,)


@requires_jax
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("index", range(len(GRAD_NAN_SYSTEMS)), ids=GRAD_NAN_IDS)
def test__raw_mode_gradient_is_finite_on_the_captured_grad_nan_systems(jnp, index, jit):
    """Red on PyAutoArray 3de624b5 (#572): the gradient is NaN on all four systems eagerly, and under jit on
    prng10 / prng14 (the jitted NaN is rounding-sensitive; prng2 / prng12 happen to pass jitted on main).
    """
    import jax

    Q, q = GRAD_NAN_SYSTEMS[index]
    w = jnp.linspace(0.5, 1.5, q.shape[0])

    grad = jax.grad(lambda Q_, q_: w @ _raw(jnp, Q_, q_), argnums=(0, 1))
    if jit:
        grad = jax.jit(grad)
    gQ, gq = grad(jnp.asarray(Q), jnp.asarray(q))

    assert np.all(np.isfinite(np.asarray(gQ))) and np.all(np.isfinite(np.asarray(gq)))
    assert np.any(np.asarray(gq) != 0.0)


def _backward_status(jnp, Q, q):
    from autoarray.util.jax_nnls import raw_forward_backward_status

    Qj, qj = jnp.asarray(Q), jnp.asarray(q)
    Q_pc, q_pc, D = (jnp.asarray(a) for a in _jacobi(Q, q))
    return [
        int(v)
        for v in raw_forward_backward_status(
            Q_pc, q_pc, Qj, qj, D, target_kappa=1.0e-11, max_iter=PRODUCTION_MAX_ITER
        )
    ]


@requires_jax
@pytest.mark.parametrize(
    "system",
    [("slam", k) for k in KEYS]
    + [("grad_nan", i) for i in range(len(GRAD_NAN_SYSTEMS))],
    ids=[f"slam-{i}" for i in IDS] + [f"grad_nan-{i}" for i in GRAD_NAN_IDS],
)
def test__raw_mode_backward_pass_converges(jnp, system):
    """The backward pass reports convergence: the tight polish of the mapped iterate converges (measured <= 6
    iterations) and the relaxed-KKT solve then converges well inside its 50-iteration cap (measured 1).
    Not a red-on-main witness (``raw_forward_backward_status`` is new with #573); the gradient test is.
    """
    kind, index = system
    Q, q = (SYSTEMS if kind == "slam" else GRAD_NAN_SYSTEMS)[index]

    relaxed_converged, relaxed_iter, polish_converged, polish_iter = _backward_status(
        jnp, Q, q
    )

    assert polish_converged == 1, polish_iter
    assert relaxed_converged == 1 and relaxed_iter < PRODUCTION_MAX_ITER, relaxed_iter
