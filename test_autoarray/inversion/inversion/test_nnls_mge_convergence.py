"""
Regression tests for the JAX positive-only (PDIP NNLS) solve on real SLaM MGE systems (PyAutoArray#571).

The fixture `files/mge_slam_nnls_systems.npz` holds 8 `(curvature_reg_matrix, data_vector)` systems captured
from the SLaM `source_lp[1]` MGE model (2 x 20 lens Gaussians with `sigma_min = pixel_scale / 10` plus 20
source Gaussians, 60 linear columns) exactly as the JAX likelihood hands them to
`reconstruction_positive_only_from` (see `files/README.md` for the generator and versions):

- keys 0-4: the Jacobi-preconditioned PDIP solve never converges, even with a 200-iteration cap;
- keys 5-6: it hits the production 50-iteration cap but converges by 200;
- key 7: a healthy system (19 iterations).

Every fixture system must converge within the production cap and reach the NumPy `fnnls_cholesky` objective
`0.5 x^T Q x - q^T x`, single and under `vmap`. The control test pins today's PDIP answer on well-conditioned
random systems so a fix cannot move converging solves.

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


@requires_jax
@pytest.mark.parametrize("key", KEYS, ids=IDS)
def test__pdip_jacobi_converges_within_production_cap(jnp, key):
    from autoarray.util.jax_nnls import solve_nnls

    Q, q = SYSTEMS[key]
    Q_pc, q_pc, D = _jacobi(Q, q)

    x, _, _, converged, pdip_iter = solve_nnls(
        jnp.asarray(Q_pc), jnp.asarray(q_pc), max_iter=PRODUCTION_MAX_ITER
    )

    assert int(converged) == 1, f"PDIP did not converge ({int(pdip_iter)} iterations)"
    assert int(pdip_iter) < PRODUCTION_MAX_ITER
    _assert_objective_reaches_fnnls(Q, q, np.asarray(x) * D)


@requires_jax
@pytest.mark.parametrize("key", KEYS, ids=IDS)
def test__reconstruction_positive_only_from__jax_matches_numpy_objective(jnp, key):
    Q, q = SYSTEMS[key]
    settings = aa.Settings()
    stats = {}

    x_jax = inversion_util.reconstruction_positive_only_from(
        data_vector=jnp.asarray(q),
        curvature_reg_matrix=jnp.asarray(Q),
        settings=settings,
        xp=jnp,
        stats=stats,
    )
    x_np = inversion_util.reconstruction_positive_only_from(
        data_vector=q, curvature_reg_matrix=Q, settings=settings, xp=np
    )

    x_jax = np.asarray(x_jax)
    assert np.all(np.isfinite(x_jax)), "JAX reconstruction is non-finite"
    obj_jax = _objective(Q, q, x_jax)
    obj_np = _objective(Q, q, np.asarray(x_np))
    assert abs(obj_jax - obj_np) <= OBJECTIVE_RTOL * abs(obj_np), (obj_jax, obj_np)

    assert stats["solver"] == "pdip"
    # Once the fix surfaces the PDIP convergence flag (PyAutoArray#571 step 4), also assert:
    # assert bool(stats["converged"])


@requires_jax
def test__pdip_jacobi_converges_under_vmap(jnp):
    import jax

    from autoarray.util.jax_nnls import solve_nnls

    scaled = [_jacobi(Q, q) for Q, q in SYSTEMS]
    Q_pc = jnp.asarray(np.stack([s[0] for s in scaled]))
    q_pc = jnp.asarray(np.stack([s[1] for s in scaled]))

    x, _, _, converged, pdip_iter = jax.vmap(
        lambda Q, q: solve_nnls(Q, q, max_iter=PRODUCTION_MAX_ITER)
    )(Q_pc, q_pc)

    converged = np.asarray(converged)
    pdip_iter = np.asarray(pdip_iter)
    failed = [
        k for k in KEYS if converged[k] != 1 or pdip_iter[k] >= PRODUCTION_MAX_ITER
    ]
    assert not failed, (
        f"failed lanes {failed}; converged {converged.tolist()}; "
        f"pdip_iter {pdip_iter.tolist()}"
    )
    for k, (Q, q) in enumerate(SYSTEMS):
        _assert_objective_reaches_fnnls(Q, q, np.asarray(x[k]) * scaled[k][2])


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
