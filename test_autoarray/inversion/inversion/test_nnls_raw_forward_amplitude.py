"""
Amplitude regression for the forward value of the ``"raw"`` positive-only PDIP mode (PyAutoArray#594).

The mapper-less JAX positive-only solve (``preconditioning="raw"``, PyAutoArray#571/#572) stops on an absolute
infinity-norm KKT residual with the data-scaled tolerance ``1e-2 * n * EPSILON * max(1, max|q|)``. The
complementarity ``s * z`` is judged against that same threshold, so a column fnnls holds at zero can keep
``x ~ tol / z`` while the solve reports "converged": the objective, the log likelihood and the KKT residual are
all blind to it, but the amplitudes (and latent fluxes derived from them) are not. The phase-1 accuracy study
(autolens_profiling #355, record ``complete/2026/09/linear-solver-accuracy-study.md``) measured it on the euclid
vis_lp system as 11.5 % of the reference flux left on reference-inactive columns (``total_source_flux``
+5.76 %, the red euclid-pipeline latent jit-vs-eager test), and >1e-3 source-flux errors on 4/8 #571 systems.

The fixture ``files/mge_solver_reference_systems.npz`` holds the 8 #571 SLaM systems (``k0``..``k7``) and the
euclid vis_lp system with their fnnls reference solutions ``x_ref`` (see ``files/README.md``). Each system is
solved through the library entry point :func:`autoarray.util.jax_nnls.solve_nnls_primal_raw_forward` (built as
``inversion_util.reconstruction_positive_only_from`` builds it) and through
``reconstruction_positive_only_from(..., preconditioning="raw")`` itself, and must satisfy:

- ``converged == 1``, ``iterations < 50``, finite;
- ``flux_inactive_rel = sum(x[x_ref <= 1e-6 max x_ref]) / sum(x_ref) <= 1e-3``;
- ``|flux_rel_all| = |sum x - sum x_ref| / |sum x_ref| <= 1e-3``;
- ``|flux_rel_source|`` (the same over ``source_column_index_list``) ``<= 1e-3`` on ``k0``..``k7`` only. The
  euclid system is excluded from this one metric: its source columns carry only ~0.4 % of the reference flux
  (a single active column), so the relative source flux is noise-dominated (+5e-2 even for the polished
  solve); its end-to-end latent is covered by the euclid-pipeline test (polished: +7.5e-5);
- ``jax.jit`` of the reconstruction equals the eager value exactly, the custom_vjp primal equals its
  differentiated forward value exactly (eager and jitted), and ``jax.grad`` of ``sum(y)`` with
  respect to ``q`` is finite and non-zero (a mis-wired custom_vjp returns zeros silently).

Thresholds were fixed from the phase-1 rows before this test was written. Raw forward (unpolished): euclid
inactive 0.115, all 0.115 (others inactive <= 5.4e-5); source k1 1.64e-3, k2 1.22e-3, k3 4.04e-3, k5 6.44e-3
(k0 5.2e-4, k4 6.7e-4, k6 9.2e-4, k7 9.5e-4). Polished forward: inactive <= 3.3e-4 (euclid), source <= 5e-5
on k0..k7, all <= 3.3e-4. The amplitude-max criterion of the task prompt is dropped: its worst value (0.219)
is shared by every accurate candidate, i.e. it measures fnnls noise on near-flat directions.

Red on the unfixed base (PyAutoArray bd03e09e, whose solver equals d4298445): euclid ``inactive`` and ``all``,
and ``source`` on k1, k2, k3, k5 -- on both paths. The jit == eager, primal == forward and gradient cases are green there too (the base
returns the unpolished ``y`` from both the primal and the fwd rule).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.inversion import inversion_util


requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)

FIXTURE = Path(__file__).parent / "files" / "mge_solver_reference_systems.npz"
PRODUCTION_MAX_ITER = 50
TARGET_KAPPA = (
    1.0e-11  # autoarray general.yaml ``nnls_target_kappa``, as the dispatch reads it
)
INACTIVE_REL = 1.0e-6
FLUX_TOL = 1.0e-3


def _load_systems():
    with np.load(FIXTURE) as data:
        meta = json.loads(str(data["meta"]))
        systems = {
            s["name"]: tuple(
                np.asarray(data[f"{p}_{s['name']}"]) for p in ("Q", "q", "x_ref")
            )
            for s in meta["systems"]
        }
    return meta, systems


META, SYSTEMS = _load_systems()
SYSTEM_META = {s["name"]: s for s in META["systems"]}
NAMES = list(SYSTEM_META)
SLAM_NAMES = [n for n in NAMES if SYSTEM_META[n]["group"] == "slam_fixture_571"]
PATHS = ["library_entry", "reconstruction_positive_only_from"]


@pytest.fixture(scope="module")
def jnp():
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jaxnnls.pdip import EPSILON

    # jaxnnls fixes its tolerance scale at import time from the default dtype; a float32-era import would
    # loosen every tolerance and hide the bias.
    assert EPSILON < 1.0e-10, EPSILON

    return jnp


def _entry(jnp, Q, q):
    """`solve_nnls_primal_raw_forward`, built exactly as the raw branch of `reconstruction_positive_only_from`."""
    from autoarray.util.jax_nnls import solve_nnls_primal_raw_forward

    d = jnp.sqrt(jnp.diag(Q))
    D = 1.0 / d
    Q_pc = (Q * D[:, None]) * D[None, :]
    q_pc = q * D
    y, converged, iterations = solve_nnls_primal_raw_forward(
        Q_pc,
        q_pc,
        Q,
        q,
        D,
        target_kappa=TARGET_KAPPA,
        solver_tol=None,
        max_iter=PRODUCTION_MAX_ITER,
    )
    return y, D, converged, iterations


def _entry_solver(jnp, Q, q):
    """The ``y`` output of `solve_nnls_primal_raw_forward` as a function of its five inputs, and those inputs."""
    from autoarray.util.jax_nnls import solve_nnls_primal_raw_forward

    D = 1.0 / jnp.sqrt(jnp.diag(Q))
    args = ((Q * D[:, None]) * D[None, :], q * D, Q, q, D)

    def solve(*a):
        return solve_nnls_primal_raw_forward(
            *a, target_kappa=TARGET_KAPPA, solver_tol=None, max_iter=PRODUCTION_MAX_ITER
        )[0]

    return solve, args


def _dispatch(jnp, Q, q, stats=None):
    return inversion_util.reconstruction_positive_only_from(
        data_vector=q,
        curvature_reg_matrix=Q,
        settings=aa.Settings(),
        xp=jnp,
        stats=stats,
        preconditioning="raw",
        solver="pdip",
    )


_SOLVED = {}


def _solve(jnp, path, name):
    """(x, converged, iterations) for one system through one path, cached per module."""
    if (path, name) not in _SOLVED:
        Q, q, _ = SYSTEMS[name]
        Qj, qj = jnp.asarray(Q), jnp.asarray(q)
        if path == "library_entry":
            y, D, converged, iterations = _entry(jnp, Qj, qj)
            x = y * D
        else:
            stats = {}
            x = _dispatch(jnp, Qj, qj, stats=stats)
            assert stats["solver"] == "pdip" and stats["preconditioning"] == "raw"
            converged, iterations = stats["converged"], stats["iterations"]
        _SOLVED[(path, name)] = (np.asarray(x), int(converged), int(iterations))
    return _SOLVED[(path, name)]


def _flux_inactive_rel(x, x_ref):
    inactive = x_ref <= INACTIVE_REL * x_ref.max()
    return float(np.sum(x[inactive]) / np.sum(x_ref))


def _flux_rel(x, x_ref, index=slice(None)):
    return float((np.sum(x[index]) - np.sum(x_ref[index])) / abs(np.sum(x_ref[index])))


def test__fixture_is_the_reference_system_set():
    assert FIXTURE.stat().st_size < 400_000
    assert NAMES == [f"k{k}" for k in range(8)] + ["euclid_vis_lp"]
    assert META["provenance"]["corpus_commit"] == "3ad68af"
    for name, (Q, q, x_ref) in SYSTEMS.items():
        assert Q.shape == (60, 60) and q.shape == (60,) and x_ref.shape == (60,)
        np.testing.assert_allclose(Q, Q.T, rtol=0, atol=1e-8 * np.abs(Q).max())
        assert np.all(x_ref >= 0.0)
        assert np.isclose(np.abs(q).max(), SYSTEM_META[name]["max_abs_q"], rtol=1e-12)


@requires_jax
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("path", PATHS)
def test__raw_forward_converges(jnp, path, name):
    x, converged, iterations = _solve(jnp, path, name)

    assert converged == 1, f"not converged ({iterations} iterations)"
    assert iterations < PRODUCTION_MAX_ITER
    assert np.all(np.isfinite(x))


@requires_jax
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("path", PATHS)
def test__raw_forward_inactive_column_flux(jnp, path, name):
    x, _, _ = _solve(jnp, path, name)
    value = _flux_inactive_rel(x, SYSTEMS[name][2])

    assert value <= FLUX_TOL, f"flux_inactive_rel = {value:.3e}"


@requires_jax
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("path", PATHS)
def test__raw_forward_total_flux(jnp, path, name):
    x, _, _ = _solve(jnp, path, name)
    value = _flux_rel(x, SYSTEMS[name][2])

    assert abs(value) <= FLUX_TOL, f"flux_rel_all = {value:.3e}"


@requires_jax
@pytest.mark.parametrize("name", SLAM_NAMES)
@pytest.mark.parametrize("path", PATHS)
def test__raw_forward_source_flux(jnp, path, name):
    """k0..k7 only: see the module docstring for why the euclid system is excluded from this metric."""
    x, _, _ = _solve(jnp, path, name)
    value = _flux_rel(
        x, SYSTEMS[name][2], index=SYSTEM_META[name]["source_column_index_list"]
    )

    assert abs(value) <= FLUX_TOL, f"flux_rel_source = {value:.3e}"


@requires_jax
@pytest.mark.parametrize("name", NAMES)
def test__raw_forward_jit_matches_eager(jnp, name):
    """``jax.jit`` of the reconstruction returns the eager value exactly, end-to-end and for the solver alone."""
    import jax

    Q, q, _ = SYSTEMS[name]
    Qj, qj = jnp.asarray(Q), jnp.asarray(q)

    def f(Q_, q_):
        return _dispatch(jnp, Q_, q_)

    np.testing.assert_array_equal(np.asarray(jax.jit(f)(Qj, qj)), np.asarray(f(Qj, qj)))

    # The solver alone, on inputs built outside the traced function: tracing the Jacobi scaling together with
    # the final ``x / D`` lets XLA reassociate them and moves ``y`` by 1 ULP (seen on the unfixed base too),
    # which is not what this test is about.
    solve, args = _entry_solver(jnp, Qj, qj)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(solve)(*args)), np.asarray(solve(*args))
    )


@requires_jax
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("name", NAMES)
def test__raw_forward_primal_matches_differentiated_forward(jnp, name, jit):
    """The custom_vjp primal (plain calls) and its fwd rule (any differentiated call) return the same ``y``
    bit-for-bit, so a value does not change when a gradient is taken through it."""
    import jax

    Q, q, _ = SYSTEMS[name]
    solve, args = _entry_solver(jnp, jnp.asarray(Q), jnp.asarray(q))

    def primal_out(*a):
        return solve(*a)

    def fwd_out(*a):
        return jax.vjp(solve, *a)[0]

    if jit:
        primal_out, fwd_out = jax.jit(primal_out), jax.jit(fwd_out)

    np.testing.assert_array_equal(
        np.asarray(fwd_out(*args)), np.asarray(primal_out(*args))
    )


@requires_jax
@pytest.mark.parametrize("name", NAMES)
def test__raw_forward_gradient_is_finite_and_non_zero(jnp, name):
    import jax

    Q, q, _ = SYSTEMS[name]
    Qj = jnp.asarray(Q)

    def f(q_):
        y, _, _, _ = _entry(jnp, Qj, q_)
        return jnp.sum(y)

    for grad in (jax.grad(f), jax.jit(jax.grad(f))):
        gq = np.asarray(grad(jnp.asarray(q)))
        assert np.all(np.isfinite(gq))
        assert np.any(gq != 0.0)
