import numpy as np
import pytest

from autoarray.util.cholesky_funcs import cholinsertlast
from autoarray.util.fnnls import fnnls_cholesky


# A (near-)degenerate source mesh puts two vertices at (near-)identical
# positions, which gives two (near-)identical columns in the mapping matrix and
# so a normal-equations matrix that is singular to working precision. The
# Cholesky insertion's Schur complement is then zero up to rounding, and which
# side of zero it lands on depends only on floating-point summation order --
# i.e. on the BLAS thread count, which is why the original failure was
# reproducible on CI but not locally.
#
# Every one of those roundings is the same degenerate matrix, so all of them
# must fail the same way. The regression these tests lock down is that two of
# the three used to return NaN *without raising*, which let a NaN
# reconstruction escape into the adapt image and resurface much later as a
# qhull "Points cannot contain NaN" in the next pixelization stage.


def _normal_equations(n, n_data, jitter, seed):
    """Normal equations for a mapping matrix whose columns 0 and 1 belong to
    (near-)coincident mesh vertices. `jitter=0.0` makes them exactly equal."""
    rng = np.random.default_rng(seed)
    mapping = rng.random((n_data, n))
    mapping[:, 1] = mapping[:, 0] + jitter * rng.standard_normal(n_data)
    return mapping.T @ mapping, mapping.T @ rng.random(n_data)


def _insert_duplicate_last(ZTZ):
    """Factorise every column but the duplicate, then insert the duplicate --
    the exact situation `fnnls_cholesky` reaches via `cholinsertlast`."""
    from scipy import linalg as slg

    n = ZTZ.shape[0]
    order = [0] + list(range(2, n)) + [1]
    U = slg.cholesky(ZTZ[np.ix_(order[:-1], order[:-1])])
    return U, ZTZ[order[-1]][order]


@pytest.mark.parametrize("seed", range(12))
def test__cholinsertlast__singular_insertion_never_yields_an_unusable_pivot(seed):
    # On a singular insertion the Schur complement is zero up to rounding and
    # lands on either side of zero depending only on summation order. Before
    # the fix, a tiny negative raised but an exact zero produced a zero pivot,
    # which the following solve divides by -- NaN, with nothing raised.
    #
    # The invariant is therefore NOT "always raise" (a small positive pivot is
    # still a usable pivot, and rejecting it would change likelihood
    # evaluations). It is: either raise, or return a strictly positive finite
    # pivot. A zero or non-finite pivot must never be returned.
    ZTZ, _ = _normal_equations(n=12, n_data=40, jitter=0.0, seed=seed)

    U, x = _insert_duplicate_last(ZTZ)

    try:
        S = cholinsertlast(U, x)
    except np.linalg.LinAlgError:
        return

    assert S[-1, -1] > 0.0
    assert np.all(np.isfinite(S))


def test__cholinsertlast__rejects_a_non_positive_schur_complement():
    # The exact-zero pivot is the silent case the fix exists to close, so pin
    # it directly rather than relying on a seed happening to produce it.
    from autoarray.util.cholesky_funcs import _pivot_from_schur

    for schur in (0.0, -0.0, -1e-16, -1.0):
        with pytest.raises(np.linalg.LinAlgError):
            _pivot_from_schur(schur=schur, diagonal=13.0, index=11)


def test__cholinsertlast__passes_a_positive_schur_complement_through_unchanged():
    # Anything positive must return the bitwise-identical pivot the old raw
    # `math.sqrt` returned, so no working fit changes.
    import math

    from autoarray.util.cholesky_funcs import _pivot_from_schur

    for schur in (1.776357e-15, 1e-8, 0.5, 13.0):
        assert _pivot_from_schur(
            schur=schur, diagonal=13.0, index=11
        ) == math.sqrt(schur)


@pytest.mark.parametrize("seed", range(12))
def test__cholinsertlast__well_conditioned_insertion_still_succeeds(seed):
    # The guard must not reject an ordinary, non-degenerate insertion.
    rng = np.random.default_rng(seed)
    mapping = rng.random((40, 12))
    ZTZ = mapping.T @ mapping

    U, x = _insert_duplicate_last(ZTZ)
    S = cholinsertlast(U, x)

    assert S.shape == (12, 12)
    assert S[-1, -1] > 0.0
    assert np.all(np.isfinite(S))


@pytest.mark.parametrize("jitter", [0.0, 1e-15, 1e-12, 1e-9])
def test__fnnls_cholesky__never_returns_a_non_finite_solution(jitter):
    # The producer regression: across the whole near-degenerate band, the
    # solver must either return a finite solution or raise -- never hand back
    # NaN as though it were a valid reconstruction.
    solved = 0

    for seed in range(40):
        ZTZ, ZTx = _normal_equations(n=12, n_data=40, jitter=jitter, seed=seed)

        # The band really is degenerate. This is a property of the fixture, so it
        # holds on every machine -- unlike *how the solver responds* to it, which
        # is decided by floating-point summation order (see the module comment).
        assert np.linalg.cond(ZTZ) > 1.0e12

        try:
            P_initial = np.linalg.solve(ZTZ, ZTx) > 0
        except np.linalg.LinAlgError:
            P_initial = np.zeros(ZTZ.shape[0], dtype=bool)

        try:
            reconstruction = fnnls_cholesky(ZTZ, ZTx.T, P_initial=P_initial)
        except np.linalg.LinAlgError:
            continue

        solved += 1

        assert np.all(np.isfinite(reconstruction))

    # Sanity: at least one seed must have reached the finiteness assertion above,
    # otherwise this test passes vacuously.
    #
    # NOT `raised > 0`. The invariant here is the same one
    # `test__cholinsertlast__singular_insertion_never_yields_an_unusable_pivot`
    # states: either raise, or return something finite. Raising is one *allowed*
    # outcome, not a required one, so requiring it of some seed asserts on which
    # side of zero the runner's BLAS happens to round the Schur complement. At
    # `jitter=0.0` exactly one of these 40 seeds raises on a single-threaded local
    # BLAS and none do on the CI runners, so `raised > 0` was a coin flip that
    # failed CI on a green library.
    assert solved > 0


def test__fnnls_cholesky__well_conditioned_problem_is_unaffected():
    # No false positives: a well-conditioned problem whose non-negativity
    # constraints bind hard must still solve, and solve non-negatively.
    for seed in range(40):
        rng = np.random.default_rng(seed)
        mapping = rng.random((60, 20))
        ZTZ = mapping.T @ mapping
        ZTZ[np.diag_indices(20)] += 1e-8

        # a truth vector with many negative entries makes the constraints bind
        truth = rng.normal(size=20)
        truth[rng.random(20) < 0.6] *= -1.0
        ZTx = ZTZ @ truth

        reconstruction = fnnls_cholesky(
            ZTZ, ZTx.T, P_initial=np.linalg.solve(ZTZ, ZTx) > 0
        )

        assert np.all(np.isfinite(reconstruction))
        assert np.all(reconstruction >= 0.0)


def test__degenerate_failure_is_caught_by_the_inversion_guard():
    # `reconstruction_positive_only_from` guards the solver with
    # `except (RuntimeError, np.linalg.LinAlgError, ValueError)`. The type
    # raised for a degenerate matrix has to fall inside that tuple, otherwise
    # the failure escapes the inversion machinery instead of being converted
    # into an InversionException / resample signal.
    assert issubclass(np.linalg.LinAlgError, (RuntimeError, ValueError))
