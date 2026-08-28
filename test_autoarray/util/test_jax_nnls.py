import importlib
import sys

import numpy as np
import pytest

import autoarray as aa


def test__jax_nnls_module_never_imports_jax_at_module_level(monkeypatch):
    # Library unit tests are NumPy-only: importing the module must succeed
    # even when jax / jaxnnls are unimportable. A None entry in sys.modules
    # makes any `import jax` raise ImportError, so a module-level import
    # would fail this reload.
    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.setitem(sys.modules, "jaxnnls", None)

    module = importlib.reload(importlib.import_module("autoarray.util.jax_nnls"))

    assert hasattr(module, "solve_nnls")
    assert hasattr(module, "solve_nnls_primal")


def test__settings_nnls_knobs_default_off():
    # The Settings defaults must reproduce jaxnnls's own hard-coded
    # behaviour (tolerance formula, MAX_ITER = 50) so a default Settings
    # object — or none at all — does not alter any solve.
    settings = aa.Settings()

    assert settings.nnls_solver_tol is None
    assert settings.nnls_max_iter is None

    settings = aa.Settings(nnls_solver_tol=1e-6, nnls_max_iter=30)

    assert settings.nnls_solver_tol == 1e-6
    assert settings.nnls_max_iter == 30


def test__reconstruction_positive_only_from__numpy_path_ignores_knobs():
    # The xp=np path (fnnls_cholesky) is untouched by the JAX solver knobs;
    # solve a small system whose unconstrained solution has a negative
    # component and check the non-negative solution, with and without
    # knob-carrying settings.
    data_vector = np.array([1.0, 1.0, 2.0])

    curvature_reg_matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])

    for settings in [None, aa.Settings(nnls_solver_tol=1e-6, nnls_max_iter=30)]:
        reconstruction = aa.util.inversion.reconstruction_positive_only_from(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            settings=settings,
        )

        assert np.all(reconstruction >= 0.0)
        # Unconstrained solution is [1, -1, 3]; the NNLS solution zeroes the
        # negative component and re-solves the free ones.
        assert reconstruction == pytest.approx(np.array([0.5, 0.0, 2.0]), 1.0e-4)


@pytest.fixture(autouse=True)
def _clear_nnls_memo():
    from autoarray.inversion.inversion.nnls_memo import _nnls_passive_set_memo

    _nnls_passive_set_memo.clear()
    yield
    _nnls_passive_set_memo.clear()


def _small_positive_only_system():
    # Unconstrained solution is [1, -1, 3]; the NNLS solution is [0.5, 0, 2].
    data_vector = np.array([1.0, 1.0, 2.0])
    curvature_reg_matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
    return data_vector, curvature_reg_matrix


def test__reconstruction_positive_only_from__warm_start_memo_records_the_passive_set():
    from autoarray.inversion.inversion.nnls_memo import (
        _nnls_passive_set_memo,
        memo_key,
    )

    data_vector, curvature_reg_matrix = _small_positive_only_system()

    reconstruction = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=aa.Settings(nnls_warm_start_memo=True),
        fingerprint="mesh",
    )

    assert reconstruction == pytest.approx(np.array([0.5, 0.0, 2.0]), 1.0e-4)

    key = memo_key(n=3, fingerprint="mesh")

    entry = _nnls_passive_set_memo[key]

    assert np.array_equal(entry.passive_set, np.array([0, 2]))
    # The dense-sign start of this system is exactly right, so the reference
    # error fraction it hands the guard is zero.
    assert entry.dense_error_fraction == 0.0


def test__reconstruction_positive_only_from__warm_start_memo_seeds_the_next_solve(
    monkeypatch,
):
    from autoarray.inversion.inversion import nnls_memo

    data_vector, curvature_reg_matrix = _small_positive_only_system()
    settings = aa.Settings(nnls_warm_start_memo=True)

    first = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=settings,
        fingerprint="mesh",
    )

    # Spy on the memo lookup so the second solve is shown to actually consume
    # the seed, not merely to leave the memo populated.
    seeds = []
    passive_set_get = nnls_memo.passive_set_get

    def _spy(**kwargs):
        seed = passive_set_get(**kwargs)
        seeds.append(seed)
        return seed

    monkeypatch.setattr(nnls_memo, "passive_set_get", _spy)

    second = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=settings,
        fingerprint="mesh",
    )

    assert len(seeds) == 1
    assert np.array_equal(seeds[0].passive_set, np.array([0, 2]))
    assert second == pytest.approx(first, rel=1e-10, abs=1e-12)


def test__reconstruction_positive_only_from__warm_start_memo_on_by_default():
    from autoarray.inversion.inversion.nnls_memo import (
        _nnls_passive_set_memo,
        memo_key,
    )

    data_vector, curvature_reg_matrix = _small_positive_only_system()

    # The memo ships on, so default settings plus a fingerprint memoize.
    reconstruction = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=aa.Settings(),
        fingerprint="mesh",
    )

    assert reconstruction == pytest.approx(np.array([0.5, 0.0, 2.0]), 1.0e-4)
    assert list(_nnls_passive_set_memo) == [memo_key(n=3, fingerprint="mesh")]


def test__reconstruction_positive_only_from__warm_start_memo_opt_outs():
    from autoarray.inversion.inversion.nnls_memo import _nnls_passive_set_memo

    data_vector, curvature_reg_matrix = _small_positive_only_system()

    # No settings object at all, and an explicit opt-out, both leave the memo
    # untouched even though the default is on.
    for settings in [None, aa.Settings(nnls_warm_start_memo=False)]:
        aa.util.inversion.reconstruction_positive_only_from(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            settings=settings,
            fingerprint="mesh",
        )

    # A caller that supplies no fingerprint cannot be memoized either, since
    # there is nothing identifying the index space the passive set lives in.
    aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=aa.Settings(),
    )

    assert _nnls_passive_set_memo == {}


def test__reconstruction_positive_only_from__warm_start_memo_disabled_by_env(
    monkeypatch,
):
    from autoarray.inversion.inversion.nnls_memo import _nnls_passive_set_memo

    monkeypatch.setenv("AUTOARRAY_NNLS_WARM_START", "0")

    data_vector, curvature_reg_matrix = _small_positive_only_system()

    reconstruction = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=aa.Settings(nnls_warm_start_memo=True),
        fingerprint="mesh",
    )

    assert reconstruction == pytest.approx(np.array([0.5, 0.0, 2.0]), 1.0e-4)
    assert _nnls_passive_set_memo == {}


# ===================================================================
# Relative fallback guard on a memo seed (Settings.nnls_warm_start_error_tolerance)
# ===================================================================


def _solve_capturing_stats(monkeypatch, settings, fingerprint="mesh"):
    """
    One `reconstruction_positive_only_from` on the small positive-only system,
    returning (reconstruction, stats).

    `seed_source` / `warm_start_fallback` are written into the stats dict AFTER
    `fnnls_cholesky` returns, so the dict must be held by reference and read
    once the call has finished -- reading it inside the wrapper would see the
    solver's keys only.
    """
    import autoarray.util.fnnls as fnnls_mod

    original = fnnls_mod.fnnls_cholesky
    captured = []

    def _wrapped(ZTZ, ZTx, P_initial=np.zeros(0, dtype=int), stats=None):
        captured.append(stats)
        return original(ZTZ, ZTx, P_initial, stats=stats)

    monkeypatch.setattr(fnnls_mod, "fnnls_cholesky", _wrapped)

    data_vector, curvature_reg_matrix = _small_positive_only_system()

    reconstruction = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=settings,
        fingerprint=fingerprint,
    )

    monkeypatch.setattr(fnnls_mod, "fnnls_cholesky", original)

    return reconstruction, captured[-1]


def test__warm_start_guard__seed_worse_than_tolerance_is_dropped_and_next_solve_is_dense(
    monkeypatch,
):
    from autoarray.inversion.inversion.nnls_memo import (
        _nnls_passive_set_memo,
        memo_key,
        passive_set_put,
    )

    key = memo_key(n=3, fingerprint="mesh")

    # A deliberately wrong seed against a deliberately small reference: the
    # true passive set is [0, 2], so seeding [1] gets all three entries wrong
    # (fraction 1.0) against a dense-sign reference of 0.1 -- 1.0 > 1.5 * 0.1.
    passive_set_put(key=key, passive_set=np.array([1]), dense_error_fraction=0.1)

    settings = aa.Settings(nnls_warm_start_memo=True)
    assert settings.nnls_warm_start_error_tolerance == 1.5

    reconstruction, stats = _solve_capturing_stats(monkeypatch, settings)

    assert reconstruction == pytest.approx(np.array([0.5, 0.0, 2.0]), 1.0e-4)
    assert stats["seed_source"] == "memo"
    assert stats["warm_start_fallback"] is True
    assert _nnls_passive_set_memo == {}

    # With the entry dropped, the next solve for the key restarts from the
    # dense-sign start and refreshes the reference.
    _, stats = _solve_capturing_stats(monkeypatch, settings)

    assert stats["seed_source"] == "dense"
    assert stats["warm_start_fallback"] is False

    entry = _nnls_passive_set_memo[key]

    assert np.array_equal(entry.passive_set, np.array([0, 2]))
    assert entry.dense_error_fraction == 0.0


def test__warm_start_guard__seed_within_tolerance_keeps_the_entry_and_the_reference(
    monkeypatch,
):
    from autoarray.inversion.inversion.nnls_memo import (
        _nnls_passive_set_memo,
        memo_key,
        passive_set_put,
    )

    key = memo_key(n=3, fingerprint="mesh")

    # Seeding every entry passive gets exactly one of three wrong (fraction
    # 1/3), which is inside 1.5 * 0.5.
    passive_set_put(key=key, passive_set=np.array([0, 1, 2]), dense_error_fraction=0.5)

    reconstruction, stats = _solve_capturing_stats(
        monkeypatch, aa.Settings(nnls_warm_start_memo=True)
    )

    assert reconstruction == pytest.approx(np.array([0.5, 0.0, 2.0]), 1.0e-4)
    assert stats["seed_source"] == "memo"
    assert stats["warm_start_fallback"] is False

    entry = _nnls_passive_set_memo[key]

    assert np.array_equal(entry.passive_set, np.array([0, 2]))
    # Only a dense-sign solve refreshes the reference, so it is carried through
    # the seeded solve unchanged.
    assert entry.dense_error_fraction == 0.5


@pytest.mark.parametrize("tolerance", [float("inf"), 0.0, -1.0])
def test__warm_start_guard__disabled_tolerance_never_drops(monkeypatch, tolerance):
    from autoarray.inversion.inversion.nnls_memo import (
        _nnls_passive_set_memo,
        memo_key,
        passive_set_put,
    )

    key = memo_key(n=3, fingerprint="mesh")

    passive_set_put(key=key, passive_set=np.array([1]), dense_error_fraction=0.1)

    _, stats = _solve_capturing_stats(
        monkeypatch,
        aa.Settings(
            nnls_warm_start_memo=True, nnls_warm_start_error_tolerance=tolerance
        ),
    )

    assert stats["seed_source"] == "memo"
    assert stats["warm_start_fallback"] is False
    assert _nnls_passive_set_memo[key].dense_error_fraction == 0.1


def test__warm_start_guard__a_perfect_dense_reference_does_not_breach_on_a_perfect_seed(
    monkeypatch,
):
    from autoarray.inversion.inversion.nnls_memo import (
        _nnls_passive_set_memo,
        memo_key,
    )

    settings = aa.Settings(nnls_warm_start_memo=True)

    # The dense-sign start of this system is exact, so the reference is 0.0 and
    # the guard degenerates to `frac > 0`. A seed that is also exact must not
    # breach it -- a perfect dense start is cheap to keep, not a reason to drop.
    _, stats = _solve_capturing_stats(monkeypatch, settings)

    assert stats["seed_source"] == "dense"

    key = memo_key(n=3, fingerprint="mesh")

    assert _nnls_passive_set_memo[key].dense_error_fraction == 0.0

    _, stats = _solve_capturing_stats(monkeypatch, settings)

    assert stats["seed_source"] == "memo"
    assert stats["warm_start_fallback"] is False
    assert key in _nnls_passive_set_memo
