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

    curvature_reg_matrix = np.array(
        [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]]
    )

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
    curvature_reg_matrix = np.array(
        [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]]
    )
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

    assert np.array_equal(_nnls_passive_set_memo[key], np.array([0, 2]))


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
    assert np.array_equal(seeds[0], np.array([0, 2]))
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
