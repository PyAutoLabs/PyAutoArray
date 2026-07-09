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
