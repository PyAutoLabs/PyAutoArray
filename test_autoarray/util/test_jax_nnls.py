import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

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


def test__nnls_solver_config_keys_have_upstream_default_values():
    # The packaged general.yaml defaults must reproduce jaxnnls's own
    # hard-coded behaviour (tolerance formula and MAX_ITER = 50) so that
    # installing this change alone does not alter any solve. The packaged
    # file is parsed directly because the test suite (like workspaces)
    # pushes its own shadowing config which omits the nnls keys — the
    # solver reads missing keys via try/except fallbacks to these values.
    packaged = Path(aa.__file__).parent / "config" / "general.yaml"
    inversion = yaml.safe_load(packaged.read_text())["inversion"]

    assert inversion["nnls_solver_tol"] is None
    assert inversion["nnls_max_iter"] == 50


def test__reconstruction_positive_only_from__numpy_path():
    # The xp=np path (fnnls_cholesky) is untouched by the JAX solver knobs;
    # solve a small system whose unconstrained solution has a negative
    # component and check the non-negative solution.
    data_vector = np.array([1.0, 1.0, 2.0])

    curvature_reg_matrix = np.array(
        [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]]
    )

    reconstruction = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
    )

    assert np.all(reconstruction >= 0.0)
    # Unconstrained solution is [1, -1, 3]; the NNLS solution zeroes the
    # negative component and re-solves the free ones.
    assert reconstruction == pytest.approx(np.array([0.5, 0.0, 2.0]), 1.0e-4)
