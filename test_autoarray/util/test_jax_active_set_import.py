"""
Import guard for `autoarray.util.jax_active_set`, kept apart from `test_jax_active_set.py` because that module
skips wholesale when jax is absent, and this test must run exactly there.
"""

import importlib
import sys


def test__jax_active_set_module_never_imports_jax_at_module_level(monkeypatch):
    # Library unit tests are NumPy-only: importing the module must succeed even when jax is
    # unimportable. A None entry in sys.modules makes any `import jax` raise ImportError, so a
    # module-level import would fail this reload.
    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.setitem(sys.modules, "jaxnnls", None)

    module = importlib.reload(importlib.import_module("autoarray.util.jax_active_set"))

    for name in (
        "masked_solve",
        "certify",
        "active_set_search",
        "solve_certified",
        "solve_certified_with_fallback",
    ):
        assert hasattr(module, name)
