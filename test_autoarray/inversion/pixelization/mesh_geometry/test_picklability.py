"""Pickle round-trip tests for AbstractMeshGeometry subclasses.

Required by subprocess visualization (PyAutoFit #1279, Phase 4 of the JAX
visualization roadmap). A populated FitImaging must be sendable over an
mp.Process+Queue or ProcessPoolExecutor boundary — historically blocked
by `self._xp = xp` (module attribute) on AbstractMeshGeometry, since
Python's pickle cannot serialise module references.

Fix: `_xp` is now a property derived from a boolean `_use_jax` flag.
This file is the regression test for that invariant.
"""

import importlib.util
import pickle

import numpy as np
import pytest

from autoarray.inversion.mesh.mesh_geometry.rectangular import MeshGeometryRectangular
from autoarray.inversion.mesh.mesh_geometry.delaunay import MeshGeometryDelaunay


def _jax_installed() -> bool:
    return importlib.util.find_spec("jax") is not None


@pytest.mark.parametrize("cls", [MeshGeometryRectangular, MeshGeometryDelaunay])
def test_pickle_round_trip_numpy_backend(cls):
    """A numpy-backed MeshGeometry instance must round-trip through pickle
    with `_xp` restored to the numpy module."""
    mg = cls.__new__(cls)
    mg._use_jax = False

    restored = pickle.loads(pickle.dumps(mg))

    assert restored._use_jax is False
    assert restored._xp is np


@pytest.mark.skipif(not _jax_installed(), reason="jax not installed")
@pytest.mark.parametrize("cls", [MeshGeometryRectangular, MeshGeometryDelaunay])
def test_pickle_round_trip_jax_backend(cls):
    """A JAX-backed MeshGeometry instance must round-trip through pickle
    with `_xp` restored to the jax.numpy module."""
    import jax.numpy as jnp

    mg = cls.__new__(cls)
    mg._use_jax = True

    restored = pickle.loads(pickle.dumps(mg))

    assert restored._use_jax is True
    assert restored._xp is jnp


def test_use_jax_inferred_from_xp_kwarg_in_init():
    """The __init__ continues to accept an `xp=` kwarg but stores it as
    a boolean — modules never land on the instance."""
    import types

    # Use __new__ + manual __init__ call with stubbed positional args to
    # avoid pulling in Mesh / Grid construction. The invariant under test
    # is post-__init__ state.
    mg = MeshGeometryRectangular.__new__(MeshGeometryRectangular)
    MeshGeometryRectangular.__init__(
        mg,
        mesh=None,
        mesh_grid=None,
        data_grid=None,
        xp=np,
    )
    assert mg._use_jax is False
    # No module attribute should exist on the instance.
    module_attrs = [
        name for name, val in vars(mg).items() if isinstance(val, types.ModuleType)
    ]
    assert module_attrs == [], f"unexpected module attrs on instance: {module_attrs}"
