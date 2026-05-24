"""Unit test for the ``AbstractMaker.__init__`` xp/grid mismatch ``ValueError``.

Library unit tests stay NumPy-only per [[feedback_no_jax_in_unit_tests]]. The
``grid.use_jax`` flag is a plain Python bool on the autoarray side, so we can
simulate a "JAX-backed grid" with a mock object that has ``use_jax=True``
without importing JAX.
"""
import types

import numpy as np
import pytest

from autoarray.structures.decorators.abstract import AbstractMaker


def _dummy_func(obj, grid, xp):
    return xp.zeros(1)


class _NumpyGrid:
    use_jax = False


class _JaxGrid:
    use_jax = True


def test_raises_on_xp_np_with_jnp_backed_grid():
    grid = _JaxGrid()
    with pytest.raises(ValueError) as exc_info:
        AbstractMaker(func=_dummy_func, obj=None, grid=grid, xp=np)
    msg = str(exc_info.value)
    assert "xp=np but the input grid is JAX-backed" in msg
    assert "_dummy_func" in msg


def test_does_not_raise_on_xp_np_with_numpy_grid():
    grid = _NumpyGrid()
    maker = AbstractMaker(func=_dummy_func, obj=None, grid=grid, xp=np)
    assert maker.use_jax is False


def test_does_not_raise_on_xp_np_with_grid_lacking_use_jax_attr():
    """getattr fallback: a grid without ``use_jax`` attribute (e.g. a plain
    ndarray-like) does not trip the guard — only explicit ``use_jax=True``
    counts as a mismatch."""
    grid = types.SimpleNamespace()  # no use_jax attr
    maker = AbstractMaker(func=_dummy_func, obj=None, grid=grid, xp=np)
    assert maker.use_jax is False
