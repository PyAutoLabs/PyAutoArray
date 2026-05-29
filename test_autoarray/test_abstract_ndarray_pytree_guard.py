"""Regression: cached_property descriptors on AbstractNDArray subclasses
must be filtered from ``instance_flatten`` so derived caches never reach
the JAX pytree leaves.

NumPy-only per the project rule [[feedback_no_jax_in_unit_tests]]:
exercise the ``instance_flatten`` classmethod directly (which is what
the JAX pytree path delegates to) and assert composition is correct.
"""

import functools

import numpy as np

from autoarray.abstract_ndarray import AbstractNDArray


class _FakeArray(AbstractNDArray):
    """Minimal AbstractNDArray subclass that adds a ``@cached_property``
    returning a string. Used to assert the guard filters it from
    ``instance_flatten``."""

    __no_flatten__ = ("use_jax",)

    def __init__(self, array):
        # Skip AbstractNDArray.__init__ to avoid the JAX-registration path
        # — we only need the dict-shape for the flatten test.
        self._array = np.asarray(array)
        self._is_transformed = False
        self.use_jax = False

    @property
    def native(self):
        # AbstractNDArray declares ``native`` abstract; the body is
        # irrelevant to the flatten path so just echo ``_array``.
        return self._array

    @functools.cached_property
    def heavy_summary(self):
        return "a-pretty-printed-summary-of-the-array"


def test_instance_flatten_excludes_cached_property_names():
    """``AbstractNDArray.instance_flatten`` unions the class-level
    ``__no_flatten__`` with the result of
    ``autoconf.tools.decorators.cached_property_names`` so derived
    cached strings stay out of the pytree leaves.

    This pins the structural defense that follows PyAutoFit#1300: the
    leak surfaces today only on the Model side, but the same opt-out
    filter shape on AbstractNDArray descendants would break ``jax.jit``
    the moment anyone added a ``@cached_property`` returning a
    non-array value to a Fit class."""

    arr = _FakeArray([1.0, 2.0, 3.0])

    # Trigger the cached property: it writes "...summary..." into __dict__.
    _ = arr.heavy_summary
    assert arr.__dict__["heavy_summary"] == "a-pretty-printed-summary-of-the-array"

    leaves, keys = _FakeArray.instance_flatten(arr)

    # The pre-existing __no_flatten__ exclusion ("use_jax") still applies.
    assert "use_jax" not in keys
    # The new cached_property exclusion fires too.
    assert "heavy_summary" not in keys
    # No string leaves anywhere.
    assert not any(isinstance(leaf, str) for leaf in leaves)


def test_instance_flatten_preserves_array_data():
    """Sanity check: filtering cached_property names does not collateral-
    damage real array data. The underlying numpy array must still appear
    in the leaves."""

    arr = _FakeArray([1.0, 2.0, 3.0])
    _ = arr.heavy_summary  # poison the cache before flattening

    leaves, keys = _FakeArray.instance_flatten(arr)

    assert "_array" in keys
    array_index = keys.index("_array")
    np.testing.assert_array_equal(leaves[array_index], np.asarray([1.0, 2.0, 3.0]))
