"""Regression: registering a class that another route already registered
with JAX must not raise.

Building a JAX ``Fitness`` registers ``Galaxy`` through
``autofit.jax.register_model``; ``autolens.jax.register_tracer_classes`` then
reaches ``register_instance_pytree(Galaxy)``. JAX raises
``ValueError: Duplicate custom PyTreeDef type registration`` on the second
registration, which broke every static-lattice JAX test run after a JAX
``Fitness`` in the same process (PyAutoArray#693).

NumPy-only per the project rule [[feedback_no_jax_in_unit_tests]]: the JAX
registration call is replaced with a stub that raises the error JAX raises.
"""

import pytest

import autonerves.jax_wrapper as jax_wrapper

from autoarray import abstract_ndarray


def _duplicate(cls, *args, **kwargs):
    raise ValueError(f"Duplicate custom PyTreeDef type registration for {cls}.")


def _other_error(cls, *args, **kwargs):
    raise ValueError("some other registration failure")


def test__register_instance_pytree__class_registered_elsewhere__no_error(monkeypatch):
    class _RegisteredElsewhere:
        pass

    monkeypatch.setattr(jax_wrapper, "register_pytree_node", _duplicate)

    abstract_ndarray.register_instance_pytree(_RegisteredElsewhere)

    assert _RegisteredElsewhere in abstract_ndarray._pytree_registered_classes


def test__register_as_pytree__class_registered_elsewhere__no_error(monkeypatch):
    class _RegisteredElsewhere:
        @classmethod
        def instance_flatten(cls, instance):
            return (), ()

        @classmethod
        def instance_unflatten(cls, aux, children):
            return cls()

    monkeypatch.setattr(jax_wrapper, "register_pytree_node", _duplicate)

    abstract_ndarray._register_as_pytree(_RegisteredElsewhere)

    assert _RegisteredElsewhere in abstract_ndarray._pytree_registered_classes


def test__register_instance_pytree__other_value_error__raises(monkeypatch):
    class _Broken:
        pass

    monkeypatch.setattr(jax_wrapper, "register_pytree_node", _other_error)

    with pytest.raises(ValueError, match="some other registration failure"):
        abstract_ndarray.register_instance_pytree(_Broken)

    assert _Broken not in abstract_ndarray._pytree_registered_classes
