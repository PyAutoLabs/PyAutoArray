from __future__ import annotations

from copy import copy

from abc import ABC
from abc import abstractmethod

import numpy as np



from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.structures.abstract_structure import Structure

from autoconf import conf


def to_new_array(func):
    """
    Decorator for functions that returns an array. The array is wrapped in a new instance of the class.

    Parameters
    ----------
    func
        The function to be decorated.

    Returns
    -------
    The decorated function.
    """

    def wrapper(self, *args, **kwargs) -> "AbstractNDArray":
        return self.with_new_array(func(self, *args, **kwargs))

    return wrapper


def unwrap_array(func):
    """
    Decorator for functions that take an array as an argument. If the argument is an AbstractNDArray, the underlying
    array is used instead.

    Parameters
    ----------
    func
        The function to be decorated.

    Returns
    -------
    The decorated function.
    """

    def wrapper(self, other):
        try:
            return func(self, other.array)
        except AttributeError:
            pass
        return func(self, other)

    return wrapper


_pytree_registered_classes: set = set()


def _register_as_pytree(cls):
    """Register ``cls`` with ``jax.tree_util`` via the lazy autoconf wrapper.

    Gated: only called when a subclass instance is constructed on the JAX path
    (``xp is not np``). The registration is class-scoped via
    ``_pytree_registered_classes`` so each subclass pays the cost at most once
    regardless of how many instances are created. No-ops if JAX is not installed.
    """
    if cls in _pytree_registered_classes:
        return
    from autoconf.jax_wrapper import register_pytree_node

    register_pytree_node(cls, cls.instance_flatten, cls.instance_unflatten)
    _pytree_registered_classes.add(cls)


def register_instance_pytree(cls, no_flatten=()):
    """Register any class with ``jax.tree_util`` via ``__dict__`` flattening.

    Generic counterpart to :func:`_register_as_pytree` for classes that are
    *not* ``AbstractNDArray`` subclasses but still need to round-trip through
    ``jax.jit`` (e.g. ``FitImaging``, ``Tracer``, ``Imaging``). Attributes are
    partitioned using ``no_flatten``:

    * Names **not** in ``no_flatten`` ride as pytree children — JAX traces them
      and can substitute new values on unflatten (dynamic per fit).
    * Names **in** ``no_flatten`` ride as ``aux_data`` — JAX treats them as
      opaque Python objects, closing over the original reference across the
      JIT boundary. Appropriate for per-analysis constants (dataset, settings,
      cosmology, adapt images).

    Reconstructs via ``cls.__new__`` + ``setattr`` (side-effect-free — no
    ``__init__`` re-entry). Idempotent.
    """
    if cls in _pytree_registered_classes:
        return
    from autoconf.jax_wrapper import register_pytree_node

    no_flatten_set = frozenset(no_flatten)

    def flatten(instance):
        dyn: list = []
        static: list = []
        for key, value in sorted(instance.__dict__.items()):
            if key in no_flatten_set:
                static.append((key, value))
            else:
                dyn.append((key, value))
        dyn_keys = tuple(k for k, _ in dyn)
        dyn_values = tuple(v for _, v in dyn)
        static_items = tuple(static)
        return dyn_values, (dyn_keys, static_items)

    def unflatten(aux, children):
        dyn_keys, static_items = aux
        new = cls.__new__(cls)
        for key, value in zip(dyn_keys, children):
            setattr(new, key, value)
        for key, value in static_items:
            setattr(new, key, value)
        return new

    register_pytree_node(cls, flatten, unflatten)
    _pytree_registered_classes.add(cls)


class AbstractNDArray(ABC):

    __no_flatten__ = ()

    def __init__(self, array, xp=np):

        self._is_transformed = False

        while isinstance(array, AbstractNDArray):
            array = array.array
        self._array = array

        self.use_jax = xp is not np

        if self.use_jax:
            _register_as_pytree(type(self))

    @property
    def is_transformed(self) -> bool:
        return self._is_transformed

    @is_transformed.setter
    def is_transformed(self, value: bool):
        self._is_transformed = value

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    def invert(self):
        new = self.copy()
        new._array = self._xp.invert(new._array)
        return new

    @classmethod
    def instance_flatten(cls, instance):
        """
        Flatten an instance of an autoarray class into a tuple of its attributes (i.e.. a pytree)
        """
        keys, values = zip(
            *sorted(
                {
                    key: value
                    for key, value in instance.__dict__.items()
                    if key not in cls.__no_flatten__
                }.items()
            )
        )
        return values, keys

    @classmethod
    def instance_unflatten(cls, aux_data, children):
        """
        Unflatten a tuple of attributes (i.e. a pytree) into an instance of an autoarray class
        """
        instance = cls.__new__(cls)
        for key, value in zip(aux_data, children):
            setattr(instance, key, value)
        return instance

    def with_new_array(self, array: np.ndarray) -> "AbstractNDArray":
        """
        Copy this object but give it a new array.

        This is used to ensure that when an array is modified, associated
        attributes such as pixel size are retained.

        Parameters
        ----------
        array
            The new array that is given to the copied object.

        Returns
        -------

        """
        new_array = self.copy()
        new_array._array = array
        return new_array

    def copy(self):
        new = copy(self)
        return new

    def __copy__(self):
        """
        When copying an autoarray also copy its underlying array.
        """
        new = self.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new._array = self._array.copy()
        return new

    def __deepcopy__(self, memo):
        """
        When copying an autoarray also copy its underlying array.
        """
        new = self.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new._array = self._array.copy()
        return new

    def __iter__(self):
        return iter(self._array)

    @to_new_array
    def sqrt(self):
        return self._xp.sqrt(self._array)

    @property
    def array(self):
        return self._array

    @unwrap_array
    def __lt__(self, other):
        return self._array < other

    @unwrap_array
    def __le__(self, other):
        return self._array <= other

    @unwrap_array
    def __gt__(self, other):
        return self._array > other

    @unwrap_array
    def __ge__(self, other):
        return self._array >= other

    @unwrap_array
    def __eq__(self, other):
        return self._array == other

    @to_new_array
    @unwrap_array
    def __pow__(self, other):
        return self._array**other

    @to_new_array
    @unwrap_array
    def __add__(self, other):
        return self._array + other

    @to_new_array
    @unwrap_array
    def __radd__(self, other):
        return other + self._array

    @to_new_array
    @unwrap_array
    def __sub__(self, other):
        return self._array - other

    @to_new_array
    @unwrap_array
    def __rsub__(self, other):
        return other - self._array

    @unwrap_array
    def __ne__(self, other):
        return self._array != other

    @to_new_array
    @unwrap_array
    def __mul__(self, other):
        return self._array * other

    @to_new_array
    @unwrap_array
    def __rmul__(self, other):
        return other * self._array

    @to_new_array
    def __neg__(self):
        return -self._array

    def __invert__(self):
        return ~self._array

    def __divmod__(self, other):
        return divmod(self._array, other)

    def __rdivmod__(self, other):
        return divmod(other, self._array)

    @to_new_array
    @unwrap_array
    def __truediv__(self, other):
        return self._array / other

    @to_new_array
    @unwrap_array
    def __rtruediv__(self, other):
        return other / self._array

    @to_new_array
    def __abs__(self):
        return abs(self._array)

    def sum(self, *args, **kwargs):
        return self._array.sum(*args, **kwargs)

    def __float__(self):
        return float(self._array)

    @property
    @abstractmethod
    def native(self) -> Structure:
        """
        Returns the data structure in its `native` format which contains all unmaksed values to the native dimensions.
        """

    @property
    def shape(self):
        try:
            return self._array.shape
        except AttributeError:
            return ()

    @property
    def size(self):
        return self._array.size

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def ndim(self):
        return self._array.ndim

    def max(self, *args, **kwargs):
        return self._array.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self._array.min(*args, **kwargs)

    @to_new_array
    def reshape(self, *args, **kwargs):
        return self._array.reshape(*args, **kwargs)

    def __getattr__(self, item):
        if item != "__setstate__":
            try:
                return getattr(self._array, item)
            except AttributeError:
                pass
        raise AttributeError(
            f"{self.__class__.__name__} does not have attribute {item}"
        )

    def __getitem__(self, item):

        result = self._array[item]

        if isinstance(item, slice):
            result = self.with_new_array(result)

        try:
            import jax.numpy as jnp

            if isinstance(result, jnp.ndarray):
                result = self.with_new_array(result)
        except ImportError:
            pass

        return result

    def __setitem__(self, key, value):

        if isinstance(self._array, np.ndarray):
            self._array[key] = value
        else:
            import jax.numpy as jnp

            self._array = jnp.where(key, value, self._array)

    def __repr__(self):
        return repr(self._array).replace(
            "array",
            self.__class__.__name__,
        )

    def __array__(self, dtype=None, copy=None):
        # numpy >=1.26 requires __array__ to return an actual np.ndarray
        # ("object __array__ method not producing an array" otherwise),
        # and numpy 2.0 added the `copy` kwarg. self._array can be a
        # jax.Array on the JAX path (set in `_array = jnp.where(...)` etc.)
        # — coerce to numpy so the protocol contract holds.
        arr = self._array
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy:
            arr = arr.copy()
        return arr

    def __len__(self):
        return len(self._array)

    @to_new_array
    def astype(self, dtype):
        return self._array.astype(dtype)

    @property
    @to_new_array
    def real(self):
        return self._array.real

    @property
    @to_new_array
    def imag(self):
        return self._array.imag

    def all(self):
        return self._array.all()
