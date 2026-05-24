import numpy as np

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class AbstractMaker:
    def __init__(self, func, obj, grid, xp=np, *args, **kwargs):
        """
        Makes 2D data structures from an input function and grid, ensuring that the structure of the input grid is
        paired to the structure of the output data structure.

        This is used by the `to_array`, `to_grid` and `to_vector_yx` decorators to ensure that the input grid and output
        data structure are consistent.

        There are two types of consistent data structures and therefore decorated function mappings:

        - Uniform: 2D structures defined on a uniform grid of data points, for example the `Array2D` and `Grid2D`
        objects. Both structures are defined according to a `Mask2D`, which the maker object ensures is passed through
        self consistently.

        - Irregular: 2D structures defined on an irregular grid of data points, for example an `ArrayIrregular`
        and `Grid2DIrregular` objects. Neither structure is defined according to a mask and the maker ensures the lack of
        a mask does not prevent the function from being evaluated.

        Parameters
        ----------
        func
            The function that is being decorated, which the maker object evaluates to create the output data structure.
        obj
            The object that the function is a method of, which is passed to the function when it is evaluated. This
            is typically the self object of the class the function is a method of.
        grid
            The grid that is passed to the function when it is evaluated, which is used to evaluate the function.
            For example, this might be the `Grid2D` object that the function checks the typing of when determining
            the output data structure.
        args
            Any arguments that are passed to the function when it is evaluated.
        kwargs
            Any keyword arguments that are passed to the function when it is evaluated.
        """

        # JAX-pairing-rule guard: if the caller declared xp=np but handed in a
        # jnp-backed grid, that's almost always a user error inside an
        # ``@jax.jit`` body — they forgot to pass ``xp=jnp`` to the library
        # method. The function body would silently host-transfer the tracer
        # through ``__array__()`` (slow) or fail at the JIT boundary because
        # the ``if xp is np:`` guard would wrap the result in an ``aa.Array2D``.
        # Raise here so the user discovers the pairing rule the first time
        # they break it. See ``admin_jammy/notes/jax_interface.md`` §4.8.
        if xp is np and getattr(grid, "use_jax", False):
            raise ValueError(
                f"Called {getattr(func, '__qualname__', repr(func))} with "
                f"xp=np but the input grid is JAX-backed (grid.use_jax=True). "
                f"Inside @jax.jit, pass xp=jnp explicitly to the library call. "
                f"See the autolens_workspace `scripts/guides/lens_calc.py` "
                f"`__JAX__` section (Phase 5d) for the JIT-it-yourself pattern."
            )

        self.func = func
        self.obj = obj
        self.grid = grid
        self.args = args
        self.kwargs = kwargs

        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    @property
    def mask(self) -> Mask2D:
        return self.grid.mask

    @property
    def over_sample_size(self) -> np.ndarray:
        return self.grid.over_sample_size

    def via_grid_2d(self, result):
        raise NotImplementedError

    def via_grid_2d_irr(self, result):
        raise NotImplementedError

    @property
    def evaluate_func(self):
        return self.func(self.obj, self.grid, self._xp, *self.args, **self.kwargs)

    @property
    def result(self):
        """
        The result of the function that is being decorated, which this function converts to the output data structure
        that is consistent with the input grid.

        This function calls one of two methods, depending on the type of the input grid:

        - `via_grid_2d`: If the input grid is a `Grid2D` object.
        - `via_grid_2d_irr`: If the input grid is a `Grid2DIrregular` object.

        If the input is a raw ndarray (e.g. numpy or JAX), the function result is returned unchanged.
        """

        if isinstance(self.grid, Grid2D):
            return self.via_grid_2d(self.evaluate_func)
        elif isinstance(self.grid, Grid2DIrregular):
            return self.via_grid_2d_irr(self.evaluate_func)

        return self.evaluate_func
