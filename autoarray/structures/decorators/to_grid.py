from functools import wraps
import numpy as np
from typing import List, Union

from autoarray.structures.decorators.abstract import AbstractMaker
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class GridMaker(AbstractMaker):
    # The private ``_over_sampled`` / ``_over_sampler`` attributes are read here
    # instead of the public properties on purpose: when ``result`` is already a
    # ``Grid2D`` (e.g. the chained ``@to_grid`` methods of the spherical mass
    # profiles) touching the properties *materialises* them, and building the
    # over sampled grid runs a per-pixel Python loop that dominates the runtime
    # of an otherwise millisecond deflection calculation. Only a value that was
    # explicitly passed in by the caller should propagate to the new grid; a
    # ``None`` lets the new ``Grid2D`` compute its own lazily, if anything ever
    # asks for it.
    def via_grid_2d(self, result) -> Union[Grid2D, List[Grid2D]]:
        if not isinstance(result, list):
            return Grid2D(
                values=result,
                mask=self.mask,
                over_sample_size=self.over_sample_size,
                over_sampled=getattr(result, "_over_sampled", None),
                over_sampler=getattr(result, "_over_sampler", None),
            )

        return [
            Grid2D(
                values=res,
                mask=self.mask,
                over_sample_size=self.over_sample_size,
                over_sampled=getattr(res, "_over_sampled", None),
                over_sampler=getattr(res, "_over_sampler", None),
            )
            for res in result
        ]

    def via_grid_2d_irr(self, result) -> Union[Grid2DIrregular, List[Grid2DIrregular]]:
        if not isinstance(result, list):
            return Grid2DIrregular(values=result)
        return [Grid2DIrregular(values=res) for res in result]


def to_grid(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of coordinates and return a 2D ndarray
    which is converted to a `Grid2D` or `Grid2DIrregular` object.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of coordinates.

    Returns
    -------
        A function that has its outputs homogenized to `Grid2D` or `Grid2DIrregular` objects.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular],
        xp=np,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Grid2D, Grid2DIrregular, List]:
        return GridMaker(func=func, obj=obj, grid=grid, xp=xp, *args, **kwargs).result

    return wrapper
