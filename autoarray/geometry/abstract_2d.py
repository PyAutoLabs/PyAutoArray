import numpy as np
from typing import Tuple


class AbstractGeometry2D:
    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """
        The extent of the 2D geometry in scaled units, returned as (x_min, x_max, y_min, y_max).

        This format matches the ``extent`` argument of ``matplotlib.pyplot.imshow``, with x and y
        swapped relative to the usual (y,x) PyAutoArray convention. Subclasses must implement this.
        """
        raise NotImplementedError

    @property
    def extent_square(self) -> Tuple[float, float, float, float]:
        """
        The extent of the 2D geometry in scaled units, expanded so that the y and x ranges are equal.

        This ensures that a uniform grid with square pixels can be laid over this extent, which is
        useful for constructing interpolation grids used in visualization. The centre of the extent
        is preserved; whichever axis has the smaller range is expanded symmetrically to match the
        larger one.

        Returns
        -------
        (x_min, x_max, y_min, y_max)
            The square extent in scaled units following the matplotlib ``imshow`` convention.
        """

        y_mean = 0.5 * (self.extent[2] + self.extent[3])
        y_half_length = 0.5 * (self.extent[3] - self.extent[2])

        x_mean = 0.5 * (self.extent[0] + self.extent[1])
        x_half_length = 0.5 * (self.extent[1] - self.extent[0])

        half_length = np.max([y_half_length, x_half_length])

        y0 = y_mean - half_length
        y1 = y_mean + half_length

        x0 = x_mean - half_length
        x1 = x_mean + half_length

        return (x0, x1, y0, y1)
