from typing import Tuple

from autoarray.geometry.abstract_2d import AbstractGeometry2D


class Geometry2DIrregular(AbstractGeometry2D):
    def __init__(
        self,
        shape_native_scaled: Tuple[float, float],
        scaled_maxima: Tuple[float, float],
        scaled_minima: Tuple[float, float],
    ):
        """
        A 2D geometry for an irregular (non-uniform) grid of (y,x) coordinates.

        Unlike `Geometry2D` which is derived from a regular pixel grid with a fixed `shape_native`
        and `pixel_scales`, this class stores pre-computed extent information directly. It is used
        by `Grid2DIrregular` and similar structures to define their bounding box for visualization
        and coordinate-space calculations.

        Because the coordinates are irregular there is no uniform pixel scale, so the geometry
        is defined purely by the scaled extent (minima, maxima) of the point distribution.

        Parameters
        ----------
        shape_native_scaled
            The (y, x) extent of the geometry in scaled units, i.e.
            (scaled_maxima[0] - scaled_minima[0], scaled_maxima[1] - scaled_minima[1]).
        scaled_maxima
            The maximum (y,x) scaled coordinates of the irregular grid.
        scaled_minima
            The minimum (y,x) scaled coordinates of the irregular grid.
        """
        self.shape_native_scaled = shape_native_scaled
        self.scaled_maxima = scaled_maxima
        self.scaled_minima = scaled_minima

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """
        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module, which is why the x and y coordinates are swapped compared to
        the normal PyAutoArray convention.
        """
        return (
            self.scaled_minima[1],
            self.scaled_maxima[1],
            self.scaled_minima[0],
            self.scaled_maxima[0],
        )
