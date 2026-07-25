from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D
    from autoarray.structures.grids.uniform_2d import Grid2D

import numpy as np

from autoarray.geometry.abstract_2d import AbstractGeometry2D

from autoarray import type as ty
from autoarray.geometry import geometry_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Geometry2D(AbstractGeometry2D):
    def __init__(
        self,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        A 2D geometry, representing a uniform rectangular grid of (y,x) coordinates which has ``shape_native``.

        This class is used for converting coordinates from pixel-units to scaled coordinates via
        the geometry's (y,x) ``pixel_scales`` conversion factor and its (y,x) ``origin``.

        Parameters
        ----------
        shape_native
            The 2D shape of the array in its ``native`` format (and its 2D mask) whose 2D geometry this object
            represents.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        self.shape_native = shape_native
        self.pixel_scales = pixel_scales
        self.origin = origin

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The (y,x) 2D native shape of the geometry in scaled units.

        This is computed by multiplying the 2D ``shape_native`` (units ``pixels``) with
        the ``pixel_scales`` (units ``scaled/pixels``) conversion factor.
        """
        return (
            float(self.pixel_scales[0] * self.shape_native[0]),
            float(self.pixel_scales[1] * self.shape_native[1]),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        """
        The maximum (y,x) scaled coordinates of the 2D geometry.

        For example, if the geometry's most positive scaled y value is 10.0 and most positive scaled x value is
        20.0, this returns (10.0, 20.0).
        """
        return (
            (self.shape_native_scaled[0] / 2.0) + self.origin[0],
            (self.shape_native_scaled[1] / 2.0) + self.origin[1],
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        """
        The minimum (y,x) scaled coordinates of the 2D geometry.

        For example, if the geometry's most negative scaled y value is -10.0 and most negative scaled x value is
        -20.0, this returns (-10.0, -20.0).
        """
        return (
            (-(self.shape_native_scaled[0] / 2.0)) + self.origin[0],
            (-(self.shape_native_scaled[1] / 2.0)) + self.origin[1],
        )

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """
        The extent of the geometry in scaled units, returned as a tuple (x_min, x_max, y_min, y_max).

        This format is identical to the ``extent`` input of the ``matplotlib`` method ``imshow`` (and other methods).
        It is used for visualization in the plot module, which is why the x and y coordinates are swapped compared to
        the normal convention.
        """
        return (
            self.scaled_minima[1],
            self.scaled_maxima[1],
            self.scaled_minima[0],
            self.scaled_maxima[0],
        )

    @property
    def central_pixel_coordinates(self) -> Tuple[float, float]:
        """
        Returns the central pixel coordinates of the 2D geometry (and therefore a 2D data structure
        like an ``Array2D``) from the shape of that data structure.

        Examples of the central pixels are as follows:

        - For a 3x3 image, the central pixel is pixel [1, 1].
        - For a 4x4 image, the central pixel is [1.5, 1.5].
        """
        return geometry_util.central_pixel_coordinates_2d_from(
            shape_native=self.shape_native
        )

    @property
    def central_scaled_coordinates(self) -> Tuple[float, float]:
        """
        The central (y,x) coordinates of the 2D geometry in scaled units.

        Computed from the shape and pixel scales of the geometry, shifted by the origin. This is
        the scaled-unit coordinate that sits at the geometric centre of the grid.

        Returns
        -------
        Tuple[float, float]
            The (y,x) scaled coordinates of the geometric centre.
        """
        return geometry_util.central_scaled_coordinate_2d_from(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    def pixel_coordinates_2d_from(
        self, scaled_coordinates_2d: Tuple[float, float]
    ) -> Tuple[int, int]:
        """
        Convert a 2D (y,x) scaled coordinate to a 2D (y,x) pixel coordinate, returned as integers that
        identify the pixel containing the scaled coordinate (without sub-pixel decimal offsets).

        The conversion is performed according to the 2D geometry on a uniform grid, where the pixel
        coordinate origin is at the top-left corner, such that pixel [0,0] corresponds to the highest
        (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the grid.

        The geometry's origin is applied so that scaled coordinates are correctly shifted before the
        pixel index calculation.

        Parameters
        ----------
        scaled_coordinates_2d
            The 2D (y,x) coordinates in scaled units to convert to pixel coordinates.

        Returns
        -------
        Tuple[int, int]
            The 2D (y,x) integer pixel coordinates.
        """
        return geometry_util.pixel_coordinates_2d_from(
            scaled_coordinates_2d=scaled_coordinates_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    def scaled_coordinates_2d_from(
        self, pixel_coordinates_2d: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Convert a 2D (y,x) pixel coordinate to a 2D (y,x) scaled coordinate.

        The conversion is performed according to the 2D geometry on a uniform grid, where the pixel
        coordinate origin is at the top-left corner, such that pixel [0,0] corresponds to the highest
        (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the grid.

        The geometry's origin is applied so that the returned scaled coordinate is correctly shifted
        relative to the coordinate system's centre.

        Parameters
        ----------
        pixel_coordinates_2d
            The 2D (y,x) pixel coordinates to convert to scaled units.

        Returns
        -------
        Tuple[float, float]
            The 2D (y,x) coordinates in scaled units.
        """

        return geometry_util.scaled_coordinates_2d_from(
            pixel_coordinates_2d=pixel_coordinates_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    def pixel_coordinates_wcs_2d_from(
        self, scaled_coordinates_2d: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Convert a 2D (y,x) scaled coordinate to a 2D (y,x) WCS/FITS-style pixel coordinate.

        The returned pixel coordinates follow the standard WCS convention:

        - Coordinates are 1-based rather than 0-based, so that the centre of the top-left pixel is at (y, x) = (1.0, 1.0).
        - Coordinates refer to pixel centres, not pixel corners.
        - Values are continuous floats, so the fractional part encodes the sub-pixel offset from the pixel centre.

        This differs from integer pixel-index conversions (e.g. ``pixel_coordinates_2d_from``), which return 0-based
        indices associated with pixel corners/top-left positions.

        The mapping from scaled coordinates to WCS pixel coordinates is defined by this geometry's ``origin``: scaled
        coordinates are first shifted by the specified origin(s) before being converted using the pixel scale and
        array shape. Changing ``origin`` therefore translates the returned WCS pixel coordinates by a constant offset.

        Parameters
        ----------
        scaled_coordinates_2d
            The 2D (y,x) coordinates in scaled units to be converted to WCS-style pixel coordinates.

        Returns
        -------
        A 2D (y,x) WCS pixel coordinate, expressed as 1-based, pixel-centre, floating-point values.
        """
        return geometry_util.pixel_coordinates_wcs_2d_from(
            scaled_coordinates_2d=scaled_coordinates_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    def scaled_coordinate_2d_to_scaled_at_pixel_centre_from(
        self, scaled_coordinate_2d: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Convert a 2D (y,x) scaled coordinate to a 2D scaled coordinate at the centre of the pixel it is located in.

        This is performed by converting the scaled coordinate to a pixel coordinate and then back to a scaled coordinate.

        For example, if a scaled coordinate is (0.5, 0.5) and it falls within a pixel whose centre is at (0.75, 0.75),
        this function would return (0.75, 0.75).

        Parameters
        ----------
        scaled_coordinate_2d
            The 2D (y,x) coordinates in scaled units which are converted to pixel coordinates.

        Returns
        -------
        The 2D (y,x) pixel-value coordinate at the centre of the pixel the input scaled coordinate is located in.
        """

        pixel_coordinate_2d = self.pixel_coordinates_2d_from(
            scaled_coordinates_2d=scaled_coordinate_2d
        )
        return self.scaled_coordinates_2d_from(pixel_coordinates_2d=pixel_coordinate_2d)

    def grid_pixels_2d_from(self, grid_scaled_2d: Grid2D) -> Grid2D:
        """
        Convert a grid of 2D (y,x) scaled coordinates to a grid of 2D (y,x) pixel values, which are returned as floats
        and include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        highest y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_2d
            A grid of (y,x) coordinates in scaled units.

        Returns
        -------
        Grid2D
            A grid of (y,x) pixel-value coordinates as floats including the sub-pixel decimal offset.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_pixels_2d = geometry_util.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_2d.array,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return Grid2D(values=grid_pixels_2d, mask=grid_scaled_2d.mask)

    def grid_pixel_centres_2d_from(self, grid_scaled_2d: Grid2D) -> Grid2D:
        """
        Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel values, which are
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        highest y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_2d
            The grid of (y,x) coordinates in scaled units.

        Returns
        -------
        Grid2D
            A grid of (y,x) integer pixel coordinates identifying which pixel each scaled coordinate
            falls within.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_pixel_centres_1d = geometry_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

        return Grid2D(values=grid_pixel_centres_1d, mask=grid_scaled_2d.mask)

    def grid_pixel_indexes_2d_from(self, grid_scaled_2d: Grid2D) -> Array2D:
        """
        Convert a grid of (y,x) scaled coordinates to an array of pixel 1D indexes, which are returned as integers.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        highest y scaled coordinate value and lowest x scaled coordinate.

        For example:

        - The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
        - The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
        - The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_2d
            The grid of (y,x) coordinates in scaled units.

        Returns
        -------
        Array2D
            A 1D array of flat pixel indexes, where each value is the integer index of the pixel
            (counting left-to-right, top-to-bottom) that the corresponding scaled coordinate falls in.
        """

        from autoarray.structures.arrays.uniform_2d import Array2D

        grid_pixel_indexes_2d = geometry_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        ).astype("int")

        return Array2D(values=grid_pixel_indexes_2d, mask=grid_scaled_2d.mask)

    def grid_scaled_2d_from(self, grid_pixels_2d: Grid2D) -> Grid2D:
        """
        Convert a grid of (y,x) pixel coordinates to a grid of (y,x) scaled values.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        highest y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_pixels_2d
            The grid of (y,x) coordinates in pixels.

        Returns
        -------
        Grid2D
            A grid of (y,x) coordinates in scaled units.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_scaled_1d = geometry_util.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return Grid2D(values=grid_scaled_1d, mask=grid_pixels_2d.mask)
