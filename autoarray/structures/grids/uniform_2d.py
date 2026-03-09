from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

from autoconf import conf
from autoconf.fitsable import ndarray_via_fits_from

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util
from autoarray.operators.over_sampling import over_sample_util

from autoarray import exc
from autoarray import type as ty


class Grid2D(Structure):
    def __init__(
        self,
        values: Union[np.ndarray, List],
        mask: Mask2D,
        store_native: bool = False,
        over_sample_size: Union[int, Array2D] = 4,
        over_sampled: Optional[Grid2D] = None,
        over_sampler=None,
        xp=np,
        *args,
        **kwargs,
    ):
        """
        A grid of 2D (y,x) coordinates, which are paired to a uniform 2D mask of pixels. Each entry
        on the grid corresponds to the (y,x) coordinates at the centre of a pixel of an unmasked pixel.

        A `Grid2D` is ordered such that pixels begin from the top-row (e.g. index [0, 0]) of the corresponding mask
        and go right and down. The positive y-axis is upwards and positive x-axis to the right.

        The grid can be stored in two formats:

        - slimmed: all masked entries are removed so the ndarray is shape [total_unmasked_coordinates**2, 2]
        - native: it retains the original shape of the grid so the ndarray is
          shape [total_y_coordinates, total_x_coordinates, 2].


        __Slim__

        The Grid2D is an ndarray of shape [total_unmasked_coordinates, 2], therefore when `slim` the shape of
        the grid is 2, not 1.

        The first element of the ndarray corresponds to the pixel index and second element the y or x coordinate value.

        For example:

        - grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 10 pixels are unmasked and are included in \
        the grid.

        .. code-block:: bash

             x x x x x x x x x x
             x x x x x x x x x x     This is an example mask.Mask2D, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the grid)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the grid)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        The mask pixel index's will come out like this (and the direction of scaled coordinates is highlighted
        around the mask.

        .. code-block:: bash

            pixel_scales = 1.0"

            <--- -ve  x  +ve -->
                                                            y      x
             x x x x x x x x x x  ^   grid[0] = [ 1.5, -0.5]
             x x x x x x x x x x  I   grid[1] = [ 1.5,  0.5]
             x x x x x x x x x x  I   grid[2] = [ 0.5, -1.5]
             x x x x 0 1 x x x x +ve  grid[3] = [ 0.5, -0.5]
             x x x 2 3 4 5 x x x  y   grid[4] = [ 0.5,  0.5]
             x x x 6 7 8 9 x x x -ve  grid[5] = [ 0.5,  1.5]
             x x x x x x x x x x  I   grid[6] = [-0.5, -1.5]
             x x x x x x x x x x  I   grid[7] = [-0.5, -0.5]
             x x x x x x x x x x \/   grid[8] = [-0.5,  0.5]
             x x x x x x x x x x      grid[9] = [-0.5,  1.5]

        __native__

        The Grid2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_coordinates, total_x_coordinates, 2]. Therefore when `native` the shape of the
        grid is 3, not 2.

        All masked entries on the grid has (y,x) values of (0.0, 0.0).

        For the following example mask:

        .. code-block:: bash

             x x x x x x x x x x
             x x x x x x x x x x     This is an example mask.Mask2D, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the grid)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the grid)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        In the above grid:

            - grid[0,0,0] = 0.0 (it is masked, thus zero)
            - grid[0,0,1] = 0.0 (it is masked, thus zero)
            - grid[3,3,0] = 0.0 (it is masked, thus zero)
            - grid[3,3,1] = 0.0 (it is masked, thus zero)
            - grid[3,4,0] = 1.5
            - grid[3,4,1] = -0.5

        **Grid2D Mapping:**

        Every set of (y,x) coordinates in a pixel of the grid maps to an unmasked pixel in the mask. For a uniform
        grid, every (y,x) coordinate directly corresponds to the location of its paired unmasked pixel.

        It is not a requirement that grid is uniform and that their coordinates align with the mask. The input grid
        could be an irregular set of (y,x) coordinates where the indexing signifies that the (y,x) coordinate
        *originates* or *is paired with* the mask's pixels but has had its value change by some aspect of the
        calculation.

        This is important for the child project *PyAutoLens*, where grids in the image-plane are ray-traced and
        deflected to perform lensing calculations. The grid indexing is used to map pixels between the image-plane and
        source-plane.

        Parameters
        ----------
        values
            The (y,x) coordinates of the grid.
        mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_native
            If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels, 2]. This avoids
            mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
        over_sample_size
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sampled
            The over sampled grid of (y,x) coordinates, which can be passed in manually because if the grid is
            not uniform (e.g. due to gravitational lensing) is cannot be computed internally in this function. If the
            over sampled grid is not passed in it is computed assuming uniformity.
        """

        values = grid_2d_util.convert_grid_2d(
            grid_2d=values, mask_2d=mask, store_native=store_native, xp=xp
        )

        super().__init__(values, xp=xp)

        self.mask = mask

        grid_2d_util.check_grid_2d(grid_2d=values)

        self._over_sample_size = over_sample_size

        self._over_sampler = over_sampler
        self._over_sampled = over_sampled

    @property
    def over_sample_size(self):

        if isinstance(self._over_sample_size, Array2D):
            return self._over_sample_size

        self._over_sample_size = (
            over_sample_util.over_sample_size_convert_to_array_2d_from(
                over_sample_size=self._over_sample_size, mask=self.mask
            )
        )

        return self._over_sample_size

    @property
    def over_sampler(self):

        if self._over_sampler is not None:
            return self._over_sampler

        from autoarray.operators.over_sampling.over_sampler import OverSampler

        self._over_sampler = OverSampler(
            sub_size=self.over_sample_size.array.astype("int"), mask=self.mask
        )

        return self._over_sampler

    @property
    def over_sampled(self):

        if self._over_sampled is not None:
            return self._over_sampled

        over_sampled = over_sample_util.grid_2d_slim_over_sampled_via_mask_from(
            mask_2d=np.array(self.mask),
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.over_sample_size.array.astype("int"),
            origin=self.mask.origin,
        )

        self._over_sampled = Grid2DIrregular(values=over_sampled)

        return self._over_sampled

    @classmethod
    def no_mask(
        cls,
        values: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        shape_native: Tuple[int, int] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 1D or 2D, automatically
        determining whether to use the 'manual_slim' or 'manual_native' methods.

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        values
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixels, 2]
            or a list of lists.
        shape_native
            The 2D shape of the mask the grid is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The origin of the grid's mask.
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        values = grid_2d_util.convert_grid(grid=values)

        if len(values.shape) == 2:
            grid_2d_util.check_grid_slim(grid=values, shape_native=shape_native)
        else:
            shape_native = (
                int(values.shape[0]),
                int(values.shape[1]),
            )

        mask = Mask2D.all_false(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return Grid2D(
            values=np.array(values),
            mask=mask,
            over_sample_size=over_sample_size,
        )

    @classmethod
    def from_yx_1d(
        cls,
        y: Union[np.ndarray, List],
        x: np.ndarray,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates as 1D y and x values.

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        y or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        shape_native
            The 2D shape of the mask the grid is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The origin of the grid's mask.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            # Make Grid2D from input np.ndarray.

            grid_2d = aa.Grid2D.from_yx_1d(
                y=np.array([1.0, 3.0, 5.0, 7.0]),
                x=np.array([2.0, 4.0, 6.0, 8.0]),
                shape_native=(2, 2),
                pixel_scales=1.0,
            )

            # Make Grid2D from input list.

           grid_2d = aa.Grid2D.from_yx_1d(
                y=[1.0, 3.0, 5.0, 7.0],
                x=[2.0, 4.0, 6.0, 8.0],
                shape_native=(2, 2),
                pixel_scales=1.0,
            )

            # Print grid's slim (masked 1D data representation) and
            # native (masked 2D data representation)

            print(grid_2d.slim)
            print(grid_2d.native)
        """
        if type(y) is list:
            y = np.asarray(y)

        if type(x) is list:
            x = np.asarray(x)

        return cls.no_mask(
            values=np.stack((y, x), axis=-1),
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            over_sample_size=over_sample_size,
        )

    @classmethod
    def from_yx_2d(
        cls,
        y: Union[np.ndarray, List],
        x: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates as 2D y and x values.

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        y or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The origin of the grid's mask.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            # Make Grid2D from input list(s).

            grid_2d = aa.Grid2D.from_yx_2d(
                y=[[1.0], [3.0]],
                x=[[2.0], [4.0]],
                pixel_scales=1.0
            )
        """
        if type(y) is list:
            y = np.asarray(y)

        if type(x) is list:
            x = np.asarray(x)

        return cls.no_mask(
            values=np.stack((y, x), axis=-1),
            pixel_scales=pixel_scales,
            origin=origin,
            over_sample_size=over_sample_size,
        )

    @classmethod
    def from_extent(
        cls,
        extent: Tuple[float, float, float, float],
        shape_native: Tuple[int, int],
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the extent of the (y,x) grid coordinates as an input
        (x0, x1, y0, y1) tuple.

        The (y,x) `shape_native` in pixels is also input which determines the resolution of the `Grid2D`.

        (The **PyAutoArray** API typically uses a (y,x) notation, however extent variables begin with x currently.
        This will be updated in a future release):

        extent = (x0, x1, y0, y1) = (2.0, 4.0, -2.0, 6.0)
        shape_native = (y,x) = (10, 20)

        Parameters
        ----------
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created.
        shape_native
            The 2D shape of the grid that is created within this extent.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The origin of the grid's mask.
        """

        x0, x1, y0, y1 = extent

        ys = np.linspace(y1, y0, shape_native[0])
        xs = np.linspace(x0, x1, shape_native[1])

        xs_grid, ys_grid = np.meshgrid(xs, ys)

        xs_grid_1d = xs_grid.ravel()
        ys_grid_1d = ys_grid.ravel()

        grid_2d = np.vstack((ys_grid_1d, xs_grid_1d)).T

        grid_2d = grid_2d.reshape((shape_native[0], shape_native[1], 2))

        pixel_scales = (
            abs(grid_2d[0, 0, 0] - grid_2d[1, 0, 0]),
            abs(grid_2d[0, 0, 1] - grid_2d[0, 1, 1]),
        )

        return Grid2D.no_mask(
            values=grid_2d,
            pixel_scales=pixel_scales,
            over_sample_size=over_sample_size,
        )

    @classmethod
    def uniform(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Create a `Grid2D` (see *Grid2D.__new__*) as a uniform grid of (y,x) values given an input `shape_native` and
        `pixel_scales` of the grid:

        Parameters
        ----------
        shape_native
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) tuple.
        origin
            The origin of the grid's mask.
        """
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return cls.no_mask(
            values=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            over_sample_size=over_sample_size,
        )

    @classmethod
    def bounding_box(
        cls,
        bounding_box: np.ndarray,
        shape_native: Tuple[int, int],
        buffer_around_corners: bool = False,
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) from an input bounding box with coordinates [y_min, y_max, x_min, x_max],
        where the shape_native is used to compute the (y,x) grid values within this bounding box.

        If buffer_around_corners=True, the grid's (y,x) values fully align with the input bounding box values. This
        means the mask's edge pixels extend beyond the bounding box by pixel_scale/2.0. If buffer_around_corners=False,
        the grid (y,x) coordinates are defined within the bounding box such that the mask's edge pixels align with
        the bouning box.

        Parameters
        ----------
        shape_native
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The origin of the grid's mask.
        buffer_around_corners
            Whether the grid is buffered such that the (y,x) values in the centre of its masks' edge pixels align
            with the input bounding box values.
        """
        y_min, y_max, x_min, x_max = bounding_box

        if not buffer_around_corners:
            pixel_scales = (
                (y_max - y_min) / (shape_native[0]),
                (x_max - x_min) / (shape_native[1]),
            )

        else:
            pixel_scales = (
                (y_max - y_min) / (shape_native[0] - 1),
                (x_max - x_min) / (shape_native[1] - 1),
            )
        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

        return cls.uniform(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            over_sample_size=over_sample_size,
        )

    @classmethod
    def from_mask(
        cls,
        mask: Mask2D,
        over_sample_size: Union[int, Array2D] = 4,
        xp=np,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in its native 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask
            The mask whose masked pixels are used to setup the grid.
        """

        grid_2d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=mask.array,
            pixel_scales=mask.pixel_scales,
            origin=mask.origin,
            xp=xp,
        )

        return Grid2D(
            values=grid_2d, mask=mask, over_sample_size=over_sample_size, xp=xp
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in its native 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask
            The mask whose masked pixels are used to setup the grid.
        """

        grid_2d = ndarray_via_fits_from(file_path=file_path, hdu=0)

        return Grid2D.no_mask(
            values=grid_2d,
            pixel_scales=pixel_scales,
            origin=origin,
            over_sample_size=over_sample_size,
        )

    @classmethod
    def blurring_grid_from(
        cls,
        mask: Mask2D,
        kernel_shape_native: Tuple[int, int],
        over_sample_size: Union[int, Array2D] = 4,
    ) -> "Grid2D":
        """
        Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked (and
        therefore have their values set to (0.0, 0.0)), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This when computing images from
        light profile objects.

        The mask's pixel_scales and origin properties are used to compute the blurring grid's (y,x)
        coordinates.

        For example, if our mask is as follows:

        .. code-block:: bash

             x x x x x x x x x xI
             x x x x x x x x x xI     This is an imaging.Mask2D, where
             x x x x x x x x x xI
             x x x x x x x x x xI     x = `True` (Pixel is masked and excluded from lens)
             x x x O O O x x x xI     O = `False` (Pixel is not masked and included in lens)
             x x x O O O x x x xI
             x x x O O O x x x xI
             x x x x x x x x x xI
             x x x x x x x x x xI
             x x x x x x x x x xI

        For a PSF of shape (3,3), the following blurring mask is computed (noting that only pixels that are direct
        neighbors of the unmasked pixels above will blur light into an unmasked pixel)

        .. code-block:: bash

             x x x x x x x x xI     This is an example grid.Mask2D, where
             x x x x x x x x xI
             x x O O O O O x xI     x = `True` (Pixel is masked and excluded from lens)
             x x O x x x O x xI     O = `False` (Pixel is not masked and included in lens)
             x x O x x x O x xI
             x x O x x x O x xI
             x x O O O O O x xI
             x x x x x x x x xI
             x x x x x x x x xI

        Thus, the blurring grid coordinates and indexes will be as follows

        .. code-block:: bash

            pixel_scales = 1.0"

            positive    negative
                                                                y     x                          y     x
             x x x  x  x  x  x  x xI  I   blurring_grid[0] = [2.0, -2.0]  blurring_grid[9] =  [-1.0, -2.0]
             x x x  x  x  x  x  x xI  I   blurring_grid[1] = [2.0, -1.0]  blurring_grid[10] = [-1.0,  2.0]
             x xI0 I1 I2 I3 I4  x xI pos  blurring_grid[2] = [2.0,  0.0]  blurring_grid[11] = [-2.0, -2.0]
             x xI5  x  x  x I6  x xI  y   blurring_grid[3] = [2.0,  1.0]  blurring_grid[12] = [-2.0, -1.0]
             x xI7  x  x  x I8  x xI  I   blurring_grid[4] = [2.0,  2.0]  blurring_grid[13] = [-2.0,  0.0]
             x xI9  x  x  x I10 x xI neg  blurring_grid[5] = [1.0, -2.0]  blurring_grid[14] = [-2.0,  1.0]
             x xI11I12I13I14I15 x xI  I   blurring_grid[6] = [1.0,  2.0]  blurring_grid[15] = [-2.0,  2.0]
             x x x  x  x  x  x  x xI  I   blurring_grid[7] = [0.0, -2.0]
             x x x  x  x  x  x  x xI  I   blurring_grid[8] = [0.0,  2.0]

        For a PSF of shape (5,5), the following blurring mask is computed (noting that pixels are 2 pixels from a
        direct unmasked pixels now blur light into an unmasked pixel)

        .. code-block:: bash

             x x x x x x x x xI     This is an example grid.Mask2D, where
             xIoIoIoIoIoIoIo xI
             xIoIoIoIoIoIoIo xI     x = `True` (Pixel is masked and excluded from lens)
             xIoIo x x xIoIo xI     O = `False` (Pixel is not masked and included in lens)
             xIoIo x x xIoIo xI
             xIoIo x x xIoIo xI
             xIoIoIoIoIoIoIo xI
             xIoIoIoIoIoIoIo xI
             x x x x x x x x xI

        Parameters
        ----------
        mask
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        blurring_mask = mask.derive_mask.blurring_from(
            kernel_shape_native=kernel_shape_native
        )

        return cls.from_mask(
            mask=blurring_mask,
            over_sample_size=over_sample_size,
        )

    def subtracted_from(
        self, offset: Tuple[(float, float), np.ndarray], xp=np
    ) -> "Grid2D":

        mask = Mask2D(
            mask=self.mask,
            pixel_scales=self.pixel_scales,
            origin=(self.origin[0] - offset[0], self.origin[1] - offset[1]),
        )

        return Grid2D(
            values=self - xp.array(offset),
            mask=mask,
            over_sample_size=self.over_sample_size,
            over_sampled=self.over_sampled - xp.array(offset),
            over_sampler=self.over_sampler,
        )
    
    def subtracted_and_rotated_from(
        self, offset: Tuple[float, float], angle: float, xp=np
    ) -> Grid2D:
        """
        Returns a new Grid2D from this grid, where the (y,x) coordinates of this grid have an offset subtracted from
        them and are then rotated by an input angle.

        1) Subtract offset: (y', x') = (y - offset_y, x - offset_x)
        2) Rotate: y'' = y' * cos(angle) - x' * sin(angle)
                   x'' = y' * sin(angle) + x' * cos(angle)

        Parameters
        ----------
        offset
            The (y,x) offset which is subtracted from the grid.
        angle
            The angle (in radians) to rotate the grid.
        """

        mask = Mask2D(
            mask=self.mask,
            pixel_scales=self.pixel_scales,
            origin=(self.origin[0] - offset[0], self.origin[1] - offset[1]),
        )
        # 1. Subtracted from
        offset_array = xp.array(offset)

        grid_subtracted = (self - offset_array).array
        #rint(dir(grid_subtracted))
        #grid_subtracted = xp.array(grid_subtracted)

        over_sampled_subtracted = (self.over_sampled - offset_array).array
        #over_sampled_subtracted = xp.array(over_sampled_subtracted)
        
        # 2. Rotate
        angle_rad = xp.deg2rad(angle)
        cos_angle = xp.cos(angle_rad)
        sin_angle = xp.sin(angle_rad)

        # Note: We manually extract columns and compute to avoid creating large rotation matrices
        grid_y = grid_subtracted[:, 0]
        grid_x = grid_subtracted[:, 1]

        grid_rotated_x = grid_x * cos_angle - grid_y * sin_angle
        grid_rotated_y = grid_x * sin_angle + grid_y * cos_angle
        
        grid_rotated = xp.stack((grid_rotated_y, grid_rotated_x), axis=-1)

        # Also rotate the over_sampled grid
        over_sub_y = over_sampled_subtracted[:, 0]
        over_sub_x = over_sampled_subtracted[:, 1]

        over_rotated_x = over_sub_x * cos_angle - over_sub_y * sin_angle
        over_rotated_y = over_sub_x * sin_angle + over_sub_y * cos_angle

        over_sampled_rotated = xp.stack((over_rotated_y, over_rotated_x), axis=-1)

        return Grid2D(
            values=grid_rotated,
            mask=mask,
            over_sample_size=self.over_sample_size,
            over_sampled=Grid2DIrregular(over_sampled_rotated),
            over_sampler=self.over_sampler,
            xp=xp
        )

    @property
    def slim(self) -> "Grid2D":
        """
        Return a `Grid2D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels, 2].

        If it is already stored in its `slim` representation  it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Grid2D`.
        """
        return Grid2D(
            values=self,
            mask=self.mask,
            over_sample_size=self.over_sample_size,
        )

    @property
    def native(self) -> "Grid2D":
        """
        Return a `Grid2D` where the data is stored in its `native` representation, which has shape
        [total_y_pixels, total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid2D`.

        This method is used in the child `Grid2D` classes to create their `native` properties.
        """
        return Grid2D(
            values=self,
            mask=self.mask,
            over_sample_size=self.over_sample_size,
            store_native=True,
        )

    @property
    def flipped(self) -> "Grid2D":
        """
        Return the grid as an ndarray of shape [total_unmasked_pixels, 2] with flipped values such that coordinates
        are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format.
        """
        return self.with_new_array(np.fliplr(self.array))

    @property
    def in_radians(self) -> "Grid2D":
        """
        Return the grid as an ndarray where all (y,x) values are converted to Radians.

        This grid is used by the interferometer module.
        """
        return (self * np.pi) / 648000.0

    def grid_2d_via_deflection_grid_from(self, deflection_grid: "Grid2D") -> "Grid2D":
        """
        Returns a new Grid2D from this grid, where the (y,x) coordinates of this grid have a grid of (y,x) values,
        termed the deflection grid, subtracted from them to determine the new grid of (y,x) values.

        This is used by PyAutoLens to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid2D(
            values=self - deflection_grid,
            mask=self.mask,
            over_sample_size=self.over_sample_size,
        )

    def blurring_grid_via_kernel_shape_from(
        self, kernel_shape_native: Tuple[int, int]
    ) -> "Grid2D":
        """
        Returns the blurring grid from a grid, via an input 2D kernel shape.

        For a full description of blurring grids, checkout *blurring_grid_from*.

        Parameters
        ----------
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return Grid2D.blurring_grid_from(
            mask=self.mask,
            kernel_shape_native=kernel_shape_native,
            over_sample_size=1,
        )

    def grid_with_coordinates_within_distance_removed_from(
        self, coordinates: Union[np.ndarray, List], distance: float
    ) -> "Grid2D":
        """Remove all coordinates from this Grid2D which are within a certain distance of an input list of coordinates.

        For example, if the grid has the coordinate (0.0, 0.0) and coordinates=[(0.0, 0.0)], distance=0.1 is input into
        this function, a new Grid2D will be created which removes the coordinate (0.0, 0.0).

        Parameters
        ----------
        coordinates : [(float, float)]
            The list of coordinates which are removed from the grid if they are within the distance threshold.
        distance
            The distance threshold that coordinates are removed if they are within that of the input coordinates.
        """

        if not isinstance(coordinates, list):
            coordinates = [coordinates]

        distance_mask = np.full(fill_value=False, shape=self.shape_native)

        for coordinate in coordinates:
            distances = self.distances_to_coordinate_from(coordinate=coordinate)

            distance_mask += distances.native < distance

        mask = Mask2D(
            mask=np.array(distance_mask),
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

        return Grid2D.from_mask(
            mask=mask,
            over_sample_size=self.over_sample_size.apply_mask(mask=mask),
        )

    def squared_distances_to_coordinate_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> Array2D:
        """
        Returns the squared distance of every coordinate on the grid from an input coordinate.

        Parameters
        ----------
        coordinate
            The (y,x) coordinate from which the squared distance of every grid (y,x) coordinate is computed.
        """
        squared_distances = self._xp.square(
            self.array[:, 0] - coordinate[0]
        ) + self._xp.square(self.array[:, 1] - coordinate[1])

        return Array2D(values=squared_distances, mask=self.mask)

    def distances_to_coordinate_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> Array2D:
        """
        Returns the distance of every coordinate on the grid from an input (y,x) coordinate.

        Parameters
        ----------
        coordinate
            The (y,x) coordinate from which the distance of every grid (y,x) coordinate is computed.
        """
        squared_distance = self.squared_distances_to_coordinate_from(
            coordinate=coordinate
        )
        distances = self._xp.sqrt(squared_distance.array)
        return Array2D(values=distances, mask=self.mask)

    def grid_2d_radial_projected_shape_slim_from(
        self, centre: Tuple[float, float] = (0.0, 0.0)
    ) -> int:
        """
        The function `grid_scaled_2d_slim_radial_projected_from()` determines a projected radial grid of points from a
        2D region of coordinates defined by an extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

        To do this, the function first performs these 3 steps:

        1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance
           of the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and
           x axes).

        2) Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
           pixel_scale in the x dimension is used).

        3) Determine the number of pixels between the centre and the edge of the region using the longest path between
           the two chosen above.

        A schematic is shown below:

        .. code-block:: bash

            -------------------
            |                 |
            |<- - -  - ->x    | x = centre
            |                 | <-> = longest radial path from centre to extent edge
            |                 |
            -------------------

        Using the centre x above, this function finds the longest radial path to the edge of the extent window.

        This function returns the integer number of pixels given by this radial grid, which is then used to create
        the radial grid.

        Parameters
        ----------
        extent
            The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
        centre : (float, flloat)
            The (y,x) central coordinate which the radial grid is traced outwards from.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factor of the 2D mask array.

        Returns
        -------
        int
            The 1D integer shape of a radial set of points sampling the longest distance from the centre to the edge of the
            extent in along the positive x-axis.
        """

        return grid_2d_util._radial_projected_shape_slim_from(
            extent=self.geometry.extent,
            centre=centre,
            pixel_scales=self.mask.pixel_scales,
        )

    def grid_2d_radial_projected_from(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        shape_slim: Optional[int] = 0,
        remove_projected_centre: bool = False,
    ) -> Grid2DIrregular:
        """
        Determine a projected radial grid of points from a 2D region of coordinates defined by an
        extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

        This functions operates as follows:

        1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance
           of the 4 paths from the (y,x) centre to the edge of the region e.g. following the positive / negative y and
           x axes.

        2) Use the pixel-scale corresponding to the direction chosen e.g. if the positive x-axis was the longest, the
           pixel_scale in the x dimension is used.

        3) Determine the number of pixels between the centre and the edge of the region using the longest path between
           the two chosen above.

        4) Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values
           iterate from the centre in increasing steps of the pixel-scale.

        5) Rotate these radial coordinates by the input `angle` clockwise.

        A schematic is shown below:

        .. code-block:: bash

            -------------------
            |                 |
            |<- - -  - ->x    | x = centre
            |                 | <-> = longest radial path from centre to extent edge
            |                 |
            -------------------

        Parameters
        ----------
        centre
            The (y,x) central coordinate which the radial grid is traced outwards from.
        angle
            The angle with which the radial coordinates are rotated clockwise.

        Returns
        -------
        Grid2DIrregular
            A radial set of points sampling the longest distance from the centre to the edge of the extent in along the
            positive x-axis.
        """

        grid_radial_projected_2d = (
            grid_2d_util.grid_scaled_2d_slim_radial_projected_from(
                extent=self.geometry.extent,
                centre=centre,
                pixel_scales=self.mask.pixel_scales,
                shape_slim=shape_slim,
            )
        )

        grid_radial_projected_2d = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid_radial_projected_2d, centre=centre, angle=angle
        )

        grid_radial_projected_2d = geometry_util.transform_grid_2d_from_reference_frame(
            grid_2d=grid_radial_projected_2d, centre=centre, angle=0.0
        )

        if remove_projected_centre:
            grid_radial_projected_2d = grid_radial_projected_2d[1:, :]

        return Grid2DIrregular(values=grid_radial_projected_2d)

    @property
    def shape_native_scaled_interior(self) -> Tuple[float, float]:
        """
        The (y,x) interior 2D shape of the grid in scaled units, computed from the minimum and maximum y and x
        values of the grid.

        This differs from the `shape_native_scaled` because the edges of the shape are at the maxima and minima
        of the grid's (y,x) values, whereas the `shape_native_scaled` uses the uniform geometry of the grid and its
        ``pixel_scales``, which means it has a buffer at each edge of half a ``pixel_scale``.
        """
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def scaled_minima(self) -> Tuple:
        """
        The (y,x) minimum values of the grid in scaled units, buffed such that their extent is further than the grid's
        extent.
        """
        return (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

    @property
    def scaled_maxima(self) -> Tuple:
        """
        The (y,x) maximum values of the grid in scaled units, buffed such that their extent is further than the grid's
        extent.
        """
        return (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

    def extent_with_buffer_from(self, buffer: float = 1.0e-8) -> List[float]:
        """
        The extent of the grid in scaled units returned as a list [x_min, x_max, y_min, y_max], where all values are
        buffed such that their extent is further than the grid's extent..

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module.
        """
        return [
            self.scaled_minima[1] - buffer,
            self.scaled_maxima[1] + buffer,
            self.scaled_minima[0] - buffer,
            self.scaled_maxima[0] + buffer,
        ]

    def padded_grid_from(self, kernel_shape_native: Tuple[int, int]) -> "Grid2D":
        """
        When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will
        be 'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the padded grid is used, which is 'buffed' such that it includes all pixels
        whose signal will be convolved into the unmasked pixels given the 2D kernel shape.

        Parameters
        ----------
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        if kernel_shape_native[0] % 2 == 0 or kernel_shape_native[1] % 2 == 0:
            raise exc.KernelException("Convolver Convolver must be odd")

        shape = self.mask.shape

        padded_shape = (
            shape[0] + kernel_shape_native[0] - 1,
            shape[1] + kernel_shape_native[1] - 1,
        )

        padded_mask = Mask2D.all_false(
            shape_native=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.origin,
        )

        pad_width = (
            (padded_shape[0] - shape[0]) // 2,
            (padded_shape[1] - shape[1]) // 2,
        )

        over_sample_size = np.pad(
            self.over_sample_size.native.array,
            pad_width,
            mode="constant",
            constant_values=1,
        )

        over_sample_size[over_sample_size == 0] = 1

        return Grid2D.from_mask(mask=padded_mask, over_sample_size=over_sample_size)

    @property
    def is_uniform(self) -> bool:
        """
        Returns if the grid is uniform, where a uniform grid is defined as a grid where all pixels are separated by
        the same pixel-scale in both the y and x directions.

        The method does not check if the x coordinates are uniformly spaced, only the y coordinates, under the
        assumption that no calculation will be performed on a grid where the y coordinates are uniformly spaced but the
        x coordinates are not. If such a case arises, the method should be updated to check both the y and x coordinates.

        Returns
        -------
        Whether the grid is uniform.
        """

        y_diff = self[:, 0][:-1] - self[:, 0][1:]
        y_diff = y_diff[y_diff != 0]

        if any(abs(y_diff - self.pixel_scales[0]) > 1.0e-8):
            return False

        return True

    def apply_over_sampling(
        self, over_sample_size: Union[int, np.ndarray]
    ) -> "AbstractDataset":
        """
        Apply new over sampling to the grid.

        This method is used to change the over sampling of the grid, for example when the user wishes to perform
        over sampling with a higher sub grid size.

        Parameters
        ----------
        over_sample_size
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        """
        if not self.is_uniform:
            raise exc.GridException(
                """
                Cannot apply over sampling to a Grid2D which is not uniform.
                """
            )

        return Grid2D(
            values=self,
            mask=self.mask,
            over_sample_size=over_sample_size,
        )
