from __future__ import annotations
import numpy as np
from typing import List, Union, Tuple

from autoarray.structures.abstract_structure import Structure
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.mask.mask_1d import Mask1D

from autoarray.structures.grids import grid_1d_util
from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util
from autoarray import type as ty


class Grid1D(Structure):
    def __init__(
        self,
        values: Union[np.ndarray, List],
        mask: Mask1D,
        store_native: bool = False,
    ):
        """
        A grid of 1D (x) coordinates, which are paired to a uniform 1D mask of pixels. Each entry
        on the grid corresponds to the (x) coordinates at the centre of a pixel of an unmasked pixel.

        A `Grid1D` is ordered such that pixels begin from the left (e.g. index [0]) of the corresponding mask
        and go right. The positive x-axis is to the right.

        The grid can be stored in two formats:

        - slimmed: all masked entries are removed so the ndarray is shape [total_unmasked_coordinates, 2]
        - native: it retains the original shape of the grid so the ndarray is
          shape [total_unmasked_coordinates, 2].


        __slim__

        The Grid1D is an ndarray of shape [total_unmasked_coordinates].

        The first element of the ndarray corresponds to the pixel index. For example:

        - grid[3] = the 4th unmasked pixel's x-coordinate.
        - grid[6] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 3 pixels are unmasked and are included in the grid.

        .. code-block:: bash

            <--- -ve  x  +ve -->

            x x x O o x O x x x

        This is an example mask.Mask1D, where:

        .. code-block:: bash

            x = `True` (Pixel is masked and excluded from the grid)
            O = `False` (Pixel is not masked and included in the grid)

        The mask pixel index's will come out like this (and the direction of scaled coordinates is highlighted
        around the mask.

        .. code-block:: bash

            pixel_scales = 1.0"

            <--- -ve  x  +ve -->

            x x x 0 1 x 2 x x x

            grid[0] = [-1.5]
            grid[1] = [-0.5]
            grid[2] = [1.5]


        __native__

        The ``Grid1D`` in its ``native`` format is stored as an ndarray of shape [total_x_coordinates], where
        masked entries have a value of 0.0.

        All masked entries on the grid has (y,x) values of (0.0, 0.0).

        For the following example mask:

        .. code-block:: bash

            x x x O O x O x x x

            - grid[0] = 0.0 (it is masked, thus zero)
            - grid[1] = 0.0 (it is masked, thus zero)
            - grid[2] = 0.0 (it is masked, thus zero)
            - grid[3] = -1.5
            - grid[4] = -0.5
            - grid[5] = 0.0 (it is masked, thus zero)
            - grid[6] = 0.5

        **Grid1D Mapping**

        Every set of (x) coordinates in a pixel of the grid maps to an unmasked pixel in the mask. For a uniform
        grid, every x coordinate directly corresponds to the location of its paired unmasked pixel.

        It is not a requirement that grid is uniform and that their coordinates align with the mask. The input grid
        could be an irregular set of x coordinates where the indexing signifies that the x coordinate
        *originates* or *is paired with* the mask's pixels but has had its value change by some aspect of the
        calculation.

        This is important for the child project *PyAutoLens*, where grids in the image-plane are ray-traced and
        deflected to perform lensing calculations. The grid indexing is used to map pixels between the image-plane and
        source-plane.

        Parameters
        ----------
        values
            The (x,) coordinates of the grid, input as an ndarray of shape [total_unmasked_pixels] or
            [total_pixels] (native format).
        mask
            The 1D mask associated with the grid, defining which pixels each grid coordinate is paired with.
        store_native
            If True, the ndarray is stored in its native format [total_pixels]. This avoids mapping large data
            arrays to and from the slim / native formats, which can be a computational bottleneck.
        """

        values = grid_1d_util.convert_grid_1d(
            grid_1d=values, mask_1d=mask, store_native=store_native
        )

        self.mask = mask

        super().__init__(values)

    @classmethod
    def no_mask(
        cls,
        values: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float] = (0.0,),
    ) -> "Grid1D":
        """
        Create a Grid1D (see *Grid1D.__new__*) by inputting the grid coordinates in 1D.

        Parameters
        ----------
        values
            The (x,) coordinates of the grid, input as an ndarray of shape [total_unmasked_pixels]
            or a list of values.
        pixel_scales
            The (x,) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a ``float``,
            it is converted to a (float,) tuple.
        origin
            The (x,) origin of the grid's coordinate system.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            # Make Grid1D from input ndarray.

            grid_1d = aa.Grid1D.no_mask(values=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0)

            # Make Grid1D from input list.

            grid_1d = aa.Grid1D.no_mask(values=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

            # Print grid's slim (masked 1D data representation) and
            # native (masked 1D data representation)

            print(grid_1d.slim)
            print(grid_1d.native)
        """

        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        values = grid_2d_util.convert_grid(grid=values)

        mask = Mask1D.all_false(
            shape_slim=values.shape[0],
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return Grid1D(values=np.array(values), mask=mask)

    @classmethod
    def from_mask(cls, mask: Mask1D) -> "Grid1D":
        """
        Create a Grid1D (see *Grid1D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in its native 1D masked values are 0.0).

        The mask's pixel_scales, and origin properties are used to compute the grid (x) coordinates.

        Parameters
        ----------
        mask
            The mask whose masked pixels are used to setup the pixel grid.
        """

        grid_1d = grid_1d_util.grid_1d_slim_via_mask_from(
            mask_1d=mask.array,
            pixel_scales=mask.pixel_scales,
            origin=mask.origin,
        )

        return Grid1D(values=np.array(grid_1d), mask=mask)

    @classmethod
    def uniform(
        cls,
        shape_native: Tuple[int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float] = (0.0, 0.0),
    ) -> "Grid1D":
        """
        Create a `Grid1D` (see `Grid`D.__new__`) as a uniform grid of (x) values given an input `shape_native` and
        `pixel_scales` of the grid.

        Parameters
        ----------
        shape_native
            The 1D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (x) scaled units to pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float,) tuple.
        origin
            The origin of the grid's mask and coordinate system.
        """
        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        grid_slim = grid_1d_util.grid_1d_slim_via_shape_slim_from(
            shape_slim=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return cls.no_mask(
            values=grid_slim,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def uniform_from_zero(
        cls, shape_native: Tuple[int], pixel_scales: ty.PixelScales
    ) -> "Grid1D":
        """
        Create a `Grid1D` (see `Grid`D.__new__`) as a uniform grid of (x) values given an input `shape_native` and
        `pixel_scales` of the grid, where the first (x) coordinate of the grid is 0.0 and all other values ascend
        positively.

        Parameters
        ----------
        shape_native
            The 1D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (x) scaled units to pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float,) tuple.
        """
        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        grid_slim = grid_1d_util.grid_1d_slim_via_shape_slim_from(
            shape_slim=shape_native,
            pixel_scales=pixel_scales,
        )

        grid_slim -= np.min(grid_slim)

        return cls.no_mask(
            values=grid_slim,
            pixel_scales=pixel_scales,
        )

    @property
    def slim(self) -> "Grid1D":
        """
        Return a `Grid1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels, 2].

        If it is already stored in its `slim` representation  the `Grid1D` is returned as it is. If not, it is
        mapped from  `native` to `slim` and returned as a new `Grid1D`.
        """
        return Grid1D(values=self, mask=self.mask)

    @property
    def native(self) -> "Grid1D":
        """
        Return a `Grid1D` where the data is stored in its `native` representation, which is an ndarray of shape
        [total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid1D`.
        """
        return Grid1D(values=self, mask=self.mask, store_native=True)

    def grid_2d_radial_projected_from(self, angle: float = 0.0) -> Grid2DIrregular:
        """
        Project the 1D grid of (y,x) coordinates to an irregular 2d grid of (y,x) coordinates. The projection works
        as follows:

        1) Map the 1D (x) coordinates to 2D along the x-axis, such that the x value of every 2D coordinate is the
        corresponding (x) value in the 1D grid, and every y value is 0.0.

        2) Rotate this projected 2D grid clockwise by the input angle.

        Parameters
        ----------
        angle
            The angle with which the project 2D grid of coordinates is rotated clockwise.

        Returns
        -------
        Grid2DIrregular
            The projected and rotated 2D grid of (y,x) coordinates.
        """

        grid = np.zeros((self.mask.pixels_in_mask, 2))
        grid[:, 1] = self.slim.array

        grid = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid, centre=(0.0, 0.0), angle=angle
        )

        return Grid2DIrregular(values=grid + 1e-6)
