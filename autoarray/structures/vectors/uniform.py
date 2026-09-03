import logging

import numpy as np
from typing import List, Optional, Tuple, Union

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.vectors.abstract import AbstractVectorYX2D

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util

from autoarray import exc
from autoarray import type as ty

logging.basicConfig()
logger = logging.getLogger(__name__)


class VectorYX2D(AbstractVectorYX2D):
    def __init__(
        self,
        values: Union[np.ndarray, List[Tuple[float, float]]],
        grid: Union[Grid2D, List],
        mask: Mask2D,
        store_native: bool = False,
    ):
        r"""
        A collection of (y,x) vectors which are located on a regular 2D grid of (y,x) coordinates.

        The vectors are paired to a uniform 2D mask of pixels. Each vector corresponds to a value at
        the centre of a pixel in an unmasked pixel.

        The `VectorYX2D` is ordered such that pixels begin from the top-row of the corresponding mask and go right
        and down. The positive y-axis is upwards and positive x-axis to the right.

        The (y,x) vectors are stored as a NumPy array which has the `slim` and `native shapes described below.
        Irrespective of this shape, the last dimension of the data structure storing the vectors is always shape 2,
        corresponding to the y and x vectors. [total_vectors, 2].

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The vectors input to this function can have any of the following forms (they will be converted to the 1D NumPy
        array structure and can be converted back using the object's properties):

        [[vector_0_y, vector_0_x], [vector_1_y, vector_1_x]]
        [(vector_0_y, vector_0_x), (vector_1_y, vector_1_x)]

        If your vector field lies on a 2D irregular grid of data the `VectorFieldIrregular2D` data structure should be
        used.


        __slim__

        The Vector2D is an ndarray of shape [total_unmasked_pixels, 2].

        The first element of the ndarray corresponds to the pixel index, for example:

        - vector[3, 0:2] = the 4th unmasked pixel's y and x values.
        - vector[6, 0:2] = the 7th unmasked pixel's y and x values.

        Below is a visual illustration of a vector, where a total of 10 pixels are unmasked and are included in
        the vector.

         x x x x x x x x x x
         x x x x x x x x x x     This is an example `Mask2D`, where:
         x x x x x x x x x x
         x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the vector)
         x x x O O O O x x x     O = `False` (Pixel is not masked and included in the vector)
         x x x O O O O x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x

        The mask pixel index's will come out like this (and the direction of scaled values is highlighted
        around the mask).

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
         x x x x x x x x x x  ^   vector[0, :] = 0
         x x x x x x x x x x  I   vector[1, :] = 1
         x x x x x x x x x x  I   vector[2, :] = 2
         x x x x 0 1 x x x x +ve  vector[3, :] = 3
         x x x 2 3 4 5 x x x  y   vector[4, :] = 4
         x x x 6 7 8 9 x x x -ve  vector[5, :] = 5
         x x x x x x x x x x  I   vector[6, :] = 6
         x x x x x x x x x x  I   vector[7, :] = 7
         x x x x x x x x x x \/   vector[8, :] = 8
         x x x x x x x x x x      vector[9, :] = 9


        __native__

        The Vector2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_values, total_x_values, 2].

        All masked entries on the vector have values of 0.0.

        For the following example mask:

         x x x x x x x x x x
         x x x x x x x x x x     This is an example `Mask2D`, where:
         x x x x x x x x x x
         x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the vector)
         x x x O O O O x x x     O = `False` (Pixel is not masked and included in the vector)
         x x x O O O O x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x

        For the example above:

            - vector[0,0, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[0,0, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[3,3, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[3,3, 0:2] = [0.0, 0.0] (it is masked, thus zero)
            - vector[3,4, 0:2] = [0, 0]
            - vector[3,4, 0:2] = [-1, -1]

        Parameters
        ----------
        values
            The 2D (y,x) vectors on a regular grid that represent the vector-field.
        grid
            The regular grid of (y,x) coordinates where each vector is located.
        mask
            The 2D mask associated with the array, defining the pixels each array value is paired with and
            originates from.
        store_native
            If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels, 2]. This avoids
            mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
        """

        values = grid_2d_util.convert_grid_2d(
            grid_2d=values, mask_2d=mask, store_native=store_native
        )

        # A `Grid2D` already paired with this mask is the grid the vectors are located on: the
        # `@to_vector_yx` decorator hands the input grid straight through, so re-converting its
        # values and rebuilding a second `Grid2D` on every profile evaluation would only pay the
        # constructor twice for the same object.
        if isinstance(grid, Grid2D) and grid.mask is mask and not store_native:
            self.grid = grid
        else:
            grid = grid_2d_util.convert_grid_2d(
                grid_2d=grid, mask_2d=mask, store_native=store_native
            )

            self.grid = Grid2D(values=grid, mask=mask)

        self.mask = mask

        super().__init__(values)

    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "grid"):
            self.grid = obj.grid

    @classmethod
    def no_mask(
        cls,
        values: Union[np.ndarray, List[List], List[Tuple]],
        pixel_scales: ty.PixelScales,
        shape_native: Optional[Tuple[int, int]] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a VectorYX2D (see *VectorYX2D.__new__*) by inputting the vector in 1D, for example:

        vectors=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        vectors=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        The `VectorYX2D` object assumes a uniform `Grid2D` which is computed from the input `shape_native`,
        `pixel_scales` and `origin`.

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the `shape_native` must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        values
            The (y,x) vectors input as an ndarray of shape [total_unmasked_pixels, 2] or a list of lists.
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
            if shape_native is None:
                raise exc.VectorYXException(
                    f"""
                    The input vectors are not in their native shape (an ndarray / list of 
                    shape [total_y_pixels, total_x_pixels, 2]) and the shape_native parameter has not been input the 
                    VectorYX2D function.
    
                    Either change the input array to be its native shape or input its shape_native input the function.
    
                    The shape of the input array is {values.shape}
                    """
                )

            if shape_native and len(shape_native) != 2:
                raise exc.GridException(
                    """
                    The input shape_native parameter is not a tuple of type (int, int).
                    """
                )

        else:
            shape_native = (
                int(values.shape[0]),
                int(values.shape[1]),
            )

        grid = Grid2D.uniform(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        mask = Mask2D.all_false(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return cls(values=values, grid=grid, mask=mask)

    @classmethod
    def from_mask(cls, values: Union[np.ndarray, List], mask: Mask2D) -> "VectorYX2D":
        """
        Create a VectorYX2D (see *VectorYX2D.__new__*) by inputting the vectors in 1D or 2D with its mask,
        for example:

        mask = Mask2D([[True, False, False, False])
        array=np.array([1.0, 2.0, 3.0])

        Parameters
        ----------
        values
            The values of the array input as an ndarray of shape [total_unmasked_pixels] or a list of
            lists.
        mask
            The mask whose masked pixels are used to setup the pixel grid.
        """

        grid = Grid2D.from_mask(mask=mask)

        return VectorYX2D(values=values, grid=grid, mask=mask)

    @classmethod
    def full(
        cls,
        fill_value: float,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a `VectorYX2D` (see `AbstractVectorYX2D.__new__`) where all values are filled with an input fill value,
        analogous to the method np.full().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The origin of the grid's mask.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """

        return cls.no_mask(
            values=np.full(
                fill_value=fill_value, shape=(shape_native[0], shape_native[1], 2)
            ),
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def ones(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a `VectorYX2D` (see `AbstractVectorYX2D.__new__`) where all values are filled with 1.0, analogous to
        the method np.ones().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=1.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @classmethod
    def zeros(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2D":
        """
        Create a `VectorYX2D` (see `AbstractVectorYX2D.__new__`) where all values are filled with 1.0, analogous to
        the method np.zeros().

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=0.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @property
    def slim(self) -> "VectorYX2D":
        """
        Return a `VectorYX2D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels, 2].

        If it is already stored in its `slim` representation it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Array2D`.
        """
        return VectorYX2D(values=self, grid=self.grid.slim, mask=self.mask)

    @property
    def native(self) -> "VectorYX2D":
        """
        Return a `VectorYX2D` where the data is stored in its `native` representation, which is an ndarray of shape
        [total_y_pixels, total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid2D`.
        """
        return VectorYX2D(
            values=self,
            grid=self.grid.native,
            mask=self.mask,
            store_native=True,
        )

    def apply_mask(self, mask: Mask2D) -> "VectorYX2D":
        return VectorYX2D.from_mask(values=self.native, mask=mask)

    @property
    def magnitudes(self) -> Array2D:
        """
        Returns the magnitude of every vector which are computed as sqrt(y**2 + x**2).
        """
        return Array2D(
            values=np.sqrt(self.array[:, 0] ** 2.0 + self.array[:, 1] ** 2.0),
            mask=self.mask,
        )

    @property
    def y(self) -> Array2D:
        """
        Returns the y vector values as an `Array2D` object.
        """
        return Array2D(values=self.slim[:, 0], mask=self.mask)

    @property
    def x(self) -> Array2D:
        """
        Returns the y vector values as an `Array2D` object.
        """
        return Array2D(values=self.slim[:, 1], mask=self.mask)
