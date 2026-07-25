from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_1d import Grid1D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.mask.derive.grid_2d import DeriveGrid2D
from autoarray.mask.derive.indexes_2d import DeriveIndexes2D
from autoarray.mask.derive.mask_2d import DeriveMask2D

from autoarray import exc


class Structure(AbstractNDArray, ABC):
    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask

    @property
    @abstractmethod
    def slim(self) -> "Structure":
        """
        Returns the data structure in its `slim` format which flattens all unmasked values to a 1D array.
        """

    @property
    def geometry(self):
        """
        The geometry object of the mask associated with this structure, which defines coordinate conversions
        between pixel units and scaled units.
        """
        return self.mask.geometry

    @property
    def derive_grid(self) -> DeriveGrid2D:
        """
        The ``DeriveGrid2D`` object of the mask, used to compute derived grids of (y,x) coordinates such as
        the edge grid, border grid, and full unmasked grid.
        """
        return self.mask.derive_grid

    @property
    def derive_indexes(self) -> DeriveIndexes2D:
        """
        The ``DeriveIndexes2D`` object of the mask, used to compute index arrays that map data between the
        ``slim`` (1D unmasked) and ``native`` (2D full-shape) representations.
        """
        return self.mask.derive_indexes

    @property
    def derive_mask(self) -> DeriveMask2D:
        """
        The ``DeriveMask2D`` object of the mask, used to compute derived masks such as the edge mask,
        border mask, and blurring mask.
        """
        return self.mask.derive_mask

    @property
    def shape_slim(self) -> int:
        """
        The 1D shape of the data structure in its ``slim`` representation, equal to the number of unmasked pixels.
        """
        return self.mask.shape_slim

    @property
    def shape_native(self) -> Tuple[int, ...]:
        """
        The shape of the data structure in its ``native`` representation (e.g. ``(total_y_pixels, total_x_pixels)``
        for a 2D structure).
        """
        return self.mask.shape

    @property
    def pixel_scales(self) -> Tuple[float, ...]:
        """
        The (y,x) scaled units to pixel units conversion factors of every pixel, as a tuple of floats.
        """
        return self.mask.pixel_scales

    @property
    def pixel_scale(self) -> float:
        """
        The pixel scale as a single float value. Assumes all pixel scales are equal (e.g. square pixels).
        """
        return self.mask.pixel_scale

    @property
    def header_dict(self) -> Dict:
        """
        The FITS header dictionary of the mask associated with this structure, containing pixel scale and origin entries.
        """
        return self.mask.header_dict

    @property
    def pixel_area(self):
        """
        The area of a single pixel in scaled units squared (``pixel_scales[0] * pixel_scales[1]``).

        Only valid for 2D structures; raises ``GridException`` for 1D structures.
        """
        if len(self.pixel_scales) != 2:
            raise exc.GridException("Cannot compute area of structure which is not 2D.")

        return self.pixel_scales[0] * self.pixel_scales[1]

    @property
    def total_area(self):
        """
        The total area of all unmasked pixels in scaled units squared (``total_pixels * pixel_area``).
        """
        return self.total_pixels * self.pixel_area

    @property
    def origin(self) -> Tuple[int, ...]:
        """
        The (y,x) scaled units origin of the mask's coordinate system.
        """
        return self.mask.origin

    @property
    def unmasked_grid(self) -> Union[Grid1D, Grid2D]:
        """
        A grid of (y,x) coordinates of every pixel in the full mask shape (including masked pixels), using
        the mask's geometry to compute each pixel's scaled coordinate.
        """
        return self.mask.derive_grid.all_false

    @property
    def total_pixels(self) -> int:
        """
        The total number of unmasked pixels in the data structure (its ``slim`` length).
        """
        return self.shape[0]

    def trimmed_after_convolution_from(self, kernel_shape) -> "Structure":
        """
        Trim the data structure back to its original shape after PSF convolution has been performed on a
        padded version of it.

        This is the inverse of ``padded_before_convolution_from``: first the data structure is padded so
        that edge-pixel signal is not lost during convolution, and then this method trims the result back
        to the original shape.

        Parameters
        ----------
        kernel_shape
            The 2D shape of the convolution kernel used to pad and trim the data structure.
        """
        raise NotImplementedError
