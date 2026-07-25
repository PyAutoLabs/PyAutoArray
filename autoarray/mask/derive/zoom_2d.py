from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D
    from autoarray.structures.arrays.rgb import Array2DRGB

from autoarray.structures.arrays import array_2d_util
from autoarray.structures.grids import grid_2d_util


class Zoom2D:

    def __init__(self, mask: Union[np.ndarray, List]):
        """
        Derives a zoomed in `Mask2D` object from a `Mask2D` object, which is typically used to visualize 2D arrays
        zoomed in to only the unmasked region an analysis is performed on.

        A `Mask2D` masks values which are associated with a uniform 2D rectangular grid of pixels, where unmasked
        entries (which are `False`) are used in subsequent calculations and masked values (which are `True`) are
        omitted (for a full description see the :meth:`Mask2D` class API
        documentation <autoarray.mask.mask_2d.Mask2D.__new__>`).

        The ``Zoom2D`` object computes the zoomed region around the unmasked pixels, including its centre,
        pixel offsets, scaled offsets, and extracted sub-arrays for visualization.

        Parameters
        ----------
        mask
            The `Mask2D` from which zoomed in `Mask2D` objects are derived.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                    [True,  True,  True,  True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True,  True,  True,  True, True],
                ],
                pixel_scales=1.0,
            )

            zoom_2d = aa.Zoom2D(mask=mask_2d)

            print(zoom_2d.centre)
        """
        self.mask = mask

    @property
    def centre(self) -> Tuple[float, float]:
        """
        Returns the centre of the zoomed in region, which is the average of the maximum and minimum y and x pixel values
        of the unmasked region.

        The y and x pixel values are the pixel coordinates of the unmasked region, which are derived from the
        `Mask2D` object. The pixel coordinates are in the same units as the pixel scales of the `Mask2D` object.

        Returns
        -------
        The centre of the zoomed in region.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self.mask,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

        grid = Grid2D(values=grid, mask=self.mask)

        extraction_grid_1d = self.mask.geometry.grid_pixels_2d_from(grid_scaled_2d=grid)
        y_pixels_max = np.max(extraction_grid_1d[:, 0])
        y_pixels_min = np.min(extraction_grid_1d[:, 0])
        x_pixels_max = np.max(extraction_grid_1d[:, 1])
        x_pixels_min = np.min(extraction_grid_1d[:, 1])

        return (
            ((y_pixels_max + y_pixels_min - 1.0) / 2.0),
            ((x_pixels_max + x_pixels_min - 1.0) / 2.0),
        )

    @property
    def offset_pixels(self) -> Tuple[float, float]:
        """
        Returns the (y,x) pixel offset of the centre of the zoomed region from the centre of the ``Mask2D``.

        This is the difference between the pixel coordinates of the zoomed region's centre and the pixel
        coordinates of the full mask's central pixel.

        Returns
        -------
        The (y,x) offset of the zoomed region centre from the mask centre in pixel units.
        """
        if self.mask.pixel_scales is None:
            return self.mask.geometry.central_pixel_coordinates

        return (
            self.centre[0] - self.mask.geometry.central_pixel_coordinates[0],
            self.centre[1] - self.mask.geometry.central_pixel_coordinates[1],
        )

    @property
    def offset_scaled(self) -> Tuple[float, float]:
        """
        Returns the (y,x) scaled offset of the centre of the zoomed region from the centre of the ``Mask2D``.

        This converts ``offset_pixels`` to scaled units using the mask's ``pixel_scales``.

        Returns
        -------
        The (y,x) offset of the zoomed region centre from the mask centre in scaled units.
        """
        return (
            -self.mask.pixel_scales[0] * self.offset_pixels[0],
            self.mask.pixel_scales[1] * self.offset_pixels[1],
        )

    @property
    def region(self) -> List[int]:
        """
        The zoomed region corresponding to the square encompassing all unmasked values.

        This is used to zoom in on the region of an image that is used in an analysis for visualization.

        This zoomed extraction region is a square, even if the mask is rectangular, so that extraction regions are
        always squares which is important for ensuring visualization does not have aspect ratio issues.
        """

        where = np.array(np.where(np.invert(self.mask.astype("bool"))))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)

        # Have to convert mask to bool for invert function to work.

        ylength = y1 - y0
        xlength = x1 - x0

        if ylength > xlength:
            length_difference = ylength - xlength
            x1 += int(length_difference / 2.0)
            x0 -= int(length_difference / 2.0)
        elif xlength > ylength:
            length_difference = xlength - ylength
            y1 += int(length_difference / 2.0)
            y0 -= int(length_difference / 2.0)

        return [y0, y1 + 1, x0, x1 + 1]

    @property
    def shape_native(self) -> Tuple[int, int]:
        """
        The shape of the zoomed in region in pixels.

        This is computed by subtracting the minimum and maximum y and x pixel values of the unmasked region.

        Returns
        -------
        The shape of the zoomed in region in pixels.
        """
        region = self.region
        return (region[1] - region[0], region[3] - region[2])

    def extent_from(self, buffer: int = 1) -> np.ndarray:
        """
        Compute the extent of the zoomed region in scaled coordinates, including an optional pixel buffer.

        The extent is returned as a tuple of the form ``(x_min, x_max, y_min, y_max)`` which matches the
        ``extent`` format expected by ``matplotlib.imshow``.

        Parameters
        ----------
        buffer
            The number of pixels around the unmasked region used as a buffer when computing the extent.

        Returns
        -------
        The extent of the zoomed region as ``(x_min, x_max, y_min, y_max)`` in scaled units.
        """
        from autoarray.mask.mask_2d import Mask2D

        extracted_array_2d = array_2d_util.extracted_array_2d_from(
            array_2d=np.array(self.mask),
            y0=self.region[0] - buffer,
            y1=self.region[1] + buffer,
            x0=self.region[2] - buffer,
            x1=self.region[3] + buffer,
        )

        mask = Mask2D.all_false(
            shape_native=extracted_array_2d.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.centre,
        )

        return mask.geometry.extent

    def mask_2d_from(self, buffer: int = 1) -> "Mask2D":
        """
        Extract the 2D region of a mask corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        buffer
            The number pixels around the extracted array used as a buffer.
        """
        from autoarray.mask.mask_2d import Mask2D

        extracted_mask_2d = array_2d_util.extracted_array_2d_from(
            array_2d=np.array(self.mask),
            y0=self.region[0] - buffer,
            y1=self.region[1] + buffer,
            x0=self.region[2] - buffer,
            x1=self.region[3] + buffer,
        )

        return Mask2D(
            mask=extracted_mask_2d,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def array_2d_from(self, array: Array2D, buffer: int = 1) -> Array2D:
        """
        Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        buffer
            The number pixels around the extracted array used as a buffer.
        """
        from autoarray.structures.arrays.uniform_2d import Array2D
        from autoarray.structures.arrays.rgb import Array2DRGB
        from autoarray.mask.mask_2d import Mask2D

        if isinstance(array, Array2DRGB):
            return self.array_2d_rgb_from(array=array, buffer=buffer)

        extracted_array_2d = array_2d_util.extracted_array_2d_from(
            array_2d=array.native.array,
            y0=self.region[0] - buffer,
            y1=self.region[1] + buffer,
            x0=self.region[2] - buffer,
            x1=self.region[3] + buffer,
        )

        extracted_mask_2d = array_2d_util.extracted_array_2d_from(
            array_2d=np.array(self.mask),
            y0=self.region[0] - buffer,
            y1=self.region[1] + buffer,
            x0=self.region[2] - buffer,
            x1=self.region[3] + buffer,
        )

        mask = Mask2D(
            mask=extracted_mask_2d,
            pixel_scales=array.pixel_scales,
            origin=array.mask.mask_centre,
        )

        arr = array_2d_util.convert_array_2d(array_2d=extracted_array_2d, mask_2d=mask)

        return Array2D(values=arr, mask=mask, header=array.header).native

    def array_2d_rgb_from(self, array: Array2DRGB, buffer: int = 1) -> Array2DRGB:
        """
        Extract the 2D region of an RGB array corresponding to the rectangle encompassing all unmasked values.

        This works the same as the `array_2d_from` method, but for RGB arrays, meaning that it iterates over the three
        channels of the RGB array and extracts the region for each channel separately.

        This is used to extract and visualize only the region of an RGB image that is used in an analysis.

        Parameters
        ----------
        buffer
            The number pixels around the extracted array used as a buffer.
        """
        from autoarray.structures.arrays.rgb import Array2DRGB
        from autoarray.mask.mask_2d import Mask2D

        for i in range(3):

            extracted_array_2d = array_2d_util.extracted_array_2d_from(
                array_2d=np.array(array.native[:, :, i]),
                y0=self.region[0] - buffer,
                y1=self.region[1] + buffer,
                x0=self.region[2] - buffer,
                x1=self.region[3] + buffer,
            )

            if i == 0:
                array_2d_rgb = np.zeros(
                    (extracted_array_2d.shape[0], extracted_array_2d.shape[1], 3)
                )

            array_2d_rgb[:, :, i] = extracted_array_2d

        extracted_mask_2d = array_2d_util.extracted_array_2d_from(
            array_2d=np.array(self.mask),
            y0=self.region[0] - buffer,
            y1=self.region[1] + buffer,
            x0=self.region[2] - buffer,
            x1=self.region[3] + buffer,
        )

        mask = Mask2D(
            mask=extracted_mask_2d,
            pixel_scales=array.pixel_scales,
            origin=array.mask.mask_centre,
        )

        return Array2DRGB(values=array_2d_rgb.astype("int"), mask=mask)
