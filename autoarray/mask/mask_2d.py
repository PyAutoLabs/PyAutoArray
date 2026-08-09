from __future__ import annotations
import logging
import os
from enum import Enum
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from autonerves import cached_property

from autoarray.structures.abstract_structure import Structure

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D


from autonerves.fitsable import ndarray_via_fits_from

from autoarray.mask.abstract_mask import Mask

from autoarray import exc
from autoarray import type as ty
from autoarray import validate
from autoarray.geometry.geometry_2d import Geometry2D
from autoarray.mask.derive.mask_2d import DeriveMask2D
from autoarray.mask.derive.grid_2d import DeriveGrid2D
from autoarray.mask.derive.indexes_2d import DeriveIndexes2D
from autoarray.mask.derive.zoom_2d import Zoom2D

from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask2DKeys(Enum):
    PIXSCAY = "PIXSCAY"
    PIXSCAX = "PIXSCAX"
    ORIGINY = "ORIGINY"
    ORIGINX = "ORIGINX"


class Mask2D(Mask):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        mask: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
        xp=np,
        *args,
        **kwargs,
    ):
        r"""
        A 2D mask, used for masking values which are associated with a a uniform rectangular grid of pixels.

        When applied to 2D data with the same shape, values in the mask corresponding to ``False`` entries are
        unmasked and therefore used in subsequent calculations. .

        The ``Mask2D`, has in-built functionality which:

        - Maps data structures between two data representations: `slim`` (all unmasked ``False`` values in
          a 1D ``ndarray``) and ``native`` (all unmasked values in a 2D or 3D ``ndarray``).

        - Has a ``Geometry2D`` object (defined by its (y,x) ``pixel scales`` and (y,x) ``origin``)
          which defines how coordinates are converted from pixel units to scaled units.

        - Associates Cartesian ``Grid2D`` objects of (y,x) coordinates with the data structure (e.g.
          a (y,x) grid of all unmasked pixels).

        A detailed description of the 2D mask API is provided below.

        __Slim__

        Below is a visual illustration of a ``Mask2D``, where a total of 10 pixels are unmasked (values are ``False``):

        ::

             x x x x x x x x x x
             x x x x x x x x x x     This is an example ``Mask2D``, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the array)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the array)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        The mask pixel index's are as follows (the positive / negative direction of the ``Grid2D`` objects associated
        with the mask are also shown on the y and x axes).

        ::

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x 0 1 x x x x +ve
             x x x 2 3 4 5 x x x  y
             x x x 6 7 8 9 x x x -ve
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x x x x x x x \/
             x x x x x x x x x x

        The ``Mask2D``'s ``slim`` data representation is an ``ndarray`` of shape [total_unmasked_pixels].

        For the ``Mask2D`` above the ``slim`` representation therefore contains 10 entries and two examples of these
        entries are:

        ::

            mask[3] = the 4th unmasked pixel's value.
            mask[6] = the 7th unmasked pixel's value.

        A Cartesian grid of (y,x) coordinates, corresponding to all ``slim`` values (e.g. unmasked pixels) is given
        by:


        __native__

        Masked data represented as an an ``ndarray`` of shape [total_y_values, total_x_values], where all masked
        entries have values of 0.0.

        For the following mask:

        ::

             x x x x x x x x x x
             x x x x x x x x x x     This is an example ``Mask2D``, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the array)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the array)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        The mask has the following indexes:

        ::

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x 0 1 x x x x +ve
             x x x 2 3 4 5 x x x  y
             x x x 6 7 8 9 x x x -ve
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x x x x x x x  \/
             x x x x x x x x x x

        In the above array:

        ::

            - mask[0,0] = True (it is masked)
            - mask[0,0] = True (it is masked)
            - mask[3,3] = True (it is masked)
            - mask[3,3] = True (it is masked)
            - mask[3,4] = False (not masked)
            - mask[3,5] = False (not masked)
            - mask[4,5] = False (not masked)

        **SLIM TO NATIVE MAPPING**

        The ``Mask2D`` has functionality which maps data between the ``slim`` and ``native`` data representations.

        For the example mask above, the 1D ``ndarray`` given by ``derive_indexes.slim_to_native`` is:

        ::

            slim_to_native[0] = [3,4]
            slim_to_native[1] = [3,5]
            slim_to_native[2] = [4,3]
            slim_to_native[3] = [4,4]
            slim_to_native[4] = [4,5]
            slim_to_native[5] = [4,6]
            slim_to_native[6] = [5,3]
            slim_to_native[7] = [5,4]
            slim_to_native[8] = [5,5]
            slim_to_native[9] = [5,6]

        Parameters
        ----------
        mask
            The `ndarray` of shape [total_y_pixels, total_x_pixels] containing the `bool`'s representing the
            `mask`, where `False` signifies an entry is unmasked and used in calculations.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, so `False` entries become `True`
            and vice versa.
        xp
            The array module to use (default `numpy`; pass `jax.numpy` for JAX support). Controls
            whether internal index arrays are computed on CPU or GPU.
        """

        if type(mask) is list:
            mask = xp.asarray(mask).astype("bool")

        if invert:
            mask = ~mask

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        if len(mask.shape) != 2:
            raise exc.MaskException("The input mask is not a two dimensional array")

        # Every `Mask2D` factory returns through this constructor, and `Grid2D.uniform`
        # reaches it via `Grid2D.no_mask` -> `Mask2D.all_false`, so a degenerate shape
        # is caught once here rather than at each construction site.
        validate.validate_shape_native(
            shape_native=mask.shape, name="shape_native", exc_type=exc.MaskException
        )

        super().__init__(
            mask=mask,
            origin=origin,
            pixel_scales=pixel_scales,
            xp=xp,
        )

        slim_to_native = self.derive_indexes.native_for_slim.astype("int32")
        self.slim_to_native_tuple = (slim_to_native[:, 0], slim_to_native[:, 1])

    @property
    def native_for_slim(self):
        """
        A 2D array of shape [total_unmasked_pixels, 2] that maps every unmasked pixel's slim index to its
        (y, x) native 2D index.

        For example, if ``slim_to_native[3] = [2, 5]``, the 4th unmasked pixel (slim index 3) is located at
        row 2, column 5 in the native 2D array.
        """
        return self.derive_indexes.native_for_slim

    __no_flatten__ = ("derive_indexes",)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj=obj)

        if not isinstance(obj, Mask2D):
            self.origin = (0.0, 0.0)

    @property
    def native(self) -> Structure:
        return self

    @property
    def geometry(self) -> Geometry2D:
        """
        Return the 2D geometry of the mask, representing its uniform rectangular grid of (y,x) coordinates defined by
        its ``shape_native``.
        """
        return Geometry2D(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def derive_indexes(self) -> DeriveIndexes2D:
        """
        Returns the ``DeriveIndexes2D`` object associated with the mask, which contains derived index arrays
        used to map data between ``slim`` (1D unmasked) and ``native`` (2D full-shape) representations.
        """
        return DeriveIndexes2D(mask=self, xp=self._xp)

    @property
    def derive_mask(self) -> DeriveMask2D:
        """
        Returns the ``DeriveMask2D`` object associated with the mask, which computes derived masks such as
        the edge mask, border mask, and blurring mask.
        """
        return DeriveMask2D(mask=self)

    @property
    def derive_grid(self) -> DeriveGrid2D:
        """
        Returns the ``DeriveGrid2D`` object associated with the mask, which computes derived grids of (y,x)
        coordinates such as the unmasked pixel grid, edge grid, and border grid.
        """
        return DeriveGrid2D(mask=self)

    @property
    def zoom(self) -> Zoom2D:
        """
        Returns the ``Zoom2D`` object associated with the mask, which computes the zoomed region of the mask
        around its unmasked pixels for use in visualization.
        """
        return Zoom2D(mask=self)

    @classmethod
    def all_false(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Create a mask where all pixels are `False` and therefore unmasked.

        Parameters
        ----------
        shape_native
            The 2D shape of the mask that is created.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        return cls(
            mask=np.full(shape=shape_native, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular(
        cls,
        shape_native: Tuple[int, int],
        radius: float,
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within a circle of input radius.

        The `radius` and `centre` are both input in scaled units.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the mask in units of pixels.
        radius
            The radius in scaled units of the circle within which pixels are `False` and unmasked.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centre of the circle used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
            if shape_native[0] > 16 or shape_native[1] > 16:
                shape_native = (16, 16)
                pixel_scales = 0.6
            scale_scalar = (
                pixel_scales
                if isinstance(pixel_scales, (int, float))
                else min(pixel_scales)
            )
            max_radius = min(shape_native) * scale_scalar / 2.0
            if radius > max_radius:
                radius = max_radius

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_circular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            radius=radius,
            centre=centre,
        )

        return cls(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def circular_annular(
        cls,
        shape_native: Tuple[int, int],
        inner_radius: float,
        outer_radius: float,
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an annulus of input
        inner radius and outer radius.

        The `inner_radius`, `outer_radius` and `centre` are all input in scaled units.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the mask in units of pixels.
        inner_radius
            The inner radius in scaled units of the annulus within which pixels are `False` and unmasked.
        outer_radius
            The outer radius in scaled units of the annulus within which pixels are `False` and unmasked.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centre of the annulus used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        validate.validate_radii_ordered(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            exc_type=exc.MaskException,
        )

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_circular_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            centre=centre,
        )

        return cls(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def elliptical(
        cls,
        shape_native: Tuple[int, int],
        major_axis_radius: float,
        axis_ratio: float,
        angle: float,
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an ellipse.

        The `major_axis_radius`, and `centre` are all input in scaled units.

        Parameters
        ----------
        shape_native
            The (y,x) shape of the mask in units of pixels.
        major_axis_radius
            The major-axis in scaled units of the ellipse within which pixels are unmasked.
        axis_ratio
            The axis-ratio of the ellipse within which pixels are unmasked.
        angle
            The rotation angle of the ellipse within which pixels are unmasked, (counter-clockwise from the positive
             x-axis).
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centred of the ellipse used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_elliptical_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            major_axis_radius=major_axis_radius,
            axis_ratio=axis_ratio,
            angle=angle,
            centre=centre,
        )

        return cls(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def elliptical_annular(
        cls,
        shape_native: Tuple[int, int],
        inner_major_axis_radius: float,
        inner_axis_ratio: float,
        inner_phi: float,
        outer_major_axis_radius: float,
        outer_axis_ratio: float,
        outer_phi: float,
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        centre: Tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are within an elliptical annulus of input
        inner and outer scaled major-axis and centre.

        The `outer_major_axis_radius`, `inner_major_axis_radius` and `centre` are all input in scaled units.

        Parameters
        ----------
        shape_native (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        inner_major_axis_radius
            The major-axis in scaled units of the inner ellipse within which pixels are masked.
        inner_axis_ratio
            The axis-ratio of the inner ellipse within which pixels are masked.
        inner_phi
            The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the
            positive x-axis).
        outer_major_axis_radius
            The major-axis in scaled units of the outer ellipse within which pixels are unmasked.
        outer_axis_ratio
            The axis-ratio of the outer ellipse within which pixels are unmasked.
        outer_phi
            The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the
            positive x-axis).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        centre
            The (y,x) scaled units centre of the elliptical annuli used to mask pixels.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """
        validate.validate_radii_ordered(
            inner_radius=inner_major_axis_radius,
            outer_radius=outer_major_axis_radius,
            inner_name="inner_major_axis_radius",
            outer_name="outer_major_axis_radius",
            exc_type=exc.MaskException,
        )

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = mask_2d_util.mask_2d_elliptical_annular_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            inner_major_axis_radius=inner_major_axis_radius,
            inner_axis_ratio=inner_axis_ratio,
            inner_phi=inner_phi,
            outer_major_axis_radius=outer_major_axis_radius,
            outer_axis_ratio=outer_axis_ratio,
            outer_phi=outer_phi,
            centre=centre,
        )

        return cls(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_pixel_coordinates(
        cls,
        shape_native: Tuple[int, int],
        pixel_coordinates: [[int, int]],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        buffer: int = 0,
        invert: bool = False,
    ) -> "Mask2D":
        """
        Returns a Mask2D (see *Mask2D.__new__*) where all `False` entries are defined from an input list of list of
        pixel coordinates.

        These may be buffed via an input `buffer`, whereby all entries in all 8 neighboring directions by this
        amount.

        Parameters
        ----------
        shape_native (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_coordinates : [[int, int]]
            The input lists of 2D pixel coordinates where `False` entries are created.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        buffer
            All input `pixel_coordinates` are buffed with `False` entries in all 8 neighboring directions by this
            amount.
        invert
            If `True`, the `bool`'s of the input `mask` are inverted, for example `False`'s become `True`
            and visa versa.
        """

        mask = mask_2d_util.mask_2d_via_pixel_coordinates_from(
            shape_native=shape_native,
            pixel_coordinates=pixel_coordinates,
            buffer=buffer,
        )

        return cls(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        pixel_scales: ty.PixelScales,
        hdu: int = 0,
        origin: Tuple[float, float] = (0.0, 0.0),
        resized_mask_shape: Tuple[int, int] = None,
        invert: bool = False,
    ) -> "Mask2D":
        """
        Load a ``Mask2D`` from a 2D boolean array stored in a ``.fits`` file.

        Parameters
        ----------
        file_path
            The full path of the ``.fits`` file, including the file name and extension.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a
            `float`, it is converted to a (float, float) structure.
        hdu
            The HDU number in the ``.fits`` file containing the mask array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        resized_mask_shape
            If provided, the loaded mask is resized to this (y,x) shape after loading.
        invert
            If `True`, the `bool`'s of the loaded mask are inverted, so `False` entries become `True`
            and vice versa.

        Returns
        -------
        Mask2D
            The mask loaded from the ``.fits`` file.
        """
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = ndarray_via_fits_from(file_path=file_path, hdu=hdu)

        if invert:
            mask = np.invert(mask.astype("bool"))

        mask = Mask2D(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        if resized_mask_shape is not None:
            mask = mask.resized_from(new_shape=resized_mask_shape)

        return mask

    @property
    def shape_native(self) -> Tuple[int, ...]:
        """
        The 2D shape of the mask in its native representation, equal to the shape of the underlying boolean ndarray.
        """
        return self.shape

    @cached_property
    def fft_index_for_masked_pixel(self) -> np.ndarray:
        """
        Return a mapping from masked-pixel (slim) indices to flat indices
        on the rectangular FFT grid.

        This array is used to translate between:

            - "masked pixel space" (a compact 1D indexing over unmasked pixels)
            - the 2D rectangular grid on which FFT-based convolutions are performed

        The FFT grid is assumed to be rectangular and already suitable for FFTs
        (e.g. padded and centered appropriately). Masked pixels are present on
        this grid but are ignored in computations via zero-weighting.

        Returns
        -------
        np.ndarray
            A 1D array of shape (N_unmasked,), where element `i` gives the flat
            (row-major) index into the FFT grid corresponding to the `i`-th
            unmasked pixel in slim ordering.

        Notes
        -----
        - The slim ordering is defined as the order returned by `np.where(~mask)`.
        - The flat FFT index is computed assuming row-major (C-style) ordering:
              flat_index = y * width + x
        - This method is intentionally backend-agnostic and can be used by both
          imaging and interferometer curvature pipelines.
        """
        # Boolean mask defined on the rectangular FFT grid
        mask_fft = self

        # Coordinates of unmasked pixels in the FFT grid
        ys, xs = np.where(~mask_fft)

        # Width of the FFT grid (number of columns)
        width = mask_fft.shape[1]

        # Convert (y, x) coordinates to flat row-major indices
        return (ys * width + xs).astype(np.int32)

    @cached_property
    def extent_index_for_masked_pixel(self) -> np.ndarray:
        """
        Return a mapping from masked-pixel (slim) indices to flat indices on
        the *unmasked-extent* rectangular FFT grid.

        The unmasked extent is the bounding box of unmasked pixels
        (``shape_native_masked_pixels``). This index is the interferometer
        counterpart of `fft_index_for_masked_pixel`, which uses the full native
        grid: the interferometer W~ kernel is computed on the (extent_y,
        extent_x) grid because it is translation-invariant and only the offsets
        between pairs of unmasked pixels matter — the surrounding masked region
        contributes nothing.

        Returns
        -------
        np.ndarray
            A 1D array of shape (N_unmasked,) of int32 values in
            ``[0, extent_y * extent_x)``, suitable as row indices into the
            (extent_y * extent_x, batch) scatter buffer used by
            ``InterferometerSparseOperator.curvature_matrix_diag_from``.
        """
        ys, xs = np.where(~self)
        if ys.size == 0:
            return np.zeros((0,), dtype=np.int32)

        y0, x0 = int(np.min(ys)), int(np.min(xs))
        extent_y, extent_x = self.shape_native_masked_pixels
        width = int(extent_x)
        return ((ys - y0) * width + (xs - x0)).astype(np.int32)

    def trimmed_array_from(self, padded_array, image_shape) -> Array2D:
        """
        Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        ----------
        padded_array
            A 1D array of values which were computed using a padded grid
        """

        from autoarray.structures.arrays.uniform_2d import Array2D

        pad_size_0 = self.shape[0] - image_shape[0]
        pad_size_1 = self.shape[1] - image_shape[1]
        trimmed_array = padded_array.native[
            pad_size_0 // 2 : self.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.shape[1] - pad_size_1 // 2,
        ]
        return Array2D.no_mask(
            values=trimmed_array,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    def unmasked_blurred_array_from(self, padded_array, psf, image_shape) -> Array2D:
        """
        Convolve a padded array with the PSF and trim it back to the original image shape.

        This relies on a padded array whose shape extends beyond the masked region, so that PSF convolution
        does not suffer from edge effects. The result is trimmed back to ``image_shape`` after convolution.

        Parameters
        ----------
        padded_array
            The padded ``Array2D`` (or slim 1D representation) of values to be convolved with the PSF.
        psf
            The PSF convolver used to blur the padded array.
        image_shape
            The (y,x) shape of the original (unpadded) image, used to trim the blurred result.

        Returns
        -------
        Array2D
            The blurred and trimmed ``Array2D`` of shape ``image_shape``.
        """

        blurred_image = psf.convolved_image_from(
            image=padded_array, blurring_image=None
        )

        return self.trimmed_array_from(
            padded_array=blurred_image, image_shape=image_shape
        )

    @property
    def header_dict(self) -> Dict:
        """
        Returns the pixel scales of the mask as a header dictionary, which can be written to a .fits file.

        A 2D mask has different pixel scale variables for each dimension, the header therefore contain both pixel
        scales as separate y and x entries.

        Returns
        -------
        A dictionary containing the pixel scale of the mask, which can be output to a .fits file.
        """

        return {
            Mask2DKeys.PIXSCAY: self.pixel_scales[0],
            Mask2DKeys.PIXSCAX: self.pixel_scales[1],
            Mask2DKeys.ORIGINY: self.origin[0],
            Mask2DKeys.ORIGINX: self.origin[1],
        }

    @property
    def mask_centre(self) -> Tuple[float, float]:
        """
        The (y,x) scaled coordinate centre of the unmasked pixels in the mask.

        This is computed as the mean of the (y,x) scaled coordinates of all unmasked pixels.

        Returns
        -------
        The (y,x) centre of the unmasked region of the mask in scaled units.
        """
        grid = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

        return grid_2d_util.grid_2d_centre_from(grid_2d_slim=grid)

    @property
    def shape_native_masked_pixels(self) -> Tuple[int, int]:
        """
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask.

        For example, if a mask is primarily surrounded by True entries, and there are 15 False entries going vertically
        and 12 False entries going horizontally in the central regions of the mask, then shape_masked_pixels=(15,12).
        """

        where = np.where(np.invert(self.astype("bool")))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)

        return (y1 - y0 + 1, x1 - x0 + 1)

    def rescaled_from(self, rescale_factor) -> Mask2D:
        """
        Returns the ``Mask2D`` rescaled to a bigger or small shape via input ``rescale_factor``.

        For example, for a ``rescale_factor=2.0`` the following mask:

        ::
           [[ True,  True],
            [False, False]]

        Will double in size and become:

        ::
            [[True,   True,  True,  True],
             [True,   True,  True,  True],
             [False, False, False, False],
             [False, False, False, False]]

        Parameters
        ----------
        rescale_factor
            The factor by which the ``Mask2D`` is rescaled (less than 1.0 produces a smaller mask, greater than 1.0
            produces a bigger mask).

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
            )

            print(mask_2d.rescaled_from(rescale_factor=2.0)
        """
        from autoarray.mask.mask_2d import Mask2D

        rescaled_mask = mask_2d_util.rescaled_mask_2d_from(
            mask_2d=self.array,
            rescale_factor=rescale_factor,
        )

        return Mask2D(
            mask=rescaled_mask,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    def resized_from(self, new_shape, pad_value: int = 0.0) -> Mask2D:
        """
        Returns the ``Mask2D`` resized to a small or bigger ``ndarray``, but with the same distribution of
         ``False`` and ``True`` entries.

        Resizing which increases the ``Mask2D`` shape pads it with values on its edge.

        Resizing which reduces the ``Mask2D`` shape removes entries on its edge.

        For example, for a ``new_shape=(4,4)`` the following mask:

        ::
           [[ True,  True],
            [False, False]]

        Will be padded with zeros (``False`` values) and become:

        ::
          [[True,  True,  True, True]
           [True,  True,  True, True],
           [True, False, False, True],
           [True,  True,  True, True]]

        Parameters
        ----------
        new_shape
            The new two-dimensional shape of the resized ``Mask2D``.
        pad_value
            The value new values are padded using if the resized ``Mask2D`` is bigger.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
            )

            print(mask_2d.resized_from(new_shape=(4,4))
        """

        resized_mask = array_2d_util.resized_array_2d_from(
            array_2d=self.array,
            resized_shape=new_shape,
            pad_value=pad_value,
        ).astype("bool")

        return Mask2D(
            mask=resized_mask,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def is_circular(self) -> bool:
        """
        Returns whether the unmasked region of the mask is a filled circular disc.

        The check is robust to circles whose centre is offset from the coordinate origin, including offsets that
        do not align with pixel boundaries. It rejects annular, square and other non-disc shapes.

        The algorithm:

        1) The bounding box of unmasked pixels must be square to within one pixel. Offset centres that fall between
           pixel centres can produce a one-pixel asymmetry in the bounding box, which is allowed; anything larger
           indicates a non-circular shape such as an ellipse. Tiny masks (bounding box <= 2 pixels) require an exactly
           square bounding box, since the one-pixel slack would otherwise admit 1x2 strips.
        2) The pixel at the geometric centre of the bounding box must be unmasked. This rejects annular masks whose
           inner hole is at least one pixel wide.
        3) The centre and radius of the unmasked region are inferred from the bounding box and pixel count, and a
           reference circular mask is built with those parameters. The input mask must match the reference within a
           small number of pixels (tolerance scales with mask area to absorb rim quantization). This rejects squares,
           crosses and tight annuli that slipped past the earlier checks.

        This function does not support rectangular masks and an exception will be raised if the pixel scales in each
        direction are different.
        """

        if self.pixel_scales[0] != self.pixel_scales[1]:
            raise exc.MaskException(
                """
                The is_circular function cannot be called for a mask with different pixel scales in each dimension
                (e.g. it does not support rectangular masks.
                """
            )

        where = np.where(np.invert(self.array))
        if where[0].size == 0:
            return False

        y_min, y_max = int(where[0].min()), int(where[0].max())
        x_min, x_max = int(where[1].min()), int(where[1].max())
        y_extent = y_max - y_min + 1
        x_extent = x_max - x_min + 1

        if abs(y_extent - x_extent) > 1:
            return False
        if max(y_extent, x_extent) <= 2 and y_extent != x_extent:
            return False

        cy = (y_max + y_min) // 2
        cx = (x_max + x_min) // 2
        if bool(self.array[cy, cx]):
            return False

        actual_area = int(where[0].size)
        inferred_radius_pix = np.sqrt(actual_area / np.pi)
        inferred_radius = inferred_radius_pix * self.pixel_scales[0]

        y_centre_scaled = (
            0.5 * (self.shape_native[0] - 1) - 0.5 * (y_min + y_max)
        ) * self.pixel_scales[0]
        x_centre_scaled = (
            0.5 * (x_min + x_max) - 0.5 * (self.shape_native[1] - 1)
        ) * self.pixel_scales[1]

        expected = mask_2d_util.mask_2d_circular_from(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            radius=inferred_radius,
            centre=(y_centre_scaled, x_centre_scaled),
        )

        diff_count = int(np.sum(self.array != expected))
        tolerance = max(2, int(0.1 * actual_area))
        return diff_count <= tolerance

    @property
    def circular_radius(self) -> float:
        """
        Returns the radius in scaled units of a circular mask.

        The radius is computed from the bounding box of the unmasked region, taking the larger of the y and x extents
        as the diameter in pixels. This is robust to offset centres that fall between pixel boundaries (where the
        bounding box can be asymmetric by one pixel).

        The mask is first checked that it is circular using the `is_circular` property, with an exception raised if
        it is not.

        This function does not support rectangular masks and an exception will be raised if the pixel scales in each
        direction are different.

        Returns
        -------
        The circular radius of the mask in scaled units.
        """

        if not self.is_circular:
            raise exc.MaskException(
                """
                A circular radius can only be computed for a circular mask.

                The `is_circular` property of this mask has returned False, indicating the mask is not circular.
                """
            )

        where = np.where(np.invert(self.array))
        y_extent = int(where[0].max() - where[0].min() + 1)
        x_extent = int(where[1].max() - where[1].min() + 1)
        diameter = max(y_extent, x_extent)

        return diameter * self.pixel_scales[0] / 2.0
