import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple

from autoarray import exc


class AbstractRegion:
    def __init__(self, region):
        """
        Abstract base class for a region, which defines coordinates of a region on 1D or 2D data.

        Parameters
        ----------
        region
            The coordinates on the data of the region defined using pixel coordinates.
        """
        self.region = region

    def __getitem__(self, item):
        return self.region[item]

    def __eq__(self, other):
        if self.region == other:
            return True
        return super().__eq__(other)


class Region1D(AbstractRegion):
    def __init__(self, region: Tuple[int, int]):
        """
        A region of a 1D array defined as a tuple (x0, x1) = (left-pixel, right-pixel).

        For example, the overscan of 1D data may be defined by the region (40, 50), indicating it spans 10 pixels after
        40 rows of data.

        Parameters
        ----------
        region
            The coordinates on the 1D data of the region defined following the convention (x0, y1).
        """

        if region[0] < 0 or region[1] < 0:
            raise exc.RegionException(
                "A coordinate of the Region1D was specified as negative."
            )

        if region[0] >= region[1]:
            raise exc.RegionException(
                "The first pixel index in the Region1D was equal to or greater than the second pixel index."
            )

        super().__init__(region=region)

    def __repr__(self):
        return "<Region1D {} {}>".format(*self)

    @property
    def total_pixels(self) -> int:
        return self.x1 - self.x0

    @property
    def x0(self) -> int:
        return self[0]

    @property
    def x1(self) -> int:
        return self[1]

    @property
    def slice(self) -> ArrayLike:
        return np.s_[self.x0 : self.x1]

    @property
    def x_slice(self) -> ArrayLike:
        return np.s_[self.x0 : self.x1]

    def front_region_from(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> "Region1D":
        """
        Returns a `Region1D` corresponding to the front pixels in this region in the clocking direction (e.g. the
        pixels in the region that are closest to the read-out electronics which is defined at the 1D index (0,)).

        For example, if the `Region1D` covers the pixels (5, 15) and we input `pixels=(1,3)` this will return
        the 2nd to 4th pixels within the region corresponding to (6, 8).

        Other functions use the computed region to extract the front region from 1D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel coordinates used to compute the front region.
        """

        if pixels_from_end is not None:
            pixels = (self.total_pixels - pixels_from_end, self.total_pixels)

        x_min = self.x0 + pixels[0]
        x_max = self.x0 + pixels[1]

        return Region1D((x_min, x_max))

    def trailing_region_from(self, pixels: Tuple[int, int]) -> "Region1D":
        """
        Returns a `Region1D` corresponding to the pixels trailing this region in the clocking direction (e.g. the rows
        of pixels outside this region and in the direction away from the CCD read-out electronics defined at the 1D
        index (0,)).

        For example, if the `Region1D` covers the pixels (5, 15) and we input `pixels=(1,3)` this will return
        the 2nd to 4th pixels of trailing region corresponding to (16, 18).

        Other functions use the computed region to extract the trailing region from 1D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel coordinates used to compute the trailing region.
        """
        x_min = self.x1 + pixels[0]
        x_max = self.x1 + pixels[1]

        return Region1D((x_min, x_max))


class Region2D(AbstractRegion):
    def __init__(self, region: Tuple[int, int, int, int]):
        """
        A region of a 2D array defined as a tuple (y0, y1, x0, x1) = (top-row, bottom-row, left-column, right-column).

        For example, a the parallel overscan of an image may be defined by the region (100, 120, 10, 30),
        indicating it spans 20 rows at the end of the array over the columns defined between 10 and 30.

        Parameters
        ----------
        region
            The coordinates on the 2D array of the region defined following the convention (y0, y1, x0, y1).
        """

        if region[0] < 0 or region[1] < 0 or region[2] < 0 or region[3] < 0:
            raise exc.RegionException(
                "A coordinate of the Region2D was specified as negative."
            )

        if region[0] >= region[1]:
            raise exc.RegionException(
                "The first row in the Region2D was equal to or greater than the second row."
            )

        if region[2] >= region[3]:
            raise exc.RegionException(
                "The first column in the Region2D was equal to greater than the second column."
            )

        super().__init__(region=region)

    def __getitem__(self, item):
        return self.region[item]

    def __eq__(self, other):
        if self.region == other:
            return True
        return super().__eq__(other)

    def __repr__(self):
        return "<Region2D {} {} {} {}>".format(*self)

    @property
    def total_rows(self) -> int:
        return self.y1 - self.y0

    @property
    def total_columns(self) -> int:
        return self.x1 - self.x0

    @property
    def y0(self) -> int:
        return self[0]

    @property
    def y1(self) -> int:
        return self[1]

    @property
    def x0(self) -> int:
        return self[2]

    @property
    def x1(self) -> int:
        return self[3]

    @property
    def slice(self) -> ArrayLike:
        return np.s_[self.y0 : self.y1, self.x0 : self.x1]

    @property
    def y_slice(self) -> ArrayLike:
        return np.s_[self.y0 : self.y1]

    @property
    def x_slice(self) -> ArrayLike:
        return np.s_[self.x0 : self.x1]

    @property
    def shape(self):
        return self.y1 - self.y0, self.x1 - self.x0

    def serial_x_front_range_from(self, pixels: Tuple[int, int]) -> Tuple[int, int]:
        """
        Returns pixels defining the x range from the serial front edge of the region.

        For example, if the `Region2D` covers the pixels (5, 10, 0, 20) and we input `pixels=(1,3)` this will return
        the coordinates (6,8).

        Parameters
        ----------
        pixels
            A tuple defining the pixel columns used to compute the serial front edge range.
        """
        x_coord = self.x0
        x_min = x_coord + pixels[0]
        x_max = x_coord + pixels[1]

        return x_min, x_max

    def parallel_front_region_from(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> "Region2D":
        """
        Returns a `Region2D` corresponding to the front pixels in this region in the parallel clocking direction
        (e.g. the rows of pixels in the region that are closest to the CCD read-out electronics defined at 2D
        index (0,0)).

        For example, if the `Region2D` covers the pixels (0, 10, 0, 20) and we input `pixels=(1,6)` this will return
        the 2nd to 6th rows of the region corresponding to (1, 6, 0, 20).

        Other functions use the computed region to extract the parallel front region from 2D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel rows used to compute the parallel front region.
        pixels_from_end
            Alternative row pixel index specification, which extracts this number of pixels from the end of
            the region. For example, if the region is 100 pixels and `pixels_from_end=10`, the last 10 pixels of
            the region are extracted.
        """

        if pixels_from_end is not None:
            pixels = (self.total_rows - pixels_from_end, self.total_rows)

        y_coord = self.y0
        y_min = y_coord + pixels[0]
        y_max = y_coord + pixels[1]

        return Region2D((y_min, y_max, self.x0, self.x1))

    def parallel_trailing_region_from(
        self, pixels: Tuple[int, int] = (0, 1)
    ) -> "Region2D":
        """
        Returns a `Region2D` corresponding to the pixels trailing this region in the parallel clocking direction
        (e.g. the rows of pixels outside this region and in the direction away from the CCD read-out electronics
        defined at 2D index (0,0)).

        For example, if the `Region2D` covers the pixels (0, 10, 0, 20) and we input `pixels=(1,6)` this will return
        the 2nd to 6th rows of the trailing region corresponding to (10, 16, 0, 20).

        Other functions use the computed region to extract the parallel trailing region from 2D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel rows used to compute the parallel trailing region.
        """
        y_coord = self.y1
        y_min = y_coord + pixels[0]
        y_max = y_coord + pixels[1]

        return Region2D((y_min, y_max, self.x0, self.x1))

    def parallel_full_region_from(self, shape_2d: Tuple[int, int]) -> "Region2D":
        """
        Returns a `Region2D` corresponding to pixels which are inside this region and span all columns of the CCD.

        The returned region spans every column, for example if the image has a 2D shape (10, 20) the region will span
        all 20 columns irrespective of this region's coordinates.

        For example, if the `Region2D` covers the pixels (5, 10, 10, 20) and we input `shape_2d=(40, 60) this will
        return the 1st to 5th rows of the region towards the roe over the full 2D array corresponding to (5, 10, 0, 60).

        Other functions use the computed region to extract the parallel front region from 2D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel columns which are retained in the region.
        """
        return Region2D(region=(self.y0, self.y1, 0, shape_2d[1]))

    def serial_front_region_from(
        self,
        pixels: Optional[Tuple[int, int]] = None,
        pixels_from_end: Optional[int] = None,
    ) -> "Region2D":
        """
        Returns a `Region2D` corresponding to the front pixels in this region in the serial clocking direction
        (e.g. the columns of pixels in the region that are closest to the CCD read-out electronics defined at 2D
        index (0,0)).

        For example, if the `Region2D` covers the pixels (0, 10, 0, 20) and we input `pixels=(1,6)` this will return
        the 2nd to 6th rows of the region corresponding to (1, 6, 0, 20).

        Other functions use the computed region to extract the parallel front region from 2D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel columns used to compute the serial front region.
        """

        if pixels_from_end is not None:
            pixels = (self.total_columns - pixels_from_end, self.total_columns)

        x_min, x_max = self.serial_x_front_range_from(pixels)
        return Region2D(region=(self.y0, self.y1, x_min, x_max))

    def serial_trailing_region_from(
        self, pixels: Tuple[int, int] = (0, 1)
    ) -> "Region2D":
        """
        Returns a `Region2D` corresponding to the pixels trailing this region in the serial clocking direction
        (e.g. the columns of pixels outside this region and in the direction away from the CCD read-out electronics
        defined at 2D index (0,0)).

        For example, if the `Region2D` covers the pixels (0, 10, 0, 20) and we input `pixels=(1,6)` this will return
        the 2nd to 6th rows of the trailing region corresponding to (0, 10, 20, 26).

        Other functions use the computed region to extract the parallel trailing region from 2D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel columns used to compute the parallel trailing region.
        """
        x_coord = self.x1
        x_min = x_coord + pixels[0]
        x_max = x_coord + pixels[1]

        return Region2D(region=(self.y0, self.y1, x_min, x_max))

    def serial_towards_roe_full_region_from(
        self, shape_2d: Tuple[int, int], pixels: Tuple[int, int] = (0, 1)
    ) -> "Region2D":
        """
        Returns a `Region2D` corresponding to pixels which are inside this region and towards the CCD read-out
        electronics (which **PyAutoArray** internally defines at (0, 0)).

        The returned region spans every row, for example if the image has a 2D shape (20, 10) the region will span
        all 20 rows irrespective of this region's coordinates.

        For example, if the `Region2D` covers the pixels (5, 10, 10, 20) and we input `shape_2d=(40, 60) and
        `pixels=(0,5)` this will return the 1st to 5th columns of the region towards the roe
        over the full 2D array corresponding to (0, 40, 10, 15).

        Other functions use the computed region to extract the parallel front region from 2D arrays containing data.

        Parameters
        ----------
        pixels
            A tuple defining the pixel columns which are retained in the region.
        """
        x_min, x_max = self.serial_x_front_range_from(pixels)

        return Region2D(region=(0, shape_2d[0], x_min, x_max))
