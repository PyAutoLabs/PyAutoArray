import os

import numpy as np
import pytest
import autoarray as aa
from autoarray import exc


def test__slice_1d__extract():
    arr_1d = aa.Array1D.no_mask(
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), pixel_scales=1.0
    )

    region = aa.Region1D(region=(0, 2))

    new_arr_1d = arr_1d[region.slice]

    assert (new_arr_1d == np.array([1.0, 2.0])).all()

    region = aa.Region1D(region=(1, 3))

    new_arr_1d = arr_1d[region.slice]

    assert (new_arr_1d == np.array([2.0, 3.0])).all()


def test__slice_1d__addition():
    arr_1d = aa.Array1D.no_mask(values=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0)

    image = aa.Array1D.full(fill_value=1.0, shape_native=6, pixel_scales=1.0)

    region = aa.Region1D(region=(0, 1))

    arr_1d[region.slice] += image[region.slice]

    assert (arr_1d == np.array([2.0, 2.0, 3.0, 4.0])).all()

    arr_1d = aa.Array1D.no_mask(values=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0)

    region = aa.Region1D(region=(2, 4))

    arr_1d[region.slice] += image[region.slice]

    assert (arr_1d == np.array([1.0, 2.0, 4.0, 5.0])).all()


def test__slice_1d__set_to_zeros():
    arr_1d = aa.Array1D.no_mask(values=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0)

    region = aa.Region1D(region=(0, 1))

    arr_1d[region.slice] = 0

    assert (arr_1d == np.array([0.0, 2.0, 3.0, 4.0])).all()

    arr_1d = aa.Array1D.no_mask(values=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0)

    region = aa.Region1D(region=(2, 4))

    arr_1d[region.slice] = 0

    assert (arr_1d == np.array([1.0, 2.0, 0.0, 0.0])).all()


def test__front_region_from():
    region = aa.Region1D(region=(0, 3))

    # Front edge is pixel 0, so for 1 pixel we extract 0 -> 1

    front_edge_region = region.front_region_from(pixels=(0, 1))

    assert front_edge_region == (0, 1)

    # Front edge is pixel 0, so for 2 pixels we extract 0 -> 2

    front_edge_region = region.front_region_from(pixels=(0, 2))

    assert front_edge_region == (0, 2)

    # Front edge is pixel 0, so for these 2 pixels we extract 1 ->2

    front_edge_region = region.front_region_from(pixels=(1, 3))

    assert front_edge_region == (1, 3)


def test__trailing_region_from():
    region = aa.Region1D(region=(0, 3))

    # Front edge ends pixel 3, so for 1 pixel we extract 3 -> 4

    trails_region = region.trailing_region_from(pixels=(0, 1))

    assert trails_region == (3, 4)

    # Front edge ends pixel 3, so for 2 pixels we extract 3 -> 5

    trails_region = region.trailing_region_from(pixels=(0, 2))

    assert trails_region == (3, 5)

    # Front edge ends pixel 3, so for these 2 pixels we extract 3 ->6

    trails_region = region.trailing_region_from(pixels=(1, 3))

    assert trails_region == (4, 6)


def test__exception__first_pixel_or_column_equal_too_or_bigger_than_second():
    with pytest.raises(exc.RegionException):
        aa.Region1D(region=(2, 2))

    with pytest.raises(exc.RegionException):
        aa.Region1D(region=(2, 1))


def test__exception__negative_coordinates_1d():
    with pytest.raises(exc.RegionException):
        aa.Region1D(region=(-1, 0))

    with pytest.raises(exc.RegionException):
        aa.Region1D(region=(0, -1))


def test__slice_2d__extraction():
    array = aa.Array2D.no_mask(
        values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        pixel_scales=1.0,
    )

    region = aa.Region2D(region=(0, 2, 0, 2))

    new_array = array.native[region.slice]

    assert (new_array == np.array([[1.0, 2.0], [4.0, 5.0]])).all()

    array = aa.Array2D.no_mask(
        values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        pixel_scales=1.0,
    )

    region = aa.Region2D(region=(1, 3, 0, 3))

    new_array = array.native[region.slice]

    assert (new_array == np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])).all()


def test__slice_2d__addition():
    array = aa.Array2D.no_mask(values=np.zeros((2, 2)), pixel_scales=1.0)
    array = array.native
    image = np.ones((2, 2))
    region = aa.Region2D(region=(0, 1, 0, 1))

    array[region.slice] += image[region.slice]

    assert (array == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

    array = aa.Array2D.no_mask(values=np.ones((2, 2)), pixel_scales=1.0)
    array = array.native
    image = np.ones((2, 2))
    region = aa.Region2D(region=(0, 1, 0, 1))

    array[region.slice] += image[region.slice]

    assert (array == np.array([[2.0, 1.0], [1.0, 1.0]])).all()

    array = aa.Array2D.no_mask(values=np.ones((3, 3)), pixel_scales=1.0)
    array = array.native
    image = np.ones((3, 3))
    region = aa.Region2D(region=(1, 3, 2, 3))

    array[region.slice] += image[region.slice]

    assert (
        array == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
    ).all()


def test__slice_2d__set_to_zerose():
    array = aa.Array2D.no_mask(values=np.ones((2, 2)), pixel_scales=1.0)
    array = array.native

    region = aa.Region2D(region=(0, 1, 0, 1))

    array[region.slice] = 0

    assert (array == np.array([[0.0, 1.0], [1.0, 1.0]])).all()

    array = aa.Array2D.no_mask(values=np.ones((3, 3)), pixel_scales=1.0)
    array = array.native

    region = aa.Region2D(region=(1, 3, 2, 3))

    array[region.slice] = 0

    assert (
        array == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ).all()


def test__parallel_front_region_from():
    region = aa.Region2D(region=(0, 3, 0, 3))

    front_edge = region.parallel_front_region_from(pixels=(0, 1))

    assert front_edge == (0, 1, 0, 3)

    front_edge = region.parallel_front_region_from(pixels=(0, 2))

    assert front_edge == (0, 2, 0, 3)

    front_edge = region.parallel_front_region_from(pixels=(1, 3))

    assert front_edge == (1, 3, 0, 3)


def test__parallel_front_region_from__pixels_from_end():
    region = aa.Region2D(region=(0, 3, 0, 3))

    front_edge = region.parallel_front_region_from(pixels_from_end=1)

    assert front_edge == (2, 3, 0, 3)

    front_edge = region.parallel_front_region_from(pixels_from_end=2)

    assert front_edge == (1, 3, 0, 3)

    front_edge = region.parallel_front_region_from(pixels_from_end=3)

    assert front_edge == (0, 3, 0, 3)


def test__parallel_trailing_region_from():
    region = aa.Region2D(region=(0, 3, 0, 3))

    trails = region.parallel_trailing_region_from(pixels=(0, 1))

    assert trails == (3, 4, 0, 3)

    trails = region.parallel_trailing_region_from(pixels=(0, 2))

    assert trails == (3, 5, 0, 3)

    trails = region.parallel_trailing_region_from(pixels=(1, 3))

    assert trails == (4, 6, 0, 3)


def test__parallel_full_region_from():
    region = aa.Region2D(region=(1, 3, 0, 5))

    serial_region = region.parallel_full_region_from(shape_2d=(5, 5))

    assert serial_region == (1, 3, 0, 5)

    region = aa.Region2D(region=(1, 3, 0, 5))

    serial_region = region.parallel_full_region_from(shape_2d=(5, 25))

    assert serial_region == (1, 3, 0, 25)

    region = aa.Region2D(region=(3, 5, 5, 30))

    serial_region = region.parallel_full_region_from(shape_2d=(8, 55))

    assert serial_region == (3, 5, 0, 55)


def test__serial_front_region_from():
    region = aa.Region2D(region=(0, 3, 0, 3))

    front_edge = region.serial_front_region_from(pixels=(0, 1))

    assert front_edge == (0, 3, 0, 1)

    front_edge = region.serial_front_region_from(pixels=(0, 2))

    assert front_edge == (0, 3, 0, 2)

    front_edge = region.serial_front_region_from(pixels=(1, 3))

    assert front_edge == (0, 3, 1, 3)


def test__serial_trailing_region_from():
    region = aa.Region2D(
        region=(0, 3, 0, 3)
    )  # The trails are column 3 and above, so extract 3 -> 4

    trails = region.serial_trailing_region_from(pixels=(0, 1))

    assert trails == (0, 3, 3, 4)

    # The trails are column 3 and above, so extract 3 -> 5

    trails = region.serial_trailing_region_from(pixels=(0, 2))

    assert trails == (0, 3, 3, 5)

    # The trails are column 3 and above, so extract 4 -> 6

    trails = region.serial_trailing_region_from(pixels=(1, 3))

    assert trails == (0, 3, 4, 6)


def test__serial_towards_roe_full_region_from():
    region = aa.Region2D(region=(1, 3, 0, 5))

    parallel_region = region.serial_towards_roe_full_region_from(
        shape_2d=(5, 5), pixels=(0, 1)
    )

    assert parallel_region == (0, 5, 0, 1)

    parallel_region = region.serial_towards_roe_full_region_from(
        shape_2d=(4, 4), pixels=(1, 3)
    )

    assert parallel_region == (0, 4, 1, 3)

    region = aa.Region2D(region=(1, 3, 2, 5))

    parallel_region = region.serial_towards_roe_full_region_from(
        shape_2d=(4, 4), pixels=(1, 3)
    )

    assert parallel_region == (0, 4, 3, 5)

    region = aa.Region2D(region=(1, 3, 0, 5))

    parallel_region = region.serial_towards_roe_full_region_from(
        shape_2d=(2, 5), pixels=(0, 1)
    )

    assert parallel_region == (0, 2, 0, 1)


def test__exception__first_row_or_column_equal_too_or_bigger_than_second():
    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(2, 2, 1, 2))

    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(2, 1, 2, 2))

    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(2, 1, 1, 2))

    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(0, 1, 3, 2))


def test__exception__negative_coordinates_2d():
    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(-1, 0, 1, 2))

    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(0, -1, 1, 2))

    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(0, 0, -1, 2))

    with pytest.raises(exc.RegionException):
        aa.Region2D(region=(0, 1, 2, -1))
