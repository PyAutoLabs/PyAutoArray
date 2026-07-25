import numpy as np
import pytest

import autoarray as aa


def test__central_pixel_coordinates():
    geometry = aa.Geometry2D(shape_native=(3, 3), pixel_scales=(0.1, 0.1))

    central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(3, 3)
    )

    assert geometry.central_pixel_coordinates == central_pixel_coordinates_util

    geometry = aa.Geometry2D(shape_native=(5, 3), pixel_scales=(2.0, 1.0))

    central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(5, 3)
    )

    assert geometry.central_pixel_coordinates == central_pixel_coordinates_util


def test__pixel_coordinates_2d_from():
    geometry = aa.Geometry2D(
        shape_native=(6, 7), pixel_scales=(2.4, 1.8), origin=(1.0, 1.5)
    )

    pixel_coordinates_util = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.3, 1.2),
        shape_native=(6, 7),
        pixel_scales=(2.4, 1.8),
        origins=(1.0, 1.5),
    )

    assert (
        geometry.pixel_coordinates_2d_from(scaled_coordinates_2d=(2.3, 1.2))
        == pixel_coordinates_util
    )


def test__scaled_coordinates_2d_from():
    geometry = aa.Geometry2D(
        shape_native=(6, 7), pixel_scales=(2.4, 1.8), origin=(1.0, 1.5)
    )

    pixel_coordinates_util = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(5, 4),
        shape_native=(6, 7),
        pixel_scales=(2.4, 1.8),
        origins=(1.0, 1.5),
    )

    assert (
        geometry.scaled_coordinates_2d_from(pixel_coordinates_2d=(5, 4))
        == pixel_coordinates_util
    )


def test__scaled_coordinate_2d_to_scaled_at_pixel_centre_from():
    geometry = aa.Geometry2D(
        shape_native=(5, 5),
        pixel_scales=(0.1, 0.1),
    )

    scaled_coordinate = (0.0, 0.0)

    pixel_centre = geometry.scaled_coordinate_2d_to_scaled_at_pixel_centre_from(
        scaled_coordinate_2d=scaled_coordinate,
    )

    assert pixel_centre == pytest.approx((0.0, 0.0), 1.0e-4)

    scaled_coordinate = (0.12, 0.22)

    pixel_centre = geometry.scaled_coordinate_2d_to_scaled_at_pixel_centre_from(
        scaled_coordinate_2d=scaled_coordinate,
    )

    assert pixel_centre == pytest.approx((0.1, 0.2), 1.0e-4)

    geometry = aa.Geometry2D(
        shape_native=(6, 6),
        pixel_scales=(0.2, 0.2),
    )

    scaled_coordinate = (-0.0001, -0.2001)

    pixel_centre = geometry.scaled_coordinate_2d_to_scaled_at_pixel_centre_from(
        scaled_coordinate_2d=scaled_coordinate,
    )

    assert pixel_centre == pytest.approx((-0.1, -0.3), 1.0e-4)

    geometry = aa.Geometry2D(
        shape_native=(6, 6), pixel_scales=(0.2, 0.2), origin=(1.0, 1.0)
    )

    scaled_coordinate = (1.0001, 1.2001)

    pixel_centre = geometry.scaled_coordinate_2d_to_scaled_at_pixel_centre_from(
        scaled_coordinate_2d=scaled_coordinate,
    )

    assert pixel_centre == pytest.approx((1.1, 1.3), 1.0e-4)


def test__grid_pixels_2d_slim_from():
    geometry = aa.Geometry2D(shape_native=(2, 2), pixel_scales=(2.0, 4.0))

    grid_scaled_2d = aa.Grid2D.no_mask(
        values=[[[1.0, -2.0], [1.0, 2.0]], [[-1.0, -2.0], [-1.0, 2.0]]],
        pixel_scales=geometry.pixel_scales,
    )

    grid_pixels_util = aa.util.geometry.grid_pixels_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled_2d,
        shape_native=(2, 2),
        pixel_scales=geometry.pixel_scales,
    )
    grid_pixels = geometry.grid_pixels_2d_from(grid_scaled_2d=grid_scaled_2d)

    assert (grid_pixels == grid_pixels_util).all()
    assert (grid_pixels.slim == grid_pixels_util).all()


def test__grid_pixel_centres_2d_from():
    geometry = aa.Geometry2D(shape_native=(2, 2), pixel_scales=(7.0, 2.0))

    grid_scaled_2d = aa.Grid2D.no_mask(
        values=[[[1.0, -2.0], [1.0, 2.0]], [[-1.0, -2.0], [-1.0, 2.0]]],
        pixel_scales=geometry.pixel_scales,
    )

    grid_pixels_util = aa.util.geometry.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled_2d,
        shape_native=(2, 2),
        pixel_scales=(7.0, 2.0),
    )

    grid_pixels = geometry.grid_pixel_centres_2d_from(grid_scaled_2d=grid_scaled_2d)

    assert (grid_pixels == grid_pixels_util).all()


def test__grid_pixel_indexes_2d_from():
    geometry = aa.Geometry2D(shape_native=(2, 2), pixel_scales=(2.0, 4.0))

    grid_scaled_2d = aa.Grid2D.no_mask(
        values=[[[1.0, -2.0], [1.0, 2.0]], [[-1.0, -2.0], [-1.0, 2.0]]],
        pixel_scales=geometry.pixel_scales,
    )

    grid_pixels_util = aa.util.geometry.grid_pixel_indexes_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled_2d,
        shape_native=(2, 2),
        pixel_scales=(2.0, 4.0),
    )

    grid_pixels = geometry.grid_pixel_indexes_2d_from(grid_scaled_2d=grid_scaled_2d)

    assert (grid_pixels == grid_pixels_util).all()


def test__grid_scaled_2d_from():
    geometry = aa.Geometry2D(shape_native=(2, 2), pixel_scales=(2.0, 2.0))

    grid_pixels = aa.Grid2D.no_mask(
        values=[[[0, 0], [0, 1]], [[1, 0], [1, 1]]],
        pixel_scales=geometry.pixel_scales,
    )

    grid_pixels_util = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels,
        shape_native=(2, 2),
        pixel_scales=(2.0, 2.0),
    )
    grid_pixels = geometry.grid_scaled_2d_from(grid_pixels_2d=grid_pixels)

    assert (grid_pixels == grid_pixels_util).all()
