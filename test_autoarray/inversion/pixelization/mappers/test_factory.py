import numpy as np
import pytest

import autoarray as aa


def test__rectangular_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=2.0,
        origin=(0.5, 0.5),
    )

    # Slightly manipulate input grid so sub gridding is evidence in first source pixel.
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)
    grid.over_sampled[0, 0] = -2.0
    grid.over_sampled[0, 1] = 2.0

    mesh = aa.mesh.RectangularUniform(shape=(3, 3))

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid, source_plane_mesh_grid=None
    )

    mapper = aa.Mapper(interpolator=interpolator)

    assert mapper.mesh_geometry.geometry.shape_native_scaled == pytest.approx(
        (5.0, 5.0), 1.0e-4
    )
    assert mapper.mesh_geometry.origin == pytest.approx((0.5, 0.5), 1.0e-4)
    assert mapper.mapping_matrix == pytest.approx(
        np.array(
            [
                [0.0675, 0.5775, 0.18, 0.0075, -0.065, -0.1425, 0.0, 0.0375, 0.3375],
                [0.18, -0.03, 0.0, 0.84, -0.14, 0.0, 0.18, -0.03, 0.0],
                [0.0225, 0.105, 0.0225, 0.105, 0.49, 0.105, 0.0225, 0.105, 0.0225],
                [0.0, -0.03, 0.18, 0.0, -0.14, 0.84, 0.0, -0.03, 0.18],
                [0.0, 0.0, 0.0, -0.03, -0.14, -0.03, 0.18, 0.84, 0.18],
            ]
        ),
        1.0e-4,
    )
    assert mapper.mesh_geometry.shape_native == (3, 3)


def test__delaunay_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    # Slightly manipulate input grid so sub gridding is evidence in first source pixel.
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)

    grid.over_sampled[0, 0] = -2.0
    grid.over_sampled[0, 1] = 2.0

    mesh = aa.mesh.Delaunay(pixels=5)
    image_mesh = aa.image_mesh.Overlay(shape=(3, 3))
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_plane_mesh_grid,
    )

    mapper = aa.Mapper(interpolator=interpolator)

    assert (mapper.source_plane_mesh_grid == image_plane_mesh_grid).all()
    assert mapper.mesh_geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

    assert mapper.mapping_matrix == pytest.approx(
        np.array(
            [
                [0.625, 0.0625, 0.0, 0.3125, 0.0],
                [0.0625, 0.875, 0.0, 0.0, 0.0625],
                [0.125, 0.125, 0.5, 0.125, 0.125],
                [0.0625, 0.0, 0.0, 0.875, 0.0625],
                [0.0, 0.0625, 0.0, 0.0625, 0.875],
            ]
        ),
        1.0e-2,
    )
