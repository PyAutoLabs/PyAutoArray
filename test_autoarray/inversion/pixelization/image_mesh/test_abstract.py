import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="mask")
def make_mask():
    return aa.Mask2D.circular(
        shape_native=(3, 3),
        radius=2.0,
        pixel_scales=1.0,
    )


@pytest.fixture(name="mesh_grid")
def make_mesh_grid():
    return aa.Grid2DIrregular(
        values=[
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
        ]
    )


@pytest.fixture(name="image_mesh")
def make_image_mesh():
    return aa.image_mesh.Hilbert(pixels=8)


def test__mesh_pixels_per_image_pixels_from(mask, mesh_grid, image_mesh):
    mesh_pixels_per_image_pixels = image_mesh.mesh_pixels_per_image_pixels_from(
        mask=mask, mesh_grid=mesh_grid
    )

    assert mesh_pixels_per_image_pixels.native == pytest.approx(
        np.array([[0, 0, 0], [0, 3, 2], [1, 0, 0]]), 1.0e-4
    )


def test__append_with_circle_edge_points():

    image_plane_mesh_grid = aa.Grid2DIrregular(
        values=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    )

    grid_with_circle = aa.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=(0.0, 0.0),
        radius=3.0,
        n_points=4,
    )

    assert grid_with_circle == pytest.approx(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [0.0, 3.0],
                [3.0, 0.0],
                [0.0, -3.0],
                [-3.0, 0.0],
            ]
        ),
        1.0e-4,
    )


def test__check_mesh_pixels_per_image_pixels(mask, mesh_grid, image_mesh):
    image_mesh.check_mesh_pixels_per_image_pixels(
        mask=mask,
        mesh_grid=mesh_grid,
    )

    image_mesh.check_mesh_pixels_per_image_pixels(
        mask=mask,
        mesh_grid=mesh_grid,
        image_mesh_min_mesh_pixels_per_pixel=3,
        image_mesh_min_mesh_number=1,
    )

    with pytest.raises(aa.exc.InversionException):
        image_mesh.check_mesh_pixels_per_image_pixels(
            mask=mask,
            mesh_grid=mesh_grid,
            image_mesh_min_mesh_pixels_per_pixel=5,
            image_mesh_min_mesh_number=1,
        )

    with pytest.raises(aa.exc.InversionException):
        image_mesh.check_mesh_pixels_per_image_pixels(
            mask=mask,
            mesh_grid=mesh_grid,
            image_mesh_min_mesh_pixels_per_pixel=3,
            image_mesh_min_mesh_number=2,
        )


def test__check_adapt_background_pixels(mask, mesh_grid, image_mesh):
    adapt_data = aa.Array2D.no_mask(
        values=[[0.05, 0.05, 0.05], [0.05, 0.6, 0.05], [0.05, 0.05, 0.05]],
        pixel_scales=(1.0, 1.0),
    )

    image_mesh.check_adapt_background_pixels(
        mask=mask,
        mesh_grid=mesh_grid,
        adapt_data=adapt_data,
    )

    image_mesh.check_adapt_background_pixels(
        mask=mask,
        mesh_grid=mesh_grid,
        adapt_data=adapt_data,
        image_mesh_adapt_background_percent_threshold=0.05,
        image_mesh_adapt_background_percent_check=0.9,
    )

    with pytest.raises(aa.exc.InversionException):
        image_mesh.check_adapt_background_pixels(
            mask=mask,
            mesh_grid=mesh_grid,
            adapt_data=adapt_data,
            image_mesh_adapt_background_percent_threshold=0.8,
            image_mesh_adapt_background_percent_check=0.5,
        )
