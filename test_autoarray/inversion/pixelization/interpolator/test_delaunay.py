import numpy as np
import pytest

import autoarray as aa


def test__scipy_delaunay__simplices(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7,
    )

    assert (mesh_grid.delaunay.simplices[0, :] == np.array([3, 4, 0])).all()
    assert (mesh_grid.delaunay.simplices[1, :] == np.array([3, 5, 4])).all()
    assert (mesh_grid.delaunay.simplices[-1, :] == np.array([-1, -1, -1])).all()


def test__scipy_delaunay__split(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7,
    )

    assert mesh_grid.delaunay.split_points[0, :] == pytest.approx(
        [0.45059473, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[1, :] == pytest.approx(
        [-0.25059473, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[-1, :] == pytest.approx(
        [2.1, 0.39142161], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[0, :] == pytest.approx(
        [2, 1, 0], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[1, :] == pytest.approx(
        [0, -1, -1], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[-1, :] == pytest.approx(
        [5, 1, 2], 1.0e-4
    )
