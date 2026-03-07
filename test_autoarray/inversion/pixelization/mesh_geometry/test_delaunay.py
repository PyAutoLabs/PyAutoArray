import numpy as np
import pytest

import autoarray as aa


def test__neighbors(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_geometry = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7,
    )

    neighbors = mesh_geometry.neighbors

    expected = np.array(
        [
            [1, 2, 3, 4],
            [0, 2, 3, 5],
            [0, 1, 5, -1],
            [0, 1, 4, 5],
            [0, 3, 5, -1],
            [1, 2, 3, 4],
        ]
    )

    assert all(
        set(neighbors[i]) - {-1} == set(expected[i]) - {-1}
        for i in range(neighbors.shape[0])
    )


def test__voronoi_areas_via_delaunay_from(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2DIrregular(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7.over_sampled,
    )

    voronoi_areas = mesh.voronoi_areas

    assert voronoi_areas[1] == pytest.approx(1.39137102, 1.0e-4)
    assert voronoi_areas[3] == pytest.approx(29.836324, 1.0e-4)
    assert voronoi_areas[4] == pytest.approx(-1.0, 1.0e-4)
