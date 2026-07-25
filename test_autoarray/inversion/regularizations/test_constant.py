import autoarray as aa
import numpy as np
import pytest

np.set_printoptions(threshold=np.inf)


def test__regularization_matrix_from__6_pixel_mesh_coefficient_2__diagonal_value_approximately_8():

    reg = aa.reg.Constant(coefficient=2.0)

    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mesh_geometry = aa.MeshGeometryRectangular(
        mesh=aa.mesh.RectangularUniform(shape=(3, 3)),
        mesh_grid=source_plane_mesh_grid,
        data_grid=None,
    )

    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid, mesh_geometry=mesh_geometry
    )

    regularization_matrix = reg.regularization_matrix_from(linear_obj=mapper)

    assert regularization_matrix[0, 0] == pytest.approx(8.0000001, 1.0e-4)
