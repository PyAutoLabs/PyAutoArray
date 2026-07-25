import pytest

import autoarray as aa
import numpy as np

np.set_printoptions(threshold=np.inf)


def test__regularization_matrix_from__6_pixel_mesh_coefficient_1_scale_2__correct_diagonal_value():
    reg = aa.reg.ExponentialKernel(coefficient=1.0, scale=2.0)

    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mapper = aa.m.MockMapper(source_plane_mesh_grid=source_plane_mesh_grid)

    regularization_matrix = reg.regularization_matrix_from(linear_obj=mapper)

    assert regularization_matrix[0, 0] == pytest.approx(1.71521, 1.0e-4)
