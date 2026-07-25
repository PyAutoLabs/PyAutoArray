import pytest

import autoarray as aa
import numpy as np

np.set_printoptions(threshold=np.inf)


def test__regularization_matrix():

    reg = aa.reg.MaternAdaptKernel(
        scale=0.1,
        nu=0.5,
        inner_coefficient=0.1,
        outer_coefficient=0.2,
        signal_scale=0.1,
    )

    pixel_signals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid, pixel_signals=pixel_signals
    )

    regularization_matrix = reg.regularization_matrix_from(linear_obj=mapper)

    # assert regularization_matrix[0, 0] == pytest.approx(18.7439565009, 1.0e-4)
    # assert regularization_matrix[0, 1] == pytest.approx(-8.786547368, 1.0e-4)
