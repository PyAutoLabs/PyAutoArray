import autoarray as aa
import numpy as np
import pytest


def test__regularization_weights_from__3_pixel_signals_inner_10_outer_15__matches_util():
    reg = aa.reg.Adapt(inner_coefficient=10.0, outer_coefficient=15.0)

    pixel_signals = np.array([0.21, 0.586, 0.45])

    mapper = aa.m.MockMapper(pixel_signals=pixel_signals)

    weight_list = reg.regularization_weights_from(linear_obj=mapper)

    weight_list_util = aa.util.regularization.adapt_regularization_weights_from(
        inner_coefficient=10.0, outer_coefficient=15.0, pixel_signals=pixel_signals
    )

    assert (weight_list == weight_list_util).all()


def test__regularization_matrix_from__9_pixel_grid_inner_1_outer_2__diagonal_value_approximately_18():
    reg = aa.reg.Adapt(inner_coefficient=1.0, outer_coefficient=2.0, signal_scale=1.0)

    pixel_signals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[
            [0.1, 0.1],
            [0.1, 0.2],
            [0.1, 0.3],
            [0.2, 0.1],
            [0.2, 0.2],
            [0.2, 0.3],
            [0.3, 0.1],
            [0.3, 0.2],
            [0.3, 0.3],
        ],
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    mesh_geometry = aa.MeshGeometryRectangular(
        mesh=aa.mesh.RectangularUniform(shape=(3, 3)),
        mesh_grid=source_plane_mesh_grid,
        data_grid=None,
    )

    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid,
        pixel_signals=pixel_signals,
        mesh_geometry=mesh_geometry,
    )

    regularization_matrix = reg.regularization_matrix_from(linear_obj=mapper)

    assert regularization_matrix[0, 0] == pytest.approx(18.0000000, 1.0e-4)
