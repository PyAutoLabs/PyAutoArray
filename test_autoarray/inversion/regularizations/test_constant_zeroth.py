import autoarray as aa
import numpy as np
import pytest

from autoarray.inversion.regularization.constant_zeroth import (
    constant_zeroth_regularization_matrix_from,
)


def test__regularization_matrix_from__shape_is_pixels_by_pixels():
    # 3-pixel chain (0-1, 0-2) with P=2 neighbor columns, so S != P and the
    # zeroth term must be built from S, not the neighbor count.
    neighbors = np.array([[1, 2], [0, -1], [0, -1]])
    neighbors_sizes = np.array([2, 1, 1])

    regularization_matrix = constant_zeroth_regularization_matrix_from(
        coefficient=1.0,
        coefficient_zeroth=2.0,
        neighbors=neighbors,
        neighbors_sizes=neighbors_sizes,
    )

    assert regularization_matrix.shape == (3, 3)


def test__regularization_matrix_from__no_null_mode__smallest_eigenvalue_at_least_coefficient_zeroth_squared():
    neighbors = np.array(
        [[1, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1], [0, 2, -1, -1]]
    )
    neighbors_sizes = np.array([2, 2, 2, 2])

    coefficient_zeroth = 2.0

    regularization_matrix = constant_zeroth_regularization_matrix_from(
        coefficient=1.0,
        coefficient_zeroth=coefficient_zeroth,
        neighbors=neighbors,
        neighbors_sizes=neighbors_sizes,
    )

    eigenvalues = np.linalg.eigvalsh(regularization_matrix)

    assert np.min(eigenvalues) >= coefficient_zeroth**2.0


def test__regularization_matrix_from__reduces_to_constant_matrix_plus_scaled_identity():
    neighbors = np.array(
        [[1, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1], [0, 2, -1, -1]]
    )
    neighbors_sizes = np.array([2, 2, 2, 2])

    coefficient = 3.0
    coefficient_zeroth = 2.0

    constant_matrix = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=coefficient,
        neighbors=neighbors,
        neighbors_sizes=neighbors_sizes,
    )

    constant_zeroth_matrix = constant_zeroth_regularization_matrix_from(
        coefficient=coefficient,
        coefficient_zeroth=coefficient_zeroth,
        neighbors=neighbors,
        neighbors_sizes=neighbors_sizes,
    )

    assert constant_zeroth_matrix == pytest.approx(
        constant_matrix + coefficient_zeroth**2.0 * np.identity(4), 1.0e-8
    )


def test__regularization_matrix_from__class_api_returns_without_raising():
    reg = aa.reg.ConstantZeroth(coefficient_neighbor=2.0, coefficient_zeroth=3.0)

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

    assert regularization_matrix.shape == (9, 9)
    # Constant leg diagonal (8.0000001 for coefficient 2, see test_constant.py)
    # plus the zeroth term coefficient_zeroth**2 = 9.
    assert regularization_matrix[0, 0] == pytest.approx(17.0000001, 1.0e-4)
