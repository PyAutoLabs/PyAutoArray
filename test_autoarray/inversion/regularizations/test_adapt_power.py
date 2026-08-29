"""
The ``AdaptPower`` family: the corrected siblings of the ``Adapt`` family.

Two things separate them from the legacy classes, and both are asserted here:

1. the coefficient enters the regularization matrix at ``2 * power`` (default ``lambda^2``, the
   ``Constant`` convention) rather than always at ``lambda^4``;
2. every mesh edge is scattered once rather than twice, so the non-split class equals ``Constant``
   exactly instead of being ``2 x`` it.

The legacy classes are untouched — their own tests stay as they are.
"""

import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="rectangular_mapper_9")
def make_rectangular_mapper_9():
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

    return aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid,
        pixel_signals=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        mesh_geometry=mesh_geometry,
    )


@pytest.mark.parametrize("coefficient", [0.5, 2.0, 7.0])
def test__uniform_coefficients__matrix_equals_constant_exactly(
    rectangular_mapper_9, coefficient
):
    """
    The contract ``Adapt``'s docstring always claimed and never delivered: with
    ``inner_coefficient == outer_coefficient`` the adaptive weighting is uniform, so the scheme must
    reduce to ``Constant`` of the same coefficient.
    """
    regularization_matrix = aa.reg.AdaptPower(
        inner_coefficient=coefficient, outer_coefficient=coefficient
    ).regularization_matrix_from(linear_obj=rectangular_mapper_9)

    regularization_matrix_constant = aa.reg.Constant(
        coefficient=coefficient
    ).regularization_matrix_from(linear_obj=rectangular_mapper_9)

    assert regularization_matrix == pytest.approx(
        regularization_matrix_constant, abs=1.0e-12
    )


def test__legacy_adapt_is_twice_constant__documents_the_scatter_asymmetry(
    rectangular_mapper_9,
):
    """
    A characterisation test for the legacy behaviour the ``*Power`` classes fix: ``Adapt`` scatters
    every edge twice, so it is ``2 x`` ``Constant`` (up to the shared ``1e-8`` diagonal floor).
    """
    floor = 1.0e-8 * np.eye(9)

    regularization_matrix_adapt = aa.reg.Adapt(
        inner_coefficient=3.0, outer_coefficient=3.0
    ).regularization_matrix_from(linear_obj=rectangular_mapper_9)

    regularization_matrix_constant = aa.reg.Constant(
        coefficient=9.0
    ).regularization_matrix_from(linear_obj=rectangular_mapper_9)

    assert regularization_matrix_adapt - floor == pytest.approx(
        2.0 * (regularization_matrix_constant - floor), rel=1.0e-12
    )


def test__weights_are_unsquared_by_default__and_squared_at_power_2(
    rectangular_mapper_9,
):
    weights_legacy = aa.reg.Adapt(
        inner_coefficient=1.0, outer_coefficient=2.0
    ).regularization_weights_from(linear_obj=rectangular_mapper_9)

    weights_power_1 = aa.reg.AdaptPower(
        inner_coefficient=1.0, outer_coefficient=2.0
    ).regularization_weights_from(linear_obj=rectangular_mapper_9)

    weights_power_2 = aa.reg.AdaptPower(
        inner_coefficient=1.0, outer_coefficient=2.0, power=2.0
    ).regularization_weights_from(linear_obj=rectangular_mapper_9)

    assert weights_power_1**2.0 == pytest.approx(weights_legacy, 1.0e-12)
    assert (weights_power_2 == weights_legacy).all()


def test__power_2_reproduces_legacy_matrix_up_to_the_scatter_factor(
    rectangular_mapper_9,
):
    """
    ``power=2.0`` restores the legacy ``lambda^4`` coefficient scaling. The factor-2 scatter is fixed
    regardless (that is the point of the new class), so the two matrices differ by exactly 2.
    """
    floor = 1.0e-8 * np.eye(9)

    regularization_matrix_legacy = aa.reg.Adapt(
        inner_coefficient=1.0, outer_coefficient=2.0
    ).regularization_matrix_from(linear_obj=rectangular_mapper_9)

    regularization_matrix_power = aa.reg.AdaptPower(
        inner_coefficient=1.0, outer_coefficient=2.0, power=2.0
    ).regularization_matrix_from(linear_obj=rectangular_mapper_9)

    assert regularization_matrix_legacy - floor == pytest.approx(
        2.0 * (regularization_matrix_power - floor), rel=1.0e-12
    )


def test__migration__power_class_coefficient_is_the_legacy_coefficient_squared(
    rectangular_mapper_9,
):
    """
    The documented migration ``c_new = c_old ** 2``, checked on the weights. It is exact when the
    inner and outer coefficients are equal; with differing coefficients the squaring happens after
    the interpolation, so the two are not related term by term.
    """
    weights_legacy_uniform = aa.reg.Adapt(
        inner_coefficient=2.0, outer_coefficient=2.0
    ).regularization_weights_from(linear_obj=rectangular_mapper_9)

    weights_power_uniform = aa.reg.AdaptPower(
        inner_coefficient=4.0, outer_coefficient=4.0
    ).regularization_weights_from(linear_obj=rectangular_mapper_9)

    assert weights_legacy_uniform == pytest.approx(weights_power_uniform, 1.0e-12)


def test__single_scatter_matrix__is_symmetric_positive_semi_definite():
    """
    With adaptive (non-uniform) weights the builder must still return a symmetric, positive
    semi-definite matrix — it is a weighted graph Laplacian plus a ``1e-8`` diagonal floor, so every
    row sums to the floor.
    """
    neighbors = np.array(
        [
            [1, 3, -1, -1],
            [0, 2, 4, -1],
            [1, 5, -1, -1],
            [0, 4, 6, -1],
            [1, 3, 5, 7],
            [2, 4, 8, -1],
            [3, 7, -1, -1],
            [4, 6, 8, -1],
            [5, 7, -1, -1],
        ]
    )

    weights = np.array([0.1, 5.0, 0.3, 2.0, 9.0, 0.05, 1.0, 4.0, 0.7])

    regularization_matrix = (
        aa.util.regularization.weighted_regularization_matrix_single_scatter_from(
            regularization_weights=weights, neighbors=neighbors
        )
    )

    assert regularization_matrix == pytest.approx(regularization_matrix.T, 1.0e-12)
    assert regularization_matrix.sum(axis=1) == pytest.approx(
        1.0e-8 * np.ones(9), abs=1.0e-14
    )
    assert np.linalg.eigvalsh(regularization_matrix).min() > 0.0


def test__single_scatter_matrix__ignores_padded_neighbor_entries():
    """
    ``-1`` entries are padding. The symmetric edge weight is the mean of the two endpoints, so a
    padded entry would contribute half of its own pixel's weight unless it is masked out.
    """
    neighbors_padded = np.array([[1, -1], [0, -1]])
    neighbors_unpadded = np.array([[1], [0]])

    weights = np.array([1.0, 4.0])

    matrix_padded = (
        aa.util.regularization.weighted_regularization_matrix_single_scatter_from(
            regularization_weights=weights, neighbors=neighbors_padded
        )
    )
    matrix_unpadded = (
        aa.util.regularization.weighted_regularization_matrix_single_scatter_from(
            regularization_weights=weights, neighbors=neighbors_unpadded
        )
    )

    assert matrix_padded == pytest.approx(matrix_unpadded, 1.0e-12)
