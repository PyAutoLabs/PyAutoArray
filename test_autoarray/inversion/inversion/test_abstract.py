import warnings

import numpy as np
import pytest

import autoarray as aa
from pathlib import Path

directory = Path(__file__).resolve().parent


def test__has__linear_obj_with_regularization__returns_true():
    reg = aa.m.MockRegularization()
    linear_obj = aa.m.MockLinearObj(regularization=reg)
    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj])

    assert inversion.has(cls=aa.AbstractRegularization) is True


def test__has__linear_obj_without_regularization__returns_false():
    linear_obj = aa.m.MockLinearObj(regularization=None)
    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj])

    assert inversion.has(cls=aa.AbstractRegularization) is False


def test__total_regularizations__one_regularized_one_unregularized__returns_one():
    reg = aa.m.MockRegularization()

    linear_obj_0 = aa.m.MockLinearObj(regularization=reg)
    linear_obj_1 = aa.m.MockLinearObj(regularization=None)

    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj_0, linear_obj_1])

    assert inversion.total_regularizations == 1


def test__total_regularizations__both_regularized__returns_two():
    reg = aa.m.MockRegularization()

    linear_obj_0 = aa.m.MockLinearObj(regularization=reg)

    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj_0, linear_obj_0])

    assert inversion.total_regularizations == 2


def test__total_regularizations__none_regularized__returns_zero():
    linear_obj_1 = aa.m.MockLinearObj(regularization=None)

    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj_1, linear_obj_1])

    assert inversion.total_regularizations == 0


def test__param_range_list_from__linear_obj_and_mapper__correct_ranges_per_class():
    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockLinearObj(parameters=2, regularization=None),
            aa.m.MockMapper(parameters=1, regularization=None),
        ]
    )

    assert inversion.param_range_list_from(cls=aa.LinearObj) == [[0, 2], [2, 3]]
    assert inversion.param_range_list_from(cls=aa.Mapper) == [[2, 3]]


def test__no_regularization_index_list__all_unregularized__returns_all_parameter_indices():
    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockLinearObj(parameters=2, regularization=None),
            aa.m.MockLinearObj(parameters=1, regularization=None),
        ]
    )

    assert inversion.no_regularization_index_list == [0, 1, 2]


def test__no_regularization_index_list__mixed_regularized_and_unregularized__returns_only_unregularized_indices():
    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockMapper(parameters=10, regularization=aa.m.MockRegularization()),
            aa.m.MockLinearObj(parameters=3, regularization=None),
            aa.m.MockMapper(parameters=20, regularization=aa.m.MockRegularization()),
            aa.m.MockLinearObj(parameters=4, regularization=None),
        ]
    )

    assert inversion.no_regularization_index_list == [10, 11, 12, 33, 34, 35, 36]


def test__mapping_matrix__two_mappers__concatenates_mapping_matrices_horizontally():
    mapper_0 = aa.m.MockMapper(mapping_matrix=np.ones((2, 2)))
    mapper_1 = aa.m.MockMapper(mapping_matrix=2.0 * np.ones((2, 3)))

    inversion = aa.m.MockInversion(linear_obj_list=[mapper_0, mapper_1])

    mapping_matrix = np.array([[1.0, 1.0, 2.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 2.0]])

    assert inversion.mapping_matrix == pytest.approx(mapping_matrix, 1.0e-4)


def test__curvature_matrix__via_sparse_operator__identical_to_mapping():
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
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)

    mesh_0 = aa.mesh.RectangularUniform(shape=(3, 3))
    mesh_1 = aa.mesh.RectangularUniform(shape=(4, 4))

    interpolator_0 = mesh_0.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=None,
    )

    interpolator_1 = mesh_1.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=None,
    )

    mapper_0 = aa.Mapper(interpolator=interpolator_0)
    mapper_1 = aa.Mapper(interpolator=interpolator_1)

    image = aa.Array2D.no_mask(values=np.random.random((7, 7)), pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask(values=np.random.random((7, 7)), pixel_scales=1.0)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = aa.Array2D.no_mask(values=kernel, pixel_scales=1.0)
    psf = aa.Convolver(kernel=kernel)

    dataset = aa.Imaging(data=image, noise_map=noise_map, psf=psf)

    masked_dataset = dataset.apply_mask(mask=mask)

    masked_dataset_sparse_operator = masked_dataset.apply_sparse_operator_cpu()

    inversion_sparse_operator = aa.Inversion(
        dataset=masked_dataset_sparse_operator,
        linear_obj_list=[mapper_0, mapper_1],
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_dataset,
        linear_obj_list=[mapper_0, mapper_1],
    )

    assert inversion_sparse_operator.curvature_matrix == pytest.approx(
        inversion_mapping.curvature_matrix, 1.0e-4
    )


def test__curvature_matrix_via_sparse_operator__includes_source_interpolation__identical_to_mapping():
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
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)

    mesh_0 = aa.mesh.Delaunay(pixels=9)
    mesh_1 = aa.mesh.Delaunay(pixels=16)

    image_mesh_0 = aa.image_mesh.Overlay(shape=(3, 3))
    image_mesh_1 = aa.image_mesh.Overlay(shape=(4, 4))

    image_mesh_grid_0 = image_mesh_0.image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    image_mesh_grid_1 = image_mesh_1.image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    interpolator_0 = mesh_0.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid_0,
    )

    interpolator_1 = mesh_1.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid_1,
    )

    mapper_0 = aa.Mapper(interpolator=interpolator_0)
    mapper_1 = aa.Mapper(interpolator=interpolator_1)

    image = aa.Array2D.no_mask(values=np.random.random((7, 7)), pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask(values=np.random.random((7, 7)), pixel_scales=1.0)
    kernel = aa.Array2D.no_mask(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], pixel_scales=1.0
    )

    psf = aa.Convolver(kernel=kernel)

    dataset = aa.Imaging(data=image, noise_map=noise_map, psf=psf)

    masked_dataset = dataset.apply_mask(mask=mask)

    masked_dataset_sparse_operator = masked_dataset.apply_sparse_operator_cpu()

    inversion_sparse_operator = aa.Inversion(
        dataset=masked_dataset_sparse_operator,
        linear_obj_list=[mapper_0, mapper_1],
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_dataset,
        linear_obj_list=[mapper_0, mapper_1],
    )

    assert inversion_sparse_operator.curvature_matrix == pytest.approx(
        inversion_mapping.curvature_matrix, 1.0e-4
    )


def test__curvature_reg_matrix_reduced__regularized_and_unregularized__removes_unregularized_rows_cols():
    curvature_reg_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    linear_obj_list = [
        aa.m.MockMapper(parameters=2, regularization=aa.m.MockRegularization()),
        aa.m.MockLinearObj(parameters=1, regularization=None),
    ]

    inversion = aa.m.MockInversion(
        linear_obj_list=linear_obj_list, curvature_reg_matrix=curvature_reg_matrix
    )

    assert (
        inversion.curvature_reg_matrix_reduced == np.array([[1.0, 2.0], [4.0, 5.0]])
    ).all()


def test__regularization_matrix__two_regularized_mappers__assembles_block_diagonal_matrix():
    reg_0 = aa.m.MockRegularization(regularization_matrix=np.ones((2, 2)))
    reg_1 = aa.m.MockRegularization(regularization_matrix=2.0 * np.ones((3, 3)))

    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockMapper(regularization=reg_0),
            aa.m.MockMapper(regularization=reg_1),
        ]
    )

    regularization_matrix = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )

    assert inversion.regularization_matrix == pytest.approx(regularization_matrix)


def test__reconstruction_reduced__regularized_and_unregularized__returns_only_regularized_parameters():
    linear_obj_list = [
        aa.m.MockMapper(parameters=2, regularization=aa.m.MockRegularization()),
        aa.m.MockLinearObj(parameters=1, regularization=None),
    ]

    inversion = aa.m.MockInversion(
        linear_obj_list=linear_obj_list, reconstruction=np.array([1.0, 2.0, 3.0])
    )

    assert (inversion.reconstruction_reduced == np.array([1.0, 2.0])).all()


def test__reconstruction_dict__single_linear_obj_and_mapper__splits_reconstruction_correctly():
    reconstruction = np.array([0.0, 1.0, 1.0, 1.0])

    linear_obj = aa.m.MockLinearObj(parameters=1)
    mapper = aa.m.MockMapper(parameters=3)

    inversion = aa.m.MockInversion(
        linear_obj_list=[linear_obj, mapper], reconstruction=reconstruction
    )

    assert (inversion.reconstruction_dict[linear_obj] == np.zeros(1)).all()
    assert (inversion.reconstruction_dict[mapper] == np.ones(3)).all()


def test__reconstruction_dict__multiple_linear_objs_and_mappers__splits_reconstruction_correctly():
    reconstruction = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    linear_obj = aa.m.MockLinearObj(parameters=1)
    mapper_0 = aa.m.MockMapper(parameters=2)
    mapper_1 = aa.m.MockMapper(parameters=3)

    inversion = aa.m.MockInversion(
        linear_obj_list=[linear_obj, mapper_0, mapper_1], reconstruction=reconstruction
    )

    assert (inversion.reconstruction_dict[linear_obj] == np.zeros(1)).all()
    assert (inversion.reconstruction_dict[mapper_0] == np.ones(2)).all()
    assert (inversion.reconstruction_dict[mapper_1] == 2.0 * np.ones(3)).all()


def test__regularization_weights_mapper_dict__linear_obj_before_mapper__indexes_correctly():
    # Distinct coefficients identify which linear_obj_list entry each weight set
    # came from: 7.0 → the non-mapper linear obj at index 0, 11.0 → the mapper
    # at index 1. A pre-fix `enumerate(cls_list_from(Mapper))` would pass the
    # mapper-only index 0 into `regularization_weights_from`, which reads
    # `linear_obj_list[0]` and returns the 2-element 7.0 weights.
    linear_obj = aa.m.MockLinearObj(
        parameters=2, regularization=aa.reg.Constant(coefficient=7.0)
    )
    mapper = aa.m.MockMapper(
        parameters=3, regularization=aa.reg.Constant(coefficient=11.0)
    )

    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj, mapper])

    weights = inversion.regularization_weights_mapper_dict[mapper]

    assert weights.shape == (3,)
    assert (weights == 11.0 * np.ones(3)).all()


def test__regularization_weights_mapper_dict__multiple_mappers_with_linear_obj_between__indexes_correctly():
    # Layout: [mapper_a (params=2), linear_obj (params=1), mapper_b (params=4)]
    # Each gets a distinct Constant coefficient to verify per-mapper indexing.
    mapper_a = aa.m.MockMapper(
        parameters=2, regularization=aa.reg.Constant(coefficient=3.0)
    )
    linear_obj = aa.m.MockLinearObj(
        parameters=1, regularization=aa.reg.Constant(coefficient=5.0)
    )
    mapper_b = aa.m.MockMapper(
        parameters=4, regularization=aa.reg.Constant(coefficient=13.0)
    )

    inversion = aa.m.MockInversion(
        linear_obj_list=[mapper_a, linear_obj, mapper_b],
    )

    rw_dict = inversion.regularization_weights_mapper_dict

    assert (rw_dict[mapper_a] == 3.0 * np.ones(2)).all()
    assert (rw_dict[mapper_b] == 13.0 * np.ones(4)).all()


def test__mapped_reconstructed_data_dict__single_linear_obj__returns_correct_data_and_sum():
    linear_obj_0 = aa.m.MockLinearObj()

    mapped_reconstructed_data_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        mapped_reconstructed_data_dict=mapped_reconstructed_data_dict,
        reconstruction=np.ones(3),
        reconstruction_dict=[None],
    )

    assert (inversion.mapped_reconstructed_data_dict[linear_obj_0] == np.ones(3)).all()
    assert (inversion.mapped_reconstructed_data == np.ones(3)).all()


def test__mapped_reconstructed_data_dict__two_linear_objs__sums_contributions_correctly():
    linear_obj_0 = aa.m.MockLinearObj()
    linear_obj_1 = aa.m.MockLinearObj()

    mapped_reconstructed_data_dict = {
        linear_obj_0: np.ones(2),
        linear_obj_1: 2.0 * np.ones(2),
    }

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        mapped_reconstructed_data_dict=mapped_reconstructed_data_dict,
        reconstruction=np.array([1.0, 1.0, 2.0, 2.0]),
        reconstruction_dict=[None, None],
    )

    assert (inversion.mapped_reconstructed_data_dict[linear_obj_0] == np.ones(2)).all()
    assert (
        inversion.mapped_reconstructed_data_dict[linear_obj_1] == 2.0 * np.ones(2)
    ).all()
    assert (inversion.mapped_reconstructed_data == 3.0 * np.ones(2)).all()


def test__mapped_reconstructed_operated_data_dict__single_linear_obj__returns_correct_data_and_sum():
    linear_obj_0 = aa.m.MockLinearObj()

    mapped_reconstructed_operated_data_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        mapped_reconstructed_operated_data_dict=mapped_reconstructed_operated_data_dict,
        reconstruction=np.ones(3),
        reconstruction_dict=[None],
    )

    assert (
        inversion.mapped_reconstructed_operated_data_dict[linear_obj_0] == np.ones(3)
    ).all()
    assert (inversion.mapped_reconstructed_operated_data == np.ones(3)).all()


def test__mapped_reconstructed_operated_data_dict__two_linear_objs__sums_contributions_correctly():
    linear_obj_0 = aa.m.MockLinearObj()
    linear_obj_1 = aa.m.MockLinearObj()

    mapped_reconstructed_operated_data_dict = {
        linear_obj_0: np.ones(2),
        linear_obj_1: 2.0 * np.ones(2),
    }

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        mapped_reconstructed_operated_data_dict=mapped_reconstructed_operated_data_dict,
        reconstruction=np.array([1.0, 1.0, 2.0, 2.0]),
        reconstruction_dict=[None, None],
    )

    assert (
        inversion.mapped_reconstructed_operated_data_dict[linear_obj_0] == np.ones(2)
    ).all()
    assert (
        inversion.mapped_reconstructed_operated_data_dict[linear_obj_1]
        == 2.0 * np.ones(2)
    ).all()
    assert (inversion.mapped_reconstructed_operated_data == 3.0 * np.ones(2)).all()


def test__mapped_reconstructed_operated_data__single_linear_obj__returns_correct_operated_data():
    linear_obj_0 = aa.m.MockLinearObj()

    mapped_reconstructed_operated_data_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        mapped_reconstructed_operated_data_dict=mapped_reconstructed_operated_data_dict,
        reconstruction=np.ones(3),
        reconstruction_dict=[None],
    )

    assert (
        inversion.mapped_reconstructed_operated_data_dict[linear_obj_0] == np.ones(3)
    ).all()
    assert (inversion.mapped_reconstructed_operated_data == np.ones(3)).all()


def test__mapped_reconstructed_operated_data__two_linear_objs__sums_operated_data_correctly():
    linear_obj_0 = aa.m.MockLinearObj()
    linear_obj_1 = aa.m.MockLinearObj()

    mapped_reconstructed_operated_data_dict = {
        linear_obj_0: np.ones(2),
        linear_obj_1: 2.0 * np.ones(2),
    }

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        mapped_reconstructed_operated_data_dict=mapped_reconstructed_operated_data_dict,
        reconstruction=np.array([1.0, 1.0, 2.0, 2.0]),
        reconstruction_dict=[None, None],
    )

    assert (
        inversion.mapped_reconstructed_operated_data_dict[linear_obj_0] == np.ones(2)
    ).all()
    assert (
        inversion.mapped_reconstructed_operated_data_dict[linear_obj_1]
        == 2.0 * np.ones(2)
    ).all()
    assert (inversion.mapped_reconstructed_operated_data == 3.0 * np.ones(2)).all()


def test__data_subtracted_dict__single_linear_obj__subtracts_other_contributions_from_data():
    linear_obj_0 = aa.m.MockLinearObj()

    mapped_reconstructed_operated_data_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        data=3.0 * np.ones(3),
        linear_obj_list=[linear_obj_0],
        mapped_reconstructed_operated_data_dict=mapped_reconstructed_operated_data_dict,
    )

    assert (inversion.data_subtracted_dict[linear_obj_0] == 3.0 * np.ones(3)).all()


def test__data_subtracted_dict__two_linear_objs__subtracts_other_contributions_from_data():
    linear_obj_0 = aa.m.MockLinearObj()
    linear_obj_1 = aa.m.MockLinearObj()

    mapped_reconstructed_operated_data_dict = {
        linear_obj_0: np.ones(3),
        linear_obj_1: 2.0 * np.ones(3),
    }

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        data=3.0 * np.ones(3),
        linear_obj_list=[linear_obj_0, linear_obj_1],
        mapped_reconstructed_operated_data_dict=mapped_reconstructed_operated_data_dict,
    )

    assert (inversion.data_subtracted_dict[linear_obj_0] == np.ones(3)).all()
    assert (inversion.data_subtracted_dict[linear_obj_1] == 2.0 * np.ones(3)).all()


def test__regularization_term__identity_matrix__computes_sum_of_squared_reconstruction():
    reconstruction = np.array([1.0, 1.0, 1.0])

    regularization_matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    inversion = aa.m.MockInversion(
        reconstruction=reconstruction,
        linear_obj_list=[
            aa.m.MockLinearObj(parameters=3, regularization=aa.m.MockRegularization())
        ],
        regularization_matrix=regularization_matrix,
    )

    # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

    # G_l = s_T * H * s

    # Matrix multiplication:

    # s_T * H = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [(1.0*1.0) + (1.0*0.0) + (1.0*0.0)] = [1.0, 1.0, 1.0]
    #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*1.0) + (1.0*0.0)]
    #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*0.0) + (1.0*1.0)]

    # (s_T * H) * s = [1.0, 1.0, 1.0] * [1.0] = 3.0
    #                                   [1.0]
    #                                   [1.0]

    assert inversion.regularization_term == 3.0


def test__regularization_term__tridiagonal_matrix__computes_weighted_regularization_term():
    reconstruction = np.array([2.0, 3.0, 5.0])

    regularization_matrix = np.array(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
    )

    inversion = aa.m.MockInversion(
        reconstruction=reconstruction,
        linear_obj_list=[
            aa.m.MockLinearObj(parameters=3, regularization=aa.m.MockRegularization())
        ],
        regularization_matrix=regularization_matrix,
    )

    # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

    # G_l = s_T * H * s

    # Matrix multiplication:

    # s_T * H = [2.0, 3.0, 5.0] * [2.0,  -1.0,  0.0] = [(2.0* 2.0) + (3.0*-1.0) + (5.0 *0.0)] = [1.0, -1.0, 7.0]
    #                             [-1.0,  2.0, -1.0]   [(2.0*-1.0) + (3.0* 2.0) + (5.0*-1.0)]
    #                             [ 0.0, -1.0,  2.0]   [(2.0* 0.0) + (3.0*-1.0) + (5.0 *2.0)]

    # (s_T * H) * s = [1.0, -1.0, 7.0] * [2.0] = 34.0
    #                                    [3.0]
    #                                    [5.0]

    assert inversion.regularization_term == 34.0


def test__determinant_of_positive_definite_matrix_via_cholesky__identity_matrix__matches_numpy_log_det():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
    )

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(
        inversion.log_det_curvature_reg_matrix_term, 1e-4
    )


def test__determinant_of_positive_definite_matrix_via_cholesky__tridiagonal_matrix__matches_numpy_log_det():
    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
    )

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(
        inversion.log_det_curvature_reg_matrix_term, 1e-4
    )


def test__log_det_method__default_is_cholesky_and_matches_numpy_log_det():
    # The default log_det_method must reproduce the historical Cholesky value exactly.
    assert aa.Settings().log_det_method == "cholesky"

    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
    )

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        np.log(np.linalg.det(matrix)), 1.0e-4
    )


def test__log_det_method__slogdet_matches_cholesky_on_positive_definite_matrix():
    # Where the matrix is positive-definite, "slogdet" is identical to "cholesky".
    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

    cholesky_inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
        regularization_matrix=matrix,
    )
    slogdet_inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
        regularization_matrix=matrix,
        settings=aa.Settings(log_det_method="slogdet"),
    )

    expected = np.log(np.linalg.det(matrix))

    assert cholesky_inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        expected, 1.0e-8
    )
    assert slogdet_inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        cholesky_inversion.log_det_curvature_reg_matrix_term, 1.0e-8
    )
    assert slogdet_inversion.log_det_regularization_matrix_term == pytest.approx(
        cholesky_inversion.log_det_regularization_matrix_term, 1.0e-8
    )


def test__log_det_method__slogdet_is_finite_where_cholesky_fails_on_non_positive_definite():
    # A symmetric matrix with a negative eigenvalue: Cholesky cannot factor it (raises /
    # returns NaN), but slogdet returns the finite log|det|. This is the property that
    # keeps a gradient-based search from stalling (autolens_workspace_developer#104).
    matrix = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    assert np.linalg.eigvalsh(matrix).min() < 0.0  # confirm it is not positive-definite

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
        settings=aa.Settings(log_det_method="slogdet"),
    )

    result = inversion.log_det_curvature_reg_matrix_term
    assert np.isfinite(result)
    assert result == pytest.approx(np.linalg.slogdet(matrix)[1], 1.0e-8)


def test__reconstruction_noise_map__correct_diagonal_noise_values():
    curvature_reg_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    assert inversion.reconstruction_covariance_matrix[0, 0] == pytest.approx(
        2.5, 1.0e-2
    )
    assert inversion.reconstruction_noise_map == pytest.approx(
        np.sqrt(np.array([2.5, 1.0, 0.5])), 1.0e-3
    )


def test__reconstruction_covariance_matrix__off_diagonals_are_finite_and_negative():
    """
    The off-diagonal entries of a covariance matrix are covariances and are routinely negative.

    `reconstruction_covariance_matrix` previously applied `np.sqrt` elementwise to the whole inverse, so every
    negative off-diagonal became NaN by construction -- for any matrix, however well-conditioned -- while
    emitting `RuntimeWarning: invalid value encountered in sqrt`. Only the [0, 0] diagonal element was asserted,
    so nothing caught it.
    """
    curvature_reg_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        covariance = inversion.reconstruction_covariance_matrix

    assert np.all(np.isfinite(covariance))

    # this matrix has anti-correlated pixels, so the off-diagonals are genuinely negative
    assert covariance[0, 1] < 0.0
    assert covariance == pytest.approx(np.linalg.inv(curvature_reg_matrix), 1.0e-8)


def test__reconstruction_covariance_matrix__is_accurate_and_symmetric_when_ill_conditioned():
    """
    Ground truth is exact by construction: for `A = Q diag(w) Q.T` the inverse is `Q diag(1/w) Q.T`.

    The symmetry half only guards the symmetrization line -- `0.5 * (C + C.T)` is bitwise symmetric for any C --
    so the accuracy assertion against the constructed truth is what tests the factorization itself.
    """
    rng = np.random.default_rng(1234)
    q, _ = np.linalg.qr(rng.standard_normal((25, 25)))
    eigenvalues = np.logspace(0, 9, 25)

    curvature_reg_matrix = (q * eigenvalues) @ q.T
    curvature_reg_matrix = 0.5 * (curvature_reg_matrix + curvature_reg_matrix.T)

    covariance_true = (q * (1.0 / eigenvalues)) @ q.T

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    covariance = inversion.reconstruction_covariance_matrix

    # cond ~ 1e9, so the achievable accuracy is eps * cond ~ 2e-7; the measured error is ~3e-9. This is not a
    # claim that Cholesky beats LU here -- it does not, `np.linalg.inv` measures ~7e-10 on this matrix.
    assert covariance == pytest.approx(covariance_true, abs=1.0e-7)
    assert covariance == pytest.approx(covariance.T, abs=1.0e-15)


def test__reconstruction_covariance_matrix__asymmetric_input_is_symmetrized_not_silently_upper_triangle():
    """
    `cho_factor` reads only the upper triangle, so an asymmetric input would be inverted as though its lower
    triangle matched its upper -- silently, and differing from the true inverse.
    """
    curvature_reg_matrix = np.array([[2.0, 0.5], [0.1, 2.0]])
    symmetrized = 0.5 * (curvature_reg_matrix + curvature_reg_matrix.T)

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    assert inversion.reconstruction_covariance_matrix == pytest.approx(
        np.linalg.inv(symmetrized), 1.0e-8
    )


def test__reconstruction_covariance_matrix__non_finite_matrix_raises_lin_alg_error():
    """
    scipy raises `ValueError` on a non-finite matrix, which the plotting and CSV callers do not catch -- they
    guard on `LinAlgError`. The CSV writer explicitly promises not to abort the enclosing model-fit, so the
    non-finite case is converted rather than allowed to escape.
    """
    curvature_reg_matrix = np.array([[1.0, np.nan], [np.nan, 2.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
        inversion.reconstruction_covariance_matrix


def test__reconstruction_noise_map__is_sqrt_of_covariance_diagonal():
    """
    The invariant, asserted directly rather than via hand-computed values.

    `reconstruction_noise_map` used to be `np.diagonal(...)` of an already-square-rooted matrix, which was
    correct only incidentally -- because `np.sqrt` is elementwise. It now takes the square root of the
    covariance diagonal itself, so the relationship is stated rather than emergent.
    """
    curvature_reg_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    assert inversion.reconstruction_noise_map == pytest.approx(
        np.sqrt(np.diag(inversion.reconstruction_covariance_matrix)), 1.0e-12
    )


def test__reconstruction_covariance_matrix__raises_on_a_non_positive_definite_matrix():
    """
    A covariance is only defined for a positive-definite matrix.

    `np.linalg.inv` raises only on an exactly singular matrix, so an indefinite `curvature_reg_matrix` returned
    a plausible-looking covariance with no error and no warning. The Cholesky factorization rejects it, and the
    plotting and CSV callers already catch `LinAlgError`.
    """
    # symmetric, non-singular, but indefinite (eigenvalues +1 and -1)
    curvature_reg_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    assert np.isfinite(np.linalg.inv(curvature_reg_matrix)).all()  # inv is silent here

    with pytest.raises(np.linalg.LinAlgError):
        inversion.reconstruction_covariance_matrix


def test__reconstruction_noise_map_with_covariance__is_deprecated_alias():
    curvature_reg_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    with pytest.warns(DeprecationWarning, match="reconstruction_covariance_matrix"):
        covariance = inversion.reconstruction_noise_map_with_covariance

    assert covariance == pytest.approx(
        inversion.reconstruction_covariance_matrix, 1.0e-12
    )


def test__max_pixel_list_from_and_centre__returns_top_pixels_and_brightest_centre():

    source_plane_mesh_grid = aa.Grid2DIrregular(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
    )

    mapper = aa.m.MockMapper(source_plane_mesh_grid=source_plane_mesh_grid)

    interpolator = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(pixels=4),
        mesh_grid=source_plane_mesh_grid,
        data_grid=None,
    )

    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid,
        interpolator=interpolator,
    )

    inversion = aa.m.MockInversion(
        reconstruction=np.array([2.0, 3.0, 5.0, 0.0]), linear_obj_list=[mapper]
    )

    assert inversion.max_pixel_list_from(total_pixels=2)[0] == [
        2,
        1,
    ]

    assert inversion.max_pixel_centre().in_list == [(5.0, 6.0)]


def test__max_pixel_list_from__filter_neighbors__excludes_adjacent_pixels_from_top_list():
    source_plane_mesh_grid = aa.Grid2DIrregular(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [3.0, 3.0],
        ]
    )

    mesh_geometry = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=9),
        mesh_grid=source_plane_mesh_grid,
        data_grid=None,
    )

    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=source_plane_mesh_grid,
        mesh_geometry=mesh_geometry,
    )

    inversion = aa.m.MockInversion(
        reconstruction=np.array([5.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        linear_obj_list=[mapper],
    )

    pixel_list = inversion.max_pixel_list_from(total_pixels=9, filter_neighbors=True)

    assert pixel_list[0] == [
        0,
        8,
    ]


def _zeroed_pixel_inversion(
    curvature_reg_matrix,
    use_positive_only_solver=True,
    use_edge_zeroed_pixels=True,
    with_mapper=True,
):
    """
    A `MockInversion` over a 4x4 `RectangularUniform` mesh, whose `zeroed_pixels` is the edge ring.

    16 parameters, of which the 12 edge pixels are zeroed and the 4 interior pixels [5, 6, 9, 10] are solved
    for. A 4x4 kept block is the smallest one on which "the kept block equals the inverse of the submatrix" is
    a real assertion rather than a scalar identity.
    """
    if with_mapper:
        linear_obj = aa.m.MockMapper(
            mesh=aa.mesh.RectangularUniform(shape=(4, 4)),
            parameters=16,
            regularization=aa.reg.Constant(),
        )
    else:
        linear_obj = aa.m.MockLinearObj(parameters=16, regularization=aa.reg.Constant())

    return aa.m.MockInversion(
        linear_obj_list=[linear_obj],
        curvature_reg_matrix=curvature_reg_matrix,
        settings=aa.Settings(
            use_positive_only_solver=use_positive_only_solver,
            use_edge_zeroed_pixels=use_edge_zeroed_pixels,
        ),
    )


def _spd_matrix(n=16, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def test__solve_ids_to_keep__none_unless_both_settings_and_a_mapper():
    """
    The single predicate for "did the solve subset the system?".

    It must reproduce the nesting in `reconstruction` exactly: `use_edge_zeroed_pixels` is consulted ONLY
    when `use_positive_only_solver` is on, and only when a `Mapper` is present. That scoping is deliberate
    (see the comment in `reconstruction`), so a change that made this property answer on
    `use_edge_zeroed_pixels` alone would silently start subsetting the positive-negative solve.
    """
    matrix = _spd_matrix()

    assert _zeroed_pixel_inversion(matrix).solve_ids_to_keep == pytest.approx(
        np.array([5, 6, 9, 10])
    )

    assert (
        _zeroed_pixel_inversion(matrix, use_edge_zeroed_pixels=False).solve_ids_to_keep
        is None
    )
    assert (
        _zeroed_pixel_inversion(
            matrix, use_positive_only_solver=False
        ).solve_ids_to_keep
        is None
    )
    assert _zeroed_pixel_inversion(matrix, with_mapper=False).solve_ids_to_keep is None


def test__reconstruction_covariance_matrix__formed_on_the_solved_indices():
    """
    The covariance must describe the estimator that was actually computed.

    `reconstruction` subsets `curvature_reg_matrix` to `zeroed_ids_to_keep` and scatters back exact zeros.
    Previously this property inverted the FULL matrix regardless, re-admitting the poorly-constrained boundary
    vertices the solve dropped to stay stable. Asserted structurally -- shape, which entries are NaN, and the
    kept block against an independently computed inverse of the submatrix -- rather than against baked-in
    numbers.
    """
    matrix = _spd_matrix()

    covariance = _zeroed_pixel_inversion(matrix).reconstruction_covariance_matrix

    keep = np.array([5, 6, 9, 10])
    excluded = np.setdiff1d(np.arange(16), keep)

    # shape does not depend on the settings -- callers never branch on it
    assert covariance.shape == (16, 16)

    assert np.isnan(covariance[excluded]).all()
    assert np.isnan(covariance[:, excluded]).all()

    assert covariance[np.ix_(keep, keep)] == pytest.approx(
        np.linalg.inv(matrix[np.ix_(keep, keep)]), 1.0e-8
    )


def test__reconstruction_covariance_matrix__excluded_entries_are_nan_not_zero():
    """
    NaN means "never estimated"; zero would mean "known exactly", which is the opposite claim.

    Zero is a legitimate covariance value, so a consumer cannot tell it apart from a real result. These
    parameters were held at zero by construction and never entered the solve, so the honest report is that
    there is no number.
    """
    covariance = _zeroed_pixel_inversion(_spd_matrix()).reconstruction_covariance_matrix

    assert not (covariance[0] == 0.0).any()
    assert np.isnan(covariance[0]).all()


def test__reconstruction_noise_map__nan_at_the_pixels_the_solve_zeroed():
    """
    The invariant this task exists for: the reconstruction and its noise map agree on which pixels were solved.

    Also asserts no `RuntimeWarning` escapes. `np.sqrt` of NaN propagates silently (unlike `np.sqrt` of a
    negative), so the NaN convention must not reintroduce the warning storm the elementwise-sqrt bug caused.
    """
    matrix = _spd_matrix()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        noise_map = _zeroed_pixel_inversion(matrix).reconstruction_noise_map

    keep = np.array([5, 6, 9, 10])
    excluded = np.setdiff1d(np.arange(16), keep)

    assert np.isnan(noise_map[excluded]).all()
    assert np.isfinite(noise_map[keep]).all()

    assert noise_map[keep] == pytest.approx(
        np.sqrt(np.diag(np.linalg.inv(matrix[np.ix_(keep, keep)]))), 1.0e-8
    )


def test__reconstruction_noise_map__kept_pixels_are_never_noisier_than_the_full_matrix():
    """
    Restricting the inverse is not a re-scaling of the excluded rows -- it changes the SOLVED pixels too.

    For a symmetric positive-definite `A`, `[A^-1]_keep >= (A_keep)^-1` in the positive-semidefinite ordering,
    so every kept pixel's variance is lower here than it was. The direction is guaranteed by that inequality,
    so it is asserted as a one-directional bound rather than as a magnitude: this is the value change that
    needs a release note, and a regression would show up as a kept pixel getting NOISIER.
    """
    matrix = _spd_matrix()

    restricted = _zeroed_pixel_inversion(matrix).reconstruction_noise_map
    full = _zeroed_pixel_inversion(
        matrix, use_edge_zeroed_pixels=False
    ).reconstruction_noise_map

    keep = np.array([5, 6, 9, 10])

    assert (restricted[keep] <= full[keep]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_edge_zeroed_pixels": False},
        {"use_positive_only_solver": False},
        {"with_mapper": False},
    ],
)
def test__reconstruction_covariance_matrix__full_system_path_is_unchanged(kwargs):
    """
    Every route that does NOT subset the solve must return the full inverse exactly as before, with no NaN.

    `solve_ids_to_keep` returns None rather than "every index" precisely so this path never goes through the
    indexing and scatter-back code at all.
    """
    matrix = _spd_matrix()

    covariance = _zeroed_pixel_inversion(
        matrix, **kwargs
    ).reconstruction_covariance_matrix

    assert np.isfinite(covariance).all()
    assert covariance == pytest.approx(np.linalg.inv(matrix), 1.0e-8)


def test__reconstruction_covariance_matrix__non_finite_entry_only_raises_when_the_solve_used_it():
    """
    The `LinAlgError` contract holds on the submatrix, which is what `inversion_plots.py` guards on.

    A NaN in a row the solve excluded cannot reach the factorization and does not stop the reconstruction
    either, so failing on it would make the covariance stricter than the estimator it describes. A NaN in a
    KEPT row still raises -- scipy would otherwise raise `ValueError`, which the plotting and CSV callers do
    not catch, and the CSV writer promises not to abort the enclosing model-fit.
    """
    excluded_nan = _spd_matrix()
    excluded_nan[0, 0] = np.nan  # index 0 is an edge pixel, so it is zeroed

    covariance = _zeroed_pixel_inversion(excluded_nan).reconstruction_covariance_matrix
    assert np.isfinite(covariance[np.ix_([5, 6, 9, 10], [5, 6, 9, 10])]).all()

    kept_nan = _spd_matrix()
    kept_nan[5, 5] = np.nan  # index 5 is an interior pixel, so it is solved for

    with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
        _zeroed_pixel_inversion(kept_nan).reconstruction_covariance_matrix


def _inversion_with_reconstruction(monkeypatch, inversion, reconstruction):
    """Force an inversion's reconstruction to a chosen vector, so clump finding can be tested."""
    mapper = inversion.cls_list_from(cls=aa.Mapper)[0]

    monkeypatch.setattr(
        type(inversion),
        "reconstruction_dict",
        property(lambda self: {mapper: np.asarray(reconstruction)}),
    )

    return inversion


def test__source_clumps_from__two_peaks_give_two_clumps(
    rectangular_inversion_7x7_3x3, monkeypatch
):
    # A 3x3 mesh with bright opposite corners, whose bright pixels do not touch.
    inversion = _inversion_with_reconstruction(
        monkeypatch,
        rectangular_inversion_7x7_3x3,
        [1.0, 0.6, 0.1, 0.6, 0.1, 0.1, 0.1, 0.6, 1.0],
    )

    clumps = inversion.source_clumps_from(threshold=0.5, min_pixels=1)

    assert len(clumps) == 2
    assert set(clumps[0].tolist()) == {0, 1, 3}
    assert set(clumps[1].tolist()) == {7, 8}


def test__source_clumps_from__low_threshold_merges_the_peaks_into_one_clump(
    rectangular_inversion_7x7_3x3, monkeypatch
):
    inversion = _inversion_with_reconstruction(
        monkeypatch,
        rectangular_inversion_7x7_3x3,
        [1.0, 0.6, 0.1, 0.6, 0.1, 0.1, 0.1, 0.6, 1.0],
    )

    clumps = inversion.source_clumps_from(threshold=0.05, min_pixels=1)

    assert len(clumps) == 1
    assert clumps[0].shape[0] == 9


def test__source_clumps_from__min_pixels_and_total_clumps_filter_the_clumps(
    rectangular_inversion_7x7_3x3, monkeypatch
):
    inversion = _inversion_with_reconstruction(
        monkeypatch,
        rectangular_inversion_7x7_3x3,
        [1.0, 0.6, 0.1, 0.6, 0.1, 0.1, 0.1, 0.6, 1.0],
    )

    assert len(inversion.source_clumps_from(threshold=0.5, min_pixels=3)) == 1
    assert len(inversion.source_clumps_from(threshold=0.5, min_pixels=4)) == 0
    assert (
        len(inversion.source_clumps_from(threshold=0.5, min_pixels=1, total_clumps=1))
        == 1
    )


def test__source_clumps_from__pix_indexes_bypasses_the_clump_finding(
    rectangular_inversion_7x7_3x3,
):
    clumps = rectangular_inversion_7x7_3x3.source_clumps_from(pix_indexes=[[0, 1], [8]])

    assert len(clumps) == 2
    assert (clumps[0] == np.array([0, 1])).all()
    assert (clumps[1] == np.array([8])).all()


def test__source_clumps_from__non_positive_reconstruction_gives_no_clumps(
    rectangular_inversion_7x7_3x3, monkeypatch
):
    inversion = _inversion_with_reconstruction(
        monkeypatch, rectangular_inversion_7x7_3x3, -1.0 * np.ones(9)
    )

    assert inversion.source_clumps_from(min_pixels=1) == []


def test__mappings_from__pairs_each_clump_with_its_image_regions(
    rectangular_inversion_7x7_3x3, monkeypatch
):
    inversion = _inversion_with_reconstruction(
        monkeypatch,
        rectangular_inversion_7x7_3x3,
        [1.5, 0.6, 0.1, 0.6, 0.1, 0.1, 0.1, 1.2, 2.0],
    )

    mappings = inversion.mappings_from(threshold=0.5, min_pixels=1)

    assert len(mappings) == 2

    # Ordered brightest first, so the clump containing mesh pixel 8 (value 2.0) comes first.
    assert mappings[0].peak_value == pytest.approx(2.0)
    assert set(mappings[0].pix_indexes.tolist()) == {7, 8}
    assert mappings[1].peak_value == pytest.approx(1.5)
    assert set(mappings[1].pix_indexes.tolist()) == {0}

    for mapping in mappings:
        assert len(mapping.source_contours) == mapping.pix_indexes.shape[0]
        assert len(mapping.image_regions) > 0
        assert len(mapping.source_centre) == 2
