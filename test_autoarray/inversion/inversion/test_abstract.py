import numpy as np
from os import path
import pytest

import autoarray as aa

directory = path.dirname(path.realpath(__file__))


def test__has():
    reg = aa.m.MockRegularization()
    linear_obj = aa.m.MockLinearObj(regularization=reg)
    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj])

    assert inversion.has(cls=aa.AbstractRegularization) is True

    linear_obj = aa.m.MockLinearObj(regularization=None)
    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj])

    assert inversion.has(cls=aa.AbstractRegularization) is False


def test__total_regularizations():
    reg = aa.m.MockRegularization()

    linear_obj_0 = aa.m.MockLinearObj(regularization=reg)
    linear_obj_1 = aa.m.MockLinearObj(regularization=None)

    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj_0, linear_obj_1])

    assert inversion.total_regularizations == 1

    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj_0, linear_obj_0])

    assert inversion.total_regularizations == 2

    inversion = aa.m.MockInversion(linear_obj_list=[linear_obj_1, linear_obj_1])

    assert inversion.total_regularizations == 0


def test__index_range_list_from():
    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockLinearObj(parameters=2, regularization=None),
            aa.m.MockMapper(parameters=1, regularization=None),
        ]
    )

    assert inversion.param_range_list_from(cls=aa.LinearObj) == [[0, 2], [2, 3]]
    assert inversion.param_range_list_from(cls=aa.Mapper) == [[2, 3]]


def test__no_regularization_index_list():
    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockLinearObj(parameters=2, regularization=None),
            aa.m.MockLinearObj(parameters=1, regularization=None),
        ]
    )

    assert inversion.no_regularization_index_list == [0, 1, 2]

    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockMapper(parameters=10, regularization=aa.m.MockRegularization()),
            aa.m.MockLinearObj(parameters=3, regularization=None),
            aa.m.MockMapper(parameters=20, regularization=aa.m.MockRegularization()),
            aa.m.MockLinearObj(parameters=4, regularization=None),
        ]
    )

    assert inversion.no_regularization_index_list == [10, 11, 12, 33, 34, 35, 36]


def test__mapping_matrix():
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


def test__curvature_reg_matrix_reduced():
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


def test__regularization_matrix():
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


def test__reconstruction_reduced():
    linear_obj_list = [
        aa.m.MockMapper(parameters=2, regularization=aa.m.MockRegularization()),
        aa.m.MockLinearObj(parameters=1, regularization=None),
    ]

    inversion = aa.m.MockInversion(
        linear_obj_list=linear_obj_list, reconstruction=np.array([1.0, 2.0, 3.0])
    )

    assert (inversion.reconstruction_reduced == np.array([1.0, 2.0])).all()


def test__reconstruction_dict():
    reconstruction = np.array([0.0, 1.0, 1.0, 1.0])

    linear_obj = aa.m.MockLinearObj(parameters=1)
    mapper = aa.m.MockMapper(parameters=3)

    inversion = aa.m.MockInversion(
        linear_obj_list=[linear_obj, mapper], reconstruction=reconstruction
    )

    assert (inversion.reconstruction_dict[linear_obj] == np.zeros(1)).all()
    assert (inversion.reconstruction_dict[mapper] == np.ones(3)).all()

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


def test__mapped_reconstructed_data_dict():
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


def test__mapped_reconstructed_operated_data_dict():
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


def test__mapped_reconstructed_operated_data():
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


def test__data_subtracted_dict():
    linear_obj_0 = aa.m.MockLinearObj()

    mapped_reconstructed_operated_data_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        data=3.0 * np.ones(3),
        linear_obj_list=[linear_obj_0],
        mapped_reconstructed_operated_data_dict=mapped_reconstructed_operated_data_dict,
    )

    assert (inversion.data_subtracted_dict[linear_obj_0] == 3.0 * np.ones(3)).all()

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


def test__regularization_term():
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


def test__determinant_of_positive_definite_matrix_via_cholesky():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
    )

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(
        inversion.log_det_curvature_reg_matrix_term, 1e-4
    )

    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
    )

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(
        inversion.log_det_curvature_reg_matrix_term, 1e-4
    )


def test__reconstruction_noise_map():
    curvature_reg_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    assert inversion.reconstruction_noise_map_with_covariance[0, 0] == pytest.approx(
        np.sqrt(2.5), 1.0e-2
    )
    assert inversion.reconstruction_noise_map == pytest.approx(
        np.sqrt(np.array([2.5, 1.0, 0.5])), 1.0e-3
    )


def test__max_pixel_list_from_and_centre():

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


def test__max_pixel_list_from__filter_neighbors():
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
