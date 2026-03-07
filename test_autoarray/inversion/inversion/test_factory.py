import copy
import numpy as np
import pytest
from dill import settings

import autoarray as aa


def test__inversion_imaging__via_linear_obj_func_list(masked_imaging_7x7_no_blur):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1, grid=grid, mapping_matrix=np.full(fill_value=0.5, shape=(9, 1))
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj],
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )
    assert inversion.reconstruction == pytest.approx(np.array([2.0]), 1.0e-4)

    # Overwrites use_sparse_operator to false.

    masked_imaging_7x7_no_blur_sparse_operator = (
        masked_imaging_7x7_no_blur.apply_sparse_operator_cpu()
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur_sparse_operator,
        linear_obj_list=[linear_obj],
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )
    assert inversion.reconstruction == pytest.approx(np.array([2.0]), 1.0e-4)

    # Works with multiple parameters

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=np.full(fill_value=0.5, shape=(9, 2))
    )
    linear_obj.mapping_matrix[0, 0] = 1.0

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj],
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.reconstruction == pytest.approx(np.array([0.0, 2.0]), abs=1.0e-4)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )


def test__inversion_imaging__via_mapper(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
):

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
    )

    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        7.2571757082, 1.0e-4
    )
    # assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
    #     4.609440907938719, 1.0e-4
    # )
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )

    masked_imaging_7x7_no_blur_sparse_operator = (
        masked_imaging_7x7_no_blur.apply_sparse_operator_cpu()
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur_sparse_operator,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
    )

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        7.257175708246, 1.0e-4
    )
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[delaunay_mapper_9_3x3],
    )

    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(10.6674, 1.0e-4)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur_sparse_operator,
        linear_obj_list=[delaunay_mapper_9_3x3],
    )

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(10.6674, 1.0e-4)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )


def test__inversion_imaging__via_mapper_knn(
    masked_imaging_7x7_no_blur,
    knn_mapper_9_3x3,
    regularization_adaptive_brightness_split,
):

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[knn_mapper_9_3x3],
    )

    assert knn_mapper_9_3x3.pix_indexes_for_sub_slim_index[0, :] == pytest.approx(
        [1, 0, 4, 6, 2, 5, 3, 7, 8], 1.0e-4
    )
    assert knn_mapper_9_3x3.pix_indexes_for_sub_slim_index[1, :] == pytest.approx(
        [1, 0, 2, 4, 6, 3, 5, 7, 8], 1.0e-4
    )
    assert knn_mapper_9_3x3.pix_indexes_for_sub_slim_index[2, :] == pytest.approx(
        [1, 0, 4, 6, 2, 5, 3, 7, 8], 1.0e-4
    )

    assert knn_mapper_9_3x3.pix_weights_for_sub_slim_index[0, :] == pytest.approx(
        [
            0.24139248,
            0.20182463,
            0.13465525,
            0.12882639,
            0.12169429,
            0.08682546,
            0.07062276,
            0.00982079,
            0.00433794,
        ],
        1.0e-4,
    )
    assert knn_mapper_9_3x3.pix_weights_for_sub_slim_index[1, :] == pytest.approx(
        [
            0.23255487,
            0.22727716,
            0.14466056,
            0.11643257,
            0.09868897,
            0.08878719,
            0.07744259,
            0.01010399,
            0.0040521,
        ],
        1.0e-4,
    )
    assert knn_mapper_9_3x3.pix_weights_for_sub_slim_index[2, :] == pytest.approx(
        [
            0.2334672,
            0.1785593,
            0.153417,
            0.15099354,
            0.11075057,
            0.09986048,
            0.06060822,
            0.00869774,
            0.00364596,
        ],
        1.0e-4,
    )

    assert isinstance(inversion, aa.InversionImagingMapping)

    assert inversion.regularization_matrix[0:3, 0] == pytest.approx(
        [4.00000001, -1.0, -1.0], 1.0e-4
    )
    assert inversion.regularization_matrix[0:3, 1] == pytest.approx(
        [-1.0, 3.00000001, 0.0], 1.0e-4
    )
    assert inversion.regularization_matrix[0:3, 2] == pytest.approx(
        [-1.0, 0.0, 4.00000001], 1.0e-4
    )

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        10.417803331712355, 1.0e-4
    )
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )

    mapper = copy.copy(knn_mapper_9_3x3)
    mapper.regularization = regularization_adaptive_brightness_split

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[mapper],
    )

    assert inversion.regularization_matrix[0:3, 0] == pytest.approx(
        [22.47519068, -16.373819, 8.39424766], 1.0e-4
    )
    assert inversion.regularization_matrix[0:3, 1] == pytest.approx(
        [-16.373819, 112.1402519, -13.56808248], 1.0e-4
    )
    assert inversion.regularization_matrix[0:3, 2] == pytest.approx(
        [8.39424766, -13.56808248, 26.10743213], 1.0e-4
    )


def test__inversion_imaging__via_regularizations(
    masked_imaging_7x7_no_blur,
    delaunay_mapper_9_3x3,
    regularization_constant,
    regularization_adaptive_brightness,
):
    mapper = copy.copy(delaunay_mapper_9_3x3)
    mapper.regularization = regularization_constant

    masked_imaging_7x7_no_blur_sparse_operator = (
        masked_imaging_7x7_no_blur.apply_sparse_operator_cpu()
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur_sparse_operator,
        linear_obj_list=[mapper],
    )

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        10.66747, 1.0e-4
    )
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )

    mapper = copy.copy(delaunay_mapper_9_3x3)
    mapper.regularization = regularization_adaptive_brightness

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur_sparse_operator,
        linear_obj_list=[mapper],
    )

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        47.410169, 1.0e-4
    )
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )


def test__inversion_imaging__zeroed_pixels(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
):

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.Settings(
            use_positive_only_solver=True, use_edge_zeroed_pixels=True
        ),
    )

    assert inversion.reconstruction.shape[0] == 9
    assert inversion.reconstruction[0] == 0.0
    assert inversion.reconstruction[4] > 0.0


def test__inversion_imaging__via_linear_obj_func_and_mapper(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObj(
        parameters=1,
        grid=grid,
        mapping_matrix=np.full(fill_value=0.5, shape=(9, 1)),
        regularization=None,
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(
            no_regularization_add_to_curvature_diag_value=False,
        ),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObj)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        7.2571757082469945, 1.0e-4
    )
    assert inversion.reconstruction_dict[linear_obj] == pytest.approx(
        np.array([2.0]), 1.0e-4
    )
    assert inversion.reconstruction_dict[rectangular_mapper_7x7_3x3][0] < 1.0e-4
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )

    masked_imaging_7x7_no_blur_sparse_operator = (
        masked_imaging_7x7_no_blur.apply_sparse_operator_cpu()
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur_sparse_operator,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(
            no_regularization_add_to_curvature_diag_value=False,
        ),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObj)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        7.2571757082469945, 1.0e-4
    )


def test__inversion_imaging__via_linear_obj_func_and_mapper__force_edge_pixels_to_zero(
    masked_imaging_7x7_no_blur, delaunay_mapper_9_3x3
):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObj(
        parameters=1,
        grid=grid,
        mapping_matrix=np.array(
            [[1.0], [2.0], [3.0], [2.0], [3.0], [4.0], [3.0], [1.0], [2.0]]
        ),
        regularization=None,
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, delaunay_mapper_9_3x3],
        settings=aa.Settings(
            no_regularization_add_to_curvature_diag_value=False,
        ),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObj)
    assert isinstance(inversion, aa.InversionImagingMapping)

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, delaunay_mapper_9_3x3],
        settings=aa.Settings(
            use_positive_only_solver=True,
            no_regularization_add_to_curvature_diag_value=False,
        ),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObj)
    assert isinstance(inversion, aa.InversionImagingMapping)


def test__inversion_imaging__compare_mapping_and_sparse_operator_values(
    masked_imaging_7x7, delaunay_mapper_9_3x3
):

    masked_imaging_7x7_sparse_operator = masked_imaging_7x7.apply_sparse_operator_cpu()

    inversion_sparse_operator = aa.Inversion(
        dataset=masked_imaging_7x7_sparse_operator,
        linear_obj_list=[delaunay_mapper_9_3x3],
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[delaunay_mapper_9_3x3],
        settings=aa.Settings(),
    )

    assert inversion_sparse_operator.reconstruction == pytest.approx(
        inversion_mapping.reconstruction, 1.0e-4
    )
    assert (
        inversion_sparse_operator.mapped_reconstructed_operated_data.array
        == pytest.approx(
            inversion_mapping.mapped_reconstructed_operated_data.array, 1.0e-4
        )
    )
    assert inversion_sparse_operator.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_mapping.log_det_curvature_reg_matrix_term
    )


def test__inversion_imaging__linear_obj_func_and_non_func_give_same_terms(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
):
    masked_imaging_7x7_no_blur = copy.copy(masked_imaging_7x7_no_blur)
    masked_imaging_7x7_no_blur.data[4] = 2.0

    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObj(
        parameters=2,
        grid=grid,
        mapping_matrix=np.full(fill_value=0.5, shape=(9, 2)),
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(
            use_positive_only_solver=True,
        ),
    )

    masked_imaging_7x7_no_blur = copy.copy(masked_imaging_7x7_no_blur)

    masked_imaging_7x7_no_blur.data -= (
        inversion.mapped_reconstructed_operated_data_dict[linear_obj]
    )

    inversion_no_linear_func = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.Settings(
            use_positive_only_solver=True,
        ),
    )

    assert inversion.regularization_term == pytest.approx(
        inversion_no_linear_func.regularization_term, 1.0e-4
    )
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_no_linear_func.log_det_curvature_reg_matrix_term, 1.0e-4
    )
    assert inversion.log_det_regularization_matrix_term == pytest.approx(
        inversion_no_linear_func.log_det_regularization_matrix_term, 1.0e-4
    )


def test__inversion_imaging__linear_obj_func_with_sparse_operator(
    masked_imaging_7x7,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
):

    mask = masked_imaging_7x7.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    mapping_matrix = np.full(fill_value=0.5, shape=(9, 2))
    mapping_matrix[0, 0] = 0.8
    mapping_matrix[1, 1] = 0.4

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    masked_imaging_7x7_sparse_operator = masked_imaging_7x7.apply_sparse_operator_cpu()

    inversion_sparse_operator = aa.Inversion(
        dataset=masked_imaging_7x7_sparse_operator,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    assert inversion_mapping.data_vector == pytest.approx(
        inversion_sparse_operator.data_vector, 1.0e-4
    )
    assert inversion_mapping.curvature_matrix == pytest.approx(
        inversion_sparse_operator.curvature_matrix, 1.0e-4
    )
    assert inversion_mapping.curvature_reg_matrix == pytest.approx(
        inversion_sparse_operator.curvature_reg_matrix, 1.0e-4
    )
    assert inversion_mapping.reconstruction == pytest.approx(
        inversion_sparse_operator.reconstruction, 1.0e-4
    )
    assert inversion_mapping.mapped_reconstructed_operated_data.array == pytest.approx(
        inversion_sparse_operator.mapped_reconstructed_operated_data.array, 1.0e-4
    )

    linear_obj_1 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    linear_obj_2 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            linear_obj_2,
        ],
        settings=aa.Settings(),
    )

    masked_imaging_7x7_sparse_operator = masked_imaging_7x7.apply_sparse_operator_cpu()

    inversion_sparse_operator = aa.Inversion(
        dataset=masked_imaging_7x7_sparse_operator,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            linear_obj_2,
        ],
    )

    assert inversion_mapping.data_vector == pytest.approx(
        inversion_sparse_operator.data_vector, 1.0e-4
    )
    assert inversion_mapping.curvature_matrix == pytest.approx(
        inversion_sparse_operator.curvature_matrix, 1.0e-4
    )


def test__inversion_interferometer__via_mapper(
    interferometer_7_no_fft,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
):
    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.Settings(),
    )

    assert isinstance(inversion, aa.InversionInterferometerMapping)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        1.0 + 0.0j * np.ones(shape=(7,)), 1.0e-4
    )
    assert (np.imag(inversion.mapped_reconstructed_operated_data) < 0.0001).all()
    assert (np.imag(inversion.mapped_reconstructed_operated_data) > 0.0).all()
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(10.2116, 1.0e-4)

    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[delaunay_mapper_9_3x3],
        settings=aa.Settings(),
    )

    assert isinstance(inversion, aa.InversionInterferometerMapping)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        1.0 + 0.0j * np.ones(shape=(7,)), 1.0e-4
    )
    assert (np.imag(inversion.mapped_reconstructed_operated_data) < 0.0001).all()
    assert (np.imag(inversion.mapped_reconstructed_operated_data) > 0.0).all()
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        14.49772, 1.0e-4
    )


def test__inversion_matrices__x2_mappers(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    regularization_constant,
):

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3, delaunay_mapper_9_3x3],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    assert inversion.operated_mapping_matrix[0:9, 0:9] == pytest.approx(
        rectangular_mapper_7x7_3x3.mapping_matrix, abs=1.0e-4
    )
    assert inversion.operated_mapping_matrix[0:9, 9:18] == pytest.approx(
        delaunay_mapper_9_3x3.mapping_matrix, abs=1.0e-4
    )

    operated_mapping_matrix = np.hstack(
        [
            rectangular_mapper_7x7_3x3.mapping_matrix,
            delaunay_mapper_9_3x3.mapping_matrix,
        ]
    )

    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=operated_mapping_matrix, noise_map=inversion.noise_map
    )

    assert inversion.curvature_matrix == pytest.approx(curvature_matrix, 1.0e-4)

    regularization_matrix_of_reg_0 = regularization_constant.regularization_matrix_from(
        linear_obj=rectangular_mapper_7x7_3x3
    )
    regularization_matrix_of_reg_1 = regularization_constant.regularization_matrix_from(
        linear_obj=delaunay_mapper_9_3x3
    )

    assert (
        inversion.regularization_matrix[0:9, 0:9] == regularization_matrix_of_reg_0
    ).all()

    assert (
        inversion.regularization_matrix[9:18, 9:18] == regularization_matrix_of_reg_1
    ).all()
    assert (inversion.regularization_matrix[0:9, 9:18] == np.zeros((9, 9))).all()
    assert (inversion.regularization_matrix[9:18, 0:9] == np.zeros((9, 9))).all()

    assert inversion.reconstruction_dict[rectangular_mapper_7x7_3x3][
        4
    ] == pytest.approx(0.5000029374603968, 1.0e-4)
    assert inversion.reconstruction_dict[delaunay_mapper_9_3x3][4] == pytest.approx(
        0.4999970390886761, 1.0e-4
    )
    assert inversion.reconstruction[13] == pytest.approx(0.49999703908867, 1.0e-4)

    assert inversion.mapped_reconstructed_operated_data_dict[
        rectangular_mapper_7x7_3x3
    ][4] == pytest.approx(0.5000029, 1.0e-4)
    assert inversion.mapped_reconstructed_operated_data_dict[delaunay_mapper_9_3x3][
        3
    ] == pytest.approx(0.49999704, 1.0e-4)
    assert inversion.mapped_reconstructed_operated_data[4] == pytest.approx(
        0.99999998, 1.0e-4
    )


def test__inversion_imaging__positive_only_solver(masked_imaging_7x7_no_blur):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1, grid=grid, mapping_matrix=np.full(fill_value=0.5, shape=(9, 1))
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        np.ones(9), 1.0e-4
    )
    assert inversion.reconstruction == pytest.approx(np.array([2.0]), 1.0e-4)


def test__data_linear_func_matrix_dict(
    masked_imaging_7x7,
    rectangular_mapper_7x7_3x3,
):

    mask = masked_imaging_7x7.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    mapping_matrix = np.full(fill_value=0.5, shape=(9, 2))
    mapping_matrix[0, 0] = 0.8
    mapping_matrix[1, 1] = 0.4

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    assert inversion_mapping.data_linear_func_matrix_dict[linear_obj][
        0
    ] == pytest.approx([0.075, 0.05972222], 1.0e-4)
    assert inversion_mapping.data_linear_func_matrix_dict[linear_obj][
        1
    ] == pytest.approx([0.09166667, 0.07847222], 1.0e-4)
    assert inversion_mapping.data_linear_func_matrix_dict[linear_obj][
        2
    ] == pytest.approx([0.06458333, 0.05972222], 1.0e-4)
