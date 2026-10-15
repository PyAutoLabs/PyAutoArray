import copy
import numpy as np
import pytest
from dill import settings

import autoarray as aa

from autoarray.inversion.inversion.imaging_numba.sparse import (
    InversionImagingSparseNumba,
)


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


def test__inversion_imaging__zeroed_pixels__noise_map_agrees_with_the_reconstruction(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
):
    """
    End-to-end: the reconstruction and its noise map agree on which pixels were solved.

    The unit tests in `test_abstract.py` drive this through `MockInversion` with a hand-built matrix; this
    asserts the same invariant through the real solver, mesh and index bookkeeping, which is where the two
    previously disagreed. The 3x3 mesh zeroes its 8 edge pixels, leaving only the centre solved for.
    """
    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.Settings(
            use_positive_only_solver=True, use_edge_zeroed_pixels=True
        ),
    )

    reconstruction = np.asarray(inversion.reconstruction)
    noise_map = np.asarray(inversion.reconstruction_noise_map)

    keep = np.asarray(inversion.solve_ids_to_keep)
    excluded = np.setdiff1d(np.arange(reconstruction.shape[0]), keep)

    assert keep == pytest.approx(np.array([4]))

    # the NaN set is EXACTLY the set the solve excluded -- no more, no less
    assert np.array_equal(np.flatnonzero(np.isnan(noise_map)), np.sort(excluded))

    # NaN implies an exact structural zero. The converse is deliberately NOT asserted: the
    # non-negative solver also pins pixels it DID solve for at exactly 0.0, and those keep a
    # finite noise value. Asserting the biconditional would pass here only because this 3x3
    # fixture happens to solve a single pixel -- on a real fit 603 of 784 pixels read 0.0 while
    # only the 108 excluded ones are NaN.
    assert (reconstruction[excluded] == 0.0).all()

    assert np.isfinite(noise_map[keep]).all()
    assert inversion.reconstruction_covariance_matrix.shape == (9, 9)


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


def _delaunay_nn_mapper_from(mask, pixels=25, max_neighbors=None, seed=3):
    """
    A `DelaunayNN` (Sibson) mapper on a seeded irregular mesh, so the masked pixels carry
    wide, variable-width natural-neighbour rows rather than the 3-wide Delaunay ones.
    `max_neighbors` lowers the mesh's class-level cap on the instance so rows overflow
    and carry NaN weights.
    """
    mesh = aa.mesh.DelaunayNN(pixels=pixels)

    if max_neighbors is not None:
        mesh.max_neighbors = max_neighbors

    rng = np.random.default_rng(seed=seed)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=aa.Grid2D.from_mask(mask=mask, over_sample_size=1),
        source_plane_mesh_grid=aa.Grid2DIrregular(
            rng.uniform(-4.0, 4.0, size=(pixels, 2))
        ),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=aa.reg.Constant(coefficient=1.0),
    )


def test__inversion_imaging__delaunay_nn__numba_sparse_operator_matches_mapping(
    masked_imaging_7x7,
):
    pytest.importorskip("numba")

    mapper = _delaunay_nn_mapper_from(mask=masked_imaging_7x7.mask)

    pix_sizes = np.asarray(mapper.pix_sizes_for_sub_slim_index)

    assert pix_sizes.max() > 3
    assert len(np.unique(pix_sizes)) > 1

    inversion_sparse_operator = aa.Inversion(
        dataset=masked_imaging_7x7.apply_sparse_operator_cpu(),
        linear_obj_list=[mapper],
    )
    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[mapper],
    )

    assert isinstance(inversion_sparse_operator, InversionImagingSparseNumba)
    assert isinstance(inversion_mapping, aa.InversionImagingMapping)

    np.testing.assert_allclose(
        np.asarray(inversion_sparse_operator.curvature_matrix),
        np.asarray(inversion_mapping.curvature_matrix),
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(np.asarray(inversion_mapping.curvature_matrix)).max(),
    )
    np.testing.assert_allclose(
        np.asarray(inversion_sparse_operator.reconstruction),
        np.asarray(inversion_mapping.reconstruction),
        rtol=1.0e-8,
    )
    assert inversion_sparse_operator.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_mapping.log_det_curvature_reg_matrix_term, rel=1.0e-10
    )
    assert inversion_sparse_operator.regularization_term == pytest.approx(
        inversion_mapping.regularization_term, rel=1.0e-8
    )


def test__inversion_imaging__delaunay_nn_nan_weights__numba_gives_nan_like_mapping(
    masked_imaging_7x7,
):
    """
    Overflowing Sibson rows carry NaN weights so the model sample is rejected. The mapping
    route turns them into a NaN `log_det_curvature_reg_matrix_term`; the numba sparse
    route must do the same, not silently return a finite one.
    """
    pytest.importorskip("numba")

    mapper = _delaunay_nn_mapper_from(mask=masked_imaging_7x7.mask, max_neighbors=6)

    nan_rows = np.isnan(np.asarray(mapper.pix_weights_for_sub_slim_index)).any(axis=1)

    assert nan_rows.any() and not nan_rows.all()

    inversion_sparse_operator = aa.Inversion(
        dataset=masked_imaging_7x7.apply_sparse_operator_cpu(),
        linear_obj_list=[mapper],
    )

    assert isinstance(inversion_sparse_operator, InversionImagingSparseNumba)

    for inversion in (
        inversion_sparse_operator,
        aa.Inversion(dataset=masked_imaging_7x7, linear_obj_list=[mapper]),
    ):
        assert np.isnan(np.asarray(inversion.curvature_matrix)).any()
        assert np.isnan(inversion.log_det_curvature_reg_matrix_term)


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


def test__inversion_interferometer__via_linear_obj_func_list__sparse_operator(
    interferometer_7_no_fft,
):
    mask = interferometer_7_no_fft.real_space_mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        grid=grid,
        mapping_matrix=np.random.default_rng(seed=1).normal(
            size=(mask.pixels_in_mask, 2)
        ),
    )

    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[linear_obj],
    )

    assert isinstance(inversion, aa.InversionInterferometerMapping)

    # Unlike imaging, a func-list-only interferometer inversion (e.g. an MGE with no pixelization) uses the
    # sparse operator when the dataset has one.

    inversion_sparse = aa.Inversion(
        dataset=interferometer_7_no_fft.apply_sparse_operator(use_jax=False),
        linear_obj_list=[linear_obj],
    )

    assert type(inversion_sparse) is aa.InversionInterferometerSparse
    assert inversion_sparse.data_vector == pytest.approx(inversion.data_vector, 1.0e-8)
    assert inversion_sparse.curvature_matrix == pytest.approx(
        inversion.curvature_matrix, 1.0e-8
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


class _GeometryMapper(aa.m.MockMapper):
    """
    A mapper carrying only what the interferometer route gate reads: the number of source
    pixels and the per-image-pixel non-zero counts, with no over-sampling.
    """

    def __init__(self, source_pixels, nnz, image_pixels):
        import types

        super().__init__(
            parameters=source_pixels,
            over_sampler=types.SimpleNamespace(
                sub_fraction=types.SimpleNamespace(array=np.ones(image_pixels))
            ),
        )

        sizes = np.full(image_pixels, nnz // image_pixels, dtype=np.int64)
        sizes[: nnz - int(sizes.sum())] += 1

        self._pix_sizes = sizes

    @property
    def pix_sizes_for_sub_slim_index(self):
        return self._pix_sizes


def _route_is_numba(source_pixels, nnz, image_pixels, extent_shape, settings=None):
    from autoarray.inversion.inversion.factory import _use_interferometer_numba

    mapper = _GeometryMapper(
        source_pixels=source_pixels, nnz=nnz, image_pixels=image_pixels
    )

    return _use_interferometer_numba(
        linear_obj_list=[mapper],
        settings=settings,
        xp=np,
        extent_shape=extent_shape,
    )


# Every measured point the cost model was calibrated or checked against, with the route that
# measured faster (single thread, seconds per call / per curvature build):
#
# - Aris 512x512 mask: 91,507 masked pixels, extent 341x341 (`shape_native_masked_pixels`),
#   Delaunay P=3, nnz 273,661 (PyAutoArray#628): FFT 28.3 / 44.1 / 97.3 s vs direct_conv
#   38.4 / 34.2 / 41.2 s.
# - Aris 125x125 mask: 5,496 masked pixels, extent 83x83: direct_conv 0.40 / 2.08 / 6.12 s
#   (1 thread) vs the JAX FFT route 1.96 / 6.84 / 12.0 s (2.8-3.6 cores).
# - autolens_profiling#226 bake-off (`bakeoff_v2026.8.17.1.json`), rfft2_numpy vs direct_conv.
_MEASURED_ROUTES = [
    # (label, S, nnz, image pixels, extent, faster route)
    ("aris_512_1459", 1459, 273661, 91507, (341, 341), "fft"),
    ("aris_512_2577", 2577, 273661, 91507, (341, 341), "numba"),
    ("aris_512_5746", 5746, 273661, 91507, (341, 341), "numba"),
    ("aris_125_1405", 1405, 16438, 5496, (83, 83), "numba"),
    ("aris_125_2947", 2947, 16448, 5496, (83, 83), "numba"),
    ("aris_125_5526", 5526, 16488, 5496, (83, 83), "numba"),
    ("bakeoff_sma_delaunay", 1500, 11556, 3852, (70, 70), "numba"),
    ("bakeoff_sma_rect", 784, 15408, 3852, (70, 70), "numba"),
    ("bakeoff_alma_delaunay", 1500, 46140, 15380, (140, 140), "numba"),
    ("bakeoff_alma_rect", 784, 61520, 15380, (140, 140), "fft"),
    ("bakeoff_alma_high_delaunay", 1500, 184716, 61572, (280, 280), "fft"),
    ("bakeoff_alma_high_rect", 784, 246288, 61572, (280, 280), "fft"),
]


@pytest.mark.parametrize(
    "label, source_pixels, nnz, image_pixels, extent_shape, faster",
    _MEASURED_ROUTES,
    ids=[row[0] for row in _MEASURED_ROUTES],
)
def test__interferometer_route__default_cost_model_picks_the_measured_faster_route(
    label, source_pixels, nnz, image_pixels, extent_shape, faster
):
    """
    The Witness of PyAutoArray#628: on both Aris masks the default routing picks the
    faster of numba `direct_conv` and the NumPy FFT curvature at every measured vertex
    count, and it still does at every autolens_profiling#226 bake-off cell.
    """
    pytest.importorskip("numba")

    assert aa.Settings().interferometer_numba_nnz_per_source_max is None

    is_numba = _route_is_numba(
        source_pixels=source_pixels,
        nnz=nnz,
        image_pixels=image_pixels,
        extent_shape=extent_shape,
    )

    assert is_numba == (faster == "numba")


def test__interferometer_route__crossover_moves_with_the_extent():
    """
    The old fixed gate (60 non-zeros per source column) is right at the
    autolens_profiling#226 extents and wrong on the Aris 512x512 one; the cost model
    crosses near 60 at the former and between the measured 106 (numba) and 188 (FFT) at
    the latter.
    """
    pytest.importorskip("numba")

    # alma extent (140x140), where the ~60 Delaunay crossover was measured.
    assert _route_is_numba(1500, 1500 * 50, 15380, (140, 140))
    assert not _route_is_numba(1500, 1500 * 80, 15380, (140, 140))

    # Aris 512x512 extent at 2,577 vertices: 106 non-zeros per column is numba by the cost
    # model and FFT by the old gate, which measured 1.29x slower.
    assert _route_is_numba(2577, 273661, 91507, (341, 341))
    assert not _route_is_numba(
        2577,
        273661,
        91507,
        (341, 341),
        settings=aa.Settings(interferometer_numba_nnz_per_source_max=60.0),
    )


def test__interferometer_route__settings_overrides():
    pytest.importorskip("numba")

    # Aris 512 at 1,459 vertices: the cost model picks FFT.
    geometry = dict(
        source_pixels=1459, nnz=273661, image_pixels=91507, extent_shape=(341, 341)
    )
    # Aris 512 at 5,746 vertices: the cost model picks numba.
    geometry_numba = dict(
        source_pixels=5746, nnz=273661, image_pixels=91507, extent_shape=(341, 341)
    )

    assert not _route_is_numba(**geometry)
    assert _route_is_numba(**geometry_numba)

    # A positive gate is a hard gate, whatever the extent (pre-#628 behaviour).
    assert _route_is_numba(
        **geometry,
        settings=aa.Settings(interferometer_numba_nnz_per_source_max=200.0),
    )
    assert not _route_is_numba(
        **geometry_numba,
        settings=aa.Settings(interferometer_numba_nnz_per_source_max=40.0),
    )

    # 0 is the kill switch.
    assert not _route_is_numba(
        **geometry_numba,
        settings=aa.Settings(interferometer_numba_nnz_per_source_max=0),
    )

    # The forced route wins over the cost model, the hard gate and the kill switch.
    assert _route_is_numba(**geometry, settings=aa.Settings(interferometer_route="numba"))
    assert _route_is_numba(
        **geometry_numba,
        settings=aa.Settings(
            interferometer_route="numba", interferometer_numba_nnz_per_source_max=0
        ),
    )
    assert not _route_is_numba(
        **geometry_numba, settings=aa.Settings(interferometer_route="fft")
    )

    # Without the extent the cost model cannot be evaluated and the FFT route is used.
    assert not _route_is_numba(**{**geometry_numba, "extent_shape": None})

    with pytest.raises(ValueError):
        aa.Settings(interferometer_route="direct_conv")


def test__interferometer_route__fft_length_cost_penalises_awkward_lengths():
    from autoarray.inversion.inversion.interferometer_numba import (
        inversion_interferometer_numba_util as numba_util,
    )

    def per_element(n):
        return numba_util._fft_length_cost(n) / n

    # 682 = 2 * 11 * 31 (the Aris 512 padded extent) costs far more per element than the
    # smooth 560 = 2^4 * 5 * 7, although log2 of the two differs by 3 %.
    assert per_element(682) > 2.0 * per_element(560)

    # 166 = 2 * 83 (the Aris 125 padded extent) is pocketfft's Bluestein case: the
    # estimate is capped by the Bluestein cost, below the direct mixed-radix one.
    assert numba_util._fft_length_cost(166) < 166 * (2.0 + 1.1 * 83)

    # A power of two is the cheapest per element of its size class.
    assert per_element(512) < per_element(510)
