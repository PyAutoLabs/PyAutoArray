import autoarray as aa

import numpy as np
import pytest
from pathlib import Path

directory = Path(__file__).resolve().parent


def test__curvature_matrix(rectangular_mapper_7x7_3x3):
    operated_mapping_matrix = np.array(
        [[1.0 + 1j, 1.0 + 1j, 1.0 + 1j], [1.0 + 1j, 1.0 + 1j, 1.0 + 1j]]
    )
    noise_map = np.array([1.0 + 1j, 1.0 + 1j])

    inversion = aa.m.MockInversionInterferometer(
        linear_obj_list=[aa.m.MockLinearObj(parameters=1), rectangular_mapper_7x7_3x3],
        operated_mapping_matrix=operated_mapping_matrix,
        noise_map=noise_map,
        settings=aa.Settings(no_regularization_add_to_curvature_diag_value=False),
    )

    assert inversion.curvature_matrix[0:2, 0:2] == pytest.approx(
        np.array([[4.0, 4.0], [4.0, 4.0]]), 1.0e-4
    )

    assert inversion.curvature_matrix[0, 0] - 4.0 < 1.0e-12
    assert inversion.curvature_matrix[2, 2] - 4.0 < 1.0e-12

    inversion = aa.m.MockInversionInterferometer(
        linear_obj_list=[aa.m.MockLinearObj(parameters=1), rectangular_mapper_7x7_3x3],
        operated_mapping_matrix=operated_mapping_matrix,
        noise_map=noise_map,
        settings=aa.Settings(no_regularization_add_to_curvature_diag_value=True),
    )

    assert inversion.curvature_matrix[0, 0] - 4.0 > 0.0
    assert inversion.curvature_matrix[2, 2] - 4.0 < 1.0e-12


def test__fast_chi_squared(
    interferometer_7_no_fft,
    rectangular_mapper_7x7_3x3,
):

    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.Settings(),
    )

    residual_map = aa.util.fit.residual_map_from(
        data=interferometer_7_no_fft.data,
        model_data=inversion.mapped_reconstructed_operated_data,
    )

    chi_squared_map = aa.util.fit.chi_squared_map_complex_from(
        residual_map=residual_map,
        noise_map=interferometer_7_no_fft.noise_map,
    )

    chi_squared = aa.util.fit.chi_squared_complex_from(chi_squared_map=chi_squared_map)

    assert inversion.fast_chi_squared == pytest.approx(chi_squared, 1.0e-4)


def test__operated_mapping_matrix_list__override_is_honored():
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

    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    mapping_matrix = np.ones((mask.pixels_in_mask, 1))
    override = (999.0 + 1.0j) * np.ones((n_visibilities, 1))

    linear_obj_override = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=mapping_matrix,
        operated_mapping_matrix_override=override,
    )
    linear_obj_no_override = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=mapping_matrix,
    )

    inversion = aa.Inversion(
        dataset=dataset,
        linear_obj_list=[linear_obj_override, linear_obj_no_override],
    )

    operated_mapping_matrix_list = inversion.operated_mapping_matrix_list

    assert operated_mapping_matrix_list[0] == pytest.approx(override, 1.0e-8)

    transformed_mapping_matrix = dataset.transformer.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    assert operated_mapping_matrix_list[1] == pytest.approx(
        transformed_mapping_matrix, 1.0e-8
    )

    assert inversion.operated_mapping_matrix[:, 0] == pytest.approx(
        override[:, 0], 1.0e-8
    )
    assert inversion.curvature_matrix.shape == (2, 2)
    assert inversion.data_vector.shape == (2,)


def test__operated_mapping_matrix_override__wrong_shape_raises():
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

    n_visibilities = 7
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    # A real-space shaped override (e.g. [total_mask_pixels, params]) is not valid for an
    # interferometer inversion, whose override must be in visibility space.
    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=np.ones((mask.pixels_in_mask, 1)),
        operated_mapping_matrix_override=np.ones((mask.pixels_in_mask, 1)),
    )

    inversion = aa.Inversion(dataset=dataset, linear_obj_list=[linear_obj])

    with pytest.raises(aa.exc.InversionException):
        inversion.operated_mapping_matrix_list


def test__operated_mapping_matrix_override__sparse_operator_raises():
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

    mesh = aa.mesh.Delaunay(pixels=9)
    image_mesh = aa.image_mesh.Overlay(shape=(3, 3))
    image_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=None)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )
    mapper = aa.Mapper(interpolator=interpolator)

    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset_sparse = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    ).apply_sparse_operator(use_jax=False)

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=np.ones((mask.pixels_in_mask, 1)),
        operated_mapping_matrix_override=(999.0 + 1.0j) * np.ones((n_visibilities, 1)),
    )

    with pytest.raises(aa.exc.InversionException):
        aa.Inversion(
            dataset=dataset_sparse,
            linear_obj_list=[mapper, linear_obj],
        )


def test__curvature_matrix__interferometer_sparse_operator__delaunay__identical_to_mapping():
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

    mesh = aa.mesh.Delaunay(pixels=9)
    image_mesh = aa.image_mesh.Overlay(shape=(3, 3))
    image_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=None)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )
    mapper = aa.Mapper(interpolator=interpolator)

    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    dataset_sparse = dataset.apply_sparse_operator(use_jax=False)

    inversion_sparse = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
    )

    inversion_mapping = aa.Inversion(
        dataset=dataset,
        linear_obj_list=[mapper],
    )

    assert inversion_sparse.curvature_matrix == pytest.approx(
        inversion_mapping.curvature_matrix, 1.0e-4
    )


def test__curvature_matrix__interferometer_sparse_operator__delaunay__dft_and_nufft_match():
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

    mesh = aa.mesh.Delaunay(pixels=9)
    image_mesh = aa.image_mesh.Overlay(shape=(3, 3))
    image_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=None)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )
    mapper = aa.Mapper(interpolator=interpolator)

    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset_dft = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    ).apply_sparse_operator(use_jax=False)

    dataset_nufft = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerNUFFT,
    ).apply_sparse_operator(use_jax=False)

    inversion_dft = aa.Inversion(
        dataset=dataset_dft,
        linear_obj_list=[mapper],
    )

    inversion_nufft = aa.Inversion(
        dataset=dataset_nufft,
        linear_obj_list=[mapper],
    )

    assert inversion_nufft.curvature_matrix == pytest.approx(
        inversion_dft.curvature_matrix, 1.0e-4
    )
    assert inversion_nufft.data_vector == pytest.approx(
        inversion_dft.data_vector, 1.0e-4
    )


def test__preloads_interferometer__curvature_matrix_returned_directly_and_skips_rebuild():
    """
    A `PreloadsInterferometer` injects a pre-computed `curvature_matrix` (`F`) — e.g. the datacube
    shared-state path where `F` is identical across channels. The sparse interferometer inversion
    must return it verbatim (skipping the dominant `F = LᵀW̃L` build) while leaving the per-channel
    `data_vector` unchanged.
    """
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

    mesh = aa.mesh.Delaunay(pixels=9)
    image_mesh = aa.image_mesh.Overlay(shape=(3, 3))
    image_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=None)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )
    mapper = aa.Mapper(interpolator=interpolator)

    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    dataset_sparse = dataset.apply_sparse_operator(use_jax=False)

    inversion = aa.Inversion(dataset=dataset_sparse, linear_obj_list=[mapper])
    curvature_matrix = inversion.curvature_matrix

    # A sentinel injected via the shared `curvature_matrix` is returned verbatim, proving the
    # expensive `curvature_matrix_diag` build is skipped (else the real F would be returned).
    sentinel = np.full_like(curvature_matrix, 7.0)
    inversion_sentinel = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        preloads=aa.PreloadsInterferometer(curvature_matrix=sentinel),
    )
    assert inversion_sentinel.curvature_matrix is sentinel

    # An empty `PreloadsInterferometer` (curvature_matrix left None) falls back to the standard build.
    inversion_empty_preloads = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        preloads=aa.PreloadsInterferometer(),
    )
    assert inversion_empty_preloads.curvature_matrix == pytest.approx(curvature_matrix)

    # Preloading the real F reproduces the un-preloaded curvature matrix and leaves the per-channel
    # data_vector (which does not depend on the preloaded F) unchanged.
    inversion_preloaded = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        preloads=aa.PreloadsInterferometer(curvature_matrix=curvature_matrix),
    )
    assert inversion_preloaded.curvature_matrix is curvature_matrix
    assert inversion_preloaded.data_vector == pytest.approx(inversion.data_vector)


def _sparse_parity_setup(seed=0):
    """
    The 7x7 / Delaunay / TransformerDFT setup shared by the sparse-operator parity tests below,
    returning the dense and sparse-operator datasets alongside the mask and a random generator.
    """
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

    n_visibilities = 5
    rng = np.random.default_rng(seed=seed)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    return mask, rng, dataset, dataset.apply_sparse_operator(use_jax=False)


def _mapper_from(mask, pixels, shape, regularization=None):
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)

    mesh = aa.mesh.Delaunay(pixels=pixels)
    image_mesh = aa.image_mesh.Overlay(shape=shape)
    image_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=None)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )

    return aa.Mapper(interpolator=interpolator, regularization=regularization)


def _assert_sparse_matches_mapping(dataset, dataset_sparse, linear_obj_list):
    inversion_sparse = aa.Inversion(
        dataset=dataset_sparse, linear_obj_list=linear_obj_list
    )
    inversion_mapping = aa.Inversion(dataset=dataset, linear_obj_list=linear_obj_list)

    assert isinstance(inversion_sparse, aa.InversionInterferometerSparse)
    assert isinstance(inversion_mapping, aa.InversionInterferometerMapping)

    assert inversion_sparse.curvature_matrix == pytest.approx(
        np.array(inversion_mapping.curvature_matrix), 1.0e-8
    )
    assert inversion_sparse.data_vector == pytest.approx(
        np.array(inversion_mapping.data_vector), 1.0e-8
    )
    assert inversion_sparse.reconstruction == pytest.approx(
        np.array(inversion_mapping.reconstruction), 1.0e-4
    )
    assert inversion_sparse.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_mapping.log_det_curvature_reg_matrix_term, 1.0e-6
    )


def test__interferometer_sparse_operator__func_list_and_mapper__identical_to_mapping():
    """
    A linear function list (e.g. linear light profiles) fitted simultaneously with a `Mapper` must
    reproduce the dense (mapping formalism) inversion, including the mapper-function off-diagonal
    blocks and the function-function block which the sparse path previously dropped entirely.
    """
    mask, rng, dataset, dataset_sparse = _sparse_parity_setup()

    mapper = _mapper_from(
        mask=mask,
        pixels=9,
        shape=(3, 3),
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2)),
    )

    _assert_sparse_matches_mapping(
        dataset=dataset,
        dataset_sparse=dataset_sparse,
        linear_obj_list=[linear_obj, mapper],
    )

    # The linear function list is also supported when it trails the mapper in the list, where its
    # parameters occupy the final rows / columns of the curvature matrix.
    _assert_sparse_matches_mapping(
        dataset=dataset,
        dataset_sparse=dataset_sparse,
        linear_obj_list=[mapper, linear_obj],
    )


def test__interferometer_sparse_operator__x2_mappers__identical_to_mapping():
    """
    Two `Mapper` objects fitted simultaneously require the mapper-mapper off-diagonal block
    `A_0ᵀ W~ A_1`, which the sparse path previously dropped (only the first mapper was used).
    """
    mask, rng, dataset, dataset_sparse = _sparse_parity_setup()

    mapper_0 = _mapper_from(
        mask=mask,
        pixels=9,
        shape=(3, 3),
        regularization=aa.reg.Constant(coefficient=1.0),
    )
    mapper_1 = _mapper_from(
        mask=mask,
        pixels=16,
        shape=(4, 4),
        regularization=aa.reg.Constant(coefficient=2.0),
    )

    _assert_sparse_matches_mapping(
        dataset=dataset,
        dataset_sparse=dataset_sparse,
        linear_obj_list=[mapper_0, mapper_1],
    )


def test__interferometer_sparse_operator__func_list_and_x2_mappers__identical_to_mapping():
    """
    The full mixed case: one or more linear function lists fitted simultaneously with multiple
    mappers, exercising every block of the curvature matrix at once.
    """
    mask, rng, dataset, dataset_sparse = _sparse_parity_setup()

    mapper_0 = _mapper_from(
        mask=mask,
        pixels=9,
        shape=(3, 3),
        regularization=aa.reg.Constant(coefficient=1.0),
    )
    mapper_1 = _mapper_from(
        mask=mask,
        pixels=16,
        shape=(4, 4),
        regularization=aa.reg.Constant(coefficient=2.0),
    )

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2)),
    )

    _assert_sparse_matches_mapping(
        dataset=dataset,
        dataset_sparse=dataset_sparse,
        linear_obj_list=[linear_obj, mapper_0, mapper_1],
    )

    linear_obj_1 = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 1)),
    )

    _assert_sparse_matches_mapping(
        dataset=dataset,
        dataset_sparse=dataset_sparse,
        linear_obj_list=[linear_obj, linear_obj_1, mapper_0, mapper_1],
    )


def test__interferometer_sparse_operator__x1_mapper__unchanged_by_func_list_support():
    """
    The single-mapper path is the performance-critical one and must be untouched by the
    func-list / multi-mapper block assembly: for a regularized mapper the `curvature_matrix` is
    still exactly the `curvature_matrix_diag` build, with no mirroring or diagonal stabilisation
    applied on top of it.
    """
    mask, rng, dataset, dataset_sparse = _sparse_parity_setup()

    mapper = _mapper_from(
        mask=mask,
        pixels=9,
        shape=(3, 3),
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    inversion = aa.Inversion(dataset=dataset_sparse, linear_obj_list=[mapper])

    assert inversion.no_regularization_index_list == []
    assert np.array_equal(
        np.array(inversion.curvature_matrix),
        np.array(inversion.curvature_matrix_diag),
    )

    _assert_sparse_matches_mapping(
        dataset=dataset, dataset_sparse=dataset_sparse, linear_obj_list=[mapper]
    )


def test__interferometer_sparse_operator__no_regularization_value_added_to_diag():
    """
    Linear function lists are typically unregularized, so their curvature diagonal receives the
    `no_regularization_add_to_curvature_diag_value` stabiliser. The sparse path must apply this
    exactly as the dense (mapping formalism) path does.
    """
    mask, rng, dataset, dataset_sparse = _sparse_parity_setup()

    mapper = _mapper_from(
        mask=mask,
        pixels=9,
        shape=(3, 3),
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2)),
    )

    inversion = aa.Inversion(
        dataset=dataset_sparse, linear_obj_list=[linear_obj, mapper]
    )

    assert inversion.no_regularization_index_list == [0, 1]

    value = inversion.settings.no_regularization_add_to_curvature_diag_value

    curvature_matrix = np.array(inversion.curvature_matrix)

    # Rebuilding the un-stabilised blocks directly from the operator and adding the value back on
    # reproduces the diagonal entries of the unregularized linear function parameters.
    operator = dataset_sparse.sparse_operator
    mapping_matrix = np.array(linear_obj.mapping_matrix)

    curvature_func = np.array(
        operator.curvature_matrix_func_list_from(
            curvature_weights_0=mapping_matrix,
            curvature_weights_1=mapping_matrix,
            extent_index_for_masked_pixel=mask.extent_index_for_masked_pixel,
        )
    )

    assert curvature_matrix[0, 0] == pytest.approx(curvature_func[0, 0] + value, 1.0e-8)
    assert curvature_matrix[1, 1] == pytest.approx(curvature_func[1, 1] + value, 1.0e-8)
    assert curvature_matrix[0, 1] == pytest.approx(curvature_func[0, 1], 1.0e-8)
