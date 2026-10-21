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


def test__interferometer_sparse_operator__func_list_only__identical_to_mapping():
    """
    A linear function list with no mapper (e.g. an MGE with no pixelization) is routed to the sparse
    path whenever the dataset has a sparse operator, where its curvature matrix is the func-func block
    `Bᵀ W~ B` alone and its data vector is `Bᵀ d~`, reproducing the dense (mapping formalism) inversion.
    """
    mask, rng, dataset, dataset_sparse = _sparse_parity_setup()

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=3,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 3)),
    )

    inversion_sparse = aa.Inversion(
        dataset=dataset_sparse, linear_obj_list=[linear_obj]
    )
    inversion_mapping = aa.Inversion(dataset=dataset, linear_obj_list=[linear_obj])

    assert type(inversion_sparse) is aa.InversionInterferometerSparse
    assert isinstance(inversion_mapping, aa.InversionInterferometerMapping)

    # The reconstruction comes out of a linear solve of this small, poorly conditioned system and is
    # compared (to its looser tolerance) by `_assert_sparse_matches_mapping` below.
    for name in ("curvature_matrix", "data_vector"):
        reference = np.asarray(getattr(inversion_mapping, name))

        np.testing.assert_allclose(
            np.asarray(getattr(inversion_sparse, name)),
            reference,
            rtol=1.0e-10,
            atol=1.0e-10 * np.abs(reference).max(),
            err_msg=name,
        )

    _assert_sparse_matches_mapping(
        dataset=dataset,
        dataset_sparse=dataset_sparse,
        linear_obj_list=[linear_obj],
    )


def test__interferometer_sparse_operator__sparse_dirty_image_override__used_by_data_vector():
    """
    The sparse operator caches the dirty image of the visibilities it was built from. When an inversion
    fits different visibilities (e.g. with the visibilities of ordinary light profiles subtracted), the
    `DatasetInterface` supplies their dirty image via `sparse_dirty_image`, which the data vector must use
    so that the sparse inversion reproduces the dense inversion of the subtracted visibilities.
    """
    mask, rng, dataset, dataset_sparse = _sparse_parity_setup()

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2)),
    )

    mapper = _mapper_from(
        mask=mask,
        pixels=9,
        shape=(3, 3),
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    subtracted = aa.Visibilities(
        visibilities=dataset.data.array
        - rng.normal(size=dataset.data.shape)
        - 1j * rng.normal(size=dataset.data.shape)
    )

    sparse_dirty_image = dataset.transformer.image_from(
        visibilities=aa.Visibilities(
            visibilities=subtracted.array.real * dataset.noise_map.array.real**-2.0
            + 1j * subtracted.array.imag * dataset.noise_map.array.imag**-2.0
        )
    ).array

    dataset_interface_mapping = aa.DatasetInterface(
        data=subtracted,
        noise_map=dataset.noise_map,
        grids=dataset.grids,
        transformer=dataset.transformer,
    )

    for linear_obj_list in ([linear_obj], [linear_obj, mapper]):
        inversion_mapping = aa.Inversion(
            dataset=dataset_interface_mapping, linear_obj_list=linear_obj_list
        )

        inversion_sparse = aa.Inversion(
            dataset=aa.DatasetInterface(
                data=subtracted,
                noise_map=dataset.noise_map,
                grids=dataset.grids,
                transformer=dataset.transformer,
                sparse_operator=dataset_sparse.sparse_operator,
                sparse_dirty_image=sparse_dirty_image,
            ),
            linear_obj_list=linear_obj_list,
        )

        assert isinstance(inversion_sparse, aa.InversionInterferometerSparse)

        reference = np.asarray(inversion_mapping.data_vector)

        np.testing.assert_allclose(
            np.asarray(inversion_sparse.data_vector),
            reference,
            rtol=1.0e-10,
            atol=1.0e-10 * np.abs(reference).max(),
        )

        # The control: without the override the data vector is that of the unsubtracted visibilities.
        inversion_sparse_cached = aa.Inversion(
            dataset=aa.DatasetInterface(
                data=subtracted,
                noise_map=dataset.noise_map,
                grids=dataset.grids,
                transformer=dataset.transformer,
                sparse_operator=dataset_sparse.sparse_operator,
            ),
            linear_obj_list=linear_obj_list,
        )

        assert np.abs(
            np.asarray(inversion_sparse_cached.data_vector) - reference
        ).max() > 1.0e-4 * np.abs(reference).max()


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


def _sparse_np_vs_jax_setup(mask, n_visibilities, seed, pixels, shape):
    """
    Returns the dense dataset, the sparse-operator dataset and a regularized Delaunay mapper
    for the end-to-end NumPy/JAX parity test below.

    The operator is built with `batch_size=4` so the block sweep runs more than one block
    and ends on a partial one, and with the NumPy brute-force preload builder so the two
    inversions are handed a byte-identical operator to start from.
    """
    rng = np.random.default_rng(seed=seed)

    dataset = aa.Interferometer(
        data=aa.Visibilities(
            visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
        ),
        noise_map=aa.VisibilitiesNoiseMap(
            visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
        ),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)).astype(np.float64),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    dataset_sparse = dataset.apply_sparse_operator(
        nufft_precision_operator=dataset.psf_precision_operator_from(method="numpy"),
        batch_size=4,
    )

    mapper = _mapper_from(
        mask=mask,
        pixels=pixels,
        shape=shape,
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    return dataset, dataset_sparse, mapper


def test__interferometer_sparse_operator__numpy_inversion_matches_jax_inversion():
    """
    End-to-end: `InversionInterferometerSparse(xp=np)` — which now assembles every curvature
    block with scipy rather than JAX — must reproduce the `xp=jnp` inversion.

    `curvature_matrix` and `data_vector` come straight off the operator and are pinned
    exactly. `reconstruction` and the log-determinant terms come out of a linear solve, whose
    NumPy and JAX implementations differ by more than the matrices they are handed do; the
    control at the end shows that spread is the solver pair's, by measuring the identical
    difference on the *dense* mapping inversion, which this change does not touch.
    """
    pytest.importorskip("jax")

    import jax.numpy as jnp

    cases = [
        (
            aa.Mask2D(
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
            ),
            5,
            0,
            9,
            (3, 3),
        ),
        (
            aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0),
            64,
            11,
            16,
            (4, 4),
        ),
    ]

    for mask, n_visibilities, seed, pixels, shape in cases:
        dataset, dataset_sparse, mapper = _sparse_np_vs_jax_setup(
            mask=mask,
            n_visibilities=n_visibilities,
            seed=seed,
            pixels=pixels,
            shape=shape,
        )

        inversion_np = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=[mapper], xp=np
        )
        inversion_jax = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=[mapper], xp=jnp
        )

        assert isinstance(inversion_np, aa.InversionInterferometerSparse)
        assert isinstance(inversion_jax, aa.InversionInterferometerSparse)

        for name in ("curvature_matrix", "data_vector"):
            reference = np.asarray(getattr(inversion_jax, name))

            np.testing.assert_allclose(
                np.asarray(getattr(inversion_np, name)),
                reference,
                rtol=1.0e-10,
                atol=1.0e-10 * np.abs(reference).max(),
                err_msg=name,
            )

        reconstruction = np.asarray(inversion_jax.reconstruction)

        np.testing.assert_allclose(
            np.asarray(inversion_np.reconstruction),
            reconstruction,
            rtol=1.0e-7,
            atol=1.0e-7 * np.abs(reconstruction).max(),
        )

        for name in (
            "log_det_curvature_reg_matrix_term",
            "log_det_regularization_matrix_term",
        ):
            reference = float(getattr(inversion_jax, name))

            np.testing.assert_allclose(
                float(getattr(inversion_np, name)),
                reference,
                rtol=1.0e-10,
                atol=1.0e-10 * abs(reference),
                err_msg=name,
            )

        # The control: the dense mapping inversion runs no operator code at all, so whatever
        # `xp=np` and `xp=jnp` disagree by there is the linear solver's, not the sparse
        # operator's. The sparse path must not be worse.
        reconstruction_dense_np = np.asarray(
            aa.Inversion(
                dataset=dataset, linear_obj_list=[mapper], xp=np
            ).reconstruction
        )
        reconstruction_dense_jax = np.asarray(
            aa.Inversion(
                dataset=dataset, linear_obj_list=[mapper], xp=jnp
            ).reconstruction
        )

        difference_dense = np.abs(
            reconstruction_dense_np - reconstruction_dense_jax
        ).max()
        difference_sparse = np.abs(
            np.asarray(inversion_np.reconstruction) - reconstruction
        ).max()

        assert difference_sparse <= max(10.0 * difference_dense, 1.0e-14)

        # A linear function list with no mapper (e.g. an MGE with no pixelization) takes the same
        # sparse path, with only the func-func curvature block and `Bᵀ d~` data vector.
        linear_obj = aa.m.MockLinearObjFuncList(
            parameters=3,
            mapping_matrix=np.random.default_rng(seed=seed).normal(
                size=(mask.pixels_in_mask, 3)
            ),
        )

        inversion_np = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=[linear_obj], xp=np
        )
        inversion_jax = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=[linear_obj], xp=jnp
        )

        assert type(inversion_np) is aa.InversionInterferometerSparse
        assert type(inversion_jax) is aa.InversionInterferometerSparse

        for name in ("curvature_matrix", "data_vector"):
            reference = np.asarray(getattr(inversion_jax, name))

            np.testing.assert_allclose(
                np.asarray(getattr(inversion_np, name)),
                reference,
                rtol=1.0e-10,
                atol=1.0e-10 * np.abs(reference).max(),
                err_msg=name,
            )

        # The unregularized func-list system is poorly conditioned, so the reconstruction is held to
        # the same control as above: no worse than the dense inversion's NumPy / JAX spread.
        difference_dense = np.abs(
            np.asarray(
                aa.Inversion(
                    dataset=dataset, linear_obj_list=[linear_obj], xp=np
                ).reconstruction
            )
            - np.asarray(
                aa.Inversion(
                    dataset=dataset, linear_obj_list=[linear_obj], xp=jnp
                ).reconstruction
            )
        ).max()
        difference_sparse = np.abs(
            np.asarray(inversion_np.reconstruction)
            - np.asarray(inversion_jax.reconstruction)
        ).max()

        assert difference_sparse <= max(10.0 * difference_dense, 1.0e-14)


def _count_evaluations(monkeypatch, cls, name, counts):
    """
    Wrap the body of the `cls.name` property with a counter while keeping its descriptor
    type, so a `cached_property` stays cached (and a plain `property` stays uncached) and
    the count is the number of times the body actually runs.
    """
    import functools

    descriptor = cls.__dict__[name]
    func = descriptor.fget if isinstance(descriptor, property) else descriptor.func

    @functools.wraps(func)
    def counted(self):
        counts[name] += 1
        return func(self)

    monkeypatch.setattr(cls, name, type(descriptor)(counted))


def _log_evidence_terms_from(inversion):
    """
    The inversion terms of `FitInterferometer.log_evidence`, which is what a figure of merit
    evaluates: the fast chi-squared (reads F, D and the reconstruction), the regularization
    term and both log-determinants (the cached `curvature_reg_matrix` reads F).
    """
    return (
        float(inversion.fast_chi_squared)
        + float(inversion.regularization_term)
        + float(inversion.log_det_curvature_reg_matrix_term)
        - float(inversion.log_det_regularization_matrix_term)
    )


def test__interferometer_sparse_operator__curvature_matrix_and_data_vector_evaluated_once_per_likelihood(
    monkeypatch,
):
    """
    `fast_chi_squared`, `curvature_reg_matrix` and `reconstruction` all read `curvature_matrix`
    (F) and `data_vector` (D). Each must be built once per inversion, not once per reader:
    on the NumPy path nothing merges the repeated builds, which were ~45 % of an alma call
    (autolens_profiling #326).
    """
    mask, _, _, dataset_sparse = _sparse_parity_setup()

    mapper = _mapper_from(
        mask=mask, pixels=9, shape=(3, 3), regularization=aa.reg.Constant(coefficient=1.0)
    )

    # Built directly (not via `aa.Inversion`) so the NumPy FFT class runs even when numba is
    # installed and the factory would route to `InversionInterferometerSparseNumba`.
    def inversion_from():
        return aa.InversionInterferometerSparse(
            dataset=dataset_sparse, linear_obj_list=[mapper], xp=np
        )

    reference = _log_evidence_terms_from(inversion_from())

    counts = {
        "curvature_matrix_diag": 0,
        "data_vector": 0,
        "curvature_matrix_diag_from": 0,
    }

    _count_evaluations(
        monkeypatch, aa.InversionInterferometerSparse, "curvature_matrix_diag", counts
    )
    _count_evaluations(
        monkeypatch, aa.InversionInterferometerSparse, "data_vector", counts
    )

    from autoarray.inversion.inversion.interferometer.inversion_interferometer_util import (
        InterferometerSparseOperator,
    )

    curvature_matrix_diag_from = InterferometerSparseOperator.curvature_matrix_diag_from

    def counted_curvature_matrix_diag_from(self, *args, **kwargs):
        counts["curvature_matrix_diag_from"] += 1
        return curvature_matrix_diag_from(self, *args, **kwargs)

    monkeypatch.setattr(
        InterferometerSparseOperator,
        "curvature_matrix_diag_from",
        counted_curvature_matrix_diag_from,
    )

    inversion = inversion_from()

    log_evidence_terms = _log_evidence_terms_from(inversion)

    assert counts == {
        "curvature_matrix_diag": 1,
        "data_vector": 1,
        "curvature_matrix_diag_from": 1,
    }
    assert log_evidence_terms == reference


def test__interferometer_mapping__curvature_matrix_and_data_vector_evaluated_once_per_likelihood(
    monkeypatch,
):
    """
    The dense mapping inversion's `curvature_matrix` and `data_vector` are cached per inversion
    too, as the imaging mapping inversion's are.
    """
    mask, _, dataset, _ = _sparse_parity_setup()

    mapper = _mapper_from(
        mask=mask, pixels=9, shape=(3, 3), regularization=aa.reg.Constant(coefficient=1.0)
    )

    reference = _log_evidence_terms_from(
        aa.Inversion(dataset=dataset, linear_obj_list=[mapper])
    )

    counts = {"curvature_matrix": 0, "data_vector": 0}

    for name in counts:
        _count_evaluations(
            monkeypatch, aa.InversionInterferometerMapping, name, counts
        )

    inversion = aa.Inversion(dataset=dataset, linear_obj_list=[mapper])

    assert isinstance(inversion, aa.InversionInterferometerMapping)

    log_evidence_terms = _log_evidence_terms_from(inversion)

    assert counts == {"curvature_matrix": 1, "data_vector": 1}
    assert log_evidence_terms == reference


def test__operated_mapping_matrix_list__transform_performed_once_per_linear_obj(
    interferometer_7_no_fft, rectangular_mapper_7x7_3x3, monkeypatch
):
    # The dense (mapping) route reaches `operated_mapping_matrix_list` from the
    # curvature matrix / data vector (via `operated_mapping_matrix`) and again from
    # `mapped_reconstructed_data_dict`. It must be cached so the Fourier transform of
    # each linear object's mapping matrix happens once per inversion.
    calls = []

    transformer_cls = type(interferometer_7_no_fft.transformer)
    transform_mapping_matrix = transformer_cls.transform_mapping_matrix

    def counted(self, *args, **kwargs):
        calls.append(1)
        return transform_mapping_matrix(self, *args, **kwargs)

    monkeypatch.setattr(transformer_cls, "transform_mapping_matrix", counted)

    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.Settings(),
    )

    assert isinstance(inversion, aa.InversionInterferometerMapping)

    inversion.log_det_curvature_reg_matrix_term
    inversion.reconstruction
    inversion.mapped_reconstructed_operated_data
    inversion.mapped_reconstructed_data

    assert len(calls) == 1


def _sparse_interface_setup():
    mask = aa.Mask2D.circular(shape_native=(10, 10), pixel_scales=1.0, radius=3.0)

    rng = np.random.default_rng(seed=21)
    n_visibilities = 30
    data = rng.normal(size=n_visibilities) + 1j * rng.normal(size=n_visibilities)
    sigma = rng.uniform(0.5, 2.0, size=n_visibilities)

    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=data),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=sigma + 1j * sigma),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    dataset_sparse = dataset.apply_sparse_operator(use_jax=False)

    mapper = _mapper_from(
        mask=mask, pixels=16, shape=(4, 4), regularization=aa.reg.Constant(1.0)
    )

    return dataset_sparse, mapper


def _interface_from(dataset_sparse, data):
    return aa.DatasetInterface(
        data=data,
        noise_map=dataset_sparse.noise_map,
        grids=dataset_sparse.grids,
        transformer=dataset_sparse.transformer,
        sparse_operator=dataset_sparse.sparse_operator,
    )


def test__fast_chi_squared__data_none_uses_the_sparse_operator_data_term():
    dataset_sparse, mapper = _sparse_interface_setup()

    inversion_array = aa.Inversion(
        dataset=_interface_from(dataset_sparse, data=dataset_sparse.data),
        linear_obj_list=[mapper],
    )
    inversion_scalar = aa.Inversion(
        dataset=_interface_from(dataset_sparse, data=None),
        linear_obj_list=[mapper],
    )

    assert isinstance(inversion_scalar, aa.InversionInterferometerSparse)

    # `apply_sparse_operator` computes `data_term` with the expression term 3 reduces with,
    # so the two paths agree bit-for-bit.
    assert inversion_scalar.fast_chi_squared == inversion_array.fast_chi_squared

    # And both equal the chi-squared of the residual visibilities.
    residual_map = aa.util.fit.residual_map_from(
        data=dataset_sparse.data,
        model_data=inversion_array.mapped_reconstructed_operated_data,
    )
    chi_squared = aa.util.fit.chi_squared_complex_from(
        chi_squared_map=aa.util.fit.chi_squared_map_complex_from(
            residual_map=residual_map, noise_map=dataset_sparse.noise_map
        )
    )

    assert inversion_scalar.fast_chi_squared == pytest.approx(chi_squared, 1.0e-8)


def test__fast_chi_squared__data_given__the_cached_data_term_is_ignored():
    dataset_sparse, mapper = _sparse_interface_setup()

    # Stand-in for profile-subtracted data: the interface's data differs from the data the
    # operator's `data_term` was built from, so term 3 must be reduced from the interface.
    subtracted = aa.Visibilities(visibilities=0.5 * dataset_sparse.data.array)

    inversion = aa.Inversion(
        dataset=_interface_from(dataset_sparse, data=subtracted),
        linear_obj_list=[mapper],
    )
    inversion_scalar = aa.Inversion(
        dataset=_interface_from(dataset_sparse, data=None),
        linear_obj_list=[mapper],
    )

    noise_map = dataset_sparse.noise_map.array

    term_3 = np.sum(subtracted.array.real**2.0 / noise_map.real**2.0) + np.sum(
        subtracted.array.imag**2.0 / noise_map.imag**2.0
    )

    difference = inversion.fast_chi_squared - inversion_scalar.fast_chi_squared

    assert difference == pytest.approx(
        term_3 - dataset_sparse.sparse_operator.data_term, rel=1.0e-10
    )


def test__fast_chi_squared__data_none_without_a_data_term__raises():
    dataset_sparse, mapper = _sparse_interface_setup()

    operator = dataset_sparse.sparse_operator

    operator_without_scalars = (
        aa.InterferometerSparseOperator.from_nufft_precision_operator(
            nufft_precision_operator=operator.nufft_precision_operator,
            dirty_image=operator.dirty_image,
        )
    )

    interface = aa.DatasetInterface(
        data=None,
        noise_map=dataset_sparse.noise_map,
        grids=dataset_sparse.grids,
        transformer=dataset_sparse.transformer,
        sparse_operator=operator_without_scalars,
    )

    inversion = aa.Inversion(dataset=interface, linear_obj_list=[mapper])

    with pytest.raises(aa.exc.InversionException):
        inversion.fast_chi_squared


def test__data_subtracted_dict__data_none__raises_pointing_to_inversion_with_data():
    dataset_sparse, mapper = _sparse_interface_setup()

    inversion = aa.Inversion(
        dataset=_interface_from(dataset_sparse, data=None),
        linear_obj_list=[mapper],
    )

    with pytest.raises(aa.exc.InversionException, match="inversion_with_data"):
        inversion.data_subtracted_dict


def test__fast_chi_squared__data_none__jax_matches_numpy():
    pytest.importorskip("jax")

    import jax.numpy as jnp

    dataset_sparse, mapper = _sparse_interface_setup()

    inversion_np = aa.Inversion(
        dataset=_interface_from(dataset_sparse, data=None),
        linear_obj_list=[mapper],
        xp=np,
    )
    inversion_jax = aa.Inversion(
        dataset=_interface_from(dataset_sparse, data=None),
        linear_obj_list=[mapper],
        xp=jnp,
    )

    assert float(inversion_jax.fast_chi_squared) == pytest.approx(
        float(inversion_np.fast_chi_squared), rel=1.0e-7
    )


def _profile_subtracted_from(dataset_sparse, seed=5):
    """
    A random real-space image `i_p` on the dataset's masked grid, the visibilities `d - F i_p` with its
    Fourier transform subtracted, and the sparse dirty image / data term of those visibilities computed by
    `sparse_profile_terms_from` without forming `F i_p`.
    """
    mask = dataset_sparse.real_space_mask

    rng = np.random.default_rng(seed=seed)

    image = aa.Array2D(values=rng.normal(size=mask.pixels_in_mask), mask=mask)

    subtracted = aa.Visibilities(
        visibilities=dataset_sparse.data.array
        - dataset_sparse.transformer.visibilities_from(image=image).array
    )

    operated_image, sparse_dirty_image, data_term = (
        aa.util.inversion_interferometer.sparse_profile_terms_from(
            sparse_operator=dataset_sparse.sparse_operator,
            image=image,
            extent_index_for_masked_pixel=mask.extent_index_for_masked_pixel,
        )
    )

    return image, subtracted, operated_image, sparse_dirty_image, data_term


def test__sparse_profile_terms_from__matches_the_dense_subtracted_visibilities():
    """
    The identity `sum(|d - F i_p|^2 / sigma^2) = data_term - 2 i_p^T d~ + i_p^T W~ i_p` and the dirty image
    `d~ - W~ i_p` of the subtracted visibilities, against the same quantities reduced from `d - F i_p`.
    """
    dataset_sparse, _ = _sparse_interface_setup()

    image, subtracted, operated_image, sparse_dirty_image, data_term = (
        _profile_subtracted_from(dataset_sparse)
    )

    noise_map = dataset_sparse.noise_map.array

    data_term_dense = np.sum(
        subtracted.array.real**2.0 / noise_map.real**2.0
    ) + np.sum(subtracted.array.imag**2.0 / noise_map.imag**2.0)

    assert data_term == pytest.approx(data_term_dense, rel=1.0e-10)

    sparse_dirty_image_dense = dataset_sparse.transformer.image_from(
        visibilities=aa.Visibilities(
            visibilities=subtracted.array.real * noise_map.real**-2.0
            + 1j * subtracted.array.imag * noise_map.imag**-2.0
        )
    ).array

    np.testing.assert_allclose(
        sparse_dirty_image, sparse_dirty_image_dense, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        operated_image,
        np.asarray(dataset_sparse.sparse_operator.dirty_image) - sparse_dirty_image,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test__sparse_profile_terms_from__no_operator_data_term__returns_none():
    dataset_sparse, _ = _sparse_interface_setup()

    operator = dataset_sparse.sparse_operator

    operator_without_scalars = (
        aa.InterferometerSparseOperator.from_nufft_precision_operator(
            nufft_precision_operator=operator.nufft_precision_operator,
            dirty_image=operator.dirty_image,
        )
    )

    mask = dataset_sparse.real_space_mask

    _, sparse_dirty_image, data_term = (
        aa.util.inversion_interferometer.sparse_profile_terms_from(
            sparse_operator=operator_without_scalars,
            image=np.ones(mask.pixels_in_mask),
            extent_index_for_masked_pixel=mask.extent_index_for_masked_pixel,
        )
    )

    assert data_term is None
    assert sparse_dirty_image.shape == (mask.pixels_in_mask,)


def test__fast_chi_squared__data_none__interface_data_term_overrides_the_operator():
    """
    With `data=None` the interface's own `data_term` (that of profile-subtracted visibilities) takes precedence
    over the operator's scalar (that of the raw visibilities), so the sparse inversion of the subtracted
    visibilities passed as scalars equals the one passed the visibilities themselves; without it the
    operator's scalar is used.
    """
    dataset_sparse, mapper = _sparse_interface_setup()

    _, subtracted, _, sparse_dirty_image, data_term = _profile_subtracted_from(
        dataset_sparse
    )

    def interface_from(data, data_term):
        return aa.DatasetInterface(
            data=data,
            noise_map=dataset_sparse.noise_map,
            grids=dataset_sparse.grids,
            transformer=dataset_sparse.transformer,
            sparse_operator=dataset_sparse.sparse_operator,
            sparse_dirty_image=sparse_dirty_image,
            data_term=data_term,
        )

    inversion_array = aa.Inversion(
        dataset=interface_from(data=subtracted, data_term=None),
        linear_obj_list=[mapper],
    )
    inversion_override = aa.Inversion(
        dataset=interface_from(data=None, data_term=data_term),
        linear_obj_list=[mapper],
    )
    inversion_operator = aa.Inversion(
        dataset=interface_from(data=None, data_term=None),
        linear_obj_list=[mapper],
    )

    assert isinstance(inversion_override, aa.InversionInterferometerSparse)

    assert inversion_override.fast_chi_squared == pytest.approx(
        inversion_array.fast_chi_squared, rel=1.0e-10
    )

    # Absent the override, term 3 is the operator's (unsubtracted) data term.
    difference = (
        inversion_operator.fast_chi_squared - inversion_override.fast_chi_squared
    )

    assert difference == pytest.approx(
        dataset_sparse.sparse_operator.data_term - data_term, rel=1.0e-10
    )
    assert abs(difference) > 1.0e-3

    # When the visibilities are passed, they are reduced over and the override is not read.
    inversion_array_with_override = aa.Inversion(
        dataset=interface_from(data=subtracted, data_term=0.0),
        linear_obj_list=[mapper],
    )

    assert (
        inversion_array_with_override.fast_chi_squared
        == inversion_array.fast_chi_squared
    )


def test__fast_chi_squared__data_none__interface_data_term__jax_jit_matches_numpy():
    jax = pytest.importorskip("jax")

    import jax.numpy as jnp

    dataset_sparse, mapper = _sparse_interface_setup()

    mask = dataset_sparse.real_space_mask

    image = np.random.default_rng(seed=5).normal(size=mask.pixels_in_mask)

    def fast_chi_squared_from(image, xp):
        _, sparse_dirty_image, data_term = (
            aa.util.inversion_interferometer.sparse_profile_terms_from(
                sparse_operator=dataset_sparse.sparse_operator,
                image=image,
                extent_index_for_masked_pixel=mask.extent_index_for_masked_pixel,
                xp=xp,
            )
        )

        inversion = aa.Inversion(
            dataset=aa.DatasetInterface(
                data=None,
                noise_map=dataset_sparse.noise_map,
                grids=dataset_sparse.grids,
                transformer=dataset_sparse.transformer,
                sparse_operator=dataset_sparse.sparse_operator,
                sparse_dirty_image=sparse_dirty_image,
                data_term=data_term,
            ),
            linear_obj_list=[mapper],
            xp=xp,
        )

        return inversion.fast_chi_squared

    fast_chi_squared_numpy = fast_chi_squared_from(image, xp=np)

    fast_chi_squared_jax = jax.jit(lambda i: fast_chi_squared_from(i, xp=jnp))(
        jnp.asarray(image)
    )

    assert float(fast_chi_squared_jax) == pytest.approx(
        float(fast_chi_squared_numpy), rel=1.0e-8
    )


def _array_free_setup(n_visibilities=60, seed=3):
    """
    A NUFFT dataset with random data and non-uniform (equal real/imaginary) noise, its
    in-memory `apply_sparse_operator()` counterpart, the array-free dataset streamed from
    three uneven chunks of the same visibilities, and a rectangular-mesh mapper.
    """
    from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
        overlay_grid_from,
    )

    mask = aa.Mask2D.circular(shape_native=(10, 10), pixel_scales=0.5, radius=2.0)

    rng = np.random.default_rng(seed=seed)
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)) * 1.0e5
    data = rng.normal(size=n_visibilities) + 1j * rng.normal(size=n_visibilities)
    sigma = rng.uniform(0.5, 2.0, size=n_visibilities)
    noise_map = sigma + 1j * sigma

    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=data),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=noise_map),
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerNUFFT,
    )

    chunks = (
        (uv_wavelengths[k0:k1], data[k0:k1], noise_map[k0:k1])
        for k0, k1 in ((0, 1), (1, 25), (25, n_visibilities))
    )

    dataset_stream = aa.Interferometer.from_stream(chunks, mask)

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)
    mesh = aa.mesh.RectangularUniform(shape=(4, 4))
    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(
            overlay_grid_from(shape_native=(4, 4), grid=grid)
        ),
        adapt_data=None,
    )
    mapper = aa.Mapper(
        interpolator=interpolator, regularization=aa.reg.Constant(coefficient=1.0)
    )

    return mask, dataset.apply_sparse_operator(), dataset_stream, mapper


def _assert_array_free_matches_in_memory(inversion_stream, inversion_memory, rel):
    assert isinstance(inversion_stream, aa.InversionInterferometerSparse)
    assert isinstance(inversion_memory, aa.InversionInterferometerSparse)

    for name in (
        "fast_chi_squared",
        "regularization_term",
        "log_det_curvature_reg_matrix_term",
        "log_det_regularization_matrix_term",
    ):
        assert float(getattr(inversion_stream, name)) == pytest.approx(
            float(getattr(inversion_memory, name)), rel=rel
        ), name

    reconstruction = np.asarray(inversion_memory.reconstruction)

    np.testing.assert_allclose(
        np.asarray(inversion_stream.reconstruction),
        reconstruction,
        rtol=rel,
        atol=rel * np.abs(reconstruction).max(),
    )

    log_evidence_stream = aa.m.MockFitInterferometer(
        dataset=inversion_stream.dataset, inversion=inversion_stream
    ).log_evidence
    log_evidence_memory = aa.m.MockFitInterferometer(
        dataset=inversion_memory.dataset, inversion=inversion_memory
    ).log_evidence

    assert float(log_evidence_stream) == pytest.approx(
        float(log_evidence_memory), rel=rel
    )


def test__array_free_dataset__sparse_inversion_matches_in_memory__numpy():
    pytest.importorskip("nufftax")

    _, dataset_memory, dataset_stream, mapper = _array_free_setup()

    _assert_array_free_matches_in_memory(
        aa.Inversion(dataset=dataset_stream, linear_obj_list=[mapper]),
        aa.Inversion(dataset=dataset_memory, linear_obj_list=[mapper]),
        rel=1.0e-8,
    )


def test__array_free_dataset__sparse_inversion_matches_in_memory__jax():
    pytest.importorskip("nufftax")
    pytest.importorskip("jax")

    import jax.numpy as jnp

    _, dataset_memory, dataset_stream, mapper = _array_free_setup()

    _assert_array_free_matches_in_memory(
        aa.Inversion(dataset=dataset_stream, linear_obj_list=[mapper], xp=jnp),
        aa.Inversion(dataset=dataset_memory, linear_obj_list=[mapper], xp=jnp),
        rel=1.0e-8,
    )


def test__array_free_dataset__mapped_reconstructed_operated_data_raises_typed():
    pytest.importorskip("nufftax")

    mask, dataset_memory, dataset_stream, mapper = _array_free_setup()

    inversion_stream = aa.Inversion(dataset=dataset_stream, linear_obj_list=[mapper])

    # The real-space reconstruction is still available (it needs only the mask).
    image = inversion_stream.mapped_reconstructed_data_dict[mapper]

    assert image.mask is mask

    with pytest.raises(aa.exc.InversionException, match="array-free"):
        inversion_stream.mapped_reconstructed_operated_data_dict

    # The in-memory dataset still maps to visibilities.
    inversion_memory = aa.Inversion(dataset=dataset_memory, linear_obj_list=[mapper])

    assert (
        inversion_memory.mapped_reconstructed_operated_data_dict[mapper].shape[0] == 60
    )


def test__inversion_interferometer_mask__is_the_real_space_mask_for_every_dataset_type():
    pytest.importorskip("nufftax")

    mask, dataset_memory, dataset_stream, mapper = _array_free_setup()

    dataset_interface = aa.DatasetInterface(
        data=None,
        noise_map=dataset_memory.noise_map,
        grids=dataset_memory.grids,
        transformer=dataset_memory.transformer,
        sparse_operator=dataset_memory.sparse_operator,
    )

    # A `DatasetInterface` without a transformer falls back to `grids.lp.mask`.
    dataset_interface_no_transformer = aa.DatasetInterface(
        data=None,
        noise_map=None,
        grids=dataset_memory.grids,
        sparse_operator=dataset_memory.sparse_operator,
    )

    for dataset in (
        dataset_memory,
        dataset_stream,
        dataset_interface,
        dataset_interface_no_transformer,
    ):
        inversion = aa.Inversion(dataset=dataset, linear_obj_list=[mapper])

        assert isinstance(inversion, aa.InversionInterferometerSparse)
        assert (inversion.mask == mask).all()
        assert inversion.mask.pixel_scales == mask.pixel_scales


def _over_sampled_mapper_from(mask, pixels, shape, over_sample_size):
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=over_sample_size)

    mesh = aa.mesh.Delaunay(pixels=pixels)
    image_mesh_grid = aa.image_mesh.Overlay(shape=shape).image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )

    return aa.Mapper(
        interpolator=interpolator, regularization=aa.reg.Constant(coefficient=1.0)
    )


def _dense_data_vector_from(inversion, dirty_image):
    return np.asarray(inversion.mapping_matrix).T @ np.asarray(dirty_image)


def test__interferometer_sparse_operator__data_vector_from_triplets_matches_dense_mapping_matrix():
    """
    `InversionInterferometerSparse.data_vector` computes each mapper block from the mapper's sparse
    triplets (a segment-sum over source columns) instead of the dense `mapping_matrix.T @ d~`. It must
    reproduce the dense product exactly for an over-sampled mapper (where the triplets carry the
    `sub_fraction` normalisation), for a mapper mixed with a linear function list (in either order), and
    for a `DatasetInterface` whose `sparse_dirty_image` differs from the operator's cached dirty image.
    """
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0)

    rng = np.random.default_rng(seed=3)
    n_visibilities = 32

    dataset = aa.Interferometer(
        data=aa.Visibilities(
            visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
        ),
        noise_map=aa.VisibilitiesNoiseMap(
            visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
        ),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)).astype(np.float64),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )
    dataset_sparse = dataset.apply_sparse_operator(use_jax=False)
    dirty_image = np.asarray(dataset_sparse.sparse_operator.dirty_image)

    mapper = _over_sampled_mapper_from(
        mask=mask, pixels=16, shape=(4, 4), over_sample_size=2
    )
    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2)),
    )

    for linear_obj_list in ([mapper], [mapper, linear_obj], [linear_obj, mapper]):
        inversion = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=linear_obj_list
        )

        assert type(inversion) is aa.InversionInterferometerSparse

        reference = _dense_data_vector_from(inversion, dirty_image)

        np.testing.assert_allclose(
            np.asarray(inversion.data_vector),
            reference,
            rtol=1.0e-12,
            atol=1.0e-12 * np.abs(reference).max(),
        )

    # A `DatasetInterface` with its own `sparse_dirty_image` must use it in place of the operator's.
    sparse_dirty_image = rng.normal(size=mask.pixels_in_mask)

    inversion = aa.Inversion(
        dataset=aa.DatasetInterface(
            data=dataset.data,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            transformer=dataset.transformer,
            sparse_operator=dataset_sparse.sparse_operator,
            sparse_dirty_image=sparse_dirty_image,
        ),
        linear_obj_list=[mapper, linear_obj],
    )

    reference = _dense_data_vector_from(inversion, sparse_dirty_image)

    np.testing.assert_allclose(
        np.asarray(inversion.data_vector),
        reference,
        rtol=1.0e-12,
        atol=1.0e-12 * np.abs(reference).max(),
    )

    assert (
        np.abs(_dense_data_vector_from(inversion, dirty_image) - reference).max()
        > 1.0e-4 * np.abs(reference).max()
    )


def test__interferometer_sparse_operator__data_vector_from_triplets_matches_dense_mapping_matrix__jax():
    """
    The JAX backend of the triplet data vector (a scatter-add over source columns) must reproduce the
    dense `mapping_matrix.T @ d~` for an over-sampled mapper and a mapper mixed with a linear function list.
    """
    pytest.importorskip("jax")

    import jax.numpy as jnp

    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0)

    rng = np.random.default_rng(seed=5)
    n_visibilities = 32

    dataset = aa.Interferometer(
        data=aa.Visibilities(
            visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
        ),
        noise_map=aa.VisibilitiesNoiseMap(
            visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
        ),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)).astype(np.float64),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )
    dataset_sparse = dataset.apply_sparse_operator(use_jax=False)
    dirty_image = np.asarray(dataset_sparse.sparse_operator.dirty_image)

    mapper = _over_sampled_mapper_from(
        mask=mask, pixels=16, shape=(4, 4), over_sample_size=2
    )
    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2)),
    )

    for linear_obj_list in ([mapper], [linear_obj, mapper]):
        inversion_np = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=linear_obj_list, xp=np
        )
        inversion_jax = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=linear_obj_list, xp=jnp
        )

        assert isinstance(inversion_jax, aa.InversionInterferometerSparse)

        reference = _dense_data_vector_from(inversion_np, dirty_image)

        np.testing.assert_allclose(
            np.asarray(inversion_jax.data_vector),
            reference,
            rtol=1.0e-12,
            atol=1.0e-12 * np.abs(reference).max(),
        )


def _rectangular_mapper_from(mask, shape, over_sample_size):
    from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
        overlay_grid_from,
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=over_sample_size)

    interpolator = aa.mesh.RectangularUniform(shape=shape).interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(
            overlay_grid_from(shape_native=shape, grid=grid)
        ),
        adapt_data=None,
    )

    return aa.Mapper(
        interpolator=interpolator, regularization=aa.reg.Constant(coefficient=1.0)
    )


def _mapped_reconstructed_parity_setup(seed):
    """
    A 12x12 circular-mask TransformerDFT dataset with its sparse operator, and the linear object lists
    the sparse `mapped_reconstructed_data_dict` is checked on: a Rectangular mesh, a Delaunay mesh,
    over-sampled (sub_size 2) Rectangular and Delaunay meshes, two mappers, and a mapper plus a linear
    function list (in either order).
    """
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0)

    rng = np.random.default_rng(seed=seed)
    n_visibilities = 32

    dataset = aa.Interferometer(
        data=aa.Visibilities(
            visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
        ),
        noise_map=aa.VisibilitiesNoiseMap(
            visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
        ),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)).astype(np.float64),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )
    dataset_sparse = dataset.apply_sparse_operator(use_jax=False)

    rectangular = _rectangular_mapper_from(mask=mask, shape=(4, 4), over_sample_size=1)
    delaunay = _over_sampled_mapper_from(
        mask=mask, pixels=16, shape=(4, 4), over_sample_size=1
    )
    rectangular_sub_2 = _rectangular_mapper_from(
        mask=mask, shape=(5, 5), over_sample_size=2
    )
    delaunay_sub_2 = _over_sampled_mapper_from(
        mask=mask, pixels=9, shape=(3, 3), over_sample_size=2
    )
    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2,
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2)),
    )

    linear_obj_lists = [
        [rectangular],
        [delaunay],
        [rectangular_sub_2],
        [delaunay_sub_2],
        [delaunay_sub_2, rectangular],
        [rectangular, linear_obj],
        [linear_obj, delaunay_sub_2],
    ]

    return mask, rng, dataset_sparse, linear_obj_lists


def _assert_mapped_reconstructed_data_dict_matches_dense(
    image_dict, linear_obj_list, reconstruction, inversion, rtol
):
    """
    Each entry of `image_dict` must equal the dense `mapping_matrix @ reconstruction` of its linear object,
    where `reconstruction` is the (NumPy) reconstruction split per linear object.
    """
    param_range_list = inversion.param_range_list_from(cls=aa.LinearObj)

    for linear_obj, (start, end) in zip(linear_obj_list, param_range_list):
        reference = np.asarray(linear_obj.mapping_matrix) @ np.asarray(
            reconstruction[start:end]
        )

        np.testing.assert_allclose(
            np.asarray(image_dict[linear_obj]),
            reference,
            rtol=rtol,
            atol=rtol * np.abs(reference).max(),
        )


def test__interferometer_sparse_operator__mapped_reconstructed_data_dict_matches_dense_mapping_matrix():
    """
    `InversionInterferometerSparse.mapped_reconstructed_data_dict` reconstructs each mapper's image by a
    scatter-add over the mapper's sparse triplets instead of the dense `mapping_matrix @ reconstruction`.
    It must reproduce the dense product for Rectangular and Delaunay meshes, over-sampled mappers, two
    mappers and a mapper mixed with a linear function list, and its values must be wrapped as an
    `Array2D` on the inversion's real-space mask (so the model visibilities are unchanged).
    """
    mask, rng, dataset_sparse, linear_obj_lists = _mapped_reconstructed_parity_setup(
        seed=7
    )

    for linear_obj_list in linear_obj_lists:
        inversion = aa.Inversion(
            dataset=dataset_sparse, linear_obj_list=linear_obj_list
        )

        assert isinstance(inversion, aa.InversionInterferometerSparse)

        image_dict = inversion.mapped_reconstructed_data_dict

        for linear_obj in linear_obj_list:
            assert isinstance(image_dict[linear_obj], aa.Array2D)
            assert (image_dict[linear_obj].mask == mask).all()

        _assert_mapped_reconstructed_data_dict_matches_dense(
            image_dict=image_dict,
            linear_obj_list=linear_obj_list,
            reconstruction=np.asarray(inversion.reconstruction),
            inversion=inversion,
            rtol=1.0e-10,
        )

        # The model visibilities consume the dict and must equal the transform of the dense image.
        visibilities_dict = inversion.mapped_reconstructed_operated_data_dict

        for linear_obj in linear_obj_list:
            reference = dataset_sparse.transformer.visibilities_from(
                image=aa.Array2D(
                    values=np.asarray(linear_obj.mapping_matrix)
                    @ np.asarray(inversion.reconstruction_dict[linear_obj]),
                    mask=mask,
                )
            )

            np.testing.assert_allclose(
                np.asarray(visibilities_dict[linear_obj]),
                np.asarray(reference),
                rtol=1.0e-10,
                atol=1.0e-10 * np.abs(np.asarray(reference)).max(),
            )


def test__interferometer_sparse_operator__mapped_reconstructed_data_dict_matches_dense_mapping_matrix__jax_jit():
    """
    The JAX backend of the sparse mapped reconstructed image (a `.at[rows].add` scatter over image pixels)
    must reproduce the dense `mapping_matrix @ reconstruction` under `jax.jit`, with the dirty image traced
    so the reconstruction is computed inside the jitted function.
    """
    jax = pytest.importorskip("jax")

    import jax.numpy as jnp

    mask, rng, dataset_sparse, linear_obj_lists = _mapped_reconstructed_parity_setup(
        seed=9
    )

    dirty_image = np.asarray(dataset_sparse.sparse_operator.dirty_image)

    def inversion_from(dirty_image, linear_obj_list, xp):
        return aa.Inversion(
            dataset=aa.DatasetInterface(
                data=dataset_sparse.data,
                noise_map=dataset_sparse.noise_map,
                grids=dataset_sparse.grids,
                transformer=dataset_sparse.transformer,
                sparse_operator=dataset_sparse.sparse_operator,
                sparse_dirty_image=dirty_image,
            ),
            linear_obj_list=linear_obj_list,
            xp=xp,
        )

    for linear_obj_list in linear_obj_lists:
        inversion_np = inversion_from(dirty_image, linear_obj_list, xp=np)

        assert isinstance(inversion_np, aa.InversionInterferometerSparse)

        def images_and_reconstruction_from(dirty_image):
            inversion = inversion_from(dirty_image, linear_obj_list, xp=jnp)

            image_dict = inversion.mapped_reconstructed_data_dict

            return (
                [image_dict[linear_obj].array for linear_obj in linear_obj_list],
                inversion.reconstruction,
            )

        image_list, reconstruction = jax.jit(images_and_reconstruction_from)(
            jnp.asarray(dirty_image)
        )

        # A sanity check only: the NumPy and JAX linear solves differ by more than the matrices they are
        # handed (see `test__interferometer_sparse_operator__numpy_inversion_matches_jax_inversion`), so the
        # image is compared below against the dense product of the jitted reconstruction itself.
        reference = np.asarray(inversion_np.reconstruction)

        np.testing.assert_allclose(
            np.asarray(reconstruction),
            reference,
            rtol=0.0,
            atol=1.0e-3 * np.abs(reference).max(),
        )

        _assert_mapped_reconstructed_data_dict_matches_dense(
            image_dict=dict(zip(linear_obj_list, image_list)),
            linear_obj_list=linear_obj_list,
            reconstruction=np.asarray(reconstruction),
            inversion=inversion_np,
            rtol=1.0e-12,
        )


# ----------------------------------------------------------------------------------------
# Primary beam
# ----------------------------------------------------------------------------------------


_PRIMARY_BEAM_SETTINGS = aa.Settings(no_regularization_add_to_curvature_diag_value=0.0)


def _primary_beam_parity_setup(seed=0, primary_beam="random", transformer_class=None):
    """
    The 7x7 / Delaunay setup of `_sparse_parity_setup` with a primary beam on the dataset, returning
    the mask, a random generator, the dense and sparse-operator datasets and the primary beam.

    `primary_beam="random"` draws a non-trivial beam with values in (0.2, 1]; anything else is passed
    to the dataset as is (`None`, ones, ...). The sparse operator is built with the NumPy brute-force
    preload builder, so the brute-force comparison below is limited only by float64 round-off.
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
    sigma = rng.uniform(0.5, 2.0, size=n_visibilities)
    noise_map = aa.VisibilitiesNoiseMap(visibilities=np.stack([sigma, sigma], axis=1))
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    if isinstance(primary_beam, str):
        primary_beam = aa.Array2D(
            values=np.random.default_rng(seed=seed + 100).uniform(
                0.2, 1.0, size=mask.pixels_in_mask
            ),
            mask=mask,
        )

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=transformer_class or aa.TransformerDFT,
        primary_beam=primary_beam,
    )

    dataset_sparse = dataset.apply_sparse_operator(
        nufft_precision_operator=dataset.psf_precision_operator_from(method="numpy"),
        batch_size=4,
    )

    return mask, rng, dataset, dataset_sparse, primary_beam


def _primary_beam_linear_obj_lists(mask, rng):
    """The linear-object lists exercising every block: one mapper, func list + mapper, two mappers,
    func list + two mappers and a func list on its own."""
    mapper_0 = _mapper_from(
        mask=mask, pixels=9, shape=(3, 3), regularization=aa.reg.Constant(1.0)
    )
    mapper_1 = _mapper_from(
        mask=mask, pixels=16, shape=(4, 4), regularization=aa.reg.Constant(2.0)
    )
    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 2))
    )

    return {
        "mapper": [mapper_0],
        "func_list_and_mapper": [linear_obj, mapper_0],
        "x2_mappers": [mapper_0, mapper_1],
        "func_list_and_x2_mappers": [linear_obj, mapper_0, mapper_1],
        "func_list_only": [linear_obj],
    }


def _brute_force_w_tilde_and_dirty_image_from(dataset):
    """
    `W~ = Re(F^H N^-1 F)` and `d~ = Re(F^H N^-1 d)` computed densely from the un-beamed DFT response of
    every image pixel, `T = transform_mapping_matrix(I)`.
    """
    transformer = aa.TransformerDFT(
        uv_wavelengths=dataset.uv_wavelengths, real_space_mask=dataset.real_space_mask
    )

    assert transformer.primary_beam_slim is None

    response = transformer.transform_mapping_matrix(
        mapping_matrix=np.eye(dataset.real_space_mask.pixels_in_mask)
    )

    weights_real = dataset.noise_map.array.real**-2.0
    weights_imag = dataset.noise_map.array.imag**-2.0

    w_tilde = (response.real.T * weights_real) @ response.real + (
        response.imag.T * weights_imag
    ) @ response.imag
    dirty_image = response.real.T @ (
        dataset.data.array.real * weights_real
    ) + response.imag.T @ (dataset.data.array.imag * weights_imag)

    return w_tilde, dirty_image


def test__primary_beam__sparse_curvature_and_data_vector_match_brute_force():
    """
    The witness: with a non-trivial primary beam `P`, the sparse (w-tilde) inversion's curvature matrix is
    `Lᵀ diag(P) W~ diag(P) L` and its data vector `Lᵀ diag(P) d~`, with `W~` and `d~` built densely from the
    un-beamed DFT, for every block structure (mappers, linear function lists and both).
    """
    mask, rng, dataset, dataset_sparse, primary_beam = _primary_beam_parity_setup()

    w_tilde, dirty_image = _brute_force_w_tilde_and_dirty_image_from(dataset)

    # The operator's dirty image is independent of P and is the brute-force d~.
    np.testing.assert_allclose(
        dataset_sparse.sparse_operator.dirty_image,
        dirty_image,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(dirty_image).max(),
    )

    P = primary_beam.array

    for name, linear_obj_list in _primary_beam_linear_obj_lists(mask, rng).items():
        inversion = aa.Inversion(
            dataset=dataset_sparse,
            linear_obj_list=linear_obj_list,
            settings=_PRIMARY_BEAM_SETTINGS,
        )

        assert type(inversion) is aa.InversionInterferometerSparse, name

        mapping_matrix = np.hstack(
            [np.asarray(linear_obj.mapping_matrix) for linear_obj in linear_obj_list]
        )
        beamed = P[:, None] * mapping_matrix

        curvature_matrix = beamed.T @ w_tilde @ beamed
        data_vector = beamed.T @ dirty_image

        np.testing.assert_allclose(
            np.asarray(inversion.curvature_matrix),
            curvature_matrix,
            rtol=1.0e-10,
            atol=1.0e-10 * np.abs(curvature_matrix).max(),
            err_msg=name,
        )
        np.testing.assert_allclose(
            np.asarray(inversion.data_vector),
            data_vector,
            rtol=1.0e-10,
            atol=1.0e-10 * np.abs(data_vector).max(),
            err_msg=name,
        )

        # And P genuinely changes them: the un-beamed brute force differs.
        assert not np.allclose(
            inversion.data_vector, mapping_matrix.T @ dirty_image, rtol=1.0e-3
        ), name


def test__primary_beam__none_and_ones__bit_for_bit_unchanged__numpy():
    """
    `primary_beam=None` and `primary_beam=ones` leave the sparse and mapping curvature matrices, data
    vectors and `fast_chi_squared` bit-for-bit those of a dataset built without a primary beam.
    """
    mask, rng, dataset_ref, dataset_sparse_ref, _ = _primary_beam_parity_setup(
        primary_beam=None
    )

    for primary_beam in (None, np.ones(mask.pixels_in_mask)):
        _, _, dataset, dataset_sparse, _ = _primary_beam_parity_setup(
            primary_beam=primary_beam
        )

        for name, linear_obj_list in _primary_beam_linear_obj_lists(
            mask, np.random.default_rng(seed=1)
        ).items():
            for dataset_test, dataset_reference in (
                (dataset_sparse, dataset_sparse_ref),
                (dataset, dataset_ref),
            ):
                inversion = aa.Inversion(
                    dataset=dataset_test, linear_obj_list=linear_obj_list
                )
                inversion_reference = aa.Inversion(
                    dataset=dataset_reference, linear_obj_list=linear_obj_list
                )

                assert type(inversion) is type(inversion_reference)

                for quantity in ("curvature_matrix", "data_vector", "reconstruction"):
                    assert np.array_equal(
                        np.asarray(getattr(inversion, quantity)),
                        np.asarray(getattr(inversion_reference, quantity)),
                    ), (name, quantity)

                assert inversion.fast_chi_squared == inversion_reference.fast_chi_squared

                for linear_obj in linear_obj_list:
                    assert np.array_equal(
                        inversion.mapped_reconstructed_data_dict[linear_obj].array,
                        inversion_reference.mapped_reconstructed_data_dict[
                            linear_obj
                        ].array,
                    )


def _primary_beam_jit_quantities_from(dataset_sparse, mapper, image, xp):
    """
    The data vector, curvature matrix and `fast_chi_squared` of a sparse inversion of the visibilities
    with the profile image `image` subtracted (via `sparse_profile_terms_from`), on a `DatasetInterface`
    with no visibilities -- the likelihood path a `jax.jit` fit traces.
    """
    mask = dataset_sparse.real_space_mask

    _, sparse_dirty_image, data_term = (
        aa.util.inversion_interferometer.sparse_profile_terms_from(
            sparse_operator=dataset_sparse.sparse_operator,
            image=image,
            extent_index_for_masked_pixel=mask.extent_index_for_masked_pixel,
            xp=xp,
        )
    )

    inversion = aa.Inversion(
        dataset=aa.DatasetInterface(
            data=None,
            noise_map=dataset_sparse.noise_map,
            grids=dataset_sparse.grids,
            transformer=dataset_sparse.transformer,
            sparse_operator=dataset_sparse.sparse_operator,
            sparse_dirty_image=sparse_dirty_image,
            data_term=data_term,
        ),
        linear_obj_list=[mapper],
        xp=xp,
    )

    return inversion.data_vector, inversion.curvature_matrix, inversion.fast_chi_squared


def test__primary_beam__none_and_ones__bit_for_bit_unchanged__jax_jit():
    jax = pytest.importorskip("jax")

    import jax.numpy as jnp

    mask, _, dataset_ref, dataset_sparse_ref, _ = _primary_beam_parity_setup(
        primary_beam=None
    )

    mapper = _mapper_from(
        mask=mask, pixels=9, shape=(3, 3), regularization=aa.reg.Constant(1.0)
    )
    image = jnp.asarray(np.random.default_rng(seed=5).normal(size=mask.pixels_in_mask))

    def jitted(dataset_sparse):
        return [
            np.asarray(value)
            for value in jax.jit(
                lambda i: _primary_beam_jit_quantities_from(
                    dataset_sparse, mapper, i, xp=jnp
                )
            )(image)
        ]

    reference = jitted(dataset_sparse_ref)

    mapping_reference = aa.Inversion(
        dataset=dataset_ref, linear_obj_list=[mapper], xp=jnp
    )

    for primary_beam in (None, np.ones(mask.pixels_in_mask)):
        _, _, dataset, dataset_sparse, _ = _primary_beam_parity_setup(
            primary_beam=primary_beam
        )

        for value, value_reference in zip(jitted(dataset_sparse), reference):
            assert np.array_equal(value, value_reference)

        mapping = aa.Inversion(dataset=dataset, linear_obj_list=[mapper], xp=jnp)

        for quantity in ("curvature_matrix", "data_vector"):
            assert np.array_equal(
                np.asarray(getattr(mapping, quantity)),
                np.asarray(getattr(mapping_reference, quantity)),
            )


def test__primary_beam__jax_jit_matches_numpy_and_profile_terms_apply_P():
    """
    With a non-trivial primary beam the jitted sparse likelihood path matches NumPy, and the
    profile-subtracted terms are those of the beamed profile image `P i`: the dirty image
    `d~ - W~ P i` and data term of the un-beamed operator applied to `P i`.
    """
    jax = pytest.importorskip("jax")

    import jax.numpy as jnp

    mask, _, _, dataset_sparse, primary_beam = _primary_beam_parity_setup()
    _, _, _, dataset_sparse_unbeamed, _ = _primary_beam_parity_setup(primary_beam=None)

    mapper = _mapper_from(
        mask=mask, pixels=9, shape=(3, 3), regularization=aa.reg.Constant(1.0)
    )
    image = np.random.default_rng(seed=5).normal(size=mask.pixels_in_mask)

    numpy = _primary_beam_jit_quantities_from(dataset_sparse, mapper, image, xp=np)
    jitted = jax.jit(
        lambda i: _primary_beam_jit_quantities_from(dataset_sparse, mapper, i, xp=jnp)
    )(jnp.asarray(image))

    for value, value_numpy in zip(jitted, numpy):
        value_numpy = np.asarray(value_numpy)
        np.testing.assert_allclose(
            np.asarray(value),
            value_numpy,
            rtol=1.0e-9,
            atol=1.0e-9 * np.abs(value_numpy).max(),
        )

    extent_index = mask.extent_index_for_masked_pixel

    terms = aa.util.inversion_interferometer.sparse_profile_terms_from(
        sparse_operator=dataset_sparse.sparse_operator,
        image=image,
        extent_index_for_masked_pixel=extent_index,
    )
    terms_unbeamed = aa.util.inversion_interferometer.sparse_profile_terms_from(
        sparse_operator=dataset_sparse_unbeamed.sparse_operator,
        image=image * primary_beam.array,
        extent_index_for_masked_pixel=extent_index,
    )

    for value, value_unbeamed in zip(terms, terms_unbeamed):
        np.testing.assert_allclose(value, value_unbeamed, rtol=1.0e-12, atol=1.0e-12)


def test__primary_beam__sparse_matches_mapping_and_fast_chi_squared_matches_residuals():
    """
    With a primary beam, the sparse inversion reproduces the dense (mapping) one -- whose transformer applies
    `P` in `transform_mapping_matrix` -- and its `fast_chi_squared` equals the chi-squared of the residual
    visibilities of `mapped_reconstructed_operated_data`, which the transformer forms as `F P (L s)` from the
    un-beamed (true-sky) reconstructed image `L s`.
    """
    mask, rng, dataset, dataset_sparse, primary_beam = _primary_beam_parity_setup()

    for name, linear_obj_list in _primary_beam_linear_obj_lists(mask, rng).items():
        if name != "func_list_only":
            _assert_sparse_matches_mapping(
                dataset=dataset,
                dataset_sparse=dataset_sparse,
                linear_obj_list=linear_obj_list,
            )

        inversion_sparse = aa.Inversion(
            dataset=dataset_sparse,
            linear_obj_list=linear_obj_list,
            settings=_PRIMARY_BEAM_SETTINGS,
        )
        inversion_mapping = aa.Inversion(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=_PRIMARY_BEAM_SETTINGS,
        )

        for quantity in ("curvature_matrix", "data_vector"):
            reference = np.asarray(getattr(inversion_mapping, quantity))

            np.testing.assert_allclose(
                np.asarray(getattr(inversion_sparse, quantity)),
                reference,
                rtol=1.0e-10,
                atol=1.0e-10 * np.abs(reference).max(),
                err_msg=(name, quantity),
            )

        # The reconstructed image is the true sky `L s` (no P).
        reconstruction_dict = inversion_sparse.source_quantity_dict_from(
            source_quantity=inversion_sparse.reconstruction
        )

        for linear_obj in linear_obj_list:
            np.testing.assert_allclose(
                inversion_sparse.mapped_reconstructed_data_dict[linear_obj].array,
                np.asarray(linear_obj.mapping_matrix) @ reconstruction_dict[linear_obj],
                rtol=1.0e-12,
                atol=1.0e-12,
            )

        residual_map = aa.util.fit.residual_map_from(
            data=dataset.data,
            model_data=inversion_sparse.mapped_reconstructed_operated_data,
        )
        chi_squared = aa.util.fit.chi_squared_complex_from(
            chi_squared_map=aa.util.fit.chi_squared_map_complex_from(
                residual_map=residual_map, noise_map=dataset.noise_map
            )
        )

        # The unregularized linear function list solves to coefficients ~1e4 (no positive-only solver
        # in the test config), so `s^T F s - 2 s^T D + d^T d` cancels to ~1e-7 relative, with or without P.
        assert float(inversion_sparse.fast_chi_squared) == pytest.approx(
            float(chi_squared), rel=1.0e-6
        ), name


def test__primary_beam__dft_and_nufft_sparse_inversions_match():
    pytest.importorskip("nufftax")

    mask, rng, _, dataset_sparse_dft, _ = _primary_beam_parity_setup()
    _, _, _, dataset_sparse_nufft, _ = _primary_beam_parity_setup(
        transformer_class=aa.TransformerNUFFT
    )

    assert np.array_equal(
        dataset_sparse_nufft.transformer.primary_beam_slim,
        dataset_sparse_dft.transformer.primary_beam_slim,
    )

    for name, linear_obj_list in _primary_beam_linear_obj_lists(mask, rng).items():
        inversion_dft = aa.Inversion(
            dataset=dataset_sparse_dft,
            linear_obj_list=linear_obj_list,
            settings=_PRIMARY_BEAM_SETTINGS,
        )
        inversion_nufft = aa.Inversion(
            dataset=dataset_sparse_nufft,
            linear_obj_list=linear_obj_list,
            settings=_PRIMARY_BEAM_SETTINGS,
        )

        for quantity in ("curvature_matrix", "data_vector"):
            reference = np.asarray(getattr(inversion_dft, quantity))

            np.testing.assert_allclose(
                np.asarray(getattr(inversion_nufft, quantity)),
                reference,
                rtol=1.0e-8,
                atol=1.0e-8 * np.abs(reference).max(),
                err_msg=(name, quantity),
            )

        np.testing.assert_allclose(
            inversion_nufft.mapped_reconstructed_operated_data.array,
            inversion_dft.mapped_reconstructed_operated_data.array,
            rtol=1.0e-6,
            atol=1.0e-8,
            err_msg=name,
        )


def test__primary_beam__array_free_from_sparse_terms_matches_in_memory():
    pytest.importorskip("nufftax")

    mask, dataset_memory_unbeamed, dataset_stream_unbeamed, mapper = _array_free_setup()

    primary_beam = aa.Array2D(
        values=np.random.default_rng(seed=9).uniform(0.2, 1.0, size=mask.pixels_in_mask),
        mask=mask,
    )

    dataset_memory = aa.Interferometer(
        data=dataset_memory_unbeamed.data,
        noise_map=dataset_memory_unbeamed.noise_map,
        uv_wavelengths=dataset_memory_unbeamed.uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerNUFFT,
        primary_beam=primary_beam,
    ).apply_sparse_operator()

    dataset_terms = aa.Interferometer.from_sparse_terms(
        dataset_stream_unbeamed.sparse_terms, mask, primary_beam=primary_beam
    )

    inversion_terms = aa.Inversion(dataset=dataset_terms, linear_obj_list=[mapper])
    inversion_memory = aa.Inversion(dataset=dataset_memory, linear_obj_list=[mapper])

    _assert_array_free_matches_in_memory(inversion_terms, inversion_memory, rel=1.0e-8)

    # The primary beam changes the fit.
    assert float(inversion_memory.fast_chi_squared) != pytest.approx(
        float(
            aa.Inversion(
                dataset=dataset_memory_unbeamed, linear_obj_list=[mapper]
            ).fast_chi_squared
        ),
        rel=1.0e-6,
    )


def test__primary_beam__dirty_model_image_natural_applies_P():
    mask, _, _, dataset_sparse, primary_beam = _primary_beam_parity_setup()
    _, _, _, dataset_sparse_unbeamed, _ = _primary_beam_parity_setup(primary_beam=None)

    image = np.random.default_rng(seed=2).normal(size=mask.pixels_in_mask)

    np.testing.assert_allclose(
        aa.fit.fit_interferometer.dirty_model_image_natural_from(
            dataset=dataset_sparse, image=image
        ).array,
        aa.fit.fit_interferometer.dirty_model_image_natural_from(
            dataset=dataset_sparse_unbeamed, image=image * primary_beam.array
        ).array,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test__primary_beam__numba_route_is_skipped_and_direct_construction_raises():
    pytest.importorskip("numba")

    from autoarray.inversion.inversion.interferometer_numba.sparse import (
        InversionInterferometerSparseNumba,
    )

    mask, _, _, dataset_sparse, _ = _primary_beam_parity_setup()
    _, _, _, dataset_sparse_unbeamed, _ = _primary_beam_parity_setup(primary_beam=None)

    mapper = _mapper_from(
        mask=mask, pixels=9, shape=(3, 3), regularization=aa.reg.Constant(1.0)
    )
    settings = aa.Settings(interferometer_route="numba")

    assert isinstance(
        aa.Inversion(
            dataset=dataset_sparse_unbeamed, linear_obj_list=[mapper], settings=settings
        ),
        InversionInterferometerSparseNumba,
    )
    assert (
        type(
            aa.Inversion(
                dataset=dataset_sparse, linear_obj_list=[mapper], settings=settings
            )
        )
        is aa.InversionInterferometerSparse
    )

    with pytest.raises(aa.exc.InversionException, match="primary beam"):
        InversionInterferometerSparseNumba(
            dataset=dataset_sparse, linear_obj_list=[mapper], settings=settings
        )
