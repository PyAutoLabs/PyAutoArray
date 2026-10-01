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
