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


def test__apply_sparse_operator__delaunay_mapper__raises_not_implemented():
    """``InversionInterferometerSparse.curvature_matrix_diag`` must raise a
    ``NotImplementedError`` rather than silently mis-computing when paired
    with a Delaunay mapper.

    The interferometer sparse-operator curvature path
    (``InterferometerSparseOperator.curvature_matrix_via_sparse_operator_from``)
    was only validated against ``Rectangular*`` meshes (single source pixel per
    image pixel, weight 1). On a Delaunay mapper (three source pixels per image
    pixel via barycentric interpolation, weights summing to 1) the returned
    curvature matrix disagrees with the mapping path by ~34% Frobenius norm and
    the regularized matrix loses positive-definiteness, raising a numpy
    ``LinAlgError`` deep inside ``Inversion.log_det_curvature_reg_matrix_term``.
    The guard at the entry point catches the mis-use early with a clear message.
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
    image_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

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

    inversion = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
    )

    with pytest.raises(NotImplementedError, match=r"Delaunay"):
        _ = inversion.curvature_matrix
