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
