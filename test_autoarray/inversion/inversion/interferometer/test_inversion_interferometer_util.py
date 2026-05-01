import autoarray as aa
import numpy as np
import pytest


def test__data_vector_via_transformed_mapping_matrix_from():
    mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    data_real = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_real = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_real_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_real,
            noise_map=noise_map_real,
        )
    )

    data_imag = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_imag = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_imag_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_imag,
            noise_map=noise_map_imag,
        )
    )

    data_vector_complex_via_blurred = (
        data_vector_real_via_blurred + data_vector_imag_via_blurred
    )

    transformed_mapping_matrix = np.array(
        [
            [1.0 + 1.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [1.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 1.0 + 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    data = np.array(
        [4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j, 16.0 + 16.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )
    noise_map = np.array(
        [2.0 + 2.0j, 1.0 + 1.0j, 1.0 + 1.0j, 4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )

    data_vector_via_transformed = aa.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=data,
        noise_map=noise_map,
    )

    assert (data_vector_complex_via_blurred == data_vector_via_transformed).all()
