import numpy as np
import pytest

import autoarray as aa


# ---------------------------------------------------------------------------
# residual_map_from
# ---------------------------------------------------------------------------


def test__residual_map_from__identical_data_and_model__all_zero_residuals():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    assert (residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()


def test__residual_map_from__different_data_and_model__correct_signed_residuals():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    assert (residual_map == np.array([-1.0, 0.0, 1.0, 2.0])).all()


# ---------------------------------------------------------------------------
# residual_map_with_mask_from
# ---------------------------------------------------------------------------


def test__residual_map_with_mask_from__masked_pixels__zero_at_masked_locations():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    assert (residual_map == np.array([0.0, 0.0, 1.0, 0.0])).all()


# ---------------------------------------------------------------------------
# normalized_residual_map_from
# ---------------------------------------------------------------------------


def test__normalized_residual_map_from__identical_data_and_model__all_zero():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    normalized_residual_map = aa.util.fit.normalized_residual_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert normalized_residual_map == pytest.approx(
        np.array([0.0, 0.0, 0.0, 0.0]), 1.0e-4
    )


def test__normalized_residual_map_from__different_data_and_model__divided_by_noise_map():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    normalized_residual_map = aa.util.fit.normalized_residual_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert normalized_residual_map == pytest.approx(
        np.array([-(1.0 / 2.0), 0.0, (1.0 / 2.0), (2.0 / 2.0)]), 1.0e-4
    )


# ---------------------------------------------------------------------------
# normalized_residual_map_with_mask_from
# ---------------------------------------------------------------------------


def test__normalized_residual_map_with_mask_from__masked_pixels__zero_at_masked_locations():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    normalized_residual_map = aa.util.fit.normalized_residual_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    assert normalized_residual_map == pytest.approx(
        np.array([0.0, 0.0, (1.0 / 2.0), 0.0]), abs=1.0e-4
    )


# ---------------------------------------------------------------------------
# normalized_residual_map_complex_from
# ---------------------------------------------------------------------------


def test__normalized_residual_map_complex_from__complex_data__correct_real_and_imaginary_parts():
    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    noise_map = np.array([2.0 + 2.0j, 2.0 + 2.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 12.0j])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    normalized_residual_map = aa.util.fit.normalized_residual_map_complex_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (normalized_residual_map == np.array([0.5 - 1.0j, 0.5 - 1.0j])).all()


# ---------------------------------------------------------------------------
# chi_squared_map_from
# ---------------------------------------------------------------------------


def test__chi_squared_map_from__identical_data_and_model__all_zero():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.0, 0.0, 0.0, 0.0])).all()


def test__chi_squared_map_from__different_data_and_model__squared_normalized_residuals():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (
        chi_squared_map
        == np.array([(1.0 / 2.0) ** 2.0, 0.0, (1.0 / 2.0) ** 2.0, (2.0 / 2.0) ** 2.0])
    ).all()


# ---------------------------------------------------------------------------
# chi_squared_map_with_mask_from
# ---------------------------------------------------------------------------


def test__chi_squared_map_with_mask_from__masked_pixels__zero_at_masked_locations():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.0, 0.0, (1.0 / 2.0) ** 2.0, 0.0])).all()


# ---------------------------------------------------------------------------
# chi_squared_map_complex_from
# ---------------------------------------------------------------------------


def test__chi_squared_map_complex_from__complex_data__correct_real_and_imaginary_chi_squared():
    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    noise_map = np.array([2.0 + 2.0j, 2.0 + 2.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 12.0j])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    chi_squared_map = aa.util.fit.chi_squared_map_complex_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.25 + 1.0j, 0.25 + 1.0j])).all()


# ---------------------------------------------------------------------------
# chi_squared_with_noise_covariance_from
# ---------------------------------------------------------------------------


def test__chi_squared_with_noise_covariance_from__known_residual_and_inverse__correct_chi_squared():
    residual_map = aa.Array2D.no_mask([[1.0, 1.0], [2.0, 2.0]], pixel_scales=1.0)

    noise_covariance_matrix_inv = np.array(
        [
            [1.0, 1.0, 4.0, 0.0],
            [0.0, 1.0, 9.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
        ]
    )

    chi_squared = aa.util.fit.chi_squared_with_noise_covariance_from(
        residual_map=residual_map,
        noise_covariance_matrix_inv=noise_covariance_matrix_inv,
    )

    assert chi_squared == 43


# ---------------------------------------------------------------------------
# chi_squared_with_mask_fast_from
# ---------------------------------------------------------------------------


def test__chi_squared_with_mask_fast_from__1d_data_with_mask__equals_standard_computation():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    noise_map = np.array([1.0, 2.0, 3.0, 4.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_with_mask_from(
        mask=mask, chi_squared_map=chi_squared_map
    )
    chi_squared_fast = aa.util.fit.chi_squared_with_mask_fast_from(
        data=data,
        noise_map=noise_map,
        mask=mask,
        model_data=model_data,
    )

    assert chi_squared == pytest.approx(chi_squared_fast, 1.0e-4)


def test__chi_squared_with_mask_fast_from__2d_data_with_mask__equals_standard_computation():
    data = np.array([[10.0, 10.0], [10.0, 10.0]])
    mask = np.array([[True, False], [False, True]])
    noise_map = np.array([[1.0, 2.0], [3.0, 4.0]])
    model_data = np.array([[11.0, 10.0], [9.0, 8.0]])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_with_mask_from(
        mask=mask, chi_squared_map=chi_squared_map
    )
    chi_squared_fast = aa.util.fit.chi_squared_with_mask_fast_from(
        data=data,
        noise_map=noise_map,
        mask=mask,
        model_data=model_data,
    )

    assert chi_squared == pytest.approx(chi_squared_fast, 1.0e-4)


# ---------------------------------------------------------------------------
# log_likelihood_from
# ---------------------------------------------------------------------------


def test__log_likelihood_from__identical_data_and_model__correct_value_from_noise_normalization():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)
    noise_normalization = aa.util.fit.noise_normalization_from(noise_map=noise_map)
    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    expected_chi_squared = 0.0
    expected_noise_normalization = 4 * np.log(2.0 * np.pi * (2.0**2.0))

    assert log_likelihood == pytest.approx(
        -0.5 * (expected_chi_squared + expected_noise_normalization), 1.0e-4
    )


def test__log_likelihood_from__different_data_and_model_uniform_noise__correct_value():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)
    noise_normalization = aa.util.fit.noise_normalization_from(noise_map=noise_map)
    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0.25, 0, 0.25, 1.0
    expected_chi_squared = (
        ((1.0 / 2.0) ** 2.0) + 0.0 + ((1.0 / 2.0) ** 2.0) + ((2.0 / 2.0) ** 2.0)
    )
    expected_noise_normalization = 4 * np.log(2.0 * np.pi * (2.0**2.0))

    assert log_likelihood == pytest.approx(
        -0.5 * (expected_chi_squared + expected_noise_normalization), 1.0e-4
    )


def test__log_likelihood_from__different_data_and_model_varied_noise__correct_value():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([1.0, 2.0, 3.0, 4.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)
    noise_normalization = aa.util.fit.noise_normalization_from(noise_map=noise_map)
    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0
    expected_chi_squared = 1.0 + (1.0 / (3.0**2.0)) + 0.25
    expected_noise_normalization = (
        np.log(2 * np.pi * (1.0**2.0))
        + np.log(2 * np.pi * (2.0**2.0))
        + np.log(2 * np.pi * (3.0**2.0))
        + np.log(2 * np.pi * (4.0**2.0))
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (expected_chi_squared + expected_noise_normalization), 1e-4
    )


# ---------------------------------------------------------------------------
# log_likelihood_from with mask
# ---------------------------------------------------------------------------


def test__log_likelihood_from__with_1d_mask__excludes_masked_pixels_from_chi_squared_and_noise_normalization():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    noise_map = np.array([1.0, 2.0, 3.0, 4.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_with_mask_from(
        mask=mask, chi_squared_map=chi_squared_map
    )
    noise_normalization = aa.util.fit.noise_normalization_with_mask_from(
        mask=mask, noise_map=noise_map
    )
    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0, (1/3)**2  (pixels at idx 0 and 3 are masked)
    expected_chi_squared = 0.0 + (1.0 / 3.0) ** 2.0
    expected_noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(
        2 * np.pi * (3.0**2.0)
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (expected_chi_squared + expected_noise_normalization), 1e-4
    )


def test__log_likelihood_from__with_2d_mask__excludes_masked_pixels_from_chi_squared_and_noise_normalization():
    data = np.array([[10.0, 10.0], [10.0, 10.0]])
    mask = np.array([[True, False], [False, True]])
    noise_map = np.array([[1.0, 2.0], [3.0, 4.0]])
    model_data = np.array([[11.0, 10.0], [9.0, 8.0]])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_with_mask_from(
        mask=mask, chi_squared_map=chi_squared_map
    )
    noise_normalization = aa.util.fit.noise_normalization_with_mask_from(
        mask=mask, noise_map=noise_map
    )
    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0, (1/3)**2  (corner pixels are masked)
    expected_chi_squared = 0.0 + (1.0 / 3.0) ** 2.0
    expected_noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(
        2 * np.pi * (3.0**2.0)
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (expected_chi_squared + expected_noise_normalization), 1e-4
    )


# ---------------------------------------------------------------------------
# log_likelihood_from with complex data
# ---------------------------------------------------------------------------


def test__log_likelihood_from__complex_data__chi_squared_sums_real_and_imaginary_components():
    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    noise_map = np.array([2.0 + 1.0j, 2.0 + 1.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 12.0j])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    chi_squared_map = aa.util.fit.chi_squared_map_complex_from(
        residual_map=residual_map, noise_map=noise_map
    )
    chi_squared = aa.util.fit.chi_squared_complex_from(chi_squared_map=chi_squared_map)
    noise_normalization = aa.util.fit.noise_normalization_complex_from(
        noise_map=noise_map
    )
    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0.25 and 4.0
    expected_chi_squared = 4.25
    expected_noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(
        2 * np.pi * (1.0**2.0)
    )

    assert log_likelihood == pytest.approx(
        -0.5 * 2.0 * (expected_chi_squared + expected_noise_normalization), 1e-4
    )


# ---------------------------------------------------------------------------
# log_likelihood_with_regularization_from / log_evidence_from
# ---------------------------------------------------------------------------


def test__log_likelihood_with_regularization_from__adds_regularization_to_chi_squared_and_noise():
    likelihood_with_regularization_terms = (
        aa.util.fit.log_likelihood_with_regularization_from(
            chi_squared=3.0, regularization_term=6.0, noise_normalization=2.0
        )
    )

    assert likelihood_with_regularization_terms == -0.5 * (3.0 + 6.0 + 2.0)


def test__log_evidence_from__includes_curvature_and_regularization_determinant_terms():
    evidences = aa.util.fit.log_evidence_from(
        chi_squared=3.0,
        regularization_term=6.0,
        log_curvature_regularization_term=9.0,
        log_regularization_term=10.0,
        noise_normalization=30.0,
    )

    assert evidences == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)


# ---------------------------------------------------------------------------
# residual_flux_fraction_map_from
# ---------------------------------------------------------------------------


def test__residual_flux_fraction_map_from__identical_data_and_model__all_zero():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    residual_flux_fraction_map = aa.util.fit.residual_flux_fraction_map_from(
        residual_map=residual_map, data=data
    )

    assert (residual_flux_fraction_map == np.array([0.0, 0.0, 0.0, 0.0])).all()


def test__residual_flux_fraction_map_from__different_data_and_model__correct_fractional_residuals():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)
    residual_flux_fraction_map = aa.util.fit.residual_flux_fraction_map_from(
        residual_map=residual_map, data=data
    )

    assert (residual_flux_fraction_map == np.array([-0.1, 0.0, 0.1, 0.2])).all()


# ---------------------------------------------------------------------------
# residual_flux_fraction_map_with_mask_from
# ---------------------------------------------------------------------------


def test__residual_flux_fraction_map_with_mask_from__masked_data__zero_at_masked_locations():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    residual_flux_fraction_map = aa.util.fit.residual_flux_fraction_map_with_mask_from(
        residual_map=residual_map, mask=mask, data=data
    )

    assert (residual_flux_fraction_map == np.array([0.0, 0.0, 0.1, 0.0])).all()


def test__residual_flux_fraction_map_with_mask_from__different_model__correct_unmasked_fractions():
    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    model_data = np.array([11.0, 9.0, 8.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )
    residual_flux_fraction_map = aa.util.fit.residual_flux_fraction_map_with_mask_from(
        residual_map=residual_map, mask=mask, data=data
    )

    assert (residual_flux_fraction_map == np.array([0.0, 0.1, 0.2, 0.0])).all()
