import numpy as np
import pytest

import autoarray as aa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset():
    real_space_mask = aa.Mask2D(
        mask=[[False, False], [False, False]], pixel_scales=(1.0, 1.0)
    )
    data = aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
    noise_map = aa.VisibilitiesNoiseMap(visibilities=[2.0 + 2.0j, 2.0 + 2.0j])
    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=np.ones(shape=(2, 2)),
        real_space_mask=real_space_mask,
    )
    return dataset, noise_map


def _make_identical_fit():
    dataset, noise_map = _make_dataset()
    model_data = aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
    fit = aa.m.MockFitInterferometer(
        dataset=dataset, use_mask_in_fit=False, model_data=model_data
    )
    return fit, noise_map


def _make_different_fit():
    dataset, noise_map = _make_dataset()
    model_data = aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 3.0j])
    fit = aa.m.MockFitInterferometer(
        dataset=dataset, use_mask_in_fit=False, model_data=model_data
    )
    return fit, noise_map


def _make_identical_fit_with_inversion():
    dataset, noise_map = _make_dataset()
    data = dataset.data
    model_data = aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
    chi_squared = data - model_data
    chi_squared = np.sum(
        (chi_squared.real**2.0 / dataset.noise_map.real**2.0)
        + (chi_squared.imag**2.0 / dataset.noise_map.imag**2.0)
    )
    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper()],
        data_vector=1,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
        fast_chi_squared=chi_squared,
    )
    fit = aa.m.MockFitInterferometer(
        dataset=dataset,
        use_mask_in_fit=False,
        model_data=model_data,
        inversion=inversion,
    )
    return fit, noise_map


# ---------------------------------------------------------------------------
# Tests: identical visibilities, no masking
# ---------------------------------------------------------------------------


def test__data__identical_visibilities__returns_correct_complex_data():
    fit, _ = _make_identical_fit()

    assert (fit.data == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()


def test__noise_map__uniform_complex_noise__returns_correct_noise_map():
    fit, _ = _make_identical_fit()

    assert (fit.noise_map == np.array([2.0 + 2.0j, 2.0 + 2.0j])).all()


def test__signal_to_noise_map__identical_visibilities__returns_correct_complex_snr():
    fit, _ = _make_identical_fit()

    assert (fit.signal_to_noise_map == np.array([0.5 + 1.0j, 1.5 + 2.0j])).all()


def test__residual_map__identical_visibilities__all_zero_residuals():
    fit, _ = _make_identical_fit()

    assert (fit.residual_map == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()


def test__normalized_residual_map__identical_visibilities__all_zero_normalized_residuals():
    fit, _ = _make_identical_fit()

    assert (fit.normalized_residual_map == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()


def test__chi_squared_map__identical_visibilities__all_zero_chi_squared():
    fit, _ = _make_identical_fit()

    assert (fit.chi_squared_map == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()


def test__chi_squared__identical_visibilities__is_zero():
    fit, _ = _make_identical_fit()

    assert fit.chi_squared == 0.0


def test__reduced_chi_squared__identical_visibilities__is_zero():
    fit, _ = _make_identical_fit()

    assert fit.reduced_chi_squared == 0.0


def test__noise_normalization__uniform_complex_noise__correct_log_formula():
    fit, _ = _make_identical_fit()

    assert fit.noise_normalization == pytest.approx(
        4.0 * np.log(2 * np.pi * 2.0**2.0), 1.0e-4
    )


def test__log_likelihood__identical_visibilities__negative_half_noise_normalization():
    fit, _ = _make_identical_fit()

    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


# ---------------------------------------------------------------------------
# Tests: different visibilities, no masking
# ---------------------------------------------------------------------------


def test__residual_map__different_visibilities__correct_complex_residuals():
    fit, _ = _make_different_fit()

    assert (fit.residual_map == np.array([0.0 + 0.0j, 0.0 + 1.0j])).all()


def test__normalized_residual_map__different_visibilities__correct_complex_normalized_residuals():
    fit, _ = _make_different_fit()

    assert (fit.normalized_residual_map == np.array([0.0 + 0.0j, 0.0 + 0.5j])).all()


def test__chi_squared_map__different_visibilities__correct_complex_chi_squared_map():
    fit, _ = _make_different_fit()

    assert (fit.chi_squared_map == np.array([0.0 + 0.0j, 0.0 + 0.25j])).all()


def test__chi_squared__different_visibilities__correct_value():
    fit, _ = _make_different_fit()

    assert fit.chi_squared == 0.25


def test__reduced_chi_squared__different_visibilities__divided_by_visibility_count():
    fit, _ = _make_different_fit()

    assert fit.reduced_chi_squared == 0.25 / 2.0


def test__log_likelihood__different_visibilities__correct_value():
    fit, _ = _make_different_fit()

    assert fit.noise_normalization == pytest.approx(
        4.0 * np.log(2 * np.pi * 2.0**2.0), 1.0e-4
    )
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


# ---------------------------------------------------------------------------
# Tests: identical visibilities with inversion
# ---------------------------------------------------------------------------


def test__chi_squared__identical_visibilities_with_inversion__is_zero():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.chi_squared == 0.0


def test__reduced_chi_squared__identical_visibilities_with_inversion__is_zero():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.reduced_chi_squared == 0.0


def test__log_likelihood_with_regularization__interferometer_with_inversion__adds_regularization_term():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.log_likelihood_with_regularization == -0.5 * (
        fit.chi_squared + 2.0 + fit.noise_normalization
    )


def test__log_evidence__interferometer_with_inversion__uses_chi_squared_reg_and_determinant_terms():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.log_evidence == -0.5 * (
        fit.chi_squared + 2.0 + 3.0 - 4.0 + fit.noise_normalization
    )


def test__figure_of_merit__interferometer_with_inversion__equals_log_evidence():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.figure_of_merit == fit.log_evidence


# ---------------------------------------------------------------------------
# Tests: dirty image quantities (transformer applied to fit quantities)
# ---------------------------------------------------------------------------


def test__dirty_image__equals_transformer_image_from_interferometer_data(
    transformer_7x7_7, interferometer_7, fit_interferometer_7
):
    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_image = transformer_7x7_7.image_from(visibilities=interferometer_7.data)

    assert (fit_interferometer_7.dirty_image == dirty_image).all()


def test__dirty_noise_map__equals_transformer_image_from_noise_map(
    transformer_7x7_7, interferometer_7, fit_interferometer_7
):
    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_noise_map = transformer_7x7_7.image_from(
        visibilities=interferometer_7.noise_map
    )

    assert (fit_interferometer_7.dirty_noise_map == dirty_noise_map).all()


def test__dirty_signal_to_noise_map__equals_transformer_image_from_signal_to_noise_map(
    transformer_7x7_7, interferometer_7, fit_interferometer_7
):
    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_signal_to_noise_map = transformer_7x7_7.image_from(
        visibilities=interferometer_7.signal_to_noise_map
    )

    assert (
        fit_interferometer_7.dirty_signal_to_noise_map == dirty_signal_to_noise_map
    ).all()


def test__dirty_model_image__equals_transformer_image_from_model_data(
    transformer_7x7_7, interferometer_7, fit_interferometer_7
):
    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_model_image = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.model_data
    )

    assert (fit_interferometer_7.dirty_model_image == dirty_model_image).all()


def test__dirty_residual_map__equals_transformer_image_from_residual_map(
    transformer_7x7_7, interferometer_7, fit_interferometer_7
):
    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_residual_map = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.residual_map
    )

    assert (fit_interferometer_7.dirty_residual_map == dirty_residual_map).all()


def test__dirty_normalized_residual_map__equals_transformer_image_from_normalized_residual_map(
    transformer_7x7_7, interferometer_7, fit_interferometer_7
):
    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_normalized_residual_map = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.normalized_residual_map
    )

    assert (
        fit_interferometer_7.dirty_normalized_residual_map
        == dirty_normalized_residual_map
    ).all()


def test__dirty_chi_squared_map__equals_transformer_image_from_chi_squared_map(
    transformer_7x7_7, interferometer_7, fit_interferometer_7
):
    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_chi_squared_map = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.chi_squared_map
    )

    assert (fit_interferometer_7.dirty_chi_squared_map == dirty_chi_squared_map).all()
