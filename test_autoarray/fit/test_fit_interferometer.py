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


# ---------------------------------------------------------------------------
# Tests: noise normalization cached on the sparse operator
# ---------------------------------------------------------------------------


def _sparse_dataset(noise_normalization):
    dataset, _ = _make_dataset()

    sparse_operator = aa.InterferometerSparseOperator.from_nufft_precision_operator(
        nufft_precision_operator=np.ones((4, 4)),
        dirty_image=np.zeros(4),
        noise_normalization=noise_normalization,
    )

    return aa.Interferometer(
        data=dataset.data,
        noise_map=dataset.noise_map,
        uv_wavelengths=dataset.uv_wavelengths,
        real_space_mask=dataset.real_space_mask,
        sparse_operator=sparse_operator,
    )


def test__noise_normalization__reads_the_sparse_operator_scalar_when_present():
    dataset = _sparse_dataset(noise_normalization=123.0)

    fit = aa.m.MockFitInterferometer(
        dataset=dataset,
        use_mask_in_fit=False,
        model_data=aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j]),
    )

    assert fit.noise_normalization == 123.0


def test__noise_normalization__falls_back_to_the_noise_map_without_the_scalar():
    dataset = _sparse_dataset(noise_normalization=None)

    fit = aa.m.MockFitInterferometer(
        dataset=dataset,
        use_mask_in_fit=False,
        model_data=aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j]),
    )

    assert fit.noise_normalization == pytest.approx(
        4.0 * np.log(2 * np.pi * 4.0), 1.0e-12
    )


def test__noise_normalization__a_replaced_noise_map_ignores_the_scalar():
    dataset = _sparse_dataset(noise_normalization=123.0)

    noise_map = aa.VisibilitiesNoiseMap(visibilities=[1.0 + 1.0j, 1.0 + 1.0j])

    fit = aa.m.MockFitInterferometer(
        dataset=dataset,
        use_mask_in_fit=False,
        model_data=aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j]),
        noise_map=noise_map,
    )

    assert fit.noise_normalization == pytest.approx(4.0 * np.log(2 * np.pi), 1.0e-12)


def test__noise_normalization__apply_sparse_operator_scalar_matches_array_path():
    dataset, _ = _make_dataset()

    model_data = aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])

    fit_array = aa.m.MockFitInterferometer(
        dataset=dataset, use_mask_in_fit=False, model_data=model_data
    )
    fit_sparse = aa.m.MockFitInterferometer(
        dataset=dataset.apply_sparse_operator(),
        use_mask_in_fit=False,
        model_data=model_data,
    )

    assert fit_sparse.dataset.sparse_operator.noise_normalization is not None
    assert fit_sparse.noise_normalization == fit_array.noise_normalization


def _array_free_fit_setup():
    from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
        overlay_grid_from,
    )

    mask = aa.Mask2D.circular(shape_native=(10, 10), pixel_scales=0.5, radius=2.0)

    rng = np.random.default_rng(seed=7)
    n_visibilities = 50
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)) * 1.0e5
    data = rng.normal(size=n_visibilities) + 1j * rng.normal(size=n_visibilities)
    sigma = rng.uniform(0.5, 2.0, size=n_visibilities)
    noise_map = sigma + 1j * sigma

    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=data),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=noise_map),
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
    )

    dataset_stream = aa.Interferometer.from_stream(
        [
            (uv_wavelengths[k0:k1], data[k0:k1], noise_map[k0:k1])
            for k0, k1 in ((0, 20), (20, n_visibilities))
        ],
        mask,
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)
    mesh = aa.mesh.RectangularUniform(shape=(4, 4))
    mapper = aa.Mapper(
        interpolator=mesh.interpolator_from(
            source_plane_data_grid=grid,
            source_plane_mesh_grid=aa.Grid2DIrregular(
                overlay_grid_from(shape_native=(4, 4), grid=grid)
            ),
            adapt_data=None,
        ),
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    return dataset.apply_sparse_operator(), dataset_stream, mapper


def test__fit_interferometer__array_free_dataset__log_evidence_works_and_maps_raise():
    pytest.importorskip("nufftax")

    dataset_memory, dataset_stream, mapper = _array_free_fit_setup()

    fit_stream = aa.m.MockFitInterferometer(
        dataset=dataset_stream,
        inversion=aa.Inversion(dataset=dataset_stream, linear_obj_list=[mapper]),
    )
    fit_memory = aa.m.MockFitInterferometer(
        dataset=dataset_memory,
        inversion=aa.Inversion(dataset=dataset_memory, linear_obj_list=[mapper]),
    )

    assert fit_stream.log_evidence == pytest.approx(fit_memory.log_evidence, rel=1.0e-8)
    assert fit_stream.figure_of_merit == pytest.approx(
        fit_memory.figure_of_merit, rel=1.0e-8
    )
    assert fit_stream.noise_normalization == pytest.approx(
        fit_memory.noise_normalization, rel=1.0e-12
    )

    for name in (
        "mask",
        "transformer",
        "residual_map",
        "normalized_residual_map",
        "chi_squared_map",
        "signal_to_noise_map",
        "chi_squared",
        "dirty_image",
        "dirty_noise_map",
        "dirty_residual_map",
        "dirty_chi_squared_map",
    ):
        with pytest.raises(aa.exc.DatasetException, match="array-free"):
            getattr(fit_stream, name)


def test__fit_interferometer__sparse_log_evidence_never_touches_visibility_maps(
    monkeypatch,
):
    """
    Spy: on the in-memory sparse dataset (which *has* the arrays), make every
    visibility-space quantity of the fit raise, and check `log_evidence` /
    `figure_of_merit` still evaluate -- i.e. the sparse likelihood does not reach them.
    """
    pytest.importorskip("nufftax")

    dataset_memory, _, mapper = _array_free_fit_setup()

    touched = []

    def spy(name):
        def fget(self):
            touched.append(name)
            raise AssertionError(f"log_evidence touched {name}")

        return property(fget)

    for name in (
        "mask",
        "transformer",
        "residual_map",
        "normalized_residual_map",
        "chi_squared_map",
        "signal_to_noise_map",
        "chi_squared",
    ):
        monkeypatch.setattr(aa.FitInterferometer, name, spy(name))

    fit = aa.m.MockFitInterferometer(
        dataset=dataset_memory,
        inversion=aa.Inversion(dataset=dataset_memory, linear_obj_list=[mapper]),
    )

    assert np.isfinite(fit.log_evidence)
    assert fit.figure_of_merit == fit.log_evidence
    assert touched == []


def test__dirty_model_image_natural_from__matches_transformer_natural_dirty_image(
    interferometer_7,
):
    from autoarray.fit.fit_interferometer import dirty_model_image_natural_from

    dataset = interferometer_7.apply_sparse_operator()

    image = aa.Array2D(
        values=np.random.default_rng(seed=1).normal(
            size=dataset.real_space_mask.pixels_in_mask
        ),
        mask=dataset.real_space_mask,
    )

    dirty_model_image = dirty_model_image_natural_from(dataset=dataset, image=image)

    # The natural dirty image of the model visibilities, built exactly as
    # `Interferometer.dirty_image_natural` builds it from the data.
    noise_map = dataset.noise_map.array
    visibilities = dataset.transformer.visibilities_from(image=image).array
    weighted = aa.Visibilities(
        visibilities=visibilities.real * noise_map.real**-2.0
        + 1j * visibilities.imag * noise_map.imag**-2.0
    )
    expected = dataset.transformer.image_from(visibilities=weighted) / float(
        np.sum(noise_map.real**-2.0)
    )

    assert isinstance(dirty_model_image, aa.Array2D)
    assert dirty_model_image.mask is dataset.real_space_mask
    np.testing.assert_allclose(
        dirty_model_image.array,
        expected.array,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(expected.array).max(),
    )


def test__dirty_model_image_natural_from__no_sparse_operator__raises(
    interferometer_7,
):
    from autoarray.fit.fit_interferometer import dirty_model_image_natural_from

    image = aa.Array2D.ones(
        shape_native=interferometer_7.real_space_mask.shape_native,
        pixel_scales=interferometer_7.real_space_mask.pixel_scales,
    ).apply_mask(mask=interferometer_7.real_space_mask)

    with pytest.raises(aa.exc.DatasetException, match="sparse_operator"):
        dirty_model_image_natural_from(dataset=interferometer_7, image=image)


class _ProfileImageFit(aa.m.MockFitInterferometer):
    """
    A fit whose model visibilities are the Fourier transform of a real-space `image`, overriding the
    `sparse_chi_squared` hook with the data-term identity as PyAutoGalaxy / PyAutoLens light-profile fits do.
    """

    def __init__(self, dataset, image):
        super().__init__(dataset=dataset)
        self.image = image

    @property
    def sparse_chi_squared(self):
        return aa.util.inversion_interferometer.sparse_profile_terms_from(
            sparse_operator=self.dataset.sparse_operator,
            image=self.image,
            extent_index_for_masked_pixel=self.dataset.real_space_mask.extent_index_for_masked_pixel,
        )[2]


def test__fit_interferometer__array_free_dataset__sparse_chi_squared_hook():
    """
    On an array-free dataset `chi_squared` (hence `log_likelihood`) is read from the `sparse_chi_squared` hook
    when a subclass provides it, matching the in-memory fit of the same model visibilities; the maps still
    raise. The hook is not consulted when the fit has data.
    """
    pytest.importorskip("nufftax")

    dataset_memory, dataset_stream, _ = _array_free_fit_setup()

    mask = dataset_memory.real_space_mask

    image = aa.Array2D(
        values=np.random.default_rng(seed=2).normal(size=mask.pixels_in_mask),
        mask=mask,
    )

    fit_memory = aa.m.MockFitInterferometer(
        dataset=dataset_memory,
        model_data=dataset_memory.transformer.visibilities_from(image=image),
    )
    fit_stream = _ProfileImageFit(dataset=dataset_stream, image=image)

    assert fit_stream.chi_squared == pytest.approx(fit_memory.chi_squared, rel=1.0e-8)
    assert fit_stream.log_likelihood == pytest.approx(
        fit_memory.log_likelihood, rel=1.0e-8
    )
    assert fit_stream.figure_of_merit == pytest.approx(
        fit_memory.figure_of_merit, rel=1.0e-8
    )

    for name in ("residual_map", "chi_squared_map", "normalized_residual_map"):
        with pytest.raises(aa.exc.DatasetException, match="array-free"):
            getattr(fit_stream, name)

    # With data present the hook is ignored and the chi-squared-map is summed.
    fit_memory_hook = _ProfileImageFit(dataset=dataset_memory, image=None)
    fit_memory_hook._model_data = fit_memory.model_data

    assert fit_memory_hook.chi_squared == fit_memory.chi_squared

    # The base class provides no hook.
    assert aa.m.MockFitInterferometer(dataset=dataset_stream).sparse_chi_squared is None
