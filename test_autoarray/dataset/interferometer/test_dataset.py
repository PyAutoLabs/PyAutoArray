import numpy as np

import autoarray as aa
import pytest

from autoarray.operators import transformer
from pathlib import Path

test_data_path = Path(Path(__file__).resolve().parent) / "files"


def test__dirty_image__shape_native_matches_real_space_mask(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    assert dataset.dirty_image.shape_native == (7, 7)
    assert (dataset.transformer.image_from(visibilities=dataset.data)).all()


def test__dirty_noise_map__shape_native_matches_real_space_mask(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    assert dataset.dirty_noise_map.shape_native == (7, 7)
    assert (dataset.transformer.image_from(visibilities=dataset.noise_map)).all()


def test__dirty_signal_to_noise_map__shape_native_matches_real_space_mask(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    assert dataset.dirty_signal_to_noise_map.shape_native == (7, 7)
    assert (
        dataset.transformer.image_from(visibilities=dataset.signal_to_noise_map)
    ).all()


def test__from_fits__raise_error_dft_visibilities_limit__threads_kwarg(
    tmp_path, mask_2d_7x7
):
    """``from_fits`` must forward ``raise_error_dft_visibilities_limit`` to the
    ``Interferometer`` constructor so callers loading large DFT-based datasets can opt out
    of the >10,000-visibility safety check (e.g. for profiling the JAX-traceable DFT path).
    """
    from astropy.io import fits

    n_visibilities = 10_001
    visibilities = np.ones((n_visibilities, 2), dtype=np.float64)
    noise_map = np.ones((n_visibilities, 2), dtype=np.float64)
    uv_wavelengths = np.zeros((n_visibilities, 2), dtype=np.float64)

    data_path = tmp_path / "data.fits"
    noise_map_path = tmp_path / "noise_map.fits"
    uv_path = tmp_path / "uv_wavelengths.fits"

    for arr, path in (
        (visibilities, data_path),
        (noise_map, noise_map_path),
        (uv_wavelengths, uv_path),
    ):
        fits.PrimaryHDU(data=arr).writeto(path, overwrite=True)

    with pytest.raises(aa.exc.DatasetException):
        aa.Interferometer.from_fits(
            data_path=data_path,
            noise_map_path=noise_map_path,
            uv_wavelengths_path=uv_path,
            real_space_mask=mask_2d_7x7,
            transformer_class=transformer.TransformerDFT,
        )

    dataset = aa.Interferometer.from_fits(
        data_path=data_path,
        noise_map_path=noise_map_path,
        uv_wavelengths_path=uv_path,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
        raise_error_dft_visibilities_limit=False,
    )

    assert dataset.uv_wavelengths.shape[0] == n_visibilities
    assert type(dataset.transformer) == transformer.TransformerDFT


def test__from_fits__all_files_in_one_fits__load_using_different_hdus(mask_2d_7x7):
    dataset = aa.Interferometer.from_fits(
        real_space_mask=mask_2d_7x7,
        data_path=Path(test_data_path) / "3x2_multiple_hdu.fits",
        visibilities_hdu=0,
        noise_map_path=Path(test_data_path) / "3x2_multiple_hdu.fits",
        noise_map_hdu=1,
        uv_wavelengths_path=Path(test_data_path) / "3x2_multiple_hdu.fits",
        uv_wavelengths_hdu=2,
    )

    assert (dataset.data == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])).all()
    assert (dataset.noise_map == np.array([2.0 + 2.0j, 2.0 + 2.0j, 2.0 + 2.0j])).all()
    assert (dataset.uv_wavelengths[:, 0] == 3.0 * np.ones(3)).all()
    assert (dataset.uv_wavelengths[:, 1] == 3.0 * np.ones(3)).all()


def test__output_all_arrays(mask_2d_7x7, tmp_path):
    test_data_path = Path(Path(__file__).resolve().parent) / "files"

    dataset = aa.Interferometer.from_fits(
        real_space_mask=mask_2d_7x7,
        data_path=Path(test_data_path) / "3x2_ones_twos.fits",
        noise_map_path=Path(test_data_path) / "3x2_threes_fours.fits",
        uv_wavelengths_path=Path(test_data_path) / "3x2_fives_sixes.fits",
    )

    from autoarray.dataset.plot.interferometer_plots import fits_interferometer

    fits_interferometer(
        dataset=dataset,
        data_path=tmp_path / "data.fits",
        noise_map_path=tmp_path / "noise_map.fits",
        uv_wavelengths_path=tmp_path / "uv_wavelengths.fits",
        overwrite=True,
    )

    dataset = aa.Interferometer.from_fits(
        real_space_mask=mask_2d_7x7,
        data_path=tmp_path / "data.fits",
        noise_map_path=tmp_path / "noise_map.fits",
        uv_wavelengths_path=tmp_path / "uv_wavelengths.fits",
    )

    assert (dataset.data == np.array([1.0 + 2.0j, 1.0 + 2.0j, 1.0 + 2.0j])).all()
    assert (dataset.noise_map == np.array([3.0 + 4.0j, 3.0 + 4.0j, 3.0 + 4.0j])).all()
    assert (dataset.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
    assert (dataset.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()


def test__transformer__dft_class__returns_transformer_dft_instance(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    interferometer_7 = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
    )

    assert type(interferometer_7.transformer) == transformer.TransformerDFT


def test__transformer__nufft_class__returns_transformer_nufft_instance(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    interferometer_7 = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerNUFFT,
    )

    assert type(interferometer_7.transformer) == transformer.TransformerNUFFT


def test__apply_sparse_operator__dft_and_nufft_dirty_image_match(mask_2d_7x7):
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
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
    ).apply_sparse_operator(use_jax=False)

    dataset_nufft = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerNUFFT,
    ).apply_sparse_operator(use_jax=False)

    assert dataset_nufft.sparse_operator.dirty_image == pytest.approx(
        dataset_dft.sparse_operator.dirty_image, 1.0e-4
    )


def test__apply_sparse_operator__unequal_real_imag_noise__raises_exception(mask_2d_7x7):
    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )

    noise_map_array = np.ones((n_visibilities, 2), dtype=np.float64)
    noise_map_array[2, 1] = 2.0

    noise_map = aa.VisibilitiesNoiseMap(visibilities=noise_map_array)
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
    )

    with pytest.raises(aa.exc.DatasetException):
        dataset.apply_sparse_operator(use_jax=False)


def test__apply_sparse_operator__non_uniform_but_equal_real_imag_noise__is_applied(
    mask_2d_7x7,
):
    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )

    sigma = np.array([1.0, 2.0, 0.5, 3.0, 1.5], dtype=np.float64)
    noise_map = aa.VisibilitiesNoiseMap(visibilities=np.stack((sigma, sigma), axis=-1))
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
    ).apply_sparse_operator(use_jax=False)

    assert dataset.sparse_operator is not None


def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=aa.Visibilities.ones(shape_slim=(19,)),
        noise_map=2.0 * aa.Visibilities.ones(shape_slim=(19,)),
        uv_wavelengths=3.0 * np.ones((19, 2)),
        real_space_mask=mask_2d_7x7,
    )

    real_space_mask = aa.Mask2D.all_false(
        shape_native=(19, 19),
        pixel_scales=1.0,
        invert=True,
    )
    real_space_mask[9, 9] = False

    assert (dataset.data == 1.0 + 1.0j * np.ones((19,))).all()
    assert (dataset.noise_map == 2.0 + 2.0j * np.ones((19,))).all()
    assert (dataset.uv_wavelengths == 3.0 * np.ones((19, 2))).all()


def test__apply_sparse_operator__disable_jax_overrides_an_explicit_use_jax(
    monkeypatch, mask_2d_7x7
):
    # `PYAUTO_DISABLE_JAX=1` is a harness-level override, not a preference: it is
    # the documented way to force the NumPy path and the smoke profiles set it so
    # a fast run does not pay a JIT compile (2.3-3.2 s per interferometer script).
    # A script demonstrating the production path says `use_jax=True`, and that
    # must not defeat the harness.
    n_visibilities = 5
    rng = np.random.default_rng(seed=0)
    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    recorded = []
    original = aa.Interferometer.psf_precision_operator_from

    def spy(self, *args, use_jax=False, **kwargs):
        recorded.append(use_jax)
        # Always compute on the NumPy path, so the assertion is about the flag
        # that arrives here and never about whether JAX is installed.
        return original(self, *args, use_jax=False, **kwargs)

    monkeypatch.setattr(aa.Interferometer, "psf_precision_operator_from", spy)

    def dataset():
        return aa.Interferometer(
            data=data,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            real_space_mask=mask_2d_7x7,
            transformer_class=transformer.TransformerDFT,
        )

    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    dataset().apply_sparse_operator(use_jax=True)

    assert recorded == [False]

    # Every other value of the variable, and its absence, leave the caller's
    # choice alone -- the predicate compares against the exact string "1".
    for value in ["0", "true", "True"]:
        monkeypatch.setenv("PYAUTO_DISABLE_JAX", value)
        dataset().apply_sparse_operator(use_jax=True)

    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)
    dataset().apply_sparse_operator(use_jax=True)

    assert recorded == [False, True, True, True, True]

    # The default builder is the type-1 NUFFT, which runs on JAX too (nufftax is a JAX library),
    # so the same kill switch has to demote it to the NumPy brute force rather than to the JAX
    # one. Nothing above tests that: `use_jax` only ever selected between the two brute forces.
    monkeypatch.setattr(aa.Interferometer, "psf_precision_operator_from", original)

    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    operator_under_kill_switch = np.asarray(dataset().psf_precision_operator_from())

    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)
    operator_via_numpy = np.asarray(
        dataset().psf_precision_operator_from(method="numpy")
    )

    np.testing.assert_array_equal(operator_under_kill_switch, operator_via_numpy)


def _interferometer_for_precision_operator(mask_2d_7x7, transformer_class):
    n_visibilities = 5
    rng = np.random.default_rng(seed=0)

    return aa.Interferometer(
        data=aa.Visibilities(
            visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
        ),
        noise_map=aa.VisibilitiesNoiseMap(
            visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
        ),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)).astype(np.float64),
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer_class,
    )


def test__psf_precision_operator_from__nufft_default_matches_the_numpy_brute_force(
    mask_2d_7x7,
):
    dataset = _interferometer_for_precision_operator(
        mask_2d_7x7, transformer.TransformerDFT
    )

    operator_via_numpy = np.asarray(dataset.psf_precision_operator_from(method="numpy"))
    operator_default = np.asarray(dataset.psf_precision_operator_from())

    # Mixed tolerance: a type-1 NUFFT bounds its error against the sum of the weights, so the
    # near-zero entries carry no relative accuracy guarantee and need the absolute floor.
    np.testing.assert_allclose(
        operator_default,
        operator_via_numpy,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(operator_via_numpy[0, 0]),
    )

    # `nufft_chunk_size` is a memory ceiling, not an approximation.
    np.testing.assert_allclose(
        np.asarray(dataset.psf_precision_operator_from(nufft_chunk_size=2)),
        operator_default,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(operator_via_numpy[0, 0]),
    )


def test__psf_precision_operator_from__eps_and_chunk_size_default_to_the_transformers(
    mask_2d_7x7, monkeypatch
):
    recorded = []

    original = aa.util.inversion_interferometer.nufft_precision_operator_from

    def spy(*args, eps, chunk_size, **kwargs):
        recorded.append((eps, chunk_size))
        return original(*args, eps=eps, chunk_size=chunk_size, **kwargs)

    monkeypatch.setattr(
        aa.util.inversion_interferometer, "nufft_precision_operator_from", spy
    )

    # A `TransformerNUFFT` has already chosen an accuracy and a memory ceiling; the precision
    # operator spreads the same visibilities with the same library, so it inherits them.
    dataset_nufft = _interferometer_for_precision_operator(
        mask_2d_7x7, transformer.TransformerNUFFT
    )
    dataset_nufft.transformer.eps = 1.0e-9
    dataset_nufft.transformer.chunk_size = 3

    dataset_nufft.psf_precision_operator_from()

    assert recorded[-1] == (1.0e-9, 3)

    # A `TransformerDFT` has neither, so the builder's own defaults are used.
    _interferometer_for_precision_operator(
        mask_2d_7x7, transformer.TransformerDFT
    ).psf_precision_operator_from()

    assert recorded[-1] == (1.0e-12, None)

    # An explicit value always wins over both.
    dataset_nufft.psf_precision_operator_from(eps=1.0e-11, nufft_chunk_size=4)

    assert recorded[-1] == (1.0e-11, 4)


def test__apply_sparse_operator__method_and_nufft_kwargs_reach_the_builder(mask_2d_7x7):
    dataset = _interferometer_for_precision_operator(
        mask_2d_7x7, transformer.TransformerDFT
    )

    operator_via_numpy = np.asarray(dataset.psf_precision_operator_from(method="numpy"))

    dataset_via_numpy = dataset.apply_sparse_operator(method="numpy")
    dataset_default = dataset.apply_sparse_operator()
    dataset_chunked = dataset.apply_sparse_operator(nufft_chunk_size=2, eps=1.0e-12)

    # The operator only keeps `Khat`, so the plumbing is checked through it: routing to the brute
    # force and to the NUFFT must give the same operator to the pin's tolerance.
    np.testing.assert_allclose(
        np.asarray(dataset_default.sparse_operator.Khat),
        np.asarray(dataset_via_numpy.sparse_operator.Khat),
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(operator_via_numpy[0, 0]),
    )
    np.testing.assert_allclose(
        np.asarray(dataset_chunked.sparse_operator.Khat),
        np.asarray(dataset_default.sparse_operator.Khat),
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(operator_via_numpy[0, 0]),
    )
