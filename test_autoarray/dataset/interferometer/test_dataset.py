import dataclasses
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


def _random_interferometer(mask, transformer_class, n_visibilities=40, seed=5):
    rng = np.random.default_rng(seed=seed)
    data = rng.normal(size=n_visibilities) + 1j * rng.normal(size=n_visibilities)
    sigma = rng.uniform(0.5, 2.0, size=n_visibilities)

    return aa.Interferometer(
        data=aa.Visibilities(visibilities=data),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=sigma + 1j * sigma),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)) * 5.0e4,
        real_space_mask=mask,
        transformer_class=transformer_class,
    )


def test__apply_sparse_operator__populates_data_term_and_noise_normalization(
    mask_2d_7x7,
):
    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerDFT)

    sparse_operator = dataset.apply_sparse_operator(use_jax=False).sparse_operator

    data = dataset.data.array
    noise_map = dataset.noise_map.array

    assert sparse_operator.data_term == np.sum(
        data.real**2.0 / noise_map.real**2.0
    ) + np.sum(data.imag**2.0 / noise_map.imag**2.0)
    assert (
        sparse_operator.noise_normalization
        == aa.util.fit.noise_normalization_complex_from(noise_map=noise_map)
    )


def test__apply_sparse_operator__complex64_data__data_term_is_reduced_in_complex128(
    mask_2d_7x7,
):
    n_visibilities = 7
    rng = np.random.default_rng(seed=1)
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)) * 5.0e4

    values = np.full(n_visibilities, 10001.0 + 0.0j)
    sigma = np.full(n_visibilities, 1.0 + 1.0j)

    def dataset_from(dtype):
        return aa.Interferometer(
            data=aa.Visibilities(visibilities=np.asarray(values, dtype=dtype)),
            noise_map=aa.VisibilitiesNoiseMap(
                visibilities=np.asarray(sigma, dtype=dtype)
            ),
            uv_wavelengths=uv_wavelengths,
            real_space_mask=mask_2d_7x7,
            transformer_class=transformer.TransformerDFT,
        )

    dataset_c64 = dataset_from(np.complex64)
    assert dataset_c64.data.array.dtype == np.complex64

    data_term_c64 = dataset_c64.apply_sparse_operator(
        use_jax=False
    ).sparse_operator.data_term
    data_term_c128 = dataset_from(np.complex128).apply_sparse_operator(
        use_jax=False
    ).sparse_operator.data_term

    assert data_term_c128 == 700140007.0
    assert data_term_c64 == data_term_c128

    # The dataset itself is not promoted.
    assert dataset_c64.data.array.dtype == np.complex64

    pytest.importorskip("nufftax")

    terms = aa.util.inversion_interferometer.sparse_terms_from_chunks(
        [(uv_wavelengths, dataset_c64.data.array, dataset_c64.noise_map.array)],
        real_space_mask=mask_2d_7x7,
    )

    assert terms.data_term == pytest.approx(data_term_c64, rel=1.0e-12)


def test__apply_sparse_operator_from_chunks__matches_apply_sparse_operator(
    interferometer_7_lop, mask_2d_7x7
):
    pytest.importorskip("nufftax")

    for dataset in (
        interferometer_7_lop,
        _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT),
        _random_interferometer(mask_2d_7x7, transformer.TransformerDFT),
    ):
        uv_wavelengths = dataset.uv_wavelengths
        data = dataset.data.array
        noise_map = dataset.noise_map.array
        n = uv_wavelengths.shape[0]

        edges = [0, 1, n // 2, n]
        chunks = [
            (uv_wavelengths[k0:k1], data[k0:k1], noise_map[k0:k1])
            for k0, k1 in zip(edges[:-1], edges[1:])
        ]

        dataset_one_shot = dataset.apply_sparse_operator()
        dataset_chunked = dataset.apply_sparse_operator_from_chunks(chunks)

        one_shot = dataset_one_shot.sparse_operator
        chunked = dataset_chunked.sparse_operator

        np.testing.assert_allclose(
            chunked.nufft_precision_operator,
            one_shot.nufft_precision_operator,
            rtol=1.0e-12,
            atol=1.0e-12 * np.abs(one_shot.nufft_precision_operator).max(),
        )
        np.testing.assert_allclose(
            chunked.dirty_image,
            one_shot.dirty_image,
            rtol=1.0e-12,
            atol=1.0e-12 * np.abs(one_shot.dirty_image).max(),
        )
        assert chunked.dirty_image.shape == one_shot.dirty_image.shape
        assert chunked.data_term == pytest.approx(one_shot.data_term, rel=1.0e-12)
        assert chunked.noise_normalization == pytest.approx(
            one_shot.noise_normalization, rel=1.0e-12
        )
        assert chunked.batch_size == one_shot.batch_size

        assert dataset_chunked.data is dataset.data
        assert dataset_chunked.transformer is dataset.transformer


def test__apply_sparse_operator_from_chunks__inversion_is_sparse_and_matches(
    mask_2d_7x7,
):
    pytest.importorskip("nufftax")

    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT)

    chunks = [
        (dataset.uv_wavelengths[k0:k1], dataset.data[k0:k1], dataset.noise_map[k0:k1])
        for k0, k1 in ((0, 13), (13, 40))
    ]

    grid = aa.Grid2D.from_mask(mask=mask_2d_7x7, over_sample_size=1)
    mesh = aa.mesh.Delaunay(pixels=9)
    image_mesh_grid = aa.image_mesh.Overlay(shape=(3, 3)).image_plane_mesh_grid_from(
        mask=mask_2d_7x7, adapt_data=None
    )
    mapper = aa.Mapper(
        interpolator=mesh.interpolator_from(
            source_plane_data_grid=grid, source_plane_mesh_grid=image_mesh_grid
        ),
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    inversion_one_shot = aa.Inversion(
        dataset=dataset.apply_sparse_operator(), linear_obj_list=[mapper]
    )
    inversion_chunked = aa.Inversion(
        dataset=dataset.apply_sparse_operator_from_chunks(chunks),
        linear_obj_list=[mapper],
    )

    assert isinstance(inversion_chunked, aa.InversionInterferometerSparse)

    assert inversion_chunked.fast_chi_squared == pytest.approx(
        inversion_one_shot.fast_chi_squared, rel=1.0e-10
    )
    assert inversion_chunked.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_one_shot.log_det_curvature_reg_matrix_term, rel=1.0e-10
    )


def test__apply_sparse_operator_from_chunks__unequal_real_imag_noise__raises(
    mask_2d_7x7,
):
    pytest.importorskip("nufftax")

    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT)

    noise_map = dataset.noise_map.array.copy()
    noise_map[30] = noise_map[30].real + 1.5j * noise_map[30].real

    chunks = [
        (dataset.uv_wavelengths[k0:k1], dataset.data.array[k0:k1], noise_map[k0:k1])
        for k0, k1 in ((0, 20), (20, 40))
    ]

    with pytest.raises(aa.exc.DatasetException):
        dataset.apply_sparse_operator_from_chunks(chunks)


def _chunks_of(dataset, edges):
    return [
        (
            dataset.uv_wavelengths[k0:k1],
            dataset.data.array[k0:k1],
            dataset.noise_map.array[k0:k1],
        )
        for k0, k1 in zip(edges[:-1], edges[1:])
    ]


def test__from_stream__array_free_dataset_carries_terms_and_operator(mask_2d_7x7):
    pytest.importorskip("nufftax")

    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT)
    chunks = _chunks_of(dataset, [0, 13, 40])

    dataset_stream = aa.Interferometer.from_stream(chunks, mask_2d_7x7)

    assert dataset_stream.data is None
    assert dataset_stream.noise_map is None
    assert dataset_stream.uv_wavelengths is None
    assert dataset_stream.transformer is None
    assert dataset_stream.is_array_free
    assert dataset_stream.real_space_mask is mask_2d_7x7
    assert dataset_stream.mask is mask_2d_7x7
    assert dataset_stream.shape_slim is None

    terms = dataset_stream.sparse_terms

    assert terms.n_vis == 40
    assert terms.shape_native == mask_2d_7x7.shape_native
    assert terms.pixel_scales == mask_2d_7x7.pixel_scales
    assert terms.origin == mask_2d_7x7.origin
    assert terms.eps == 1.0e-12
    assert terms.transformer_class_name == "TransformerNUFFT"

    one_shot = dataset.apply_sparse_operator().sparse_operator

    assert dataset_stream.sparse_operator.data_term == pytest.approx(
        one_shot.data_term, rel=1.0e-12
    )
    assert dataset_stream.sparse_operator.noise_normalization == pytest.approx(
        one_shot.noise_normalization, rel=1.0e-12
    )

    dataset_terms = aa.Interferometer.from_sparse_terms(terms, mask_2d_7x7)

    assert dataset_terms.transformer is None
    assert dataset_terms.data is None
    assert dataset_terms.sparse_terms is terms
    assert dataset_terms.sparse_operator.data_term == terms.data_term


def test__from_sparse_terms__mask_shape_mismatch__raises(mask_2d_7x7):
    pytest.importorskip("nufftax")

    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT)
    terms = aa.util.inversion_interferometer.sparse_terms_from_chunks(
        _chunks_of(dataset, [0, 40]), real_space_mask=mask_2d_7x7
    )

    other_mask = aa.Mask2D.circular(shape_native=(10, 10), pixel_scales=1.0, radius=3.0)

    with pytest.raises(aa.exc.DatasetException):
        aa.Interferometer.from_sparse_terms(terms, other_mask)


def test__from_sparse_terms__mask_pixel_scales_and_origin_mismatch__raises(mask_2d_7x7):
    pytest.importorskip("nufftax")

    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT)
    terms = aa.util.inversion_interferometer.sparse_terms_from_chunks(
        _chunks_of(dataset, [0, 40]), real_space_mask=mask_2d_7x7
    )

    # Same shape_native, different pixel_scales / origin.
    for name, kwargs in (
        ("pixel_scales", dict(pixel_scales=2.0 * mask_2d_7x7.pixel_scales[0])),
        (
            "origin",
            dict(pixel_scales=mask_2d_7x7.pixel_scales, origin=(0.5, -0.5)),
        ),
    ):
        other_mask = aa.Mask2D(mask=np.asarray(mask_2d_7x7), **kwargs)

        with pytest.raises(aa.exc.DatasetException, match=name):
            aa.Interferometer.from_sparse_terms(terms, other_mask)

    # Unrecorded provenance skips the check.
    terms_unrecorded = dataclasses.replace(terms, pixel_scales=None, origin=None)
    aa.Interferometer.from_sparse_terms(
        terms_unrecorded,
        aa.Mask2D(mask=np.asarray(mask_2d_7x7), pixel_scales=5.0, origin=(1.0, 1.0)),
    )


def test__array_free__array_properties_raise_typed_exception(mask_2d_7x7):
    pytest.importorskip("nufftax")

    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT)
    dataset_stream = aa.Interferometer.from_stream(
        _chunks_of(dataset, [0, 40]), mask_2d_7x7
    )

    for name in (
        "amplitudes",
        "phases",
        "uv_distances",
        "dirty_image",
        "dirty_noise_map",
        "signal_to_noise_map",
        "dirty_signal_to_noise_map",
    ):
        with pytest.raises(aa.exc.DatasetException, match="array-free"):
            getattr(dataset_stream, name)

    with pytest.raises(aa.exc.DatasetException, match="array-free"):
        dataset_stream.psf_precision_operator_from()

    with pytest.raises(aa.exc.DatasetException, match="array-free"):
        dataset_stream.apply_sparse_operator()

    with pytest.raises(aa.exc.DatasetException, match="array-free"):
        dataset_stream.apply_sparse_operator_from_chunks(_chunks_of(dataset, [0, 40]))


def test__dirty_image_natural_and_dirty_beam__stream_and_in_memory_agree(
    mask_2d_7x7,
):
    pytest.importorskip("nufftax")

    for transformer_class in (transformer.TransformerNUFFT, transformer.TransformerDFT):
        dataset = _random_interferometer(mask_2d_7x7, transformer_class)

        dataset_stream = aa.Interferometer.from_stream(
            _chunks_of(dataset, [0, 7, 40]),
            mask_2d_7x7,
            transformer_class=transformer_class,
        )

        weights = dataset.noise_map.array.real**-2.0

        # The in-memory path computes the images from its arrays (no `sparse_terms`).
        assert getattr(dataset, "sparse_terms", None) is None

        for name in ("dirty_image_natural", "dirty_beam"):
            in_memory = getattr(dataset, name)
            streamed = getattr(dataset_stream, name)

            assert streamed.shape_native == (7, 7)
            np.testing.assert_allclose(
                streamed.array,
                in_memory.array,
                rtol=1.0e-12,
                atol=1.0e-12 * np.abs(in_memory.array).max(),
                err_msg=name,
            )

        # The natural dirty image is the weighted adjoint normalised by sum(w), not the
        # unweighted `dirty_image`, which is unchanged.
        np.testing.assert_allclose(
            dataset.dirty_image_natural.array * np.sum(weights),
            dataset.apply_sparse_operator().sparse_operator.dirty_image,
            rtol=1.0e-10,
            atol=1.0e-10 * np.abs(dataset.dirty_image_natural.array).max() * np.sum(weights),
        )
        np.testing.assert_array_equal(
            dataset.dirty_image.array,
            dataset.transformer.image_from(visibilities=dataset.data).array,
        )


def test__apply_sparse_operator_from_chunks__result_carries_sparse_terms(mask_2d_7x7):
    pytest.importorskip("nufftax")

    dataset = _random_interferometer(mask_2d_7x7, transformer.TransformerNUFFT)

    dataset_chunked = dataset.apply_sparse_operator_from_chunks(
        _chunks_of(dataset, [0, 20, 40])
    )

    terms = dataset_chunked.sparse_terms

    assert isinstance(terms, aa.SparseTerms)
    assert terms.n_vis == 40
    assert terms.transformer_class_name == "TransformerNUFFT"
    assert terms.data_term == dataset_chunked.sparse_operator.data_term

    # The in-memory `apply_sparse_operator` path records no terms.
    assert dataset.apply_sparse_operator().sparse_terms is None
