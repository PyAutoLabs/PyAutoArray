import logging
import re

import numpy as np
import pytest

import autoarray as aa
from autoarray import numba_util
from autoarray.dataset import lazy_operator
from autoarray.operators import transformer

IMAGING_LOGGER = "autoarray.dataset.imaging.dataset"
INTERFEROMETER_LOGGER = "autoarray.dataset.interferometer.dataset"

ON = aa.Settings(lazy_sparse_operator="auto")


@pytest.fixture(autouse=True)
def _no_ambient_setup(monkeypatch):
    # An ambient `PYAUTO_SETUP=auto` turns the policy on; every test sets its own.
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)


def _interferometer(mask_2d_7x7, noise_imag_factor=1.0, n_vis=7):
    rng = np.random.default_rng(1)
    data = rng.normal(size=n_vis) + 1j * rng.normal(size=n_vis)
    sigma = rng.uniform(1.0, 2.0, size=n_vis)
    noise_map = sigma + 1j * sigma * noise_imag_factor
    uv_wavelengths = rng.uniform(-2.0e5, 2.0e5, size=(n_vis, 2))

    return aa.Interferometer(
        data=aa.Visibilities(visibilities=data),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=noise_map),
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
    )


def _assert_imaging_operator_equal(operator, reference):
    np.testing.assert_array_equal(
        operator.psf_precision_operator_sparse,
        reference.psf_precision_operator_sparse,
    )
    np.testing.assert_array_equal(operator.indexes, reference.indexes)
    np.testing.assert_array_equal(operator.lengths, reference.lengths)


def _assert_interferometer_operator_close(operator, reference, rtol=1.0e-12):
    np.testing.assert_allclose(
        operator.nufft_precision_operator,
        reference.nufft_precision_operator,
        rtol=rtol,
        atol=rtol * np.abs(reference.nufft_precision_operator).max(),
    )
    np.testing.assert_allclose(
        operator.dirty_image,
        reference.dirty_image,
        rtol=rtol,
        atol=rtol * np.abs(reference.dirty_image).max(),
    )
    assert operator.data_term == pytest.approx(reference.data_term, rel=rtol)
    assert operator.noise_normalization == pytest.approx(
        reference.noise_normalization, rel=rtol
    )


# Policy


def test__settings__lazy_sparse_operator__default_off_and_validated():
    assert aa.Settings().lazy_sparse_operator == "off"
    assert aa.Settings(lazy_sparse_operator="auto").lazy_sparse_operator == "auto"
    assert aa.Settings(lazy_sparse_operator=False).lazy_sparse_operator == "off"

    with pytest.raises(ValueError):
        aa.Settings(lazy_sparse_operator="sometimes")


def test__policy__config_and_pyauto_setup(monkeypatch):
    assert lazy_operator.lazy_sparse_operator_enabled() is False

    monkeypatch.setattr(
        aa.Settings,
        "_inversion_config_value",
        lambda self, key, default: "auto" if key == "lazy_sparse_operator" else default,
    )
    assert lazy_operator.lazy_sparse_operator_enabled() is True
    # An explicit Settings value wins over the config.
    assert (
        lazy_operator.lazy_sparse_operator_enabled(
            settings=aa.Settings(lazy_sparse_operator="off")
        )
        is False
    )

    monkeypatch.undo()
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)

    for value, enabled in (
        ("auto", True),
        (" AUTO ", True),
        ("", False),
        ("cpu", False),
        ("numba_cpu_sparse_fp64", False),
        ("not-a-setup", False),
    ):
        monkeypatch.setenv("PYAUTO_SETUP", value)
        assert lazy_operator.lazy_sparse_operator_enabled() is enabled, value

    monkeypatch.setenv("PYAUTO_SETUP", "auto")
    assert (
        lazy_operator.lazy_sparse_operator_enabled(
            settings=aa.Settings(lazy_sparse_operator="off")
        )
        is False
    )


def test__visibility_budget__default():
    assert lazy_operator.visibility_budget() == lazy_operator.VISIBILITY_BUDGET == 1e6


def test__is_jax_tracer():
    jax = pytest.importorskip("jax")

    assert lazy_operator.is_jax_tracer(np.ones(3)) is False

    seen = []

    def f(x):
        seen.append(lazy_operator.is_jax_tracer(x))
        return x

    jax.jit(f)(np.ones(3))

    assert seen == [True]


# Imaging


def test__imaging__policy_off__none_and_nothing_built(masked_imaging_7x7, monkeypatch):
    called = []
    monkeypatch.setattr(
        aa.Imaging,
        "apply_sparse_operator_cpu",
        lambda self: called.append(1),
    )

    assert masked_imaging_7x7.sparse_operator_for(xp=np, needs_operator=True) is None
    assert masked_imaging_7x7._lazy_sparse_operator is None
    assert called == []


def test__imaging__policy_on__built_once_cached_reused_and_logged(
    masked_imaging_7x7, caplog
):
    pytest.importorskip("numba")

    with caplog.at_level(logging.INFO, logger=IMAGING_LOGGER):
        operator = masked_imaging_7x7.sparse_operator_for(
            xp=np, needs_operator=True, settings=ON
        )

    build_lines = [
        r.message
        for r in caplog.records
        if r.message.startswith("Automatic compute setup: built")
    ]
    assert len(build_lines) == 1
    assert re.fullmatch(
        r"Automatic compute setup: built the numba sparse operator for this dataset "
        r"\(9 image pixels, \d+\.\d\d s; cached; lazy_sparse_operator: auto\)",
        build_lines[0],
    ), build_lines[0]

    from autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util import (
        SparseLinAlgImagingNumba,
    )

    assert isinstance(operator, SparseLinAlgImagingNumba)
    assert masked_imaging_7x7._lazy_sparse_operator is operator
    # The attached operator is untouched: the cache is private.
    assert masked_imaging_7x7.sparse_operator is None

    caplog.clear()

    with caplog.at_level(logging.INFO, logger=IMAGING_LOGGER):
        assert (
            masked_imaging_7x7.sparse_operator_for(
                xp=np, needs_operator=True, settings=ON
            )
            is operator
        )

    assert not any("Automatic compute setup" in r.message for r in caplog.records)

    _assert_imaging_operator_equal(
        operator, masked_imaging_7x7.apply_sparse_operator_cpu().sparse_operator
    )


def test__imaging__policy_on_via_config_and_pyauto_setup(
    masked_imaging_7x7, monkeypatch, caplog
):
    pytest.importorskip("numba")

    monkeypatch.setattr(
        aa.Settings,
        "_inversion_config_value",
        lambda self, key, default: "auto" if key == "lazy_sparse_operator" else default,
    )
    assert (
        masked_imaging_7x7.sparse_operator_for(xp=np, needs_operator=True) is not None
    )

    monkeypatch.undo()
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)

    dataset = masked_imaging_7x7.apply_over_sampling(over_sample_size_lp=2)
    dataset._lazy_sparse_operator = None
    assert dataset.sparse_operator_for(xp=np, needs_operator=True) is None

    monkeypatch.setenv("PYAUTO_SETUP", "auto")
    with caplog.at_level(logging.INFO, logger=IMAGING_LOGGER):
        assert dataset.sparse_operator_for(xp=np, needs_operator=True) is not None

    # The notice names what turned the lazy build on.
    assert any(
        r.message.startswith("Automatic compute setup: built the numba sparse operator")
        and r.message.endswith("; cached; PYAUTO_SETUP=auto)")
        for r in caplog.records
    )


def test__imaging__gates__return_none(masked_imaging_7x7, monkeypatch):
    pytest.importorskip("numba")

    # needs_operator False (e.g. MGE only).
    assert (
        masked_imaging_7x7.sparse_operator_for(xp=np, needs_operator=False, settings=ON)
        is None
    )

    # An explicit dense setup.
    dense = masked_imaging_7x7.apply_setup("numba_cpu_dense_fp64")
    assert dense.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is None
    assert dense._lazy_sparse_operator is None

    # An oversampled PSF, which apply_sparse_operator_cpu rejects.
    oversampled = masked_imaging_7x7.apply_over_sampling(
        over_sample_size_pixelization=2
    )
    oversampled.psf = aa.Convolver(
        kernel=oversampled.psf.kernel, convolve_over_sample_size=2
    )
    assert (
        oversampled.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is None
    )

    # A multiprocessing worker never builds.
    monkeypatch.setattr(numba_util, "parallel_kernel_allowed", lambda: False)
    assert (
        masked_imaging_7x7.sparse_operator_for(xp=np, needs_operator=True, settings=ON)
        is None
    )
    assert masked_imaging_7x7._lazy_sparse_operator is None


def test__imaging__worker__inherits_parent_cache(masked_imaging_7x7, monkeypatch):
    pytest.importorskip("numba")

    operator = masked_imaging_7x7.sparse_operator_for(
        xp=np, needs_operator=True, settings=ON
    )

    monkeypatch.setattr(numba_util, "parallel_kernel_allowed", lambda: False)

    assert (
        masked_imaging_7x7.sparse_operator_for(xp=np, needs_operator=True, settings=ON)
        is operator
    )


def test__imaging__jax_xp__none(masked_imaging_7x7):
    jnp = pytest.importorskip("jax.numpy")

    assert (
        masked_imaging_7x7.sparse_operator_for(xp=jnp, needs_operator=True, settings=ON)
        is None
    )
    assert masked_imaging_7x7._lazy_sparse_operator is None


def test__imaging__attached_operator_returned_as_is(masked_imaging_7x7):
    pytest.importorskip("numba")

    dataset = masked_imaging_7x7.apply_sparse_operator_cpu()

    for settings in (None, ON):
        assert (
            dataset.sparse_operator_for(xp=np, needs_operator=True, settings=settings)
            is dataset.sparse_operator
        )

    assert dataset._lazy_sparse_operator is None


def test__imaging__cache_carry_and_drop(imaging_7x7, mask_2d_7x7):
    pytest.importorskip("numba")

    dataset = imaging_7x7.apply_mask(mask=mask_2d_7x7)
    operator = dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON)
    assert operator is not None

    assert (
        dataset.apply_over_sampling(over_sample_size_lp=2)._lazy_sparse_operator
        is operator
    )

    assert dataset.apply_mask(mask=mask_2d_7x7)._lazy_sparse_operator is None
    assert dataset.apply_setup("numba_cpu_dense_fp64")._lazy_sparse_operator is None
    assert (
        dataset.trimmed_after_convolution_from(
            kernel_shape=(3, 3)
        )._lazy_sparse_operator
        is None
    )

    unmasked = imaging_7x7.apply_over_sampling(over_sample_size_lp=2)
    unmasked._lazy_sparse_operator = operator
    assert unmasked.apply_noise_scaling(mask=mask_2d_7x7)._lazy_sparse_operator is None


# Interferometer


def test__interferometer__policy_off__none_and_nothing_built(mask_2d_7x7, monkeypatch):
    dataset = _interferometer(mask_2d_7x7)

    called = []
    monkeypatch.setattr(
        aa.Interferometer,
        "apply_sparse_operator",
        lambda self, **kwargs: called.append(1),
    )

    assert dataset.sparse_operator_for(xp=np, needs_operator=True) is None
    assert dataset._lazy_sparse_operator is None
    assert called == []


def test__interferometer__policy_on__built_once_cached_matches_apply_sparse_operator(
    mask_2d_7x7, caplog
):
    pytest.importorskip("finufft")

    dataset = _interferometer(mask_2d_7x7)

    with caplog.at_level(logging.INFO, logger=INTERFEROMETER_LOGGER):
        operator = dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON)

    build_lines = [
        r.message
        for r in caplog.records
        if r.message.startswith("Automatic compute setup: built")
    ]
    assert len(build_lines) == 1
    assert re.fullmatch(
        r"Automatic compute setup: built the w-tilde sparse operator for this dataset "
        r"\(7 visibilities, \d+\.\d\d s; cached; lazy_sparse_operator: auto\)",
        build_lines[0],
    ), build_lines[0]

    assert isinstance(operator, aa.InterferometerSparseOperator)
    assert dataset.sparse_operator is None
    assert (
        dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is operator
    )

    _assert_interferometer_operator_close(
        operator, dataset.apply_sparse_operator().sparse_operator, rtol=1.0e-14
    )


def test__interferometer__above_budget__chunked_builder(
    mask_2d_7x7, monkeypatch, caplog
):
    pytest.importorskip("finufft")

    dataset = _interferometer(mask_2d_7x7)

    monkeypatch.setattr(lazy_operator, "VISIBILITY_BUDGET", 3)
    monkeypatch.setattr(lazy_operator, "CHUNK_VISIBILITIES", 3)

    chunk_lengths = []
    from_chunks = aa.Interferometer.apply_sparse_operator_from_chunks

    def spy(self, chunks, **kwargs):
        chunks = list(chunks)
        chunk_lengths.extend(len(chunk[0]) for chunk in chunks)
        return from_chunks(self, chunks, **kwargs)

    monkeypatch.setattr(aa.Interferometer, "apply_sparse_operator_from_chunks", spy)

    with caplog.at_level(logging.INFO, logger=INTERFEROMETER_LOGGER):
        operator = dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON)

    assert chunk_lengths == [3, 3, 1]
    assert any(
        "(7 visibilities, chunked, " in r.message
        and r.message.startswith("Automatic compute setup: built the w-tilde")
        for r in caplog.records
    )

    _assert_interferometer_operator_close(
        operator, dataset.apply_sparse_operator().sparse_operator
    )


def test__interferometer__unequal_real_imag_noise__none_one_time_warning_never_pooled(
    mask_2d_7x7, monkeypatch
):
    dataset = _interferometer(mask_2d_7x7, noise_imag_factor=1.5)
    noise_map = dataset.noise_map

    called = []
    monkeypatch.setattr(
        aa.Interferometer,
        "apply_sparse_operator",
        lambda self, **kwargs: called.append(kwargs),
    )

    with pytest.warns(UserWarning, match="real and imaginary noise"):
        assert (
            dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is None
        )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is None
        )

    assert called == []
    assert dataset.noise_map is noise_map
    assert dataset._lazy_sparse_operator is None


def test__interferometer__gates__return_none(mask_2d_7x7, monkeypatch):
    dataset = _interferometer(mask_2d_7x7)

    assert dataset.sparse_operator_for(xp=np, needs_operator=False, settings=ON) is None

    dense = dataset.apply_setup("numba_cpu_dense_fp64")
    assert dense.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is None

    monkeypatch.setattr(numba_util, "parallel_kernel_allowed", lambda: False)
    assert dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is None
    assert dataset._lazy_sparse_operator is None


def test__interferometer__jax_xp__none(mask_2d_7x7):
    jnp = pytest.importorskip("jax.numpy")

    dataset = _interferometer(mask_2d_7x7)

    assert dataset.sparse_operator_for(xp=jnp, needs_operator=True, settings=ON) is None


def test__interferometer__attached_operator_returned_as_is(mask_2d_7x7):
    pytest.importorskip("finufft")

    dataset = _interferometer(mask_2d_7x7).apply_sparse_operator()

    assert (
        dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON)
        is dataset.sparse_operator
    )
    assert dataset._lazy_sparse_operator is None


def test__interferometer__apply_setup_drops_cache(mask_2d_7x7):
    pytest.importorskip("finufft")

    dataset = _interferometer(mask_2d_7x7)
    assert (
        dataset.sparse_operator_for(xp=np, needs_operator=True, settings=ON) is not None
    )

    assert dataset.apply_setup("numba_cpu_dense_fp64")._lazy_sparse_operator is None
