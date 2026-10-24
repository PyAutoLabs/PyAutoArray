"""
The CPU (finufft) NUFFT backend of the NumPy path: its orientation against a direct sum, its parity
with nufftax through `TransformerNUFFT` and the precision-operator builder, its fallback to nufftax,
and a fresh-process guard that the whole NumPy interferometer route never starts JAX.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import autoarray as aa
from autoarray.operators import nufft_cpu


def _relative_max_difference(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def _direct_type1(x, y, c, n_modes, isign):
    """`f[m2, m1] = sum_k c_k exp(isign * i(m1 * x_k + m2 * y_k))` on the centred mode grid."""
    n1, n2 = n_modes
    m1 = np.arange(n1) - n1 // 2
    m2 = np.arange(n2) - n2 // 2
    phase = np.exp(
        isign
        * 1j
        * (m1[None, None, :] * x[:, None, None] + m2[None, :, None] * y[:, None, None])
    )
    return np.sum(c[:, None, None] * phase, axis=0)


def test__nufft2d1_and_nufft2d2__match_direct_sums_in_nufftax_orientation():
    pytest.importorskip("finufft")

    rng = np.random.default_rng(seed=0)
    x = rng.uniform(-np.pi, np.pi, size=30)
    y = rng.uniform(-np.pi, np.pi, size=30)
    c = rng.normal(size=30) + 1j * rng.normal(size=30)
    n_modes = (6, 5)  # non-square, odd and even, so a transposed result cannot pass

    f = nufft_cpu.nufft2d1(x, y, c, n_modes, 1.0e-12, 1)

    assert f.shape == (5, 6)
    assert _relative_max_difference(f, _direct_type1(x, y, c, n_modes, +1)) < 1.0e-10

    grid = rng.normal(size=(5, 6)) + 1j * rng.normal(size=(5, 6))

    vis = nufft_cpu.nufft2d2(x, y, grid, 1.0e-12, -1)

    m1 = np.arange(6) - 3
    m2 = np.arange(5) - 2
    expected = np.array(
        [
            np.sum(grid * np.exp(-1j * (m1[None, :] * xk + m2[:, None] * yk)))
            for xk, yk in zip(x, y)
        ]
    )

    assert vis.shape == (30,)
    assert _relative_max_difference(vis, expected) < 1.0e-10

    # A stacked grid transforms in one call, one row per grid.
    stacked = nufft_cpu.nufft2d2(x, y, np.stack([grid, 2.0 * grid]), 1.0e-12, -1)

    assert stacked.shape == (2, 30)
    assert _relative_max_difference(stacked[1], 2.0 * expected) < 1.0e-10


def _transformer_setup(seed=3):
    rng = np.random.default_rng(seed=seed)

    mask = aa.Mask2D.circular(shape_native=(10, 11), pixel_scales=0.2, radius=0.9)
    uv_wavelengths = rng.normal(size=(53, 2)) * 2.0e5

    image = aa.Array2D(values=rng.normal(size=mask.pixels_in_mask), mask=mask)
    visibilities = aa.Visibilities(visibilities=rng.normal(size=(53, 2)))
    mapping_matrix = rng.normal(size=(mask.pixels_in_mask, 4))
    primary_beam = aa.Array2D(
        values=rng.uniform(0.2, 1.0, size=mask.pixels_in_mask), mask=mask
    )

    return mask, uv_wavelengths, image, visibilities, mapping_matrix, primary_beam


def _transforms_from(transformer, image, visibilities, mapping_matrix):
    return (
        np.asarray(transformer.visibilities_from(image=image).array),
        np.asarray(transformer.image_from(visibilities=visibilities).array),
        np.asarray(transformer.transform_mapping_matrix(mapping_matrix=mapping_matrix)),
    )


def test__transformer_nufft__numpy_path_finufft_matches_nufftax(monkeypatch):
    """
    Every NumPy-path transform of `TransformerNUFFT` on finufft matches the same transform on
    nufftax (reached through the fallback), chunked and unchunked, with and without a primary
    beam -- and matches the JAX path, which is nufftax by construction.
    """
    pytest.importorskip("finufft")
    pytest.importorskip("nufftax")
    import jax.numpy as jnp

    mask, uv_wavelengths, image, visibilities, mapping_matrix, primary_beam = (
        _transformer_setup()
    )

    for chunk_size in (None, 16):
        for beam in (None, primary_beam):
            transformer = aa.TransformerNUFFT(
                uv_wavelengths=uv_wavelengths,
                real_space_mask=mask,
                chunk_size=chunk_size,
                primary_beam=beam,
            )

            finufft_transforms = _transforms_from(
                transformer, image, visibilities, mapping_matrix
            )

            with monkeypatch.context() as m:
                m.setattr(nufft_cpu, "_load_finufft", lambda: None)
                m.setattr(nufft_cpu, "_fallback_warned", True)
                nufftax_transforms = _transforms_from(
                    transformer, image, visibilities, mapping_matrix
                )

            for finufft_transform, nufftax_transform in zip(
                finufft_transforms, nufftax_transforms
            ):
                assert (
                    _relative_max_difference(finufft_transform, nufftax_transform)
                    < 1.0e-10
                )

            jax_vis = transformer.visibilities_from(image=image, xp=jnp)

            assert (
                _relative_max_difference(
                    finufft_transforms[0], np.asarray(jax_vis.array)
                )
                < 1.0e-10
            )


def _precision_operator_inputs(seed=5):
    rng = np.random.default_rng(seed=seed)

    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=0.2, radius=0.9)
    uv_wavelengths = rng.normal(size=(200, 2)) * 2.0e5
    noise_map_real = rng.uniform(0.5, 2.0, size=200)

    return dict(
        noise_map_real=noise_map_real,
        uv_wavelengths=uv_wavelengths,
        shape_masked_pixels_2d=mask.shape_native_masked_pixels,
        grid_radians_2d=mask.derive_grid.all_false.in_radians.native.array,
    )


def _assert_matches_mixed(operator, reference):
    np.testing.assert_allclose(
        operator, reference, rtol=1.0e-10, atol=1.0e-10 * np.abs(reference[0, 0])
    )


def test__nufft_precision_operator_from__default_finufft_matches_brute_force_and_nufftax():
    pytest.importorskip("finufft")

    inputs = _precision_operator_inputs()

    operator_default = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_from(**inputs)
    )
    operator_via_np = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_from(
            method="numpy", **inputs
        )
    )

    _assert_matches_mixed(operator_default, operator_via_np)

    # Chunking only splits the linear sum.
    _assert_matches_mixed(
        np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_from(
                chunk_size=37, **inputs
            )
        ),
        operator_via_np,
    )

    pytest.importorskip("nufftax")

    operator_use_jax = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_from(
            use_jax=True, **inputs
        )
    )

    _assert_matches_mixed(operator_default, operator_use_jax)


def test__nufft_cpu__falls_back_to_nufftax_with_one_warning_when_finufft_is_missing(
    monkeypatch, caplog
):
    pytest.importorskip("nufftax")

    rng = np.random.default_rng(seed=1)
    x = rng.uniform(-np.pi, np.pi, size=20)
    y = rng.uniform(-np.pi, np.pi, size=20)
    c = rng.normal(size=20) + 0j

    monkeypatch.setattr(nufft_cpu, "_load_finufft", lambda: None)
    monkeypatch.setattr(nufft_cpu, "_fallback_warned", False)

    with caplog.at_level("WARNING"):
        f = nufft_cpu.nufft2d1(x, y, c, (6, 5), 1.0e-12, 1)
        nufft_cpu.nufft2d2(x, y, f, 1.0e-12, -1)

    assert isinstance(f, np.ndarray)
    assert f.shape == (5, 6)
    assert _relative_max_difference(f, _direct_type1(x, y, c, (6, 5), +1)) < 1.0e-10

    warnings = [r for r in caplog.records if "finufft" in r.getMessage()]

    assert len(warnings) == 1
    assert "JAX" in warnings[0].getMessage()
    assert "segmentation fault" in warnings[0].getMessage()


def test__nufft_cpu__no_backend_raises(monkeypatch):
    monkeypatch.setattr(nufft_cpu, "_load_finufft", lambda: None)
    monkeypatch.setattr(nufft_cpu, "backend_available", lambda: False)

    from autoarray.operators import transformer

    monkeypatch.setattr(transformer, "_load_nufftax", lambda: None)

    with pytest.raises(ModuleNotFoundError) as exc:
        aa.TransformerNUFFT(
            uv_wavelengths=np.array([[0.2, 1.0], [0.5, 1.1]]),
            real_space_mask=aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.1),
        )

    assert "pip install finufft" in str(exc.value)

    with pytest.raises(ModuleNotFoundError):
        nufft_cpu.nufft2d1(np.zeros(2), np.zeros(2), np.ones(2) + 0j, (4, 4))


_GUARD_SCRIPT = textwrap.dedent(
    """
    import sys

    import numpy as np

    import autoarray as aa

    rng = np.random.default_rng(seed=0)
    n_vis = 300

    mask = aa.Mask2D.circular(shape_native=(16, 16), pixel_scales=0.2, radius=1.2)
    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=rng.normal(size=(n_vis, 2))),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((n_vis, 2))),
        uv_wavelengths=rng.normal(size=(n_vis, 2)) * 2.0e5,
        real_space_mask=mask,
        transformer_class=aa.TransformerNUFFT,
    )

    operator = dataset.psf_precision_operator_from()
    dataset_sparse = dataset.apply_sparse_operator(nufft_precision_operator=operator)
    dataset.apply_sparse_operator()

    transformer = dataset.transformer
    image = aa.Array2D(values=rng.normal(size=mask.pixels_in_mask), mask=mask)
    transformer.visibilities_from(image=image)
    transformer.image_from(visibilities=dataset.data)
    transformer.transform_mapping_matrix(
        mapping_matrix=rng.normal(size=(mask.pixels_in_mask, 3))
    )

    assert dataset_sparse.sparse_operator is not None

    print("nufftax_imported", "nufftax" in sys.modules)

    backends = None
    if "jax" in sys.modules:
        from jax._src import xla_bridge

        backends = len(xla_bridge._backends)
    print("jax_imported", "jax" in sys.modules)
    print("xla_backends", backends)
    """
)


def test__numpy_interferometer_route__never_imports_nufftax_or_starts_jax():
    """
    The whole NumPy interferometer route -- precision operator, sparse operator (dirty image),
    forward, adjoint and mapping-matrix transforms -- run in a fresh process must neither import
    nufftax nor start JAX's XLA backend. A parent that has started JAX before a multiprocessing
    search forks its numba workers crashes them (PyAutoArray#654), so this is the property the
    finufft backend exists for.
    """
    pytest.importorskip("finufft")

    env = dict(os.environ)
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.pop("PYAUTO_DISABLE_JAX", None)

    result = subprocess.run(
        [sys.executable, "-c", _GUARD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr[-4000:]

    lines = dict(
        line.split(" ", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("nufftax_imported", "jax_imported", "xla_backends"))
    )

    assert lines["nufftax_imported"] == "False"
    assert lines["xla_backends"] in ("None", "0"), result.stdout


_FORK_SCRIPT = textwrap.dedent(
    """
    import multiprocessing

    import numpy as np

    from autoarray.operators import nufft_cpu


    def transform(seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-np.pi, np.pi, size=20000)
        y = rng.uniform(-np.pi, np.pi, size=20000)
        f = nufft_cpu.nufft2d1(x, y, rng.normal(size=20000) + 0j, (64, 64))
        return float(np.abs(nufft_cpu.nufft2d2(x, y, f)).sum())


    if __name__ == "__main__":
        transform(0)
        with multiprocessing.get_context("fork").Pool(2) as pool:
            pool.map(transform, range(4))
        print("forked_ok")
    """
)


def test__nufft_cpu__forked_children_after_a_multithreaded_parent_do_not_deadlock(
    tmp_path,
):
    """
    libgomp deadlocks a forked child that opens a multi-threaded OpenMP region after its parent
    ran one -- the Nautilus pattern of building the operator in the parent and transforming in
    forked workers. `nufft_cpu` pins forked children to one thread; without that this hangs.
    """
    pytest.importorskip("finufft")

    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable on this platform")

    script = tmp_path / "fork_script.py"
    script.write_text(_FORK_SCRIPT)

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    # Own session, so a deadlocked run's forked workers are killed with it rather than orphaned.
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, 9)
        process.communicate()
        pytest.fail("forked nufft_cpu workers deadlocked after a multi-threaded parent")

    assert process.returncode == 0, stderr[-4000:]
    assert "forked_ok" in stdout
