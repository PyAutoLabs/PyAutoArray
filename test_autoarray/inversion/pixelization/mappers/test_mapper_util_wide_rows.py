"""
Skipping the padding of wide mapping rows (PyAutoArray#629).

`DelaunayNN` (Sibson) mapping tables are `SIBSON_MAX_NEIGHBORS` (32) columns wide but hold at most ~15 live
entries per row. `adaptive_pixel_signals_from` and `data_slim_to_pixelization_unique_from` now visit only the
first `pix_sizes[i]` entries of every row. These tests pin that the results are bit-identical to the padded
algorithms they replaced, on wide `DelaunayNN`-shaped tables and on 3-wide `Delaunay` tables alike.
"""

import importlib.util

import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.mappers import mapper_numba_util, mapper_util

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="jax not installed"
)


def _padded_scatter_reference(
    pixels,
    pixel_weights,
    signal_scale,
    pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index,
    slim_index_for_sub_slim_index,
    adapt_data,
    exclude_zero_weight_mappings=False,
):
    """The NumPy branch of `adaptive_pixel_signals_from` before PyAutoArray#629: an `np.add.at` over every padded
    slot, with masked entries sent to a discarded out-of-bounds slot."""
    M_sub, B = pix_indexes_for_sub_slim_index.shape
    flat_pixidx = pix_indexes_for_sub_slim_index.reshape(-1)
    flat_weights = pixel_weights.reshape(-1)
    I_sub = np.repeat(np.arange(M_sub), B)
    valid = (np.arange(B)[None, :] < pix_size_for_sub_slim_index[:, None]).reshape(-1)
    if exclude_zero_weight_mappings:
        valid = valid & (flat_weights != 0)
    flat_weights = np.where(valid, flat_weights, 0.0)
    flat_pixidx = np.where(valid, flat_pixidx, pixels)
    flat_data_vals = np.take(adapt_data[slim_index_for_sub_slim_index], I_sub, axis=0)
    flat_contrib = flat_data_vals * flat_weights
    pixel_signals = np.zeros((pixels + 1,))
    pixel_counts = np.zeros((pixels + 1,))
    np.add.at(pixel_signals, flat_pixidx, flat_contrib)
    np.add.at(pixel_counts, flat_pixidx, valid.astype(float))
    pixel_signals = pixel_signals[:pixels]
    pixel_counts = pixel_counts[:pixels]
    pixel_counts = np.where(pixel_counts > 0, pixel_counts, 1.0)
    pixel_signals = pixel_signals / pixel_counts
    max_sig = np.max(pixel_signals)
    max_sig = np.where(max_sig > 0, max_sig, 1.0)
    pixel_signals = pixel_signals / max_sig
    return pixel_signals**signal_scale


def _unique_full_reset_reference(
    data_pixels,
    pix_indexes_for_sub_slim_index,
    pix_sizes_for_sub_slim_index,
    pix_weights_for_sub_slim_index,
    pix_pixels,
    sub_size,
):
    """`data_slim_to_pixelization_unique_from` before PyAutoArray#629, in plain Python: the whole `pix_check`
    array is reset for every data pixel."""
    sub_fraction = 1.0 / (sub_size**2.0)
    width = int(np.max(pix_sizes_for_sub_slim_index)) * int(np.max(sub_size)) ** 2
    data_to_pix_unique = -1 * np.ones((data_pixels, width))
    data_weights = np.zeros((data_pixels, width))
    pix_lengths = np.zeros(data_pixels)
    ip_sub_start = 0
    for ip in range(data_pixels):
        pix_check = -1 * np.ones(pix_pixels)
        pix_size = 0
        ip_sub_end = ip_sub_start + sub_size[ip] ** 2
        for ip_sub in range(ip_sub_start, ip_sub_end):
            for k in range(pix_sizes_for_sub_slim_index[ip_sub]):
                pix = pix_indexes_for_sub_slim_index[ip_sub, k]
                weight = pix_weights_for_sub_slim_index[ip_sub, k]
                if pix_check[pix] > -0.5:
                    data_weights[ip, int(pix_check[pix])] += sub_fraction[ip] * weight
                else:
                    data_to_pix_unique[ip, pix_size] = pix
                    data_weights[ip, pix_size] += sub_fraction[ip] * weight
                    pix_check[pix] = pix_size
                    pix_size += 1
        ip_sub_start = ip_sub_end
        pix_lengths[ip] = pix_size
    return data_to_pix_unique, data_weights, pix_lengths


def _padded_tables(width, min_size, max_size, data_pixels=60, sub=2, pixels=40, seed=1):
    """Sibson-shaped tables: `width` columns, `size` live entries per row (distinct pixels, positive weights
    summing to 1 with a few exact zeros), `-1` / `0.0` padding beyond `size`."""
    rng = np.random.default_rng(seed)
    sub_pixels = data_pixels * sub**2
    sizes = rng.integers(min_size, max_size + 1, size=sub_pixels).astype(np.int32)
    indexes = np.full((sub_pixels, width), -1, dtype=np.int32)
    weights = np.zeros((sub_pixels, width))
    for row, size in enumerate(sizes):
        indexes[row, :size] = rng.choice(pixels, size=size, replace=False)
        live = rng.uniform(0.0, 1.0, size=size)
        live[rng.uniform(size=size) < 0.1] = 0.0
        weights[row, :size] = live / max(live.sum(), 1.0e-300)
    return dict(
        pixels=pixels,
        indexes=indexes,
        sizes=sizes,
        weights=weights,
        slim=np.repeat(np.arange(data_pixels), sub**2),
        adapt=rng.uniform(0.0, 2.0, size=data_pixels),
        sub_size=np.full(data_pixels, sub),
        data_pixels=data_pixels,
    )


TABLES = {
    "delaunay_nn_32_wide": dict(width=32, min_size=1, max_size=15),
    "delaunay_3_wide": dict(width=3, min_size=3, max_size=3),
    "knn_10_wide_variable": dict(width=10, min_size=1, max_size=10),
}


def _signal_kwargs(tables, exclude):
    return dict(
        pixels=tables["pixels"],
        pixel_weights=tables["weights"],
        signal_scale=0.7,
        pix_indexes_for_sub_slim_index=tables["indexes"],
        pix_size_for_sub_slim_index=tables["sizes"],
        slim_index_for_sub_slim_index=tables["slim"],
        adapt_data=tables["adapt"],
        exclude_zero_weight_mappings=exclude,
    )


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("exclude", [True, False])
@pytest.mark.parametrize("table", sorted(TABLES))
def test__adaptive_pixel_signals_from__live_entries_only__bit_identical_to_padded_scatter(
    table, exclude, use_numba, monkeypatch
):
    tables = _padded_tables(**TABLES[table])
    kwargs = _signal_kwargs(tables, exclude)

    monkeypatch.setattr(mapper_util, "_use_numba", lambda: use_numba)

    signals = mapper_util.adaptive_pixel_signals_from(**kwargs)

    assert np.array_equal(signals, _padded_scatter_reference(**kwargs))


def test__adaptive_pixel_signals_from__nan_in_padding_only__ignored_like_padded_scatter(
    monkeypatch,
):
    tables = _padded_tables(**TABLES["delaunay_nn_32_wide"])
    weights = tables["weights"].copy()
    weights[:, -1] = np.nan  # padding beyond every row's size
    tables["weights"] = weights
    kwargs = _signal_kwargs(tables, exclude=True)

    for use_numba in (True, False):
        monkeypatch.setattr(mapper_util, "_use_numba", lambda: use_numba)
        signals = mapper_util.adaptive_pixel_signals_from(**kwargs)
        assert np.isfinite(signals).all()
        assert np.array_equal(signals, _padded_scatter_reference(**kwargs))


@requires_jax
def test__adaptive_pixel_signals_from__wide_rows__jax_matches_numpy():
    import jax.numpy as jnp

    tables = _padded_tables(**TABLES["delaunay_nn_32_wide"])
    kwargs = _signal_kwargs(tables, exclude=True)

    numpy_signals = mapper_util.adaptive_pixel_signals_from(**kwargs)
    jax_signals = np.asarray(mapper_util.adaptive_pixel_signals_from(**kwargs, xp=jnp))

    np.testing.assert_allclose(jax_signals, numpy_signals, rtol=1.0e-12, atol=0.0)


@pytest.mark.parametrize("table", sorted(TABLES))
def test__data_slim_to_pixelization_unique_from__touched_reset__bit_identical_to_full_reset(
    table,
):
    tables = _padded_tables(**TABLES[table])
    kwargs = dict(
        data_pixels=tables["data_pixels"],
        pix_indexes_for_sub_slim_index=tables["indexes"],
        pix_sizes_for_sub_slim_index=tables["sizes"],
        pix_weights_for_sub_slim_index=tables["weights"],
        pix_pixels=tables["pixels"],
        sub_size=tables["sub_size"],
    )

    result = mapper_numba_util.data_slim_to_pixelization_unique_from(**kwargs)
    expected = _unique_full_reset_reference(**kwargs)

    for actual, reference in zip(result, expected):
        assert actual.shape == reference.shape
        assert np.array_equal(actual, reference)


def _mappers(sub_size=2):
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=0.5, radius=2.6)
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=sub_size)
    rng = np.random.default_rng(seed=4)
    mesh_grid = aa.Grid2DIrregular(rng.uniform(-3.5, 3.5, size=(30, 2)))
    adapt_data = aa.Array2D(
        values=rng.uniform(0.0, 1.0, size=mask.pixels_in_mask), mask=mask
    )
    mappers = {}
    for name, mesh in (
        ("delaunay", aa.mesh.Delaunay(pixels=30)),
        ("delaunay_nn", aa.mesh.DelaunayNN(pixels=30)),
    ):
        interpolator = mesh.interpolator_from(
            source_plane_data_grid=grid,
            source_plane_mesh_grid=mesh_grid,
            adapt_data=adapt_data,
        )
        mappers[name] = aa.Mapper(
            interpolator=interpolator,
            regularization=aa.reg.Constant(coefficient=1.0),
        )
    return mappers


@pytest.mark.parametrize("name", ["delaunay", "delaunay_nn"])
def test__mapper__real_delaunay_and_delaunay_nn_tables__unchanged(name, monkeypatch):
    """Real `Delaunay` (3-wide) and `DelaunayNN` (32-wide) mapper tables: pixel signals through both NumPy
    routes and the unique mappings equal the padded / full-reset algorithms bit for bit.
    """
    mapper = _mappers()[name]

    indexes = np.asarray(mapper.pix_indexes_for_sub_slim_index)
    sizes = np.asarray(mapper.pix_sizes_for_sub_slim_index)
    weights = np.asarray(mapper.pix_weights_for_sub_slim_index)

    if name == "delaunay_nn":
        assert indexes.shape[1] == 32 and sizes.max() > 3

    kwargs = dict(
        pixels=mapper.pixels,
        pixel_weights=weights,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=indexes,
        pix_size_for_sub_slim_index=sizes,
        slim_index_for_sub_slim_index=np.asarray(mapper.over_sampler.slim_for_sub_slim),
        adapt_data=np.asarray(mapper.adapt_data.array),
        exclude_zero_weight_mappings=mapper.interpolator.exclude_zero_weight_mappings,
    )
    expected_signals = _padded_scatter_reference(**kwargs)

    for use_numba in (True, False):
        monkeypatch.setattr(mapper_util, "_use_numba", lambda: use_numba)
        assert np.array_equal(
            mapper.pixel_signals_from(signal_scale=1.0), expected_signals
        )

    expected_unique = _unique_full_reset_reference(
        data_pixels=mapper.over_sampler.mask.pixels_in_mask,
        pix_indexes_for_sub_slim_index=indexes,
        pix_sizes_for_sub_slim_index=sizes,
        pix_weights_for_sub_slim_index=weights,
        pix_pixels=mapper.params,
        sub_size=np.asarray(mapper.over_sampler.sub_size).astype(int),
    )
    unique = mapper.unique_mappings
    for actual, reference in zip(
        (unique.data_to_pix_unique, unique.data_weights, unique.pix_lengths),
        expected_unique,
    ):
        assert np.array_equal(actual, reference)


def test__failed_row_kept_at_size_one__nan_reaches_signals_and_unique_weights(
    monkeypatch,
):
    """A failed Sibson row carries NaN weights. At `size == 0` every consumer skipped them (the evidence came
    out finite and wrong); at `size == 1` (PyAutoArray#629) the NaN reaches the unique-mapping weights that
    build the curvature matrix, and the pixel signals that build the regularization."""
    tables = _padded_tables(**TABLES["delaunay_nn_32_wide"])
    indexes, sizes, weights = (
        tables["indexes"].copy(),
        tables["sizes"].copy(),
        tables["weights"].copy(),
    )
    failed_row = 5
    indexes[failed_row, :] = -1
    indexes[failed_row, 0] = 3
    weights[failed_row, :] = np.nan

    def outputs(size):
        sizes[failed_row] = size
        unique = mapper_numba_util.data_slim_to_pixelization_unique_from(
            data_pixels=tables["data_pixels"],
            pix_indexes_for_sub_slim_index=indexes,
            pix_sizes_for_sub_slim_index=sizes,
            pix_weights_for_sub_slim_index=weights,
            pix_pixels=tables["pixels"],
            sub_size=tables["sub_size"],
        )
        signals = mapper_util.adaptive_pixel_signals_from(
            pixels=tables["pixels"],
            pixel_weights=weights,
            signal_scale=1.0,
            pix_indexes_for_sub_slim_index=indexes,
            pix_size_for_sub_slim_index=sizes,
            slim_index_for_sub_slim_index=tables["slim"],
            adapt_data=tables["adapt"],
            exclude_zero_weight_mappings=True,
        )
        return unique[1], signals

    for use_numba in (True, False):
        monkeypatch.setattr(mapper_util, "_use_numba", lambda: use_numba)

        data_weights, signals = outputs(size=0)
        assert np.isfinite(data_weights).all() and np.isfinite(signals).all()

        data_weights, signals = outputs(size=1)
        assert np.isnan(data_weights[failed_row // 4]).any()
        assert np.isnan(signals).any()
