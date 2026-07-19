import numpy as np

import autoarray as aa

from autoarray.util.dataset_util import (
    cap_array_2d_for_small_datasets,
    SMALL_DATASETS_SHAPE_NATIVE,
    SMALL_DATASETS_PIXEL_SCALES,
)


def _array_2d(shape, fill=1.0, pixel_scales=0.08):
    return aa.Array2D.no_mask(
        values=fill * np.ones(shape), pixel_scales=pixel_scales
    )


def test__env_unset__returns_inputs_unchanged(monkeypatch):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    array = _array_2d((150, 150))
    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is array
    assert pixel_scales == 0.08


def test__env_set__shape_already_at_cap__returns_inputs_unchanged(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    array = _array_2d(SMALL_DATASETS_SHAPE_NATIVE, pixel_scales=0.08)
    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is array
    assert pixel_scales == 0.08


def test__env_set__shape_below_cap__returns_inputs_unchanged(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    array = _array_2d((10, 10), pixel_scales=0.08)
    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is array
    assert pixel_scales == 0.08


def test__env_set__shape_above_cap__center_crops_and_overrides_pixel_scales(
    monkeypatch,
):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    raw = np.arange(150 * 150, dtype=float).reshape(150, 150)
    array = aa.Array2D.no_mask(values=raw, pixel_scales=0.08)

    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is not array
    assert result.shape_native == SMALL_DATASETS_SHAPE_NATIVE
    assert pixel_scales == SMALL_DATASETS_PIXEL_SCALES

    cap_h, cap_w = SMALL_DATASETS_SHAPE_NATIVE
    h0, w0 = (150 - cap_h) // 2, (150 - cap_w) // 2
    expected = raw[h0:h0 + cap_h, w0:w0 + cap_w]
    assert (result.native.array == expected).all()


def test__env_set__non_square_above_cap__center_crops_to_16x16(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    array = _array_2d((100, 50))
    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result.shape_native == SMALL_DATASETS_SHAPE_NATIVE
    assert pixel_scales == SMALL_DATASETS_PIXEL_SCALES
