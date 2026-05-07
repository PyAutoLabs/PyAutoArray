import autoarray as aa
import numpy as np
import pytest

from pathlib import Path

test_data_path = Path(__file__).resolve().parent / "files"


def test__array_1d_slim_from():
    mask_1d = np.array([False, False, False, False])

    array_1d_native = np.array([1.0, 2.0, 3.0, 4.0])

    array_1d_slim = aa.util.array_1d.array_1d_slim_from(
        array_1d_native=array_1d_native,
        mask_1d=mask_1d,
    )

    assert (array_1d_slim == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([True, False, False, True, False, False])

    array_1d_native = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    array_1d_slim = aa.util.array_1d.array_1d_slim_from(
        array_1d_native=array_1d_native,
        mask_1d=mask_1d,
    )

    assert (array_1d_slim == np.array([2.0, 3.0, 5.0, 6.0])).all()


def test__array_1d_native_from():
    mask_1d = np.array([False, False, False, False])

    array_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    array_1d_native = aa.util.array_1d.array_1d_native_from(
        array_1d_slim=array_1d_slim,
        mask_1d=mask_1d,
    )

    assert (array_1d_native == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([False, False, True, True, False, False])

    array_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    array_1d_native = aa.util.array_1d.array_1d_native_from(
        array_1d_slim=array_1d_slim,
        mask_1d=mask_1d,
    )

    assert (array_1d_native == np.array([1.0, 2.0, 0.0, 0.0, 3.0, 4.0])).all()
