import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="indexes_2d_9x9")
def make_indexes_2d_9x9():
    mask_2d = aa.Mask2D(
        mask=[
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    return aa.DeriveIndexes2D(mask=mask_2d)


def test__native_index_for_slim_index(indexes_2d_9x9):
    native_index_for_slim_index_2d = (
        aa.util.mask_2d.native_index_for_slim_index_2d_from(
            mask_2d=indexes_2d_9x9.mask,
        )
    )

    assert indexes_2d_9x9.native_for_slim == pytest.approx(
        native_index_for_slim_index_2d, 1e-4
    )


def test__unmasked_1d_indexes(indexes_2d_9x9):
    unmasked_pixels_util = aa.util.mask_2d.mask_slim_indexes_from(
        mask_2d=indexes_2d_9x9.mask, return_masked_indexes=False
    )

    assert indexes_2d_9x9.unmasked_slim == pytest.approx(unmasked_pixels_util, 1e-4)


def test__masked_1d_indexes(indexes_2d_9x9):
    masked_pixels_util = aa.util.mask_2d.mask_slim_indexes_from(
        mask_2d=indexes_2d_9x9.mask, return_masked_indexes=True
    )

    assert indexes_2d_9x9.masked_slim == pytest.approx(masked_pixels_util, 1e-4)


def test__edge_1d_indexes(indexes_2d_9x9):
    edge_1d_indexes_util = aa.util.mask_2d.edge_1d_indexes_from(
        mask_2d=indexes_2d_9x9.mask
    )

    assert indexes_2d_9x9.edge_slim == pytest.approx(edge_1d_indexes_util, 1e-4)
    assert indexes_2d_9x9.edge_slim.shape[0] == indexes_2d_9x9.edge_native.shape[0]


def test__edge_2d_indexes(indexes_2d_9x9):
    assert indexes_2d_9x9.edge_native[0] == pytest.approx(np.array([1, 1]), 1e-4)
    assert indexes_2d_9x9.edge_native[10] == pytest.approx(np.array([3, 3]), 1e-4)


def test__border_1d_indexes(indexes_2d_9x9):
    border_pixels_util = aa.util.mask_2d.border_slim_indexes_from(
        mask_2d=indexes_2d_9x9.mask
    )

    assert indexes_2d_9x9.border_slim == pytest.approx(border_pixels_util, 1e-4)
    assert indexes_2d_9x9.border_slim.shape[0] == indexes_2d_9x9.border_native.shape[0]


def test__border_2d_indexes(indexes_2d_9x9):
    assert indexes_2d_9x9.border_native[0] == pytest.approx(np.array([1, 1]), 1e-4)
    assert indexes_2d_9x9.border_native[10] == pytest.approx(np.array([3, 7]), 1e-4)
