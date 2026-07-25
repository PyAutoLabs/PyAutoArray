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


def test__from_sub_size_int():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, False]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSampler(mask=mask, sub_size=2)

    assert over_sampling.sub_size.slim == pytest.approx([2, 2, 2], 1.0e-4)
    assert over_sampling.sub_size.native == pytest.approx(
        np.array([[0, 0, 0], [0, 2, 2], [0, 0, 2]]), 1.0e-4
    )


def test__sub_pixel_areas():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, True]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSampler(mask=mask, sub_size=np.array([1, 2]))

    areas = over_sampling.sub_pixel_areas

    assert areas == pytest.approx([1.0, 0.25, 0.25, 0.25, 0.25], 1.0e-4)


def test__sub_fraction():
    mask = aa.Mask2D(
        mask=[[False, False], [True, True]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSampler(
        mask=mask, sub_size=aa.Array2D(values=[1, 2], mask=mask)
    )

    assert over_sampling.sub_fraction.slim == pytest.approx([1.0, 0.25], 1.0e-4)


def test__binned_array_2d_from():
    mask = aa.Mask2D(
        mask=[[False, False], [True, True]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSampler(
        mask=mask, sub_size=aa.Array2D(values=[2, 2], mask=mask)
    )

    arr = np.array([1.0, 5.0, 7.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    binned_array_2d = over_sampling.binned_array_2d_from(array=arr)

    assert binned_array_2d.slim == pytest.approx(np.array([5.75, 10.0]), 1.0e-4)

    over_sampling = aa.OverSampler(
        mask=mask, sub_size=aa.Array2D(values=[1, 2], mask=mask)
    )

    arr = np.array([1.0, 5.0, 7.0, 10.0, 10.0])

    binned_array_2d = over_sampling.binned_array_2d_from(array=arr)

    assert binned_array_2d.slim == pytest.approx(np.array([1.0, 8.0]), 1.0e-4)


def test__slim_index_for_sub_slim_index():
    mask = aa.Mask2D(
        mask=[[True, False, True], [False, False, False], [True, False, False]],
        pixel_scales=1.0,
        sub_size=2,
    )

    over_sampling = aa.OverSampler(mask=mask, sub_size=2)

    slim_index_for_sub_slim_index_util = (
        aa.util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=np.array(mask), sub_size=np.array([2, 2, 2, 2, 2, 2])
        )
    )

    assert (over_sampling.slim_for_sub_slim == slim_index_for_sub_slim_index_util).all()
