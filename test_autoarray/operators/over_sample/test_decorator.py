import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures.mock.mock_decorators import (
    ndarray_1d_from,
    ndarray_2d_from,
)


def test__in_grid_2d__over_sample_uniform__out_ndarray_1d():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)

    over_sample_uniform = aa.OverSampler(mask=mask, sub_size=2)

    def oversample_mask_2d_from(mask: np.ndarray, sub_size: int) -> np.ndarray:

        oversample_mask = np.full(
            (mask.shape[0] * sub_size, mask.shape[1] * sub_size), True
        )

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if not mask[y, x]:
                    oversample_mask[
                        y * sub_size : (y + 1) * sub_size,
                        x * sub_size : (x + 1) * sub_size,
                    ] = False

        return oversample_mask

    mask_sub_2 = oversample_mask_2d_from(mask=np.array(mask), sub_size=2)

    mask_sub_2 = aa.Mask2D(mask=mask_sub_2, pixel_scales=(0.5, 0.5))

    grid = aa.Grid2D(values=grid_2d.over_sampled, mask=mask_sub_2)

    ndarray_1d_via_grid = obj.ndarray_1d_from(grid=grid)

    ndarray_1d_via_grid = aa.Array2D(values=ndarray_1d_via_grid, mask=mask_sub_2)

    ndarray_1d_via_grid = over_sample_uniform.binned_array_2d_from(
        array=ndarray_1d_via_grid,
    )

    assert isinstance(ndarray_1d, aa.Array2D)
    assert (ndarray_1d == ndarray_1d_via_grid).all()


def test__in_grid_2d__over_sample_uniform__binned_false_returns_sub_values():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)

    obj = aa.m.MockGrid2DLikeObj()

    values_sub = obj.ndarray_1d_from(grid=grid_2d, binned=False)

    assert values_sub.shape[0] == mask.pixels_in_mask * 4

    # Binning the unbinned values by the mean of each sub-block reproduces the
    # decorator's binned output.
    binned = obj.ndarray_1d_from(grid=grid_2d)

    assert values_sub.reshape(mask.pixels_in_mask, 4).mean(axis=1) == pytest.approx(
        np.array(binned), abs=1.0e-14
    )
