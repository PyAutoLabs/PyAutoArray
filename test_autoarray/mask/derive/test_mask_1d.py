import numpy as np
import pytest

import autoarray as aa


def test__to_mask_2d():
    mask_1d = aa.Mask1D(mask=[False, True], pixel_scales=1.0)

    derive_mask_1d = aa.DeriveMask1D(mask=mask_1d)

    mask_2d = derive_mask_1d.to_mask_2d

    assert (mask_2d == np.array([[False, True]])).all()
    assert mask_2d.pixel_scales == (1.0, 1.0)
    assert mask_2d.origin == (0.0, 0.0)


def test__unmasked_mask():
    mask_1d = aa.Mask1D(
        mask=[True, False, True, False, False, False, True, False, True],
        pixel_scales=1.0,
    )

    derive_mask_1d = aa.DeriveMask1D(mask=mask_1d)

    assert (derive_mask_1d.all_false == np.full(fill_value=False, shape=(9,))).all()
