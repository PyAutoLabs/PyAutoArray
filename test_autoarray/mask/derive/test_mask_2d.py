import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="derive_mask_2d_9x9")
def make_derive_mask_2d_9x9():
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

    return aa.DeriveMask2D(mask=mask_2d)


def test__unmasked_mask__all_false__returns_full_false_array(derive_mask_2d_9x9):
    assert (
        derive_mask_2d_9x9.all_false == np.full(fill_value=False, shape=(9, 9))
    ).all()


@pytest.mark.parametrize("kernel_shape_native", [(3, 3)])
def test__blurring_mask_from__3x3_kernel__matches_util_result(
    derive_mask_2d_9x9, kernel_shape_native
):
    blurring_mask_via_util = aa.util.mask_2d.blurring_mask_2d_from(
        mask_2d=derive_mask_2d_9x9.mask,
        kernel_shape_native=kernel_shape_native,
    )

    blurring_mask = derive_mask_2d_9x9.blurring_from(
        kernel_shape_native=kernel_shape_native
    )

    assert (blurring_mask == blurring_mask_via_util).all()


def test__edge_mask__9x9_ring_mask__edge_pixels_are_unmasked(derive_mask_2d_9x9):
    assert (
        derive_mask_2d_9x9.edge
        == np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )
    ).all()


def test__edge_buffed_mask__5x5_mask_with_centre_masked__buffed_mask_matches_util():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    derive_mask_2d = aa.DeriveMask2D(mask=mask)

    edge_buffed_mask_manual = aa.util.mask_2d.buffed_mask_2d_from(
        mask_2d=mask,
    ).astype("bool")

    assert (derive_mask_2d.edge_buffed == edge_buffed_mask_manual).all()


def test__border_mask__9x9_ring_mask__inner_pixels_are_masked(derive_mask_2d_9x9):
    assert (
        derive_mask_2d_9x9.border
        == np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )
    ).all()
