import autoarray as aa
from autoarray import util

import numpy as np
import pytest


def test__total_sub_pixels_2d_from():
    assert (
        util.over_sample.total_sub_pixels_2d_from(sub_size=np.array([2, 2, 2, 2, 2]))
        == 20
    )


def test__slim_index_for_sub_slim_index_via_mask_2d_from():
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=np.array([2])
        )
    )

    assert (slim_index_for_sub_slim_index == np.array([0, 0, 0, 0])).all()

    mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=np.array([2, 2, 2])
        )
    )

    assert (
        slim_index_for_sub_slim_index == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    ).all()

    mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=np.array([3, 3, 3])
        )
    )

    assert (
        slim_index_for_sub_slim_index
        == np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )
    ).all()


def test__grid_2d_slim_over_sampled_via_mask_from():
    mask = np.array([[True, True, False], [False, False, False], [True, True, False]])

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
    )

    assert (
        grid
        == np.array(
            [
                [3.75, 2.25],
                [3.75, 3.75],
                [2.25, 2.25],
                [2.25, 3.75],
                [0.75, -3.75],
                [0.75, -2.25],
                [-0.75, -3.75],
                [-0.75, -2.25],
                [0.75, -0.75],
                [0.75, 0.75],
                [-0.75, -0.75],
                [-0.75, 0.75],
                [0.75, 2.25],
                [0.75, 3.75],
                [-0.75, 2.25],
                [-0.75, 3.75],
                [-2.25, 2.25],
                [-2.25, 3.75],
                [-3.75, 2.25],
                [-3.75, 3.75],
            ]
        )
    ).all()

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=np.array([3])
    )

    assert (
        grid
        == np.array(
            [
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, False],
            [True, False, False, True],
            [False, True, False, True],
        ]
    )

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=np.array([2, 2, 2, 2, 2])
    )

    assert (
        grid
        == np.array(
            [
                [3.75, 3.75],
                [3.75, 5.25],
                [2.25, 3.75],
                [2.25, 5.25],
                [0.75, -2.25],
                [0.75, -0.75],
                [-0.75, -2.25],
                [-0.75, -0.75],
                [0.75, 0.75],
                [0.75, 2.25],
                [-0.75, 0.75],
                [-0.75, 2.25],
                [-2.25, -5.25],
                [-2.25, -3.75],
                [-3.75, -5.25],
                [-3.75, -3.75],
                [-2.25, 0.75],
                [-2.25, 2.25],
                [-3.75, 0.75],
                [-3.75, 2.25],
            ]
        )
    ).all()

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=np.array([2]), origin=(1.0, 1.0)
    )

    assert grid[0:4] == pytest.approx(
        np.array([[1.75, -0.5], [1.75, 2.5], [0.25, -0.5], [0.25, 2.5]]), 1e-4
    )


def test__from_manual_adapt_radial_bin():
    mask = aa.Mask2D.circular(shape_native=(5, 5), pixel_scales=2.0, radius=3.0)

    grid = aa.Grid2D.from_mask(mask=mask)

    sub_size = aa.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid, sub_size_list=[8, 4, 2], radial_list=[1.5, 2.5]
    )
    assert sub_size.native == pytest.approx(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 4, 2, 0],
                [0, 4, 8, 4, 0],
                [0, 2, 4, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        1.0e-4,
    )


def test__from_manual_adapt_radial_bin__centre_list_input():
    mask = aa.Mask2D.circular(shape_native=(5, 5), pixel_scales=2.0, radius=3.0)

    grid = aa.Grid2D.from_mask(mask=mask)

    sub_size = aa.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[8, 4, 2],
        radial_list=[1.5, 2.5],
        centre_list=[(0.0, -2.0), (0.0, 2.0)],
    )

    assert sub_size.native == pytest.approx(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 4, 2, 4, 0],
                [0, 8, 4, 8, 0],
                [0, 4, 2, 4, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        1.0e-4,
    )


def test__from_adapt():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, False]],
        pixel_scales=1.0,
    )

    data = aa.Array2D(values=[1.0, 2.0, 3.0], mask=mask)
    noise_map = aa.Array2D(values=[1.0, 2.0, 1.0], mask=mask)

    sub_size = aa.util.over_sample.over_sample_size_via_adapt_from(
        data=data,
        noise_map=noise_map,
        signal_to_noise_cut=1.5,
        sub_size_lower=2,
        sub_size_upper=4,
    )

    assert sub_size == pytest.approx([2, 2, 4], 1.0e-4)

    sub_size = aa.util.over_sample.over_sample_size_via_adapt_from(
        data=data,
        noise_map=noise_map,
        signal_to_noise_cut=0.5,
        sub_size_lower=2,
        sub_size_upper=4,
    )

    assert sub_size == pytest.approx([4, 4, 4], 1.0e-4)


def test__mask_2d_upscaled_from():
    mask = aa.Mask2D(
        mask=[[True, False], [False, True]],
        pixel_scales=(1.0, 1.0),
        origin=(1.0, 1.0),
    )

    mask_fine = util.over_sample.mask_2d_upscaled_from(mask_2d=mask, over_sample_size=2)

    assert mask_fine.shape_native == (4, 4)
    assert mask_fine.pixel_scales == (0.5, 0.5)
    assert mask_fine.origin == (1.0, 1.0)
    assert (
        np.array(mask_fine)
        == np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        )
    ).all()


def test__mask_2d_upscaled_from__size_one_is_identity():
    mask = aa.Mask2D(mask=[[True, False], [False, True]], pixel_scales=(1.0, 1.0))

    mask_fine = util.over_sample.mask_2d_upscaled_from(mask_2d=mask, over_sample_size=1)

    assert (np.array(mask_fine) == np.array(mask)).all()
    assert mask_fine.pixel_scales == (1.0, 1.0)


def test__sub_slim_to_fine_slim_from():
    # Two unmasked pixels side by side: their 2x2 sub-blocks interleave row-wise
    # on the fine grid, so the permutation is not the identity.
    mask = aa.Mask2D(mask=[[False, False]], pixel_scales=(1.0, 1.0))

    perm = util.over_sample.sub_slim_to_fine_slim_from(mask_2d=mask, over_sample_size=2)

    assert (perm == np.array([0, 1, 4, 5, 2, 3, 6, 7])).all()


def test__sub_slim_to_fine_slim_from__size_one_is_identity():
    mask = aa.Mask2D(
        mask=[[True, False, True], [False, False, False], [True, False, True]],
        pixel_scales=(1.0, 1.0),
    )

    perm = util.over_sample.sub_slim_to_fine_slim_from(mask_2d=mask, over_sample_size=1)

    assert (perm == np.arange(5)).all()


def test__sub_slim_to_fine_slim_from__is_bijection_and_round_trips():
    mask = aa.Mask2D.circular(shape_native=(7, 7), pixel_scales=1.0, radius=2.5)

    perm = util.over_sample.sub_slim_to_fine_slim_from(mask_2d=mask, over_sample_size=3)

    assert perm.size == mask.pixels_in_mask * 9
    assert np.array_equal(np.sort(perm), np.arange(perm.size))

    values_sub = np.arange(perm.size, dtype="float")
    fine_slim = np.zeros(perm.size)
    fine_slim[perm] = values_sub

    assert (fine_slim[perm] == values_sub).all()


def test__binned_to_convolve_size_from__uniform_k__equals_manual_reshape_mean():
    # 3 pixels, evaluation size 4 (k=2, s=2): each pixel's 4x4 block bins to 2x2
    # by the mean of each 2x2 group.
    s, n = 2, 4
    rng = np.random.default_rng(5)
    values = rng.random(3 * n * n)

    binned = util.over_sample.binned_to_convolve_size_from(
        values=values, sub_size=np.full(3, n), convolve_over_sample_size=s
    )

    manual = (
        values.reshape(3, n, n)
        .reshape(3, s, n // s, s, n // s)
        .mean(axis=(2, 4))
        .reshape(-1)
    )

    assert binned == pytest.approx(manual, abs=1.0e-14)


def test__binned_to_convolve_size_from__adaptive_k__and_identity_fast_path():
    s = 2
    sub_size = np.array([4, 2])  # k = [2, 1]
    values = np.arange(16 + 4, dtype="float")

    binned = util.over_sample.binned_to_convolve_size_from(
        values=values, sub_size=sub_size, convolve_over_sample_size=s
    )

    manual_p0 = values[:16].reshape(2, 2, 2, 2).mean(axis=(1, 3)).reshape(-1)
    manual_p1 = values[16:]

    assert binned == pytest.approx(np.concatenate([manual_p0, manual_p1]), abs=1.0e-14)

    # equal sizes: returned unchanged (identity object, byte-identical behaviour)
    same = util.over_sample.binned_to_convolve_size_from(
        values=values[:8], sub_size=np.full(2, s), convolve_over_sample_size=s
    )
    assert same is values[:8] or (same == values[:8]).all()

    # trailing dims (mapping-matrix rows)
    values_2d = np.stack([values, 2 * values], axis=1)
    binned_2d = util.over_sample.binned_to_convolve_size_from(
        values=values_2d, sub_size=sub_size, convolve_over_sample_size=s
    )
    assert binned_2d[:, 0] == pytest.approx(binned, abs=1.0e-14)
    assert binned_2d[:, 1] == pytest.approx(2 * binned, abs=1.0e-14)


def test__convolve_bin_segment_ids_from__divisibility_guard():
    with pytest.raises(aa.exc.GridException):
        util.over_sample.convolve_bin_segment_ids_from(
            sub_size=np.array([4, 3]), convolve_over_sample_size=2
        )
