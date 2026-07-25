import autoarray as aa
import numpy as np


def test__sets_up_scaled_alone_grid():
    grid_slim = aa.util.grid_1d.grid_1d_slim_via_shape_slim_from(
        shape_slim=(3,),
        pixel_scales=(1.0,),
    )

    assert (grid_slim == np.array([-1.0, 0.0, 1.0])).all()


def test__grid_1d_is_actual_via_via_mask_from():
    mask = np.array([False, True, False, False])

    grid_slim = aa.util.grid_1d.grid_1d_slim_via_mask_from(
        mask_1d=mask,
        pixel_scales=(3.0,),
    )

    assert (grid_slim == np.array([-4.5, 1.5, 4.5])).all()


def test__grid_1d_slim_from():
    mask_1d = np.array([False, False, False, False])

    grid_1d_native = np.array([1.0, 2.0, 3.0, 4.0])

    grid_1d_slim = aa.util.grid_1d.grid_1d_slim_from(
        grid_1d_native=grid_1d_native,
        mask_1d=mask_1d,
    )

    assert (grid_1d_slim == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([True, False, False, True, False, False])

    grid_1d_native = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    grid_1d_slim = aa.util.grid_1d.grid_1d_slim_from(
        grid_1d_native=grid_1d_native,
        mask_1d=mask_1d,
    )

    assert (grid_1d_slim == np.array([2.0, 3.0, 5.0, 6.0])).all()


def test__grid_1d_native_from():
    mask_1d = np.array([False, False, False, False])

    grid_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    grid_1d_native = aa.util.grid_1d.grid_1d_native_from(
        grid_1d_slim=grid_1d_slim,
        mask_1d=mask_1d,
    )

    assert (grid_1d_native == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([False, False, True, True, False, False])

    grid_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    grid_1d_native = aa.util.grid_1d.grid_1d_native_from(
        grid_1d_slim=grid_1d_slim,
        mask_1d=mask_1d,
    )

    assert (grid_1d_native == np.array([1.0, 2.0, 0.0, 0.0, 3.0, 4.0])).all()
