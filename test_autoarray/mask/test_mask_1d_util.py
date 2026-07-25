from autoarray import util

import numpy as np


def test__native_index_for_slim_index_1d_from():
    mask_1d = np.array([False, False, False, False])

    sub_native_index_for_sub_slim_index_1d = (
        util.mask_1d.native_index_for_slim_index_1d_from(mask_1d=mask_1d)
    )

    assert (sub_native_index_for_sub_slim_index_1d == np.array([0, 1, 2, 3])).all()

    mask_1d = np.array([False, False, True, False, False])

    sub_native_index_for_sub_slim_index_1d = (
        util.mask_1d.native_index_for_slim_index_1d_from(mask_1d=mask_1d)
    )

    assert (sub_native_index_for_sub_slim_index_1d == np.array([0, 1, 3, 4])).all()

    mask_1d = np.array([True, False, False, True, False, False])

    sub_native_index_for_sub_slim_index_1d = (
        util.mask_1d.native_index_for_slim_index_1d_from(mask_1d=mask_1d)
    )

    assert (sub_native_index_for_sub_slim_index_1d == np.array([1, 2, 4, 5])).all()
