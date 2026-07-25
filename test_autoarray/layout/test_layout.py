import os

import numpy as np
import autoarray as aa


def test__layout_1d__extract_overscan_array_1d_from():
    array = aa.Array1D.no_mask(values=[0.0, 1.0, 2.0], pixel_scales=1.0)

    layout_1d = aa.Layout1D(shape_1d=array.shape, overscan=(0, 1))

    extracted_array = layout_1d.extract_overscan_array_1d_from(array=array)

    assert (extracted_array == np.array([[0.0]])).all()

    layout_1d = aa.Layout1D(shape_1d=array.shape, overscan=(0, 2))

    extracted_array = layout_1d.extract_overscan_array_1d_from(array=array)

    assert (extracted_array.native == np.array([0.0, 1.0])).all()

    layout_1d = aa.Layout1D(shape_1d=array.shape, overscan=(2, 3))

    extracted_array = layout_1d.extract_overscan_array_1d_from(array=array)

    assert (extracted_array.native == np.array([2.0])).all()


def test__layout_2d__extract_parallel_overscan_array_2d_from():
    array = aa.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ],
        pixel_scales=1.0,
    )

    layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 1, 0, 1))

    extracted_array = layout_2d.extract_parallel_overscan_array_2d_from(array=array)

    assert (extracted_array == np.array([[0.0]])).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 3, 0, 2))

    extracted_array = layout_2d.extract_parallel_overscan_array_2d_from(array=array)

    assert (
        extracted_array.native == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
    ).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 4, 2, 3))

    extracted_array = layout_2d.extract_parallel_overscan_array_2d_from(array=array)

    assert (extracted_array.native == np.array([[2.0], [5.0], [8.0], [11.0]])).all()


def test__layout_2d__parallel_overscan_binned_array_1d_from():
    array = aa.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ],
        pixel_scales=1.0,
    )

    layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 1, 0, 1))

    binned_line = layout_2d.parallel_overscan_binned_array_1d_from(array=array)

    assert (binned_line == np.array([0.0])).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 3, 0, 2))

    binned_line = layout_2d.parallel_overscan_binned_array_1d_from(array=array)

    assert (binned_line == np.array([0.5, 3.5, 6.5])).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 4, 2, 3))

    binned_line = layout_2d.parallel_overscan_binned_array_1d_from(array=array)

    assert (binned_line == np.array([2.0, 5.0, 8.0, 11.0])).all()


def test__layout_2d__extract_serial_overscan_array_from():
    array = aa.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ],
        pixel_scales=1.0,
    )

    layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 1, 0, 1))

    extracted_array = layout_2d.extract_serial_overscan_array_from(array=array)

    assert (extracted_array == np.array([[0.0]])).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 3, 0, 2))

    extracted_array = layout_2d.extract_serial_overscan_array_from(array=array)

    assert (
        extracted_array.native == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
    ).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 4, 2, 3))

    extracted_array = layout_2d.extract_serial_overscan_array_from(array=array)

    assert (extracted_array.native == np.array([[2.0], [5.0], [8.0], [11.0]])).all()


def test__layout_2d__serial_overscan_binned_array_1d_from():
    array = aa.Array2D.no_mask(
        values=[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ],
        pixel_scales=1.0,
    )

    layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 1, 0, 1))

    binned_lined = layout_2d.serial_overscan_binned_array_1d_from(array=array)

    assert (binned_lined == np.array([0.0])).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 3, 0, 2))

    binned_lined = layout_2d.serial_overscan_binned_array_1d_from(array=array)

    assert (binned_lined == np.array([3.0, 4.0])).all()

    layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 4, 2, 3))

    binned_lined = layout_2d.serial_overscan_binned_array_1d_from(array=array)

    assert (binned_lined == np.array([6.5])).all()
