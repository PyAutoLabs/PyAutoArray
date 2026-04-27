import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "imaging"
)


def test__array_with_new_shape():
    arr = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    arr = aa.preprocess.array_with_new_shape(array=arr, new_shape=(5, 5))

    assert arr.shape_native == (5, 5)

    arr = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    arr = aa.preprocess.array_with_new_shape(array=arr, new_shape=(2, 2))

    assert arr.shape_native == (2, 2)

    arr = aa.Array2D.ones(shape_native=(10, 6), pixel_scales=1.0)

    arr = aa.preprocess.array_with_new_shape(array=arr, new_shape=(20, 14))

    assert arr.shape_native == (20, 14)


def test__array_from_electrons_per_second_to_counts():
    arr_eps = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.full(
        fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
    )

    arr_counts = aa.preprocess.array_eps_to_counts(
        array_eps=arr_eps, exposure_time_map=exposure_time_map
    )

    assert (arr_counts.native == 2.0 * np.ones((3, 3))).all()


def test__array_from_counts_to_electrons_per_second():
    arr_counts = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.full(
        fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
    )

    arr_eps = aa.preprocess.array_counts_to_eps(
        array_counts=arr_counts, exposure_time_map=exposure_time_map
    )

    assert (arr_eps.native == 0.5 * np.ones((3, 3))).all()


def test__array_from_electrons_per_second_to_adus():
    arr_eps = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.full(
        fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
    )

    arr_adus = aa.preprocess.array_eps_to_adus(
        array_eps=arr_eps, exposure_time_map=exposure_time_map, gain=2.0
    )

    assert arr_adus.native == pytest.approx(0.5 * (2.0 * np.ones((3, 3))), 1.0e-4)

    arr_adus = aa.preprocess.array_eps_to_adus(
        array_eps=arr_eps, exposure_time_map=exposure_time_map, gain=4.0
    )

    assert arr_adus.native == pytest.approx(0.25 * (2.0 * np.ones((3, 3))), 1.0e-4)


def test__array_from_adus_to_electrons_per_second():
    arr_adus = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.full(
        fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
    )

    arr_eps = aa.preprocess.array_adus_to_eps(
        array_adus=arr_adus, exposure_time_map=exposure_time_map, gain=2.0
    )

    assert (arr_eps.native == 0.5 * 2.0 * np.ones((3, 3))).all()

    arr_eps = aa.preprocess.array_adus_to_eps(
        array_adus=arr_adus, exposure_time_map=exposure_time_map, gain=4.0
    )

    assert (arr_eps.native == 0.5 * 4.0 * np.ones((3, 3))).all()


def test__noise_map_from_image_exposure_time_map():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    poisson_noise_map = aa.preprocess.noise_map_via_data_eps_and_exposure_time_map_from(
        data_eps=image, exposure_time_map=exposure_time_map
    )

    assert (poisson_noise_map.native == np.ones((3, 3))).all()

    image = aa.Array2D.full(fill_value=4.0, shape_native=(4, 2), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(4, 2), pixel_scales=1.0)
    poisson_noise_map = aa.preprocess.noise_map_via_data_eps_and_exposure_time_map_from(
        data_eps=image, exposure_time_map=exposure_time_map
    )

    assert (poisson_noise_map.native == 2.0 * np.ones((4, 2))).all()

    image = aa.Array2D.ones(shape_native=(1, 5), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.full(
        fill_value=4.0, shape_native=(1, 5), pixel_scales=1.0
    )
    poisson_noise_map = aa.preprocess.noise_map_via_data_eps_and_exposure_time_map_from(
        data_eps=image, exposure_time_map=exposure_time_map
    )

    assert (poisson_noise_map.native == 0.5 * np.ones((1, 5))).all()

    image = aa.Array2D.no_mask(
        values=np.array([[5.0, 3.0], [10.0, 20.0]]), pixel_scales=1.0
    )
    exposure_time_map = aa.Array2D.no_mask(
        np.array([[1.0, 2.0], [3.0, 4.0]]), pixel_scales=1.0
    )
    poisson_noise_map = aa.preprocess.noise_map_via_data_eps_and_exposure_time_map_from(
        data_eps=image, exposure_time_map=exposure_time_map
    )

    assert poisson_noise_map.native == pytest.approx(
        np.array(
            [
                [np.sqrt(5.0), np.sqrt(6.0) / 2.0],
                [np.sqrt(30.0) / 3.0, np.sqrt(80.0) / 4.0],
            ]
        ),
        1.0e-4,
    )


def test__noise_map_from_image_exposure_time_map_and_background_noise_map():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    background_noise_map = aa.Array2D.full(
        fill_value=3.0**0.5, shape_native=(3, 3), pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
        data_eps=image,
        exposure_time_map=exposure_time_map,
        background_noise_map=background_noise_map,
    )

    assert noise_map.native == pytest.approx(2.0 * np.ones((3, 3)), 1e-2)

    image = aa.Array2D.ones(shape_native=(2, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(2, 3), pixel_scales=1.0)
    background_noise_map = aa.Array2D.full(
        fill_value=5.0, shape_native=(2, 3), pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
        data_eps=image,
        exposure_time_map=exposure_time_map,
        background_noise_map=background_noise_map,
    )

    assert noise_map.native == pytest.approx(
        np.array(
            [
                [np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)],
                [np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0), np.sqrt(1.0 + 25.0)],
            ]
        ),
        1e-2,
    )

    image = aa.Array2D.ones(shape_native=(2, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.full(
        fill_value=2.0, shape_native=(2, 3), pixel_scales=1.0
    )
    background_noise_map = aa.Array2D.full(
        fill_value=5.0, shape_native=(2, 3), pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
        data_eps=image,
        exposure_time_map=exposure_time_map,
        background_noise_map=background_noise_map,
    )

    assert noise_map.native == pytest.approx(
        np.array(
            [
                [
                    np.sqrt(2.0 + 100.0) / 2.0,
                    np.sqrt(2.0 + 100.0) / 2.0,
                    np.sqrt(2.0 + 100.0) / 2.0,
                ],
                [
                    np.sqrt(2.0 + 100.0) / 2.0,
                    np.sqrt(2.0 + 100.0) / 2.0,
                    np.sqrt(2.0 + 100.0) / 2.0,
                ],
            ]
        ),
        1e-2,
    )

    image = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], pixel_scales=1.0
    )
    exposure_time_map = aa.Array2D.ones(shape_native=(3, 2), pixel_scales=1.0)
    background_noise_map = aa.Array2D.full(
        fill_value=12.0, shape_native=(3, 2), pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
        data_eps=image,
        exposure_time_map=exposure_time_map,
        background_noise_map=background_noise_map,
    )

    assert noise_map.native == pytest.approx(
        np.array(
            [
                [np.sqrt(1.0 + 144.0), np.sqrt(2.0 + 144.0)],
                [np.sqrt(3.0 + 144.0), np.sqrt(4.0 + 144.0)],
                [np.sqrt(5.0 + 144.0), np.sqrt(6.0 + 144.0)],
            ]
        ),
        1e-2,
    )

    image = aa.Array2D.no_mask(values=[[5.0, 3.0], [10.0, 20.0]], pixel_scales=1.0)
    exposure_time_map = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0
    )
    background_noise_map = aa.Array2D.full(
        fill_value=9.0, shape_native=((2, 2)), pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
        data_eps=image,
        exposure_time_map=exposure_time_map,
        background_noise_map=background_noise_map,
    )

    assert noise_map.native == pytest.approx(
        np.array(
            [
                [np.sqrt(5.0 + 81.0), np.sqrt(6.0 + 18.0**2.0) / 2.0],
                [np.sqrt(30.0 + 27.0**2.0) / 3.0, np.sqrt(80.0 + 36.0**2.0) / 4.0],
            ]
        ),
        1e-2,
    )

    image = aa.Array2D.no_mask(values=[[5.0, 3.0], [10.0, 20.0]], pixel_scales=1.0)
    exposure_time_map = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0
    )
    background_noise_map = aa.Array2D.no_mask(
        values=[[5.0, 6.0], [7.0, 8.0]], pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
        data_eps=image,
        exposure_time_map=exposure_time_map,
        background_noise_map=background_noise_map,
    )

    assert noise_map.native == pytest.approx(
        np.array(
            [
                [np.sqrt(5.0 + 5.0**2.0), np.sqrt(6.0 + 12.0**2.0) / 2.0],
                [np.sqrt(30.0 + 21.0**2.0) / 3.0, np.sqrt(80.0 + 32.0**2.0) / 4.0],
            ]
        ),
        1e-2,
    )


def test__noise_map_from_image_exposure_time_map_and_background_variances():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    background_variances = aa.Array2D.full(
        fill_value=3.0**0.5, shape_native=(3, 3), pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_variances_from(
        data_eps=image,
        exposure_time_map=exposure_time_map,
        background_variances=background_variances,
    )

    assert noise_map.native == pytest.approx(1.65289 * np.ones((3, 3)), 1e-2)


def test__noise_map_via_weight_map_from():
    weight_map = aa.Array2D.no_mask(
        [[1.0, 4.0, 16.0], [1.0, 4.0, 16.0]], pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_weight_map_from(
        weight_map=weight_map.native
    )

    assert (noise_map.native == np.array([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]])).all()

    weight_map = aa.Array2D.no_mask(
        [[1.0, 4.0, 0.0], [1.0, 4.0, 16.0]], pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_weight_map_from(weight_map=weight_map)

    assert (noise_map.native == np.array([[1.0, 0.5, 1.0e8], [1.0, 0.5, 0.25]])).all()


def test__noise_map_via_inverse_noise_map_from():
    inverse_noise_map = aa.Array2D.no_mask(
        [[1.0, 4.0, 16.0], [1.0, 4.0, 16.0]], pixel_scales=1.0
    )

    noise_map = aa.preprocess.noise_map_via_inverse_noise_map_from(
        inverse_noise_map=inverse_noise_map
    )

    assert (
        noise_map.native == np.array([[1.0, 0.25, 0.0625], [1.0, 0.25, 0.0625]])
    ).all()


def test__noise_map_with_offset_values_added():
    np.random.seed(1)

    noise_map = aa.Array2D.full(fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0)

    noise_map = aa.preprocess.array_with_random_uniform_values_added(
        array=noise_map, upper_limit=0.001
    )

    assert noise_map.native == pytest.approx(
        np.array(
            [
                [3.000907, 3.00044, 3.000277],
                [3.0005587, 3.001036, 3.00119],
                [3.000558, 3.00103668, 3.0011903],
            ]
        ),
        1.0e-2,
    )


def test__background_sky_level_via_edges_of_image_from():
    image = aa.Array2D.no_mask(values=np.ones((3, 3)), pixel_scales=1.0)

    background_sky_level = aa.preprocess.background_sky_level_via_edges_from(
        image=image, no_edges=1
    )

    assert background_sky_level == 1.0

    image = aa.Array2D.no_mask(
        values=[[1, 1, 1, 1], [1, 100, 100, 1], [1, 100, 100, 1], [1, 1, 1, 1]],
        pixel_scales=1.0,
    )

    background_sky_level = aa.preprocess.background_sky_level_via_edges_from(
        image=image, no_edges=1
    )

    assert background_sky_level == 1.0

    image = aa.Array2D.no_mask(
        [
            [0, 1, 1, 1, 0],
            [0, 3, 1, 3, 0],
            [0, 3, 100, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 1, 1, 1, 0],
        ],
        pixel_scales=1.0,
    )

    background_sky_level = aa.preprocess.background_sky_level_via_edges_from(
        image=image, no_edges=2
    )

    assert background_sky_level == 1.0


def test__background_noise_map_via_edges_of_image_from():
    image = aa.Array2D.no_mask(values=np.ones((3, 3)), pixel_scales=1.0)

    background_noise_map = aa.preprocess.background_noise_map_via_edges_from(
        image=image, no_edges=1
    )

    assert (
        background_noise_map.native == np.full(fill_value=0.0, shape=image.shape_native)
    ).all()


def test__background_noise_map_via_edges_of_image_from_2():
    image = aa.Array2D.no_mask(
        values=[[1, 1, 1, 1], [1, 100, 100, 1], [1, 100, 100, 1], [1, 1, 1, 1]],
        pixel_scales=1.0,
    )

    background_noise_map = aa.preprocess.background_noise_map_via_edges_from(
        image=image, no_edges=1
    )

    assert (
        background_noise_map.native == np.full(fill_value=0.0, shape=image.shape_native)
    ).all()


def test__background_noise_map_via_edges_of_image_from_3():
    image = aa.Array2D.no_mask(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 100, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        pixel_scales=1.0,
    )

    background_noise_map = aa.preprocess.background_noise_map_via_edges_from(
        image=image, no_edges=2
    )

    assert (
        background_noise_map.native == np.full(fill_value=0.0, shape=image.shape_native)
    ).all()


def test__background_noise_map_via_edges_of_image_from_4():
    image = aa.Array2D.no_mask(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 100, 12, 13],
            [14, 15, 100, 16, 17],
            [18, 19, 20, 21, 22],
            [23, 24, 25, 26, 27],
        ],
        pixel_scales=1.0,
    )

    background_noise_map = aa.preprocess.background_noise_map_via_edges_from(
        image=image, no_edges=2
    )

    assert np.allclose(
        background_noise_map.native.array,
        np.full(fill_value=np.std(np.arange(28)), shape=image.shape_native),
    )


def test__background_noise_map_via_edges_of_image_from_5():
    image = aa.Array2D.no_mask(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13],
            [14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 100, 24, 25, 26],
            [27, 28, 29, 30, 31, 32, 33],
            [34, 35, 36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45, 46, 47],
        ],
        pixel_scales=1.0,
    )

    background_noise_map = aa.preprocess.background_noise_map_via_edges_from(
        image=image, no_edges=3
    )

    assert np.allclose(
        background_noise_map.native.array,
        np.full(fill_value=np.std(np.arange(48)), shape=image.shape_native),
    )


def test__rescaled_with_odd_dimensions_from__evens_to_odds():
    kernel = np.ones((6, 6))

    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.5, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 9.0) * np.ones((3, 3)), 1.0e-4)

    kernel = np.ones((9, 9))

    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.333333333333333, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 9.0) * np.ones((3, 3)), 1.0e-4)

    kernel = np.ones((18, 6))

    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.5, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 27.0) * np.ones((9, 3)), 1.0e-4)

    kernel = np.ones((6, 18))

    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.5, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 27.0) * np.ones((3, 9)), 1.0e-4)


def test__rescaled_with_odd_dimensions_from__different_scalings():
    kernel = np.ones(shape=(2, 2))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=2.0, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 25.0) * np.ones((5, 5)), 1.0e-4)

    kernel = np.ones(shape=(40, 40))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.1, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 25.0) * np.ones((5, 5)), 1.0e-4)

    kernel = np.ones(shape=(2, 4))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=2.0, normalize=True
    )

    assert kernel == pytest.approx((1.0 / 45.0) * np.ones((5, 9)), 1.0e-4)

    kernel = np.ones(shape=(4, 2))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=2.0, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 45.0) * np.ones((9, 5)), 1.0e-4)

    kernel = np.ones(shape=(6, 4))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.5, normalize=True
    )

    assert kernel == pytest.approx((1.0 / 9.0) * np.ones((3, 3)), 1.0e-4)

    kernel = np.ones(shape=(9, 12))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.33333333333, normalize=True
    )

    assert kernel == pytest.approx((1.0 / 15.0) * np.ones((3, 5)), 1.0e-4)

    kernel = np.ones(shape=(4, 6))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.5, normalize=True
    )

    assert kernel == pytest.approx((1.0 / 9.0) * np.ones((3, 3)), 1.0e-4)

    kernel = np.ones(shape=(12, 9))
    kernel = aa.preprocess.kernel_with_odd_dimensions_from(
        kernel=kernel, rescale_factor=0.33333333333, normalize=True
    )
    assert kernel == pytest.approx((1.0 / 15.0) * np.ones((5, 3)), 1.0e-4)


def test__exposure_time_map_from_exposure_time_and_inverse_noise_map():
    exposure_time = 6.0

    background_noise_map = aa.Array2D.no_mask(
        [[0.5, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], pixel_scales=1.0
    )

    exposure_time_map = (
        aa.preprocess.exposure_time_map_via_exposure_time_and_background_noise_map_from(
            exposure_time=exposure_time, background_noise_map=background_noise_map
        )
    )

    assert (
        exposure_time_map.native
        == np.array([[3.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]])
    ).all()


def test__poisson_noise_from_data():
    data = aa.Array2D.zeros(shape_native=(2, 2), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    poisson_noise = aa.preprocess.poisson_noise_via_data_eps_from(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (poisson_noise.native == np.zeros((2, 2))).all()

    data = aa.Array2D.no_mask(values=[[10.0, 0.0], [0.0, 10.0]], pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    poisson_noise = aa.preprocess.poisson_noise_via_data_eps_from(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (
        poisson_noise.native == np.array([[(9.0 - 10.0), 0], [0, (6.0 - 10.0)]])
    ).all()

    data = aa.Array2D.full(fill_value=10.0, shape_native=(2, 2), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    poisson_noise = aa.preprocess.poisson_noise_via_data_eps_from(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    # Use known noise_map_1d map for given seed.
    assert (poisson_noise.native == np.array([[-1, -4], [-3, -1]])).all()

    data = aa.Array2D.no_mask(
        values=[[10000000.0, 0.0], [0.0, 10000000.0]], pixel_scales=1.0
    )
    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    poisson_noise = aa.preprocess.poisson_noise_via_data_eps_from(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (poisson_noise.native == np.array([[-743, 0], [0, -3783]])).all()


def test__data_with_poisson_noised_added():
    data = aa.Array2D.zeros(shape_native=(2, 2), pixel_scales=1.0)
    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    data_with_poisson_noise = aa.preprocess.data_eps_with_poisson_noise_added(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (data_with_poisson_noise.native == np.zeros((2, 2))).all()

    data = aa.Array2D.no_mask(values=[[10.0, 0.0], [0.0, 10.0]], pixel_scales=1.0)

    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    data_with_poisson_noise = aa.preprocess.data_eps_with_poisson_noise_added(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (data_with_poisson_noise.native == np.array([[9, 0], [0, 6]])).all()

    data = aa.Array2D.full(fill_value=10.0, shape_native=(2, 2), pixel_scales=1.0)

    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    data_with_poisson_noise = aa.preprocess.data_eps_with_poisson_noise_added(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (data_with_poisson_noise.native == np.array([[9, 6], [7, 9]])).all()

    data = aa.Array2D.no_mask(
        values=[[10000000.0, 0.0], [0.0, 10000000.0]], pixel_scales=1.0
    )

    exposure_time_map = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    data_with_poisson_noise = aa.preprocess.data_eps_with_poisson_noise_added(
        data_eps=data, exposure_time_map=exposure_time_map, seed=1
    )

    assert (
        data_with_poisson_noise.native == np.array([[9999257, 0.0], [0.0, 9996217.0]])
    ).all()


def test__gaussian_noise_via_shape_and_sigma_from():
    gaussian_noise = aa.preprocess.gaussian_noise_via_shape_and_sigma_from(
        shape=(9,), sigma=0.0, seed=1
    )

    assert (gaussian_noise == np.zeros((9,))).all()

    gaussian_noise = aa.preprocess.gaussian_noise_via_shape_and_sigma_from(
        shape=(9,), sigma=1.0, seed=1
    )

    assert gaussian_noise == pytest.approx(
        np.array([1.62, -0.61, -0.53, -1.07, 0.87, -2.30, 1.74, -0.76, 0.32]), 1e-2
    )


def test__data_with_gaussian_noise_added():
    data = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    data_with_noise = aa.preprocess.data_with_gaussian_noise_added(
        data=data, sigma=0.0, seed=1
    )

    assert (data_with_noise.slim == np.ones((9,))).all()

    data_with_noise = aa.preprocess.data_with_gaussian_noise_added(
        data=data, sigma=1.0, seed=1
    )

    assert data_with_noise.slim == pytest.approx(
        np.array(
            [
                1 + 1.62,
                1 - 0.61,
                1 - 0.53,
                1 - 1.07,
                1 + 0.87,
                1 - 2.30,
                1 + 1.74,
                1 - 0.76,
                1 + 0.32,
            ]
        ),
        1e-1,
    )


def test__data_with_complex_gaussian_noise_added():
    data = (1.0 + 1.0j) * np.ones(shape=(3,))

    data_with_noise = aa.preprocess.data_with_complex_gaussian_noise_added(
        data=data, sigma=0.0, seed=1
    )

    assert (data_with_noise == 1.0 + 1.0j * np.ones((3,))).all()

    data_with_noise = aa.preprocess.data_with_complex_gaussian_noise_added(
        data=data, sigma=1.0, seed=1
    )

    assert data_with_noise == pytest.approx(
        np.array([2.62434 + 0.38824j, 0.47182 - 0.07298j, 1.86540 - 1.30153j]), 1e-3
    )


def test__noise_map_with_signal_to_noise_limit_from():

    image = aa.Array2D.no_mask(values=[[20, 20], [20, 5]], pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask(values=[[5, 5], [5, 2]], pixel_scales=1.0)

    noise_map = aa.preprocess.noise_map_with_signal_to_noise_limit_from(
        data=image, noise_map=noise_map, signal_to_noise_limit=100.0
    )

    assert (noise_map.slim == np.array([5.0, 5.0, 5.0, 2.0])).all()

    noise_map = aa.Array2D.no_mask(values=[[5, 5], [5, 2]], pixel_scales=1.0)

    noise_map = aa.preprocess.noise_map_with_signal_to_noise_limit_from(
        data=image, noise_map=noise_map, signal_to_noise_limit=2.0
    )

    assert (noise_map.native == np.array([[10.0, 10.0], [10.0, 2.5]])).all()

    image = aa.Array2D.no_mask(values=[[20, 20], [5, 5]], pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask(values=[[5, 5], [2, 2]], pixel_scales=1.0)

    mask = aa.Mask2D(mask=[[True, False], [False, True]], pixel_scales=1.0)

    noise_map = aa.preprocess.noise_map_with_signal_to_noise_limit_from(
        data=image,
        noise_map=noise_map,
        signal_to_noise_limit=2.0,
        noise_limit_mask=mask,
    )

    assert (noise_map.native == np.array([[5.0, 10.0], [2.5, 2.0]])).all()


def test__visibilities_noise_map_with_signal_to_noise_limit(
    mask_2d_7x7, uv_wavelengths_7x2
):
    data = aa.Visibilities(visibilities=np.array([1 + 1j, 1 + 1j]))
    noise_map = aa.VisibilitiesNoiseMap(visibilities=np.array([1 + 0.25j, 1 + 0.25j]))

    noise_map_limit = (
        aa.preprocess.visibilities_noise_map_with_signal_to_noise_limit_from(
            data=data, noise_map=noise_map, signal_to_noise_limit=2.0
        )
    )

    assert (noise_map_limit == np.array([1.0 + 0.5j, 1.0 + 0.5j])).all()

    noise_map_limit = (
        aa.preprocess.visibilities_noise_map_with_signal_to_noise_limit_from(
            data=data, noise_map=noise_map, signal_to_noise_limit=0.25
        )
    )

    assert (noise_map_limit == np.array([4.0 + 4.0j, 4.0 + 4.0j])).all()
