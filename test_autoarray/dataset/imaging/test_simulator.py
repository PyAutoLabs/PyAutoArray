import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="image_central_delta_3x3")
def make_array_2d_7x7():
    return aa.Array2D.no_mask(
        values=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        pixel_scales=0.1,
    )


def test__via_image_from__all_features_off(image_central_delta_3x3):
    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert dataset.pixel_scales == (0.1, 0.1)


def test__via_image_from__psf_off__include_poisson_noise_in_noise_map(
    image_central_delta_3x3,
):
    image = image_central_delta_3x3 + 1.0

    simulator = aa.SimulatorImaging(
        exposure_time=20.0,
        add_poisson_noise_to_data=True,
        include_poisson_noise_in_noise_map=True,
        noise_seed=1,
    )

    dataset = simulator.via_image_from(image=image)

    assert dataset.data.native == pytest.approx(
        np.array([[0.95, 0.7, 0.75], [0.95, 1.9, 0.8], [0.95, 0.7, 0.85]]), 1e-2
    )

    assert dataset.noise_map.native == pytest.approx(
        np.array(
            [[0.218, 0.187, 0.194], [0.218, 0.308, 0.2], [0.218, 0.187, 0.206]]
        ),
        1e-2,
    )


def test__via_image_from__psf_off__noise_off_value_is_noise_value(
    image_central_delta_3x3,
):
    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
        noise_if_add_noise_false=0.2,
        noise_seed=1,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()

    assert np.allclose(dataset.noise_map.native.array, 0.2 * np.ones((3, 3)))


def test__via_image_from__psf_off__background_sky_on(image_central_delta_3x3):
    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        background_sky_level=16.0,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[-1.0, -5.0, -4.0], [-1.0, 0.0, -1.0], [-5.0, -2.0, -7.0]])
    ).all()

    assert dataset.noise_map.native[0, 0] == pytest.approx(3.87298, 1.0e-4)


def test__via_image_from__psf_on__psf_blurs_image_with_edge_trimming(
    image_central_delta_3x3,
):
    kernel = aa.Array2D.no_mask(
        values=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )
    psf = aa.Convolver(kernel=kernel)

    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        psf=psf,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
        normalize_psf=False,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    ).all()


def test__via_image_from__psf_on__disable_poisson_noise_in_data(
    image_central_delta_3x3,
):
    kernel = aa.Array2D.no_mask(
        values=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )
    psf = aa.Convolver(kernel=kernel)

    simulator = aa.SimulatorImaging(
        exposure_time=20.0,
        psf=psf,
        normalize_psf=False,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=True,
        noise_seed=1,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    ).all()

    assert dataset.noise_map.native == pytest.approx(
        np.array(
            [[0.0, 0.21794, 0.0], [0.18708, 0.28723, 0.21794], [0.0, 0.21794, 0.0]]
        ),
        1e-2,
    )


def test__via_image_from__psf_on__psf_and_noise_both_on(image_central_delta_3x3):
    image = image_central_delta_3x3 + 1.0

    kernel = aa.Array2D.no_mask(
        values=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )
    psf = aa.Convolver(kernel=kernel)

    simulator = aa.SimulatorImaging(
        exposure_time=20.0,
        psf=psf,
        add_poisson_noise_to_data=True,
        noise_seed=1,
        normalize_psf=False,
    )

    dataset = simulator.via_image_from(image=image)

    assert dataset.data.native == pytest.approx(
        np.array([[3.9, 5.35, 3.55], [5.85, 7.85, 5.5], [3.9, 5.3, 3.75]]), 1e-2
    )
