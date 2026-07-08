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


def test__via_image_from__image_is_convolved__skips_psf_convolution():
    # An already-convolved image (e.g. from an oversampled Convolver) must pass
    # through untouched by the simulator's own convolution step.
    image = aa.Array2D.no_mask(
        values=np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
        pixel_scales=1.0,
    )

    psf = aa.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=1.0, sigma=0.5, normalize=True
    )

    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        psf=psf,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
        noise_if_add_noise_false=1.0,
    )

    dataset_convolved = simulator.via_image_from(image=image)
    dataset_passthrough = simulator.via_image_from(image=image, image_is_convolved=True)

    # The pass-through equals the input exactly; the convolved one does not.
    assert np.array(dataset_passthrough.data.native) == pytest.approx(
        np.array(image.native), abs=1.0e-14
    )
    assert not np.allclose(
        np.array(dataset_convolved.data.native), np.array(image.native)
    )


def test__simulate_and_fit__oversampled_psf__consistent_with_fit_side_convolution():
    # The simulator's oversampled path (fine evaluation of the padded frame via
    # image_is_convolved=True) must agree exactly, inside the mask, with the
    # fit-side path (mask + blurring-region fine convolution) — the padding
    # guarantees all flux within kernel reach of the mask is included in both.
    s = 2
    pixel_scales = 1.0

    def gaussian_on(grid_like, sigma=1.2, centre=(0.3, -0.4)):
        arr = np.array(grid_like)
        r2 = (arr[:, 0] - centre[0]) ** 2 + (arr[:, 1] - centre[1]) ** 2
        return np.exp(-0.5 * r2 / sigma**2)

    kernel_n = 9
    c = (np.arange(kernel_n) - (kernel_n - 1) / 2.0) * (pixel_scales / s)
    yy, xx = np.meshgrid(-c, c, indexing="ij")
    kernel = np.exp(-0.5 * (yy**2 + xx**2) / 0.8**2)
    psf = aa.Convolver(
        kernel=aa.Array2D.no_mask(values=kernel, pixel_scales=pixel_scales / s),
        normalize=True,
        convolve_over_sample_size=s,
    )

    # Simulator side: evaluate the padded frame fine, convolve, trim.
    shape_native = (11, 11)
    kernel_shape = psf.kernel_shape_image_resolution
    padded_shape = (
        shape_native[0] + kernel_shape[0] - 1,
        shape_native[1] + kernel_shape[1] - 1,
    )
    padded_mask = aa.Mask2D.all_false(
        shape_native=padded_shape, pixel_scales=pixel_scales
    )
    padded_grid = aa.Grid2D.from_mask(mask=padded_mask, over_sample_size=s)

    convolved_padded = psf.convolved_image_from(
        image=gaussian_on(padded_grid.over_sampled),
        blurring_image=None,
        mask=padded_mask,
    )
    convolved_padded = aa.Array2D(values=convolved_padded, mask=padded_mask)

    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        psf=psf,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
        noise_if_add_noise_false=1.0,
    )
    dataset = simulator.via_image_from(image=convolved_padded, image_is_convolved=True)
    dataset = dataset.trimmed_after_convolution_from(kernel_shape=kernel_shape)

    assert dataset.data.shape_native == shape_native

    # Fit side: mask + blurring-region fine convolution of the same scene.
    mask = aa.Mask2D.circular(
        shape_native=shape_native, pixel_scales=pixel_scales, radius=3.5
    )
    masked = aa.Imaging(
        data=dataset.data,
        noise_map=aa.Array2D.no_mask(
            values=np.ones(shape_native), pixel_scales=pixel_scales
        ),
        psf=psf,
        over_sample_size_lp=s,
        over_sample_size_pixelization=s,
        convolve_over_sample_size_lp=s,
        convolve_over_sample_size_pixelization=s,
    ).apply_mask(mask=mask)

    blurring_grid = masked.grids.blurring
    model_data = masked.psf.convolved_image_from(
        image=gaussian_on(masked.grids.lp.over_sampled),
        blurring_image=gaussian_on(blurring_grid.over_sampled),
    )

    fit = aa.m.MockFitImaging(
        dataset=masked, use_mask_in_fit=False, model_data=model_data
    )

    assert fit.chi_squared == pytest.approx(0.0, abs=1.0e-10)
