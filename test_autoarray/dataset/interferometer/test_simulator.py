import numpy as np
import pytest

import autoarray as aa


def test__from_image__setup_with_all_features_off(
    uv_wavelengths_7x2, transformer_7x7_7, mask_2d_7x7
):
    image = aa.Array2D.no_mask(
        values=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
        pixel_scales=transformer_7x7_7.grid.pixel_scales,
    )

    simulator = aa.SimulatorInterferometer(
        exposure_time=1.0,
        transformer_class=type(transformer_7x7_7),
        uv_wavelengths=uv_wavelengths_7x2,
        noise_sigma=None,
    )

    dataset = simulator.via_image_from(image=image)

    transformer = simulator.transformer_class(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=aa.Mask2D.all_false(
            shape_native=(3, 3), pixel_scales=image.pixel_scales
        ),
    )

    visibilities = transformer.visibilities_from(image=image)

    assert dataset.data == pytest.approx(visibilities.array, 1.0e-4)


def test__setup_with_noise(uv_wavelengths_7x2, transformer_7x7_7):
    image = aa.Array2D.no_mask(
        values=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
        pixel_scales=transformer_7x7_7.grid.pixel_scales,
    )

    simulator = aa.SimulatorInterferometer(
        exposure_time=20.0,
        transformer_class=type(transformer_7x7_7),
        uv_wavelengths=uv_wavelengths_7x2,
        noise_sigma=0.1,
        noise_seed=1,
    )

    dataset = simulator.via_image_from(image=image)

    assert dataset.data[0] == pytest.approx(-0.005364 - 2.36682j, 1.0e-4)

    assert (dataset.noise_map == 0.1 + 0.1j * np.ones((7,))).all()
