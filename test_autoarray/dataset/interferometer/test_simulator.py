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


def _primary_beam_simulator_setup(transformer_class):
    mask = aa.Mask2D.all_false(shape_native=(5, 6), pixel_scales=0.3)

    rng = np.random.default_rng(seed=6)
    uv_wavelengths = rng.normal(size=(17, 2)) * 1.0e5

    image = aa.Array2D(values=rng.normal(size=mask.pixels_in_mask), mask=mask)
    primary_beam = aa.Array2D(
        values=rng.uniform(0.2, 1.0, size=mask.pixels_in_mask), mask=mask
    )

    def simulator_from(primary_beam):
        return aa.SimulatorInterferometer(
            exposure_time=1.0,
            transformer_class=transformer_class,
            uv_wavelengths=uv_wavelengths,
            noise_sigma=None,
            primary_beam=primary_beam,
        )

    return mask, image, primary_beam, simulator_from


def test__primary_beam__simulated_visibilities_equal_unbeamed_simulation_of_beamed_image():
    transformer_classes = [aa.TransformerDFT]

    try:
        import nufftax  # noqa: F401

        transformer_classes.append(aa.TransformerNUFFT)
    except ModuleNotFoundError:
        pass

    for transformer_class in transformer_classes:
        mask, image, primary_beam, simulator_from = _primary_beam_simulator_setup(
            transformer_class
        )

        dataset = simulator_from(primary_beam).via_image_from(image=image)
        dataset_unbeamed = simulator_from(None).via_image_from(
            image=aa.Array2D(values=image.array * primary_beam.array, mask=mask)
        )

        np.testing.assert_allclose(
            dataset.data.array, dataset_unbeamed.data.array, rtol=1.0e-12, atol=1.0e-12
        )

        # The simulated dataset carries the primary beam, so fitting it models the true sky.
        assert np.array_equal(dataset.primary_beam_slim, primary_beam.array)
        assert np.array_equal(dataset.transformer.primary_beam_slim, primary_beam.array)
        assert dataset_unbeamed.primary_beam_slim is None

        # With P = None the simulation is bit-for-bit the one without the argument.
        dataset_none = aa.SimulatorInterferometer(
            exposure_time=1.0,
            transformer_class=transformer_class,
            uv_wavelengths=simulator_from(None).uv_wavelengths,
            noise_sigma=None,
        ).via_image_from(image=image)

        assert np.array_equal(
            simulator_from(None).via_image_from(image=image).data.array,
            dataset_none.data.array,
        )


def test__primary_beam__simulator_jax_jit_matches_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    mask, image, primary_beam, simulator_from = _primary_beam_simulator_setup(
        aa.TransformerDFT
    )

    simulator = simulator_from(primary_beam)

    dataset_numpy = simulator.via_image_from(image=image)

    data_jax = jax.jit(
        lambda values: simulator.via_image_from(
            image=aa.Array2D(values=values, mask=mask), xp=jnp
        ).data.array
    )(jnp.asarray(image.array))

    np.testing.assert_allclose(
        np.asarray(data_jax), dataset_numpy.data.array, rtol=1.0e-10, atol=1.0e-10
    )
