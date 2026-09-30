
import pytest
from pathlib import Path
import autoarray.plot as aplt
from autoarray.dataset.plot.interferometer_plots import (
    subplot_interferometer_dataset,
    subplot_interferometer_dirty_images,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "interferometer"


def test__individual_attributes_are_output(interferometer_7, plot_path, plot_patch):
    aplt.plot_grid_2d(
        grid=interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert str(Path(plot_path) / "data.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=interferometer_7.dirty_image,
        output_path=plot_path,
        output_filename="dirty_image",
        output_format="png",
    )
    assert str(Path(plot_path) / "dirty_image.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=interferometer_7.dirty_noise_map,
        output_path=plot_path,
        output_filename="dirty_noise_map",
        output_format="png",
    )
    assert str(Path(plot_path) / "dirty_noise_map.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=interferometer_7.dirty_signal_to_noise_map,
        output_path=plot_path,
        output_filename="dirty_signal_to_noise_map",
        output_format="png",
    )
    assert str(Path(plot_path) / "dirty_signal_to_noise_map.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.plot_grid_2d(
        grid=interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert str(Path(plot_path) / "data.png") in plot_patch.paths
    assert not str(Path(plot_path) / "dirty_image.png") in plot_patch.paths


def test__subplots_are_output(interferometer_7, plot_path, plot_patch):
    subplot_interferometer_dataset(
        dataset=interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "dataset.png") in plot_patch.paths

    subplot_interferometer_dirty_images(
        dataset=interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "dirty_images.png") in plot_patch.paths


def _array_free_from(dataset):
    """
    The array-free counterpart of an in-memory `Interferometer`: the same visibilities
    streamed through `Interferometer.from_stream` in two chunks.
    """
    import numpy as np
    import autoarray as aa

    uv_wavelengths = np.asarray(dataset.uv_wavelengths)
    data = np.asarray(dataset.data.array)
    noise_map = np.asarray(dataset.noise_map.array)

    chunks = [
        (uv_wavelengths[k0:k1], data[k0:k1], noise_map[k0:k1])
        for k0, k1 in ((0, 3), (3, data.shape[0]))
    ]

    return aa.Interferometer.from_stream(chunks, dataset.real_space_mask)


def test__subplots_are_output__array_free_dataset(
    interferometer_7, plot_path, plot_patch
):
    pytest.importorskip("nufftax")

    dataset = _array_free_from(interferometer_7)

    assert dataset.is_array_free

    subplot_interferometer_dataset(
        dataset=dataset,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "dataset.png") in plot_patch.paths

    subplot_interferometer_dirty_images(
        dataset=dataset,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "dirty_images.png") in plot_patch.paths


def test__fits_interferometer__array_free_dataset_writes_natural_terms(
    interferometer_7, tmp_path
):
    pytest.importorskip("nufftax")

    import numpy as np
    from astropy.io import fits
    from autoarray.dataset.plot.interferometer_plots import fits_interferometer

    dataset = _array_free_from(interferometer_7)

    file_path = tmp_path / "dataset.fits"

    fits_interferometer(dataset=dataset, file_path=file_path)

    with fits.open(file_path) as hdu_list:
        ext_names = [hdu.header.get("EXTNAME") for hdu in hdu_list]
        dirty_image = np.asarray(hdu_list["DIRTY_IMAGE_NATURAL"].data)
        dirty_beam = np.asarray(hdu_list["DIRTY_BEAM"].data)

    assert ext_names == ["DIRTY_IMAGE_NATURAL", "DIRTY_BEAM"]
    np.testing.assert_allclose(
        dirty_image, np.asarray(dataset.dirty_image_natural.native), rtol=1.0e-6
    )
    np.testing.assert_allclose(
        dirty_beam, np.asarray(dataset.dirty_beam.native), rtol=1.0e-6
    )

    # Separate-file mode writes nothing for the absent arrays and does not raise.
    data_path = tmp_path / "data.fits"

    fits_interferometer(dataset=dataset, data_path=data_path)

    assert not data_path.exists()


def test__fits_interferometer__in_memory_dataset_unchanged(interferometer_7, tmp_path):
    from astropy.io import fits
    from autoarray.dataset.plot.interferometer_plots import fits_interferometer

    file_path = tmp_path / "dataset.fits"

    fits_interferometer(dataset=interferometer_7, file_path=file_path)

    with fits.open(file_path) as hdu_list:
        ext_names = [hdu.header.get("EXTNAME") for hdu in hdu_list]

    assert [name.lower() for name in ext_names] == [
        "data",
        "noise_map",
        "uv_wavelengths",
    ]
