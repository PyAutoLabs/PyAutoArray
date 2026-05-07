
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
