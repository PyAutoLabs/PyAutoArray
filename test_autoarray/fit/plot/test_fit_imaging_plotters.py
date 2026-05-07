import autoarray as aa
import autoarray.plot as aplt
import pytest
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "plots" / "fit_dataset"


def test__fit_quantities_are_output(fit_imaging_7x7, plot_path, plot_patch):
    aplt.plot_array_2d(
        array=fit_imaging_7x7.data,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert str(Path(plot_path) / "data.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=fit_imaging_7x7.noise_map,
        output_path=plot_path,
        output_filename="noise_map",
        output_format="png",
    )
    assert str(Path(plot_path) / "noise_map.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=fit_imaging_7x7.signal_to_noise_map,
        output_path=plot_path,
        output_filename="signal_to_noise_map",
        output_format="png",
    )
    assert str(Path(plot_path) / "signal_to_noise_map.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=fit_imaging_7x7.model_data,
        output_path=plot_path,
        output_filename="model_image",
        output_format="png",
    )
    assert str(Path(plot_path) / "model_image.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=fit_imaging_7x7.residual_map,
        output_path=plot_path,
        output_filename="residual_map",
        output_format="png",
    )
    assert str(Path(plot_path) / "residual_map.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=fit_imaging_7x7.normalized_residual_map,
        output_path=plot_path,
        output_filename="normalized_residual_map",
        output_format="png",
    )
    assert str(Path(plot_path) / "normalized_residual_map.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=fit_imaging_7x7.chi_squared_map,
        output_path=plot_path,
        output_filename="chi_squared_map",
        output_format="png",
    )
    assert str(Path(plot_path) / "chi_squared_map.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.plot_array_2d(
        array=fit_imaging_7x7.data,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    aplt.plot_array_2d(
        array=fit_imaging_7x7.model_data,
        output_path=plot_path,
        output_filename="model_image",
        output_format="png",
    )
    aplt.plot_array_2d(
        array=fit_imaging_7x7.chi_squared_map,
        output_path=plot_path,
        output_filename="chi_squared_map",
        output_format="png",
    )

    assert str(Path(plot_path) / "data.png") in plot_patch.paths
    assert str(Path(plot_path) / "noise_map.png") not in plot_patch.paths
    assert str(Path(plot_path) / "signal_to_noise_map.png") not in plot_patch.paths
    assert str(Path(plot_path) / "model_image.png") in plot_patch.paths
    assert str(Path(plot_path) / "residual_map.png") not in plot_patch.paths
    assert str(Path(plot_path) / "normalized_residual_map.png") not in plot_patch.paths
    assert str(Path(plot_path) / "chi_squared_map.png") in plot_patch.paths


def test__fit_sub_plot(fit_imaging_7x7, plot_path, plot_patch):
    aplt.subplot_fit_imaging(
        fit=fit_imaging_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "fit.png") in plot_patch.paths


def test__output_as_fits__correct_output_format(
    fit_imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    aplt.plot_array_2d(
        array=fit_imaging_7x7.data,
        output_path=plot_path,
        output_filename="data",
        output_format="fits",
    )

    image_from_plot = aa.ndarray_via_fits_from(
        file_path=Path(plot_path) / "data.fits", hdu=0
    )

    assert image_from_plot.shape == (5, 5)
