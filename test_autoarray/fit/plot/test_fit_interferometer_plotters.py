import autoarray.plot as aplt
import numpy as np
import pytest
from pathlib import Path


directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "plots" / "fit_dataset"


def test__fit_quantities_are_output(fit_interferometer_7, plot_path, plot_patch):
    uv = fit_interferometer_7.dataset.uv_distances / 10**3.0

    aplt.plot_grid_2d(
        grid=fit_interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert str(Path(plot_path) / "data.png") in plot_patch.paths

    aplt.plot_yx_1d(
        y=np.real(fit_interferometer_7.residual_map),
        x=uv,
        output_path=plot_path,
        output_filename="real_residual_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    assert str(Path(plot_path) / "real_residual_map_vs_uv_distances.png") in plot_patch.paths

    aplt.plot_yx_1d(
        y=np.real(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="real_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    assert str(Path(plot_path) / "real_chi_squared_map_vs_uv_distances.png") in plot_patch.paths

    aplt.plot_yx_1d(
        y=np.imag(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="imag_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    assert str(Path(plot_path) / "imag_chi_squared_map_vs_uv_distances.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=fit_interferometer_7.dirty_image,
        output_path=plot_path,
        output_filename="dirty_image",
        output_format="png",
    )
    assert str(Path(plot_path) / "dirty_image.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.plot_grid_2d(
        grid=fit_interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    aplt.plot_yx_1d(
        y=np.real(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="real_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    aplt.plot_yx_1d(
        y=np.imag(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="imag_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )

    assert str(Path(plot_path) / "data.png") in plot_patch.paths
    assert str(Path(plot_path) / "real_chi_squared_map_vs_uv_distances.png") in plot_patch.paths
    assert str(Path(plot_path) / "imag_chi_squared_map_vs_uv_distances.png") in plot_patch.paths
    assert str(Path(plot_path) / "real_residual_map_vs_uv_distances.png") not in plot_patch.paths


def test__fit_sub_plots(fit_interferometer_7, plot_path, plot_patch):
    aplt.subplot_fit_interferometer(
        fit=fit_interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "fit.png") in plot_patch.paths

    aplt.subplot_fit_interferometer_dirty_images(
        fit=fit_interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "fit_dirty_images.png") in plot_patch.paths
