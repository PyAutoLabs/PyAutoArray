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


def _array_free_from(dataset):
    """
    The array-free counterpart of an in-memory `Interferometer`: the same visibilities
    streamed through `Interferometer.from_stream` in two chunks.
    """
    import autoarray as aa

    uv_wavelengths = np.asarray(dataset.uv_wavelengths)
    data = np.asarray(dataset.data.array)
    noise_map = np.asarray(dataset.noise_map.array)

    chunks = [
        (uv_wavelengths[k0:k1], data[k0:k1], noise_map[k0:k1])
        for k0, k1 in ((0, 3), (3, data.shape[0]))
    ]

    return aa.Interferometer.from_stream(chunks, dataset.real_space_mask)


def test__fit_sub_plots__array_free_dataset(
    interferometer_7, plot_path, plot_patch, monkeypatch
):
    pytest.importorskip("nufftax")

    import autoarray as aa
    from autoarray.fit.plot import fit_interferometer_plots

    dataset = _array_free_from(interferometer_7)

    fit = aa.m.MockFitInterferometer(dataset=dataset)

    model_image = aa.Array2D(
        values=np.ones(dataset.real_space_mask.pixels_in_mask),
        mask=dataset.real_space_mask,
    )

    titles = []
    plot_array = fit_interferometer_plots.plot_array

    def _plot_array(array, *args, title=None, **kwargs):
        titles.append(title)
        return plot_array(array, *args, title=title, **kwargs)

    monkeypatch.setattr(fit_interferometer_plots, "plot_array", _plot_array)

    aplt.subplot_fit_interferometer(
        fit=fit,
        output_path=plot_path,
        output_format="png",
        model_image=model_image,
    )

    assert str(Path(plot_path) / "fit.png") in plot_patch.paths
    assert titles == [
        "Dirty Image (Natural)",
        "Dirty Model Image (Natural)",
        "Dirty Residual Map (Natural)",
    ]

    titles.clear()

    aplt.subplot_fit_interferometer_dirty_images(
        fit=fit,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "fit_dirty_images.png") in plot_patch.paths
    assert titles == ["Dirty Image (Natural)"]
