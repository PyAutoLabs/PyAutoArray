import autoarray as aa
import autoarray.plot as aplt
import pytest
import numpy as np
import shutil
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "structures"


def test__plot_yx_line(plot_path, plot_patch):
    aplt.plot_yx_1d(
        y=aa.Array1D.no_mask([1.0, 2.0, 3.0], pixel_scales=1.0),
        x=aa.Array1D.no_mask([0.5, 1.0, 1.5], pixel_scales=0.5),
        output_path=plot_path,
        output_filename="yx_1",
        output_format="png",
        plot_axis_type="loglog",
    )

    assert str(Path(plot_path) / "yx_1.png") in plot_patch.paths


def test__array(
    array_2d_7x7,
    mask_2d_7x7,
    grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    aplt.plot_array_2d(
        array=array_2d_7x7,
        output_path=plot_path,
        output_filename="array1",
        output_format="png",
    )

    assert str(Path(plot_path) / "array1.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=array_2d_7x7,
        output_path=plot_path,
        output_filename="array2",
        output_format="png",
    )

    assert str(Path(plot_path) / "array2.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=array_2d_7x7,
        origin=grid_2d_irregular_7x7_list,
        border=mask_2d_7x7.derive_grid.border,
        grid=grid_2d_7x7,
        positions=grid_2d_irregular_7x7_list,
        array_overlay=array_2d_7x7,
        output_path=plot_path,
        output_filename="array3",
        output_format="png",
    )

    assert str(Path(plot_path) / "array3.png") in plot_patch.paths


def test__grid(
    array_2d_7x7,
    grid_2d_7x7,
    mask_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    color_array = np.linspace(start=0.0, stop=1.0, num=grid_2d_7x7.shape_slim)

    aplt.plot_grid_2d(
        grid=grid_2d_7x7,
        indexes=[0, 1, 2],
        output_path=plot_path,
        output_filename="grid1",
        output_format="png",
        color_array=color_array,
    )

    assert str(Path(plot_path) / "grid1.png") in plot_patch.paths

    aplt.plot_grid_2d(
        grid=grid_2d_7x7,
        indexes=[0, 1, 2],
        output_path=plot_path,
        output_filename="grid2",
        output_format="png",
        color_array=color_array,
    )

    assert str(Path(plot_path) / "grid2.png") in plot_patch.paths

    aplt.plot_grid_2d(
        grid=grid_2d_7x7,
        lines=grid_2d_irregular_7x7_list,
        indexes=[0, 1, 2],
        output_path=plot_path,
        output_filename="grid3",
        output_format="png",
        color_array=color_array,
    )

    assert str(Path(plot_path) / "grid3.png") in plot_patch.paths


def test__array_rgb(
    array_2d_rgb_7x7,
    plot_path,
    plot_patch,
):
    aplt.plot_array_2d(
        array=array_2d_rgb_7x7,
        output_path=plot_path,
        output_filename="array_rgb",
        output_format="png",
    )

    assert str(Path(plot_path) / "array_rgb.png") in plot_patch.paths


def test__plot_array_rgb(
    array_2d_rgb_7x7,
    plot_path,
    plot_patch,
):
    """
    `plot_array` (the high-level function) must handle `Array2DRGB` inputs without
    applying a colormap, norm, or colorbar — all of which would raise errors or
    produce nonsense for a 3-channel image.
    """
    aplt.plot_array(
        array=array_2d_rgb_7x7,
        title="RGB Test",
        output_path=plot_path,
        output_filename="array_rgb_high_level",
        output_format="png",
    )

    assert str(Path(plot_path) / "array_rgb_high_level.png") in plot_patch.paths
