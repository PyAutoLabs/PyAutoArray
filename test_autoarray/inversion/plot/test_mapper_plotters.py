import pytest

import autoarray as aa
import autoarray.plot as aplt
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "structures"


def test__plot_mapper(
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    plot_path,
    plot_patch,
):
    aplt.plot_mapper(
        mapper=rectangular_mapper_7x7_3x3,
        output_path=plot_path,
        output_filename="mapper1",
        output_format="png",
    )

    assert str(Path(plot_path) / "mapper1.png") in plot_patch.paths


def test__plot_mapper__zoom_extent_scale__widens_extent_around_centre(
    rectangular_mapper_7x7_3x3,
    plot_path,
    plot_patch,
):
    aplt.plot_mapper(
        mapper=rectangular_mapper_7x7_3x3,
        zoom_extent_scale=2.5,
        output_path=plot_path,
        output_filename="mapper_mid_zoom",
        output_format="png",
    )

    assert str(Path(plot_path) / "mapper_mid_zoom.png") in plot_patch.paths


def test__subplot_image_and_mapper(
    imaging_7x7,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    plot_path,
    plot_patch,
):
    aplt.subplot_image_and_mapper(
        mapper=rectangular_mapper_7x7_3x3,
        image=imaging_7x7.data,
        output_path=plot_path,
        output_format="png",
    )
    assert str(Path(plot_path) / "image_and_mapper.png") in plot_patch.paths


def test__subplot_image_and_mapper__with_regions(
    imaging_7x7,
    rectangular_mapper_7x7_3x3,
    plot_path,
    plot_patch,
):
    mappings = rectangular_mapper_7x7_3x3.mappings_from(pix_indexes=[[0, 1], [8]])

    aplt.subplot_image_and_mapper(
        mapper=rectangular_mapper_7x7_3x3,
        image=imaging_7x7.data,
        regions=mappings,
        region_labels=["1", "2"],
        output_path=plot_path,
        output_filename="image_and_mapper_regions",
        output_format="png",
    )

    assert str(Path(plot_path) / "image_and_mapper_regions.png") in plot_patch.paths
