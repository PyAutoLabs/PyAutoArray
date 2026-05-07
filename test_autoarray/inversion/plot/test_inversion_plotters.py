import autoarray.plot as aplt
from autoarray.inversion.mappers.abstract import Mapper

import pytest
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "plots" / "inversion"


def test__individual_attributes_are_output_for_all_mappers(
    rectangular_inversion_7x7_3x3,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    aplt.plot_array_2d(
        array=rectangular_inversion_7x7_3x3.mapped_reconstructed_operated_data,
        output_path=plot_path,
        output_filename="reconstructed_operated_data",
        output_format="png",
    )

    assert str(Path(plot_path) / "reconstructed_operated_data.png") in plot_patch.paths

    mapper = rectangular_inversion_7x7_3x3.cls_list_from(cls=Mapper)[0]
    pixel_values = rectangular_inversion_7x7_3x3.reconstruction_dict[mapper]

    aplt.plot_mapper(
        mapper=mapper,
        solution_vector=pixel_values,
        output_path=plot_path,
        output_filename="reconstruction",
        output_format="png",
    )

    assert str(Path(plot_path) / "reconstruction.png") in plot_patch.paths


def test__inversion_subplot_of_mapper__is_output_for_all_inversions(
    imaging_7x7,
    rectangular_inversion_7x7_3x3,
    plot_path,
    plot_patch,
):
    aplt.subplot_of_mapper(
        inversion=rectangular_inversion_7x7_3x3,
        mapper_index=0,
        output_path=plot_path,
        output_format="png",
    )
    assert str(Path(plot_path) / "inversion_0.png") in plot_patch.paths

    aplt.subplot_mappings(
        inversion=rectangular_inversion_7x7_3x3,
        pixelization_index=0,
        output_path=plot_path,
        output_format="png",
    )
    assert str(Path(plot_path) / "mappings_0.png") in plot_patch.paths
