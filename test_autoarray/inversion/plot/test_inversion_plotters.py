import autoarray.plot as aplt
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.plot.inversion_plots import save_reconstruction_csv

import csv
import numpy as np
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


def test__inversion_subplot_of_mapper__singular_curvature_reg_matrix(
    rectangular_inversion_7x7_3x3,
    plot_path,
    plot_patch,
    monkeypatch,
):
    inversion = rectangular_inversion_7x7_3x3

    params = inversion.linear_obj_list[0].params

    monkeypatch.setattr(
        type(inversion),
        "reconstruction_noise_map_with_covariance",
        property(lambda self: np.sqrt(np.linalg.inv(np.zeros((params, params))))),
    )

    with pytest.raises(np.linalg.LinAlgError):
        inversion.reconstruction_noise_map_dict

    aplt.subplot_of_mapper(
        inversion=inversion,
        mapper_index=0,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "inversion_0.png") in plot_patch.paths


def test__save_reconstruction_csv__singular_curvature_reg_matrix(
    rectangular_inversion_7x7_3x3,
    tmp_path,
    monkeypatch,
):
    inversion = rectangular_inversion_7x7_3x3

    params = inversion.linear_obj_list[0].params

    monkeypatch.setattr(
        type(inversion),
        "reconstruction_noise_map_with_covariance",
        property(lambda self: np.sqrt(np.linalg.inv(np.zeros((params, params))))),
    )

    with pytest.raises(np.linalg.LinAlgError):
        inversion.reconstruction_noise_map_dict

    save_reconstruction_csv(inversion=inversion, output_path=tmp_path)

    csv_path = tmp_path / "source_plane_reconstruction_0.csv"

    assert csv_path.exists()

    with open(csv_path, mode="r") as f:
        reader = csv.reader(f)
        header_list = next(reader)
        row_list = [row for row in reader]

    # The column schema is unchanged, because consumers index the CSV by column name.

    assert header_list == ["y", "x", "reconstruction", "noise_map"]

    mapper = inversion.cls_list_from(cls=Mapper)[0]

    assert len(row_list) == len(mapper.source_plane_mesh_grid)

    for row in row_list:
        assert np.isfinite(float(row[0]))
        assert np.isfinite(float(row[1]))
        assert np.isnan(float(row[3]))
