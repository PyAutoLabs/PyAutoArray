import os
from pathlib import Path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray.structures import visibilities as vis

test_data_path = Path(__file__).resolve().parent / "files"


def test__manual__makes_visibilities_without_other_inputs():
    visibilities = aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()
    assert (visibilities.in_array == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (visibilities.amplitudes == np.array([np.sqrt(5), 5.0])).all()
    assert visibilities.phases == pytest.approx(
        np.array([1.10714872, 0.92729522]), 1.0e-4
    )

    visibilities = aa.Visibilities(visibilities=[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])).all()
    assert (
        visibilities.in_array == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ).all()


def test__manual__makes_visibilities_with_converted_input_as_list():
    visibilities = aa.Visibilities(visibilities=[[1.0, 2.0], [3.0, 4.0]])

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()
    assert (visibilities.amplitudes == np.array([np.sqrt(5), 5.0])).all()
    assert visibilities.phases == pytest.approx(
        np.array([1.10714872, 0.92729522]), 1.0e-4
    )

    visibilities = aa.Visibilities(visibilities=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])).all()


def test__full__makes_visibilities_without_other_inputs():
    visibilities = aa.Visibilities.ones(shape_slim=(2,))

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([1.0 + 1.0j, 1.0 + 1.0j])).all()

    visibilities = aa.Visibilities.full(fill_value=2.0, shape_slim=(2,))

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([2.0 + 2.0j, 2.0 + 2.0j])).all()


def test__ones_zeros__makes_visibilities_without_other_inputs():
    visibilities = aa.Visibilities.ones(shape_slim=(2,))

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([1.0 + 1.0j, 1.0 + 1.0j])).all()

    visibilities = aa.Visibilities.zeros(shape_slim=(2,))

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()


def test__from_fits__makes_visibilities_without_other_inputs():
    visibilities = aa.Visibilities.from_fits(
        file_path=test_data_path / "3x2_ones.fits", hdu=0
    )

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])).all()

    visibilities = aa.Visibilities.from_fits(
        file_path=test_data_path / "3x2_twos.fits", hdu=0
    )

    assert type(visibilities) == vis.Visibilities
    assert (visibilities.slim == np.array([2.0 + 2.0j, 2.0 + 2.0j, 2.0 + 2.0j])).all()


def test__output_to_fits():
    files_path = Path(__file__).resolve().parent / "files"

    visibilities = aa.Visibilities.from_fits(
        file_path=files_path / "3x2_ones.fits", hdu=0
    )

    output_test_path = files_path / "output_test"

    if output_test_path.exists():
        shutil.rmtree(output_test_path)

    os.makedirs(output_test_path)

    from autonerves.fitsable import output_to_fits
    output_to_fits(values=visibilities.in_array, file_path=output_test_path / "data.fits")

    visibilities_from_out = aa.Visibilities.from_fits(
        file_path=output_test_path / "data.fits", hdu=0
    )
    assert (
        visibilities_from_out.slim == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    ).all()


def test__visibilities_noise_has_attributes():
    noise_map = aa.VisibilitiesNoiseMap(visibilities=[[1.0, 2.0], [3.0, 4.0]])

    assert type(noise_map) == vis.VisibilitiesNoiseMap
    assert (noise_map.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()
    assert (noise_map.amplitudes == np.array([np.sqrt(5), 5.0])).all()
    assert noise_map.phases == pytest.approx(np.array([1.10714872, 0.92729522]), 1.0e-4)
