import logging
import numpy as np
import pytest

import autoarray as aa
from autoarray.dataset.abstract import dataset as ds

logger = logging.getLogger(__name__)


def test__dataset_takes_structures_of_different_formats():
    array = aa.Array1D.no_mask([1.0, 2.0], pixel_scales=1.0)
    noise_map = aa.Array1D.no_mask([1.0, 3.0], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert (dataset.data.native == np.array([1.0, 2.0])).all()
    assert (dataset.noise_map.native == np.array([1.0, 3.0])).all()

    array = aa.Array2D.no_mask([[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask([[1.0, 2.0], [3.0, 5.0]], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert (dataset.data.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (dataset.noise_map.native == np.array([[1.0, 2.0], [3.0, 5.0]])).all()


def test__signal_to_noise_map():
    array = aa.Array2D.no_mask([[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask([[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert dataset.signal_to_noise_map.native == pytest.approx(
        np.array([[0.1, 0.2], [0.1, 1.0]]), 1.0e-4
    )
    assert dataset.signal_to_noise_max == 1.0

    array = aa.Array2D.no_mask([[-1.0, 2.0], [3.0, -4.0]], pixel_scales=1.0)

    noise_map = aa.Array2D.no_mask([[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert dataset.signal_to_noise_map.native == pytest.approx(
        np.array([[0.0, 0.2], [0.1, 0.0]]), 1.0e-4
    )
    assert dataset.signal_to_noise_max == 0.2


def test__new_imaging_with_arrays_trimmed_via_kernel_shape():

    data = aa.Array2D.no_mask(
        values=[[20.0, 20.0, 20.0], [20.0, 5.0, 20.0], [20.0, 20.0, 20.0]],
        pixel_scales=1.0,
    )

    noise_map = aa.Array2D.no_mask(
        values=[[20.0, 20.0, 20.0], [20.0, 2.0, 20.0], [20.0, 20.0, 20.0]],
        pixel_scales=1.0,
    )

    dataset = ds.AbstractDataset(data=data, noise_map=noise_map)

    dataset_trimmed = dataset.trimmed_after_convolution_from(kernel_shape=(3, 3))

    assert (dataset_trimmed.data.native == np.array([[5.0]])).all()

    assert (dataset_trimmed.noise_map.native == np.array([[2.0]])).all()


def test__apply_over_sampling(image_7x7, noise_map_7x7):
    dataset_7x7 = aa.Imaging(
        data=image_7x7,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
        over_sample_size_pixelization=2,
    )

    # The grid and grid_pixelizaiton are a cached_property which needs to be reset,
    # Which the code below tests.

    grid_sub_2 = dataset_7x7.grids.lp
    grids_pixelization_sub_2 = dataset_7x7.grids.pixelization

    dataset_7x7.grids.__dict__["_lp"][0][0] = 100.0
    dataset_7x7.grids.__dict__["_pixelization"][0][0] = 100.0

    assert dataset_7x7.grids.lp[0][0] == pytest.approx(100.0, 1.0e-4)
    assert dataset_7x7.grids.pixelization[0][0] == pytest.approx(100.0, 1.0e-4)

    dataset_7x7 = dataset_7x7.apply_over_sampling(
        over_sample_size_lp=2,
        over_sample_size_pixelization=4,
    )

    assert dataset_7x7.over_sample_size_lp.slim[0] == 2
    assert dataset_7x7.over_sample_size_pixelization.slim[0] == 4

    assert dataset_7x7.grids.lp[0][0] == pytest.approx(3.0, 1.0e-4)
    assert dataset_7x7.grids.pixelization[0][0] == pytest.approx(3.0, 1.0e-4)
