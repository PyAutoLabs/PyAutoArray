import copy
import os

import numpy as np
import pytest
import shutil

import autoarray as aa

from autoarray import exc
from pathlib import Path

test_data_path = Path(Path(__file__).resolve().parent) / "files"


@pytest.fixture(name="test_data_path")
def make_test_data_path():
    test_data_path = Path(__file__).resolve().parent / "files" / "array" / "output_test"

    if test_data_path.exists():
        shutil.rmtree(test_data_path)

    os.makedirs(test_data_path)

    return test_data_path


def test__grid__uses_mask_and_settings__lp_grid_matches_grid_2d_7x7(
    image_7x7,
    noise_map_7x7,
    mask_2d_7x7,
    grid_2d_7x7,
):
    masked_image_7x7 = aa.Array2D(
        values=image_7x7.native,
        mask=mask_2d_7x7,
    )

    masked_noise_map_7x7 = aa.Array2D(values=noise_map_7x7.native, mask=mask_2d_7x7)

    masked_imaging_7x7 = aa.Imaging(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        over_sample_size_lp=2,
    )

    assert isinstance(masked_imaging_7x7.grids.lp, aa.Grid2D)
    assert (masked_imaging_7x7.grids.lp == grid_2d_7x7).all()
    assert (masked_imaging_7x7.grids.lp.slim == grid_2d_7x7).all()


def test__grids_pixelization__uses_mask_and_settings__default_over_sample__matches_grid_2d_7x7(
    image_7x7,
    noise_map_7x7,
    mask_2d_7x7,
    grid_2d_7x7,
):
    masked_image_7x7 = aa.Array2D(values=image_7x7.native, mask=mask_2d_7x7)

    masked_noise_map_7x7 = aa.Array2D(values=noise_map_7x7.native, mask=mask_2d_7x7)

    masked_imaging_7x7 = aa.Imaging(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
    )

    assert (masked_imaging_7x7.grids.pixelization == grid_2d_7x7).all()
    assert (masked_imaging_7x7.grids.pixelization.slim == grid_2d_7x7).all()


def test__grids_pixelization__uses_mask_and_settings__custom_over_sample__returns_grid2d_with_correct_size(
    image_7x7,
    noise_map_7x7,
    mask_2d_7x7,
    grid_2d_7x7,
):
    masked_image_7x7 = aa.Array2D(values=image_7x7.native, mask=mask_2d_7x7)

    masked_noise_map_7x7 = aa.Array2D(values=noise_map_7x7.native, mask=mask_2d_7x7)

    masked_imaging_7x7 = aa.Imaging(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        over_sample_size_lp=2,
        over_sample_size_pixelization=4,
    )

    assert isinstance(masked_imaging_7x7.grids.pixelization, aa.Grid2D)
    assert masked_imaging_7x7.grids.over_sample_size_pixelization[0] == 4


def test__grid_settings__sub_size__returns_correct_over_sample_sizes(
    image_7x7, noise_map_7x7
):
    dataset_7x7 = aa.Imaging(
        data=image_7x7,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
        over_sample_size_pixelization=4,
    )

    assert dataset_7x7.grids.over_sample_size_lp[0] == 2
    assert dataset_7x7.grids.over_sample_size_pixelization[0] == 4


def test__noise_covariance_input__noise_map_uses_diag():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    noise_covariance_matrix = np.ones(shape=(9, 9))

    dataset = aa.Imaging(data=image, noise_covariance_matrix=noise_covariance_matrix)

    noise_map = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    assert (dataset.noise_map.native == noise_map.native).all()


def test__no_noise_map__raises_exception():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(data=image)


def test__from_fits__separate_fits_files__loads_data_psf_noise_map_correctly():
    dataset = aa.Imaging.from_fits(
        pixel_scales=0.1,
        data_path=Path(test_data_path) / "3x3_ones.fits",
        psf_path=Path(test_data_path) / "3x3_twos.fits",
        noise_map_path=Path(test_data_path) / "3x3_threes.fits",
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert dataset.psf.kernel.native == pytest.approx(
        (1.0 / 9.0) * np.ones((3, 3)), 1.0e-4
    )
    assert (dataset.noise_map.native == 3.0 * np.ones((3, 3))).all()

    assert dataset.pixel_scales == (0.1, 0.1)
    assert dataset.psf.kernel.mask.pixel_scales == (0.1, 0.1)
    assert dataset.noise_map.mask.pixel_scales == (0.1, 0.1)


def test__from_fits__all_data_in_one_fits_file_multiple_hdus__loads_data_psf_noise_map_correctly():
    dataset = aa.Imaging.from_fits(
        pixel_scales=0.1,
        data_path=Path(test_data_path) / "3x3_multiple_hdu.fits",
        data_hdu=0,
        psf_path=Path(test_data_path) / "3x3_multiple_hdu.fits",
        psf_hdu=1,
        noise_map_path=Path(test_data_path) / "3x3_multiple_hdu.fits",
        noise_map_hdu=2,
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert dataset.psf.kernel.native == pytest.approx(
        (1.0 / 9.0) * np.ones((3, 3)), 1.0e-4
    )
    assert (dataset.noise_map.native == 3.0 * np.ones((3, 3))).all()

    assert dataset.pixel_scales == (0.1, 0.1)
    assert dataset.psf.kernel.mask.pixel_scales == (0.1, 0.1)
    assert dataset.noise_map.mask.pixel_scales == (0.1, 0.1)


def test__output_to_fits__round_trips_data_psf_noise_map_correctly(
    imaging_7x7, test_data_path
):

    from autoarray.dataset.plot.imaging_plots import fits_imaging

    fits_imaging(
        dataset=imaging_7x7,
        data_path=Path(test_data_path) / "data.fits",
        psf_path=Path(test_data_path) / "psf.fits",
        noise_map_path=Path(test_data_path) / "noise_map.fits",
        overwrite=True,
    )

    dataset = aa.Imaging.from_fits(
        pixel_scales=0.1,
        data_path=Path(test_data_path) / "data.fits",
        psf_path=Path(test_data_path) / "psf.fits",
        noise_map_path=Path(test_data_path) / "noise_map.fits",
    )

    assert (dataset.data.native == np.ones((7, 7))).all()
    assert dataset.psf.kernel.native[1, 1] == pytest.approx(0.33333, 1.0e-4)
    assert (dataset.noise_map.native == 2.0 * np.ones((7, 7))).all()
    assert dataset.pixel_scales == (0.1, 0.1)


def test__apply_mask__data_noise_map_psf_correctly_masked(
    imaging_7x7, mask_2d_7x7, psf_3x3
):
    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    assert (masked_imaging_7x7.data.slim == np.ones(9)).all()

    assert (
        masked_imaging_7x7.data.native == np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()

    assert (masked_imaging_7x7.noise_map.slim == 2.0 * np.ones(9)).all()
    assert (
        masked_imaging_7x7.noise_map.native
        == 2.0 * np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()

    assert masked_imaging_7x7.psf.kernel.slim == pytest.approx(
        (1.0 / 3.0) * psf_3x3.kernel.slim.array, 1.0e-4
    )

    assert type(masked_imaging_7x7.psf) == aa.Convolver


def test__apply_noise_scaling__masked_pixel_data_zeroed_and_noise_set_to_noise_value(
    imaging_7x7, mask_2d_7x7
):
    masked_imaging_7x7 = imaging_7x7.apply_noise_scaling(
        mask=mask_2d_7x7, noise_value=1e5
    )

    assert masked_imaging_7x7.data.native[4, 4] == 0.0
    assert masked_imaging_7x7.noise_map.native[4, 4] == 1e5


def test__apply_noise_scaling__use_signal_to_noise_value__noise_map_scaled_to_match_snr(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7
):

    image_7x7 = np.array(image_7x7.native.array)
    image_7x7[3, 3] = 2.0

    image_7x7 = aa.Array2D(values=image_7x7, mask=mask_2d_7x7)

    imaging_7x7 = aa.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=1,
    )

    masked_imaging_7x7 = imaging_7x7.apply_noise_scaling(
        mask=mask_2d_7x7, signal_to_noise_value=0.1, should_zero_data=False
    )

    assert masked_imaging_7x7.data.native[3, 4] == 1.0
    assert masked_imaging_7x7.noise_map.native[3, 4] == 10.0
    assert masked_imaging_7x7.data.native[3, 3] == 2.0
    assert masked_imaging_7x7.noise_map.native[3, 3] == 10.0


def test__apply_mask__noise_covariance_matrix__submatrix_extracted_for_unmasked_pixels():
    image = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=(1.0, 1.0))

    noise_covariance_matrix = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0],
        ]
    )

    mask = np.array(
        [
            [False, True],
            [True, False],
        ]
    )

    mask_2d = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    dataset = aa.Imaging(data=image, noise_covariance_matrix=noise_covariance_matrix)

    masked_dataset = dataset.apply_mask(mask=mask_2d)

    assert masked_dataset.noise_covariance_matrix == pytest.approx(
        np.array([[1.0, 1.0], [4.0, 4.0]]), 1.0e-4
    )


def test__different_imaging_without_mock_objects__customize_constructor_inputs__single_unmasked_pixel_correct():

    kernel = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=3.0)
    psf = aa.Convolver(kernel=kernel)

    dataset = aa.Imaging(
        data=aa.Array2D.ones(shape_native=(19, 19), pixel_scales=3.0),
        psf=psf,
        noise_map=aa.Array2D.full(
            fill_value=2.0, shape_native=(19, 19), pixel_scales=3.0
        ),
    )
    mask = aa.Mask2D.all_false(
        shape_native=(19, 19),
        pixel_scales=1.0,
        invert=True,
    )
    mask[9, 9] = False

    masked_dataset = dataset.apply_mask(mask=mask)

    assert masked_dataset.psf.kernel.native == pytest.approx(
        (1.0 / 49.0) * np.ones((7, 7)), 1.0e-4
    )
    assert (masked_dataset.data == np.array([1.0])).all()
    assert (masked_dataset.noise_map == np.array([2.0])).all()


def test__noise_map_unmasked_has_zeros__raises_exception():
    array = aa.Array2D.no_mask([[1.0, 2.0]], pixel_scales=1.0)

    noise_map = aa.Array2D.no_mask([[0.0, 3.0]], pixel_scales=1.0)

    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(data=array, noise_map=noise_map)


def test__noise_map_unmasked_has_negative_values__raises_exception():
    array = aa.Array2D.no_mask([[1.0, 2.0]], pixel_scales=1.0)

    noise_map = aa.Array2D.no_mask([[-1.0, 3.0]], pixel_scales=1.0)

    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(data=array, noise_map=noise_map)


def test__psf_not_odd_x_odd_kernel__raises_error():

    with pytest.raises(exc.KernelException):
        image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
        noise_map = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
        kernel = aa.Array2D.no_mask(values=[[0.0, 1.0], [1.0, 2.0]], pixel_scales=1.0)
        psf = aa.Convolver(kernel=kernel)

        dataset = aa.Imaging(
            data=image,
            noise_map=noise_map,
            psf=psf,
        )
