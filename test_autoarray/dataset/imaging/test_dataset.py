import copy

import numpy as np
import pytest

import autoarray as aa

from autoarray import exc
from pathlib import Path

test_data_path = Path(Path(__file__).resolve().parent) / "files"


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


def test__from_fits__small_datasets_env_caps_data_and_noise_map(tmp_path, monkeypatch):
    """When PYAUTO_SMALL_DATASETS=1, Imaging.from_fits center-crops data and
    noise_map to (15, 15) at pixel_scales=0.6 so they stay shape-consistent
    with masks built via Mask2D.circular under the same env var. PSF is left
    alone."""
    from astropy.io import fits

    fits.writeto(
        tmp_path / "data_30x30.fits",
        data=np.ones((30, 30), dtype=np.float64),
        overwrite=True,
    )
    fits.writeto(
        tmp_path / "noise_map_30x30.fits",
        data=2.0 * np.ones((30, 30), dtype=np.float64),
        overwrite=True,
    )
    fits.writeto(
        tmp_path / "psf_5x5.fits",
        data=(1.0 / 25.0) * np.ones((5, 5), dtype=np.float64),
        overwrite=True,
    )

    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    dataset = aa.Imaging.from_fits(
        pixel_scales=0.08,
        data_path=tmp_path / "data_30x30.fits",
        psf_path=tmp_path / "psf_5x5.fits",
        noise_map_path=tmp_path / "noise_map_30x30.fits",
    )

    assert dataset.data.shape_native == (16, 16)
    assert dataset.noise_map.shape_native == (16, 16)
    assert dataset.pixel_scales == (0.6, 0.6)
    assert dataset.psf.kernel.shape_native == (5, 5)


def test__from_fits__small_datasets_env_unset__shape_unchanged(tmp_path, monkeypatch):
    """Sanity: with the env var unset, from_fits returns the on-disk shape
    unchanged, even for files larger than the cap."""
    from astropy.io import fits

    fits.writeto(
        tmp_path / "data_30x30.fits",
        data=np.ones((30, 30), dtype=np.float64),
        overwrite=True,
    )
    fits.writeto(
        tmp_path / "noise_map_30x30.fits",
        data=2.0 * np.ones((30, 30), dtype=np.float64),
        overwrite=True,
    )

    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    dataset = aa.Imaging.from_fits(
        pixel_scales=0.08,
        data_path=tmp_path / "data_30x30.fits",
        noise_map_path=tmp_path / "noise_map_30x30.fits",
    )

    assert dataset.data.shape_native == (30, 30)
    assert dataset.noise_map.shape_native == (30, 30)
    assert dataset.pixel_scales == (0.08, 0.08)


def test__output_to_fits__round_trips_data_psf_noise_map_correctly(imaging_7x7, tmp_path):

    from autoarray.dataset.plot.imaging_plots import fits_imaging

    fits_imaging(
        dataset=imaging_7x7,
        data_path=tmp_path / "data.fits",
        psf_path=tmp_path / "psf.fits",
        noise_map_path=tmp_path / "noise_map.fits",
        overwrite=True,
    )

    dataset = aa.Imaging.from_fits(
        pixel_scales=0.1,
        data_path=tmp_path / "data.fits",
        psf_path=tmp_path / "psf.fits",
        noise_map_path=tmp_path / "noise_map.fits",
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


def test__convolve_over_sample_size__validation_and_plumbing():
    data = aa.Array2D.no_mask(values=np.ones((11, 11)), pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask(values=np.ones((11, 11)), pixel_scales=1.0)
    kernel_fine = aa.Array2D.no_mask(values=np.ones((9, 9)), pixel_scales=0.5)
    psf = aa.Convolver(kernel=kernel_fine)

    # convolve size must be a plain int.
    with pytest.raises(TypeError):
        aa.Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            over_sample_size_lp=2,
            convolve_over_sample_size_lp=2.0,
        )

    # k x s coupling: every over_sample_size entry must be divisible by the
    # convolve size — a non-divisible int raises, divisible ints and adaptive
    # arrays are legal.
    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            over_sample_size_lp=3,
            convolve_over_sample_size_lp=2,
        )

    sub_size_adaptive = np.full(fill_value=2, shape=data.shape_slim)
    sub_size_adaptive[0] = 4

    dataset_adaptive = aa.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        over_sample_size_lp=aa.Array2D(values=sub_size_adaptive, mask=data.mask),
        convolve_over_sample_size_lp=2,
    )
    assert dataset_adaptive.convolve_over_sample_size_lp == 2

    sub_size_bad = np.full(fill_value=2, shape=data.shape_slim)
    sub_size_bad[0] = 3

    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            over_sample_size_lp=aa.Array2D(values=sub_size_bad, mask=data.mask),
            convolve_over_sample_size_lp=2,
        )

    # Differing lp / pixelization convolve sizes are not supported (single PSF kernel).
    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            over_sample_size_lp=2,
            over_sample_size_pixelization=4,
            convolve_over_sample_size_lp=2,
            convolve_over_sample_size_pixelization=4,
        )

    # Valid dataset: the psf carries the convolve size and apply_mask preserves it,
    # precomputing the fine state and building the blurring grid at the fine resolution.
    dataset = aa.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        over_sample_size_lp=2,
        convolve_over_sample_size_lp=2,
    )

    assert dataset.convolve_over_sample_size_lp == 2
    assert dataset.psf.convolve_over_sample_size == 2

    mask = aa.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=3.5)
    masked = dataset.apply_mask(mask=mask)

    assert masked.convolve_over_sample_size_lp == 2
    assert masked.psf.convolve_over_sample_size == 2
    assert masked.psf._state is not None
    assert masked.psf._state.sub_slim_to_fine_slim is not None

    # The blurring grid footprint uses the kernel's image-resolution shape (5x5 for a
    # 9x9 fine kernel at s=2) and is evaluated at the fine resolution.
    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=(5, 5), allow_padding=True
    )
    assert np.array(masked.grids.blurring.over_sampled).shape == (
        blurring_mask.pixels_in_mask * 4,
        2,
    )


def test__convolve_over_sample_size__sparse_operator_guard():
    data = aa.Array2D.no_mask(values=np.ones((11, 11)), pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask(values=np.ones((11, 11)), pixel_scales=1.0)
    kernel_fine = aa.Array2D.no_mask(values=np.ones((9, 9)), pixel_scales=0.5)
    psf = aa.Convolver(kernel=kernel_fine)

    dataset = aa.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        over_sample_size_pixelization=2,
        convolve_over_sample_size_pixelization=2,
    )

    with pytest.raises(aa.exc.DatasetException):
        dataset.apply_sparse_operator()
