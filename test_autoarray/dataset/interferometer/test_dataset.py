import numpy as np
import os
import shutil

import autoarray as aa
import pytest

from autoarray.operators import transformer
from pathlib import Path

test_data_path = Path(Path(__file__).resolve().parent) / "files"


def test__dirty_image__shape_native_matches_real_space_mask(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    assert dataset.dirty_image.shape_native == (7, 7)
    assert (dataset.transformer.image_from(visibilities=dataset.data)).all()


def test__dirty_noise_map__shape_native_matches_real_space_mask(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    assert dataset.dirty_noise_map.shape_native == (7, 7)
    assert (dataset.transformer.image_from(visibilities=dataset.noise_map)).all()


def test__dirty_signal_to_noise_map__shape_native_matches_real_space_mask(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    assert dataset.dirty_signal_to_noise_map.shape_native == (7, 7)
    assert (
        dataset.transformer.image_from(visibilities=dataset.signal_to_noise_map)
    ).all()


def test__from_fits__raise_error_dft_visibilities_limit__threads_kwarg(
    tmp_path, mask_2d_7x7
):
    """``from_fits`` must forward ``raise_error_dft_visibilities_limit`` to the
    ``Interferometer`` constructor so callers loading large DFT-based datasets can opt out
    of the >10,000-visibility safety check (e.g. for profiling the JAX-traceable DFT path)."""
    from astropy.io import fits

    n_visibilities = 10_001
    visibilities = np.ones((n_visibilities, 2), dtype=np.float64)
    noise_map = np.ones((n_visibilities, 2), dtype=np.float64)
    uv_wavelengths = np.zeros((n_visibilities, 2), dtype=np.float64)

    data_path = tmp_path / "data.fits"
    noise_map_path = tmp_path / "noise_map.fits"
    uv_path = tmp_path / "uv_wavelengths.fits"

    for arr, path in (
        (visibilities, data_path),
        (noise_map, noise_map_path),
        (uv_wavelengths, uv_path),
    ):
        fits.PrimaryHDU(data=arr).writeto(path, overwrite=True)

    with pytest.raises(aa.exc.DatasetException):
        aa.Interferometer.from_fits(
            data_path=data_path,
            noise_map_path=noise_map_path,
            uv_wavelengths_path=uv_path,
            real_space_mask=mask_2d_7x7,
            transformer_class=transformer.TransformerDFT,
        )

    dataset = aa.Interferometer.from_fits(
        data_path=data_path,
        noise_map_path=noise_map_path,
        uv_wavelengths_path=uv_path,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
        raise_error_dft_visibilities_limit=False,
    )

    assert dataset.uv_wavelengths.shape[0] == n_visibilities
    assert type(dataset.transformer) == transformer.TransformerDFT


def test__from_fits__all_files_in_one_fits__load_using_different_hdus(mask_2d_7x7):
    dataset = aa.Interferometer.from_fits(
        real_space_mask=mask_2d_7x7,
        data_path=Path(test_data_path) / "3x2_multiple_hdu.fits",
        visibilities_hdu=0,
        noise_map_path=Path(test_data_path) / "3x2_multiple_hdu.fits",
        noise_map_hdu=1,
        uv_wavelengths_path=Path(test_data_path) / "3x2_multiple_hdu.fits",
        uv_wavelengths_hdu=2,
    )

    assert (dataset.data == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])).all()
    assert (dataset.noise_map == np.array([2.0 + 2.0j, 2.0 + 2.0j, 2.0 + 2.0j])).all()
    assert (dataset.uv_wavelengths[:, 0] == 3.0 * np.ones(3)).all()
    assert (dataset.uv_wavelengths[:, 1] == 3.0 * np.ones(3)).all()


def test__output_all_arrays(mask_2d_7x7):
    test_data_path = Path(Path(__file__).resolve().parent) / "files"

    dataset = aa.Interferometer.from_fits(
        real_space_mask=mask_2d_7x7,
        data_path=Path(test_data_path) / "3x2_ones_twos.fits",
        noise_map_path=Path(test_data_path) / "3x2_threes_fours.fits",
        uv_wavelengths_path=Path(test_data_path) / "3x2_fives_sixes.fits",
    )

    test_data_path = Path(Path(__file__).resolve().parent) / "files" / "array" / "output_test"

    if Path(test_data_path).exists():
        shutil.rmtree(test_data_path)

    os.makedirs(test_data_path)

    from autoarray.dataset.plot.interferometer_plots import fits_interferometer

    fits_interferometer(
        dataset=dataset,
        data_path=Path(test_data_path) / "data.fits",
        noise_map_path=Path(test_data_path) / "noise_map.fits",
        uv_wavelengths_path=Path(test_data_path) / "uv_wavelengths.fits",
        overwrite=True,
    )

    dataset = aa.Interferometer.from_fits(
        real_space_mask=mask_2d_7x7,
        data_path=Path(test_data_path) / "data.fits",
        noise_map_path=Path(test_data_path) / "noise_map.fits",
        uv_wavelengths_path=Path(test_data_path) / "uv_wavelengths.fits",
    )

    assert (dataset.data == np.array([1.0 + 2.0j, 1.0 + 2.0j, 1.0 + 2.0j])).all()
    assert (dataset.noise_map == np.array([3.0 + 4.0j, 3.0 + 4.0j, 3.0 + 4.0j])).all()
    assert (dataset.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
    assert (dataset.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()


def test__transformer__dft_class__returns_transformer_dft_instance(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    interferometer_7 = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerDFT,
    )

    assert type(interferometer_7.transformer) == transformer.TransformerDFT


def test__transformer__nufft_class__returns_transformer_nufft_instance(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
):
    interferometer_7 = aa.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        transformer_class=transformer.TransformerNUFFT,
    )

    assert type(interferometer_7.transformer) == transformer.TransformerNUFFT


def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
    mask_2d_7x7,
):
    dataset = aa.Interferometer(
        data=aa.Visibilities.ones(shape_slim=(19,)),
        noise_map=2.0 * aa.Visibilities.ones(shape_slim=(19,)),
        uv_wavelengths=3.0 * np.ones((19, 2)),
        real_space_mask=mask_2d_7x7,
    )

    real_space_mask = aa.Mask2D.all_false(
        shape_native=(19, 19),
        pixel_scales=1.0,
        invert=True,
    )
    real_space_mask[9, 9] = False

    assert (dataset.data == 1.0 + 1.0j * np.ones((19,))).all()
    assert (dataset.noise_map == 2.0 + 2.0j * np.ones((19,))).all()
    assert (dataset.uv_wavelengths == 3.0 * np.ones((19, 2))).all()
