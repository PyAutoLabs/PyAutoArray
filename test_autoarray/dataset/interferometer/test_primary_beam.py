import numpy as np
import pytest

import autoarray as aa
from autoarray.dataset.interferometer import primary_beam as primary_beam_util


def test__primary_beam_gaussian_from__peak_1_at_centre_and_half_at_half_fwhm():
    frequency_hz = 230.0e9
    dish_diameter_m = 12.0

    fwhm = primary_beam_util.primary_beam_fwhm_arcsec_from(
        frequency_hz=frequency_hz, dish_diameter_m=dish_diameter_m
    )

    # 1.13 lambda / D for ALMA's 12 m antennas at 230 GHz is ~25.3 arcsec.
    assert fwhm == pytest.approx(
        1.13 * 299792458.0 / frequency_hz / dish_diameter_m * 206264.80624709636,
        rel=1.0e-12,
    )
    assert fwhm == pytest.approx(25.3, abs=0.1)

    # An odd 41 x 41 grid whose pixel scale puts a pixel centre exactly at x = FWHM / 2.
    pixel_scale = (fwhm / 2.0) / 10.0

    mask = aa.Mask2D.all_false(shape_native=(41, 41), pixel_scales=pixel_scale)

    primary_beam = aa.primary_beam_gaussian_from(
        mask=mask, frequency_hz=frequency_hz, dish_diameter_m=dish_diameter_m
    )

    assert isinstance(primary_beam, aa.Array2D)
    assert primary_beam.native[20, 20] == pytest.approx(1.0, abs=1.0e-14)
    assert primary_beam.native[20, 30] == pytest.approx(0.5, rel=1.0e-12)
    assert primary_beam.native[10, 20] == pytest.approx(0.5, rel=1.0e-12)
    assert np.max(primary_beam.array) == pytest.approx(1.0, abs=1.0e-14)

    # An offset centre moves the peak.
    primary_beam = aa.primary_beam_gaussian_from(
        mask=mask,
        frequency_hz=frequency_hz,
        dish_diameter_m=dish_diameter_m,
        centre=(0.0, 10.0 * pixel_scale),
    )

    assert primary_beam.native[20, 30] == pytest.approx(1.0, abs=1.0e-14)
    assert primary_beam.native[20, 20] == pytest.approx(0.5, rel=1.0e-12)


def test__primary_beam_gaussian_from__pb_factor_scales_the_fwhm():
    fwhm_113 = primary_beam_util.primary_beam_fwhm_arcsec_from(
        frequency_hz=100.0e9, dish_diameter_m=12.0
    )
    fwhm_102 = primary_beam_util.primary_beam_fwhm_arcsec_from(
        frequency_hz=100.0e9, dish_diameter_m=12.0, pb_factor=1.02
    )

    assert fwhm_102 / fwhm_113 == pytest.approx(1.02 / 1.13, rel=1.0e-12)


def test__primary_beam_gaussian_from__masked_grid_matches_the_unmasked_values():
    mask_full = aa.Mask2D.all_false(shape_native=(9, 9), pixel_scales=1.0)
    mask = aa.Mask2D.circular(shape_native=(9, 9), pixel_scales=1.0, radius=3.0)

    beam_full = aa.primary_beam_gaussian_from(
        mask=mask_full, frequency_hz=1.0e12, dish_diameter_m=12.0
    )
    beam = aa.primary_beam_gaussian_from(
        mask=mask, frequency_hz=1.0e12, dish_diameter_m=12.0
    )

    assert beam.shape_slim == mask.pixels_in_mask
    np.testing.assert_allclose(
        beam.array, beam_full.native.array[mask.slim_to_native_tuple], rtol=1.0e-14
    )
