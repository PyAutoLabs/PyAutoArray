"""
Helpers for building the primary beam `P` of an interferometer dataset.

The primary beam is the direction-dependent sensitivity of each antenna's dish: the
interferometer measures the visibilities of the apparent sky `P x I`, not of the true sky
`I`. It is passed to `Interferometer(primary_beam=...)` as an image on the real-space
mask, after which the transformers and the sparse (w-tilde) inversion apply it so every
light profile, linear light profile and pixelization models the true sky.

A user-supplied beam image (e.g. one exported by CASA) is the general API; the Gaussian
here is the standard approximation to the main lobe of a uniformly illuminated dish.
"""

import numpy as np
from typing import Tuple

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D

SPEED_OF_LIGHT_M_PER_S = 299792458.0

ARCSEC_PER_RADIAN = 180.0 / np.pi * 3600.0


def primary_beam_fwhm_arcsec_from(
    frequency_hz: float, dish_diameter_m: float, pb_factor: float = 1.13
) -> float:
    """
    Returns the full width at half maximum, in arcseconds, of the Gaussian approximation to
    the primary beam of a dish of diameter `D` observing at frequency `nu`:

        FWHM = pb_factor * lambda / D = pb_factor * c / (nu * D)  [radians]

    Parameters
    ----------
    frequency_hz
        The observing frequency in Hz.
    dish_diameter_m
        The antenna dish diameter in metres.
    pb_factor
        The FWHM in units of `lambda / D`; `1.13` is the value for ALMA's 12 m antennas
        (a tapered illumination broadens the beam beyond the uniform-dish value of 1.02).
    """
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / frequency_hz

    return float(pb_factor * wavelength_m / dish_diameter_m * ARCSEC_PER_RADIAN)


def primary_beam_gaussian_from(
    mask: Mask2D,
    frequency_hz: float,
    dish_diameter_m: float,
    pb_factor: float = 1.13,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> Array2D:
    """
    Returns a circular Gaussian primary beam `P` on the unmasked pixels of a real-space mask,
    with peak value 1 at `centre` and FWHM `pb_factor * c / frequency_hz / dish_diameter_m`
    (converted to arcseconds):

        P(r) = exp(-4 ln(2) r^2 / FWHM^2)

    so `P = 0.5` at `r = FWHM / 2`. Coordinates are the pixel centres of `Grid2D.from_mask`
    (arcseconds), so the beam is evaluated on exactly the grid the transformers use.

    Parameters
    ----------
    mask
        The real-space mask of the interferometer dataset.
    frequency_hz
        The observing frequency in Hz.
    dish_diameter_m
        The antenna dish diameter in metres.
    pb_factor
        The FWHM in units of `lambda / D` (default `1.13`).
    centre
        The `(y, x)` arcsecond position of the beam peak (the pointing centre relative to the
        image origin).
    """
    fwhm = primary_beam_fwhm_arcsec_from(
        frequency_hz=frequency_hz,
        dish_diameter_m=dish_diameter_m,
        pb_factor=pb_factor,
    )

    grid = Grid2D.from_mask(mask=mask).array

    r_squared = (grid[:, 0] - centre[0]) ** 2 + (grid[:, 1] - centre[1]) ** 2

    values = np.exp(-4.0 * np.log(2.0) * r_squared / fwhm**2)

    return Array2D(values=values, mask=mask)


def primary_beam_slim_from(primary_beam, mask: Mask2D) -> np.ndarray:
    """
    Returns a primary beam as a float64 NumPy array on the slim (unmasked, 1D) grid of `mask`,
    the representation the transformers and the sparse operator multiply by.

    `primary_beam` may be an `Array2D` (on `mask`, or on another mask of the same native
    shape, in which case its native values are read at `mask`'s unmasked pixels) or a plain
    array of either the native shape `mask.shape_native` or the slim shape
    `(mask.pixels_in_mask,)`.

    Raises
    ------
    exc.DatasetException
        If the primary beam's shape matches neither the native nor the slim shape of `mask`.
    """
    from autoarray import exc

    if isinstance(primary_beam, Array2D):
        if primary_beam.mask is mask or (
            primary_beam.mask.shape_native == mask.shape_native
            and np.array_equal(np.asarray(primary_beam.mask), np.asarray(mask))
        ):
            return np.asarray(primary_beam.slim.array, dtype=np.float64)

        values = np.asarray(primary_beam.native.array, dtype=np.float64)
    else:
        values = np.asarray(getattr(primary_beam, "array", primary_beam))
        values = values.astype(np.float64)

    if values.shape == tuple(mask.shape_native):
        return np.ascontiguousarray(values[mask.slim_to_native_tuple])

    if values.shape == (mask.pixels_in_mask,):
        return np.ascontiguousarray(values)

    raise exc.DatasetException(
        f"The primary beam has shape {values.shape}, which matches neither the native shape "
        f"{tuple(mask.shape_native)} nor the slim shape ({mask.pixels_in_mask},) of the "
        f"interferometer's real-space mask. Pass the primary beam as an image on the "
        f"real-space mask (e.g. via `aa.primary_beam_gaussian_from(mask=real_space_mask, ...)`)."
    )
