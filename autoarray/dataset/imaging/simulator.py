import logging
import numpy as np
from typing import Optional, Union

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.dataset import preprocess

logger = logging.getLogger(__name__)


class SimulatorImaging:
    def __init__(
        self,
        exposure_time: float,
        background_sky_level: float = 0.0,
        subtract_background_sky: bool = True,
        psf: Convolver = None,
        use_real_space_convolution: bool = True,
        normalize_psf: bool = True,
        add_poisson_noise_to_data: bool = True,
        include_poisson_noise_in_noise_map: bool = True,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
        use_jax: bool = False,
    ):
        """
        Simulates observations of `Imaging` data, including simulating the image, noise, blurring due to the telescope
        optics via the Point Spread Function (PSF) and the background sky.

        The simulation of an `Imaging` dataset uses the following steps:

        1) Receive as input a raw image of what the data looks like before any simulation process is applied.
        2) Include diffraction due to the telescope optics by convolving the image with an input Point Spread
           Function (PSF).
        3) Use input values of the background sky level in every pixel of the image to add the background sky to
           the PSF convolved image.
        4) Add Poisson noise to the image, which represents noise due to whether photons hits the CCD and are converted
           to photo-electrons which are successfully detected by the CCD and converted to counts.
        5) Subtract the background sky from the image, so that the returned simulated dataset is background sky
           subtracted.

        The inputs of the `SimulatorImaging` object can toggle these steps on and off, for example if `psf=None` the
        PSF convolution step is omitted.

        The inputs `add_poisson_noise_to_data` and `include_poisson_noise_in_noise_map` provide flexibility in
        handling Poisson noise during simulations:

        - `add_poisson_noise_to_data` determines whether Poisson noise is added to the simulated data.
        - `include_poisson_noise_in_noise_map` decides if Poisson noise is included in the `noise_map` of the dataset.

        These options allow you to create a dataset without adding Poisson noise to the data while still using
        a `noise_map` that represents the noise levels expected if Poisson noise had been included.

        This approach is particularly useful for diagnostic purposes. By simulating noise-free data but retaining
        realistic noise levels in the `noise_map`, you can test whether the model-fitting process accurately recovers
        the input parameters. Unlike noisy data, where random Poisson noise introduces offsets in each simulation,
        this method ensures the posterior peaks precisely around the input values. This is a powerful way to identify
        potential systematic biases in a model-fitting analysis.

        Parameters
        ----------
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        subtract_background_sky
            If `True`, the background sky level is subtracted from the simulated dataset, otherwise it is left in.
        psf
            An array describing the PSF kernel of the image, if None no PSF convolution is applied.
        use_real_space_convolution
            If `True`, the PSF convolution is performed in real space, if `False` it is performed in Fourier space.
            Real space convolution avoids aliasing effects that can occur in Fourier space convolution, but is slower.
            If the PSF kernel is small (e.g. 11x11) then real space convolution is recommended, whereas if the PSF
            kernel is large (e.g. 51x51) then Fourier space convolution with care to pad the image to avoid aliasing is recommended.
        normalize_psf
            If `True`, the PSF kernel is normalized so all values sum to 1.0.
        add_poisson_noise_to_data
            Whether Poisson noise corresponding to photon count statistics on the imaging observation is added.
        include_poisson_noise_in_noise_map
            Whether Poisson noise is included in the noise-map of the dataset. If `False` the noise-map is filled with
            the value `noise_if_add_noise_false`.
        noise_if_add_noise_false
            If noise is not added to the simulated dataset a `noise_map` must still be returned. This value gives
            the value of noise assigned to every pixel in the noise-map.
        noise_seed
            The random seed used to add random noise, where -1 corresponds to a random seed every run.
        use_jax
            If ``True``, ``via_image_from`` defaults ``xp`` to ``jax.numpy`` and the
            simulator's internal noise generation routes through ``jax.random``.
            The returned ``Imaging`` carries ``jax.Array`` data — useful for
            JAX-eager batched simulation (parameter sweeps, mock-data studies).
            Defaults to ``False``.
        """

        if psf is not None:
            if psf is not None and normalize_psf:
                psf = psf.normalized
            self.psf = psf
        else:
            self.psf = Convolver.no_blur(pixel_scales=1.0)

        self.use_real_space_convolution = use_real_space_convolution
        self.exposure_time = exposure_time
        self.background_sky_level = background_sky_level
        self.subtract_background_sky = subtract_background_sky
        self.add_poisson_noise_to_data = add_poisson_noise_to_data
        self.include_poisson_noise_in_noise_map = include_poisson_noise_in_noise_map
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed
        self.use_jax = use_jax

    @property
    def _xp(self):
        """The array module the simulator runs against by default. ``jnp`` when
        ``use_jax=True``, ``np`` otherwise. ``via_image_from`` falls back to this
        when the caller does not pass ``xp=`` explicitly."""
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    def via_image_from(
        self,
        image: Array2D,
        over_sample_size: Optional[Union[int, np.ndarray]] = None,
        xp=None,
    ) -> Imaging:
        """
        Simulate an `Imaging` dataset from an input image.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring.

        Parameters
        ----------
        image
            The 2D image from which the Imaging dataset is simulated (e.g. a model galaxy image
            before any telescope effects are applied). Must be an `Array2D`.
        over_sample_size
            If provided, the returned dataset has its over-sampling updated via `apply_over_sampling`.
            Should be an `Array2D` of integer sub-grid sizes with the same shape as the image.
        xp
            The array module to use for PSF convolution. When ``None`` (the default),
            falls back to ``self._xp`` — which is ``jnp`` if the simulator was constructed
            with ``use_jax=True`` and ``np`` otherwise. Pass explicitly to override.

        Returns
        -------
        Imaging
            The simulated imaging dataset with PSF convolution, noise and background sky applied.
        """
        if xp is None:
            xp = self._xp

        exposure_time_map = Array2D.full(
            fill_value=self.exposure_time,
            shape_native=image.shape_native,
            pixel_scales=image.pixel_scales,
        )

        background_sky_map = Array2D.full(
            fill_value=self.background_sky_level,
            shape_native=image.shape_native,
            pixel_scales=image.pixel_scales,
        )

        if self.use_real_space_convolution:
            image = self.psf.convolved_image_via_real_space_from(
                image=image,
                blurring_image=None,
                xp=xp,
            )
        else:
            image = self.psf.convolved_image_from(
                image=image, blurring_image=None, xp=xp
            )

        image = image + background_sky_map

        image_with_poisson_noise = preprocess.data_eps_with_poisson_noise_added(
            data_eps=image,
            exposure_time_map=exposure_time_map,
            seed=self.noise_seed,
            xp=xp,
        )

        if self.add_poisson_noise_to_data:
            image = image_with_poisson_noise

        if self.include_poisson_noise_in_noise_map:
            noise_map = preprocess.noise_map_via_data_eps_and_exposure_time_map_from(
                data_eps=image_with_poisson_noise, exposure_time_map=exposure_time_map
            )

        else:
            noise_map = Array2D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=image.shape_native,
                pixel_scales=image.pixel_scales,
            )

        # NaN-noise guard is a NumPy-side runtime check (Python `if` on a JAX
        # tracer triggers TracerBoolConversionError). Under JAX it is the user's
        # responsibility to confirm exposure_time / background_sky_level are
        # large enough — JAX silently propagates NaN through the trace.
        if xp is np and np.isnan(noise_map.array).any():
            raise exc.DatasetException(
                "The noise-map has NaN values in it. This suggests your exposure time and / or"
                "background sky levels are too low, creating signal counts at or close to 0.0."
            )

        if self.subtract_background_sky:
            image = image - background_sky_map

        mask = Mask2D.all_false(
            shape_native=image.shape_native,
            pixel_scales=image.pixel_scales,
            origin=image.origin,
        )

        image = Array2D(values=image.native, mask=mask)

        dataset = Imaging(
            data=image,
            psf=self.psf,
            noise_map=noise_map,
            check_noise_map=False,
        )

        if over_sample_size is not None:

            dataset = dataset.apply_over_sampling(
                over_sample_size_lp=over_sample_size.native,
            )

        return dataset
