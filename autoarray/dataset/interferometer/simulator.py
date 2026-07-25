import numpy as np

from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.operators.transformer import TransformerDFT
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray import exc
from autoarray.dataset import preprocess


class SimulatorInterferometer:
    def __init__(
        self,
        uv_wavelengths,
        exposure_time: float,
        transformer_class=TransformerDFT,
        noise_sigma=0.1,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
        use_jax: bool = False,
    ):
        """
        Simulates observations of `Interferometer` data, including transforming a real-space image to
        complex-valued visibilities in Fourier space and optionally adding complex Gaussian noise.

        The simulation of an `Interferometer` dataset uses the following steps:

        1) Receive as input a real-space image (e.g. a model galaxy or lens system) and a set of UV-plane
           baselines (uv_wavelengths) describing the interferometer configuration.
        2) Fourier transform the real-space image to the UV-plane using the configured transformer class
           (DFT or NUFFT) to produce model visibilities.
        3) Optionally add complex Gaussian noise to the visibilities, with the noise level controlled by
           `noise_sigma`. The noise is added independently to the real and imaginary parts.
        4) Create a constant noise map with value `noise_sigma` for every visibility. If noise is not
           added (`noise_sigma=None`), the noise map is filled with `noise_if_add_noise_false` instead.

        The returned `Interferometer` dataset contains the simulated visibilities, noise map,
        uv_wavelengths and real-space mask, and can be used directly with `FitInterferometer`.

        Parameters
        ----------
        uv_wavelengths
            The (u, v) baseline coordinates of the interferometer in units of wavelengths. This is a
            2D array of shape (total_visibilities, 2) where each row is a (u, v) baseline pair. These
            define the Fourier-space sampling of the observation.
        exposure_time
            The exposure time of the simulated interferometer observation in seconds.
        transformer_class
            The class used to perform the Fourier transform between real space and the UV-plane. The
            default `TransformerDFT` is suitable for small datasets (fewer than ~10,000 visibilities).
            For larger datasets use `TransformerNUFFT` for efficiency.
        noise_sigma
            The standard deviation of the complex Gaussian noise added to each visibility. Noise is
            added independently to the real and imaginary components. If `None`, no noise is added to
            the visibilities but a noise map is still created using `noise_if_add_noise_false`.
        noise_if_add_noise_false
            The noise value assigned to every visibility in the noise map when `noise_sigma=None`
            (i.e. when no noise is added to the data). This gives the noise map a non-zero value
            so that downstream fits remain well-defined.
        noise_seed
            The random seed used for noise generation. A value of -1 uses a different random seed
            on every run, producing different noise realisations each time.
        use_jax
            If ``True``, ``via_image_from`` defaults ``xp`` to ``jax.numpy`` and
            the simulator's internal complex-Gaussian noise generation routes
            through ``jax.random``. The returned ``Interferometer`` carries
            ``jax.Array`` visibilities. Mirror of ``SimulatorImaging.use_jax``.
        """

        self.uv_wavelengths = uv_wavelengths
        self.exposure_time = exposure_time
        self.transformer_class = transformer_class
        self.noise_sigma = noise_sigma
        self.noise_if_add_noise_false = noise_if_add_noise_false
        self.noise_seed = noise_seed
        self.use_jax = use_jax

    @property
    def _xp(self):
        """The array module the simulator runs against by default. ``jnp`` when
        ``use_jax=True``, ``np`` otherwise. ``via_image_from`` falls back to
        this when the caller does not pass ``xp=`` explicitly."""
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    def via_image_from(self, image, xp=None):
        """
        Simulate an `Interferometer` dataset from an input real-space image.

        The steps of the simulation process are described in the `SimulatorInterferometer` `__init__`
        docstring. In brief: the image is Fourier transformed to visibilities using the configured
        transformer class and the uv_wavelengths baselines, then complex Gaussian noise is optionally
        added and a constant noise map is created.

        Parameters
        ----------
        image
            The 2D real-space image from which the interferometer dataset is simulated (e.g. the
            surface brightness of a galaxy or lens system). Must be an `Array2D` with an associated
            mask that defines the real-space region used for the Fourier transform.
        xp
            The array module. When ``None`` (the default), falls back to ``self._xp`` —
            ``jnp`` if the simulator was constructed with ``use_jax=True``, ``np``
            otherwise. Pass explicitly to override.

        Returns
        -------
        Interferometer
            The simulated interferometer dataset containing visibilities, noise map, uv_wavelengths
            and the real-space mask derived from the input image.
        """
        if xp is None:
            xp = self._xp

        transformer = self.transformer_class(
            uv_wavelengths=self.uv_wavelengths, real_space_mask=image.mask
        )

        visibilities = transformer.visibilities_from(image=image)

        if self.noise_sigma is not None:
            visibilities = preprocess.data_with_complex_gaussian_noise_added(
                data=visibilities, sigma=self.noise_sigma, seed=self.noise_seed, xp=xp
            )
            noise_map = VisibilitiesNoiseMap.full(
                fill_value=self.noise_sigma, shape_slim=(visibilities.shape[0],)
            )
        else:
            noise_map = VisibilitiesNoiseMap.full(
                fill_value=self.noise_if_add_noise_false,
                shape_slim=(visibilities.shape[0],),
            )

        # NaN-noise guard is NumPy-side only — Python `if <tracer>:` triggers
        # TracerBoolConversionError under JAX. JAX users must confirm
        # noise_sigma is positive themselves.
        if xp is np and np.isnan(noise_map).any():
            raise exc.DatasetException(
                "The noise-map has NaN values in it. This suggests your exposure time and / or"
                "background sky levels are too low, creating signal counts at or close to 0.0."
            )

        return Interferometer(
            data=visibilities,
            noise_map=noise_map,
            uv_wavelengths=transformer.uv_wavelengths,
            real_space_mask=image.mask,
            transformer_class=self.transformer_class,
        )
