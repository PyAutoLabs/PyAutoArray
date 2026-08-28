import logging
import numpy as np
from typing import Optional

from autonerves.fitsable import ndarray_via_fits_from
from autonerves import cached_property

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.grids import GridsDataset
from autoarray.inversion.inversion.interferometer.inversion_interferometer_util import (
    InterferometerSparseOperator,
)
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray import exc
from autoarray.inversion.inversion.interferometer import (
    inversion_interferometer_util,
)

logger = logging.getLogger(__name__)


class Interferometer(AbstractDataset):
    def __init__(
        self,
        data: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        uv_wavelengths: np.ndarray,
        real_space_mask: Mask2D,
        transformer_class=TransformerNUFFT,
        sparse_operator: Optional[InterferometerSparseOperator] = None,
        raise_error_dft_visibilities_limit: bool = True,
    ):
        """
        An interferometer dataset, containing the visibilities data, noise-map, real-space msk, Fourier transformer and
        associated quantities for calculations like the grid.

        This object is the input to the `FitInterferometer` object, which fits the dataset with model visibilities
        and quantifies the goodness-of-fit via a residual map, likelihood, chi-squared and other quantities.

        The following quantities of the interferometer data are available and used for the following tasks:

        - `data`: The visibilities data, which shows the signal that is analysed and fitted with model visibilities.

        - `noise_map`: The RMS standard deviation error in every visibility, which is used to compute the chi-squared
        value and likelihood of a fit.

        - `uv_wavelengths`: The baselines of the interferometer which are used to Fourier transform a real space
        image to the uv-plane.

        `real_space_mask`: Defines in real space where the signal is present. This mask is used to transform images to
        Fourier space via the Fourier transform. The grids contained in the settings are aligned with this mask.

        The dataset also has a number of (y,x) grids of coordinates associated with it, which map to the centres
        of its image pixels. They are used for performing calculations which map directly to the data and have
        over sampling calculations built in which approximate the 2D line integral of these calculations within a
        pixel. This is explained in more detail in the `GridsDataset` class.

        Parameters
        ----------
        data
            The array of the visibilities data containing the signal that is fitted.
        noise_map
            An array describing the RMS standard deviation error in each visibility used for computing quantities like the
            chi-squared in a fit.
        uv_wavelengths
            The baselines of the interferometer which are used to Fourier transform a real space
            image to the uv-plane.
        real_space_mask
            Defines in real space where the signal is present. This mask is used to transform images to
            Fourier space via the Fourier transform. The grids contained in the settings are aligned with this mask.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        transformer_class
            The class of the Fourier Transform which maps images from real space to Fourier space visibilities and
            the uv-plane.
        sparse_operator
            A precomputed `InterferometerSparseOperator` containing the NUFFT precision matrix for efficient
            pixelized source reconstruction. This is computed via `apply_sparse_operator()` and can be passed
            here directly to avoid recomputing it (e.g. when loading a cached result from disk).
        raise_error_dft_visibilities_limit
            If `True`, an exception is raised if the dataset has more than 10,000 visibilities and
            `transformer_class=TransformerDFT`. The DFT is too slow for large datasets and `TransformerNUFFT`
            should be used instead. Set to `False` to suppress this check.
        """
        self.real_space_mask = real_space_mask

        super().__init__(
            data=data,
            noise_map=noise_map,
            over_sample_size_lp=1,
            over_sample_size_pixelization=1,
        )

        self.uv_wavelengths = uv_wavelengths

        self.transformer = transformer_class(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
        )

        self.grids = GridsDataset(
            mask=self.real_space_mask,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
        )

        self.sparse_operator = sparse_operator

        if raise_error_dft_visibilities_limit:
            if (
                self.uv_wavelengths.shape[0] > 10000
                and transformer_class == TransformerDFT
            ):
                raise exc.DatasetException(
                    """
                    Interferometer datasets with more than 10,000 visibilities should use the TransformerNUFFT class for 
                    efficient Fourier transforms between real and uv-space. The DFT (Discrete Fourier Transform) is too slow for 
                    large datasets.
                    
                    If you are certain you want to use the TransformerDFT class, you can disable this error by passing 
                    the input `raise_error_dft_visibilities_limit=False` when loading the Interferometer dataset.
                    """
                )

    @classmethod
    def from_fits(
        cls,
        data_path,
        noise_map_path,
        uv_wavelengths_path,
        real_space_mask,
        visibilities_hdu=0,
        noise_map_hdu=0,
        uv_wavelengths_hdu=0,
        transformer_class=TransformerNUFFT,
        raise_error_dft_visibilities_limit: bool = True,
    ):
        """
        Load an interferometer dataset from multiple .fits files.

        The visibilities (complex-valued Fourier-space data), noise map and uv_wavelengths (baseline
        coordinates) are each loaded from separate .fits files. A real-space mask defining the sky
        region used for Fourier transforms must be supplied separately.

        The visibilities are assumed to be stored as a 2D array of shape (total_visibilities, 2) where
        column 0 is the real component and column 1 is the imaginary component. The noise map has the
        same shape. The uv_wavelengths are a 2D array of shape (total_visibilities, 2) with columns
        corresponding to the (u, v) baseline coordinates in units of wavelengths.

        Parameters
        ----------
        data_path
            The path to the .fits file containing the visibility data
            (e.g. '/path/to/visibilities.fits').
        noise_map_path
            The path to the .fits file containing the visibility noise map
            (e.g. '/path/to/noise_map.fits').
        uv_wavelengths_path
            The path to the .fits file containing the (u, v) baseline coordinates in units of
            wavelengths (e.g. '/path/to/uv_wavelengths.fits').
        real_space_mask
            A `Mask2D` defining the real-space region of the sky that contains signal. This mask
            determines the pixel grid used by the Fourier transformer and the coordinate grids
            associated with the dataset.
        visibilities_hdu
            The HDU index in the visibilities .fits file from which data is loaded.
        noise_map_hdu
            The HDU index in the noise map .fits file from which data is loaded.
        uv_wavelengths_hdu
            The HDU index in the uv_wavelengths .fits file from which data is loaded.
        transformer_class
            The class of the Fourier Transform which maps images from real space to Fourier space
            visibilities. Defaults to `TransformerNUFFT` for efficiency with large datasets.
        raise_error_dft_visibilities_limit
            If True (default), raise a `DatasetException` when ``transformer_class`` is
            `TransformerDFT` and the dataset has more than 10,000 visibilities. Set to False to
            opt into the slow DFT path at ALMA-scale (e.g. when profiling the JAX-traceable
            DFT path before a JIT-friendly NUFFT is available).

        Returns
        -------
        Interferometer
            The interferometer dataset loaded from the .fits files.
        """

        visibilities = Visibilities.from_fits(file_path=data_path, hdu=visibilities_hdu)

        noise_map = VisibilitiesNoiseMap.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu
        )

        uv_wavelengths = ndarray_via_fits_from(
            file_path=uv_wavelengths_path, hdu=uv_wavelengths_hdu
        )

        return Interferometer(
            real_space_mask=real_space_mask,
            data=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            transformer_class=transformer_class,
            raise_error_dft_visibilities_limit=raise_error_dft_visibilities_limit,
        )

    def apply_sparse_operator(
        self,
        nufft_precision_operator=None,
        batch_size: int = 128,
        chunk_k: int = 2048,
        show_progress: bool = False,
        show_memory: bool = False,
        use_jax: bool = False,
    ):
        """
        Precompute the NUFFT precision operator for efficient pixelized source reconstruction.

        The sparse linear algebra formalism precomputes the Fourier Transform response matrix for all
        visibility baselines, enabling fast repeated evaluation during model fitting. This avoids
        recomputing the full NUFFT on every likelihood call.

        The resulting `InterferometerSparseOperator` is stored on the returned `Interferometer` dataset
        and is used automatically by `FitInterferometer` when performing pixelized reconstructions via
        the inversion module.

        Computing the NUFFT precision matrix from scratch can be very slow (runtime scales with both
        the number of visibilities and the real-space mask resolution — potentially hours for large
        datasets). The result can be cached to disk and reloaded to avoid recomputation.

        Both `TransformerDFT` and `TransformerNUFFT` are supported here and agree to ~3e-13 relative.
        Which is faster is governed by the product `N_vis * N_pix`, because the DFT setup cost scales
        as `O(N_vis * N_pix)` whereas the NUFFT scales as `O((N_vis + N_pix) log N)` on top of a fixed
        ~2s overhead:

        - below `N_vis * N_pix ~ 1e7`, `TransformerDFT` is faster (measured 0.2-0.7x the NUFFT time)
        - above it, `TransformerNUFFT` is faster (1.2-1.9x at 1e7-1e8)
        - above `~1e8`, the DFT's `O(N_vis * N_pix)` allocation makes it infeasible rather than merely
          slow — extrapolating a measured 446 MB at N_vis=4e3 / N_pix=1e4 gives ~109 GB at 1M
          visibilities, whereas the NUFFT path allocates nothing beyond its working buffers.

        As a rule of thumb at a typical 64x64 mask the crossover sits near 5,000 visibilities, but on a
        coarser mask the DFT stays ahead well beyond that — it is the product that matters, not the
        visibility count alone.

        Parameters
        ----------
        nufft_precision_operator
            An already computed NUFFT precision matrix for this dataset (e.g. loaded from disk via
            `np.load`) to avoid an expensive recomputation. If `None` it is computed from scratch
            by calling `psf_precision_operator_from()`.
        batch_size
            The number of real-space pixels processed per batch when building the sparse operator.
            Reducing this lowers peak memory usage at the cost of speed.
        chunk_k
            The number of visibilities processed per chunk when computing the NUFFT precision matrix
            inside `psf_precision_operator_from()`. Reducing this lowers peak memory usage.
        show_progress
            If `True`, a progress bar is displayed while computing the NUFFT precision matrix.
        show_memory
            If `True`, memory usage statistics are printed while computing the NUFFT precision matrix.
        use_jax
            If `True`, JAX is used to accelerate the NUFFT precision matrix computation.

        Precondition
        ------------
        Every visibility must have equal real and imaginary noise sigma
        (`noise_map.real == noise_map.imag`). The sparse operator's precision operator
        `W~ = Re(F^H W F)` is built from the real-part sigma alone (see
        `psf_precision_operator_from`, which passes `noise_map_real` to
        `nufft_precision_operator_from`), a reduction that is exact only under that
        equality. With unequal sigmas the sparse curvature matrix silently disagrees with
        the dense `InversionInterferometerMapping` path, so this method raises a
        `DatasetException` rather than returning a wrong operator.

        Returns
        -------
        Interferometer
            A new `Interferometer` dataset with the precomputed `InterferometerSparseOperator` attached,
            enabling efficient pixelized source reconstruction via the sparse linear algebra formalism.

        Raises
        ------
        exc.DatasetException
            If any visibility has unequal real and imaginary noise sigma.
        """

        noise_map_real = np.asarray(self.noise_map.real)
        noise_map_imag = np.asarray(self.noise_map.imag)

        if not np.allclose(noise_map_real, noise_map_imag):

            unequal = ~np.isclose(noise_map_real, noise_map_imag)

            denominator = np.maximum(np.abs(noise_map_real), np.abs(noise_map_imag))
            relative_difference = np.abs(noise_map_real - noise_map_imag) / np.where(
                denominator == 0.0, 1.0, denominator
            )

            raise exc.DatasetException(
                "The sparse operator cannot be applied to this interferometer dataset because its "
                "noise-map has unequal real and imaginary sigma.\n\n"
                "The sparse operator's precision operator `W~ = Re(F^H W F)` is built from the "
                "real-part noise sigma only (see `psf_precision_operator_from`, which passes "
                "`noise_map_real` to `nufft_precision_operator_from`). That reduction is exact only "
                "when every visibility has equal real and imaginary sigma "
                "(`sigma_real == sigma_imag`).\n\n"
                f"This dataset has {int(np.count_nonzero(unequal))} of {noise_map_real.size} "
                "visibilities where the real and imaginary sigma differ (maximum relative difference "
                f"{np.max(relative_difference):.3e}), so the sparse curvature matrix would silently "
                "disagree with the dense path.\n\n"
                "Either equalise the real and imaginary noise sigma of every visibility, or fit "
                "without calling `apply_sparse_operator()` — the dense "
                "`InversionInterferometerMapping` path handles unequal real and imaginary sigmas "
                "correctly."
            )

        if nufft_precision_operator is None:

            logger.info(
                "INTERFEROMETER - Computing NUFFT Precision Operator; runtime scales with visibility count and mask resolution, CPU run times may exceed hours."
            )

            n_vis = self.uv_wavelengths.shape[0]
            n_pix = self.real_space_mask.pixels_in_mask

            if n_vis * n_pix < 10**7 and not isinstance(
                self.transformer, TransformerDFT
            ):
                logger.info(
                    f"INTERFEROMETER - This dataset is small for the NUFFT setup path "
                    f"(N_vis x N_pix = {n_vis * n_pix:.1e}, below the ~1e7 crossover). "
                    f"`TransformerDFT` computes this operator faster below that point; "
                    f"`TransformerNUFFT` wins above it, and above ~1e8 it is the only "
                    f"option that fits in memory. Both are supported here."
                )

            nufft_precision_operator = self.psf_precision_operator_from(
                chunk_k=chunk_k,
                show_progress=show_progress,
                show_memory=show_memory,
                use_jax=use_jax,
            )

        dirty_image = self.transformer.image_from(
            visibilities=self.data.real * self.noise_map.real**-2.0
            + 1j * self.data.imag * self.noise_map.imag**-2.0,
        )

        sparse_operator = inversion_interferometer_util.InterferometerSparseOperator.from_nufft_precision_operator(
            nufft_precision_operator=nufft_precision_operator,
            dirty_image=dirty_image.array,
            batch_size=batch_size,
        )

        return Interferometer(
            real_space_mask=self.real_space_mask,
            data=self.data,
            noise_map=self.noise_map,
            uv_wavelengths=self.uv_wavelengths,
            transformer_class=lambda uv_wavelengths, real_space_mask: self.transformer,
            sparse_operator=sparse_operator,
        )

    def psf_precision_operator_from(
        self,
        chunk_k: int = 2048,
        show_progress: bool = False,
        show_memory: bool = False,
        use_jax: bool = False,
    ):
        """
        Compute the NUFFT precision matrix for this interferometer dataset.

        The precision matrix encodes the response of every real-space pixel to every visibility
        baseline, weighted by the noise map. It is the core precomputed quantity required for
        efficient pixelized source reconstruction via the sparse linear algebra formalism.

        This computation can be very slow for large datasets (runtime scales with the number of
        visibilities multiplied by the number of unmasked real-space pixels). For datasets with
        tens of thousands of visibilities and high-resolution masks, computation can take hours
        on a CPU. The result should be saved to disk and reloaded rather than recomputed on each
        run. Use `apply_sparse_operator(nufft_precision_operator=...)` to attach a cached result.

        Parameters
        ----------
        chunk_k
            The number of visibilities processed per chunk. Reducing this lowers peak memory
            usage during computation at the cost of speed.
        show_progress
            If `True`, a progress bar is shown during computation.
        show_memory
            If `True`, memory usage statistics are printed during computation.
        use_jax
            If `True`, JAX is used to accelerate the computation.

        Returns
        -------
        np.ndarray
            The NUFFT precision matrix of shape (total_pixels, total_pixels) where total_pixels
            is the number of unmasked real-space pixels.
        """
        return inversion_interferometer_util.nufft_precision_operator_from(
            noise_map_real=self.noise_map.array.real,
            uv_wavelengths=self.uv_wavelengths,
            shape_masked_pixels_2d=self.transformer.grid.mask.shape_native_masked_pixels,
            grid_radians_2d=self.transformer.grid.mask.derive_grid.all_false.in_radians.native.array,
            chunk_k=chunk_k,
            show_memory=show_memory,
            show_progress=show_progress,
            use_jax=use_jax,
        )

    @property
    def mask(self):
        """
        The real-space mask of the interferometer dataset.

        For an interferometer, this is the `real_space_mask` which defines the region of sky that
        contains signal. It is used as the spatial domain for the Fourier transform, determining
        the pixel grid size and coordinate grids.
        """
        return self.real_space_mask

    @property
    def amplitudes(self):
        """
        The amplitudes of the complex visibilities, defined as the absolute value of each visibility:
        amplitude = sqrt(real^2 + imag^2).
        """
        return self.data.amplitudes

    @property
    def phases(self):
        """
        The phases of the complex visibilities in radians, defined as arctan(imag / real) for
        each visibility.
        """
        return self.data.phases

    @property
    def uv_distances(self):
        """
        The radial distance of each visibility baseline from the origin of the UV-plane, in units
        of wavelengths. Computed as sqrt(u^2 + v^2) for each (u, v) baseline pair.
        """
        return np.sqrt(
            np.square(self.uv_wavelengths[:, 0]) + np.square(self.uv_wavelengths[:, 1])
        )

    @property
    def dirty_image(self):
        """
        The dirty image, computed as the inverse Fourier transform of the observed visibilities.

        This is the raw image obtained by back-projecting the visibilities without any deconvolution.
        It provides a quick visual representation of the data but is convolved with the synthesized
        beam (the Fourier transform of the UV-plane sampling function).
        """
        return self.transformer.image_from(visibilities=self.data)

    @property
    def dirty_noise_map(self):
        """
        The dirty noise map, computed as the inverse Fourier transform of the noise map visibilities.

        Provides a real-space representation of the noise levels in the dirty image.
        """
        return self.transformer.image_from(visibilities=self.noise_map)

    @property
    def dirty_signal_to_noise_map(self):
        """
        The dirty signal-to-noise map, computed as the inverse Fourier transform of the
        complex signal-to-noise visibility map.
        """
        return self.transformer.image_from(visibilities=self.signal_to_noise_map)

    @property
    def signal_to_noise_map(self):
        """
        The complex signal-to-noise map of the visibilities.

        Computed separately for the real and imaginary components as data / noise_map. Values
        below zero are clamped to zero, as negative signal-to-noise is not physically meaningful.

        Unlike the base class implementation (which operates on real-valued data), this override
        handles the complex nature of interferometric visibilities by treating the real and
        imaginary parts independently.
        """
        signal_to_noise_map_real = np.divide(
            np.real(self.data.array), np.real(self.noise_map.array)
        )
        signal_to_noise_map_real[signal_to_noise_map_real < 0] = 0.0
        signal_to_noise_map_imag = np.divide(
            np.imag(self.data.array), np.imag(self.noise_map.array)
        )
        signal_to_noise_map_imag[signal_to_noise_map_imag < 0] = 0.0

        return self.data.with_new_array(
            signal_to_noise_map_real + 1j * signal_to_noise_map_imag
        )

    @property
    def psf(self):
        """
        Returns `None` for interferometer datasets.

        Interferometers do not have a Point Spread Function in the same sense as imaging datasets.
        The equivalent quantity is the synthesized beam, which is determined by the UV-plane coverage
        and is not stored explicitly. This property exists to satisfy the `AbstractDataset` interface.
        """
        return None
