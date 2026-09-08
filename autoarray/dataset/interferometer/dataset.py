import logging
import numpy as np
from typing import Optional

from autonerves.fitsable import ndarray_via_fits_from
from autonerves import cached_property

try:
    from autonerves.test_mode import disable_jax
except ImportError:
    # `disable_jax()` arrives in the autonerves release that closes
    # PyAutoNerves#159. Importing it unconditionally would make an autonerves
    # older than that release an `ImportError` at module load -- and a
    # `--no-deps` install, an editable checkout or a hand-built virtualenv can
    # all put one on the path regardless of the floor in `pyproject.toml`,
    # which constrains resolution only. This is the same trade `dataset_util`
    # records against `SMALL_DATASETS_HEADER_KEY`: degrade to the predicate's
    # own one-line body rather than fail hard to avoid restating it. Delete the
    # fallback when the floor names a release carrying the predicate.
    import os

    def disable_jax():
        return os.environ.get("PYAUTO_DISABLE_JAX") == "1"


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
        method: str = "nufft",
        eps: Optional[float] = None,
        nufft_chunk_size: Optional[int] = None,
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

        The default builder (`method="nufft"`) computes the precision operator as a type-1 NUFFT, so
        it costs `O(N_vis * nspread^2 + M log M)` for `M = 4 * Ny * Nx` — seconds even at a million
        visibilities. The brute-force builders (`method="numpy"` / `"jax"`, and the `use_jax` kwarg)
        are `O(N_vis * N_pix)` and can take minutes to hours; they are kept as the reference the
        NUFFT builder is pinned against. Either way the result can be cached to disk and reloaded
        via `nufft_precision_operator=`.

        Parameters
        ----------
        nufft_precision_operator
            An already computed NUFFT precision matrix for this dataset (e.g. loaded from disk via
            `np.load`) to avoid an expensive recomputation. If `None` it is computed from scratch
            by calling `psf_precision_operator_from()`.
        batch_size
            The number of real-space pixels processed per batch when building the sparse operator.
            Reducing this lowers peak memory usage at the cost of speed.
        method
            Which builder computes the precision operator: `"nufft"` (default, the type-1 NUFFT),
            `"numpy"` or `"jax"` (the brute-force reference builders).
        eps
            The requested NUFFT precision of the `"nufft"` builder. `None` takes the transformer's
            own `eps` when it is a `TransformerNUFFT`, else `1e-12`.
        nufft_chunk_size
            The visibility chunk size of the `"nufft"` builder, a memory ceiling rather than an
            optimisation. `None` takes the transformer's own `chunk_size` when it is a
            `TransformerNUFFT`, else no chunking.
        chunk_k
            The number of visibilities processed per chunk by the brute-force builders when computing
            the NUFFT precision matrix inside `psf_precision_operator_from()`. Reducing this lowers
            peak memory usage.
        show_progress
            If `True`, a progress bar is displayed while computing the NUFFT precision matrix.
        show_memory
            If `True`, memory usage statistics are printed while computing the NUFFT precision matrix.
        use_jax
            If `True`, JAX is used to accelerate the NUFFT precision matrix computation.

            `PYAUTO_DISABLE_JAX=1` overrides this to `False`. That variable is a
            harness-level switch, not a preference: it is the documented way to force the
            NumPy path (the workspace `start_here` guides name it beside `use_jax=False`),
            and the smoke profiles set it so a fast run does not pay a JIT compile. An
            explicit `use_jax=True` in a script -- which is the right thing for a script
            demonstrating the production path to say -- must therefore not defeat it, or
            the harness pays 2.3-3.2 s of compile for a backend it asked to disable.

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

        if disable_jax():
            use_jax = False

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

            logger.info("INTERFEROMETER - Computing NUFFT Precision Operator.")

            n_vis = self.uv_wavelengths.shape[0]
            n_pix = self.real_space_mask.pixels_in_mask

            if method != "nufft" or use_jax:
                logger.info(
                    f"INTERFEROMETER - The precision operator is being built by a brute-force "
                    f"builder, which is O(N_vis x N_pix) = O({n_vis * n_pix:.1e}) and can take "
                    f"minutes to hours. The default `method='nufft'` builds the same array as a "
                    f"type-1 NUFFT in seconds."
                )

            if isinstance(self.transformer, TransformerDFT):
                # This is about the transformer, not the precision operator: the operator is
                # built by `nufft_precision_operator_from` either way and does not go through
                # the transformer at all. The DFT transformer allocates O(N_vis x N_pix) for
                # every subsequent transform, which is what becomes infeasible at scale --
                # extrapolating a measured 446 MB at N_vis=4e3 / N_pix=1e4 gives ~109 GB at a
                # million visibilities, whereas `TransformerNUFFT` allocates nothing beyond its
                # working buffers. Below `N_vis x N_pix ~ 1e7` the DFT is the faster transform
                # (0.2-0.7x the NUFFT time) and there is nothing to warn about.
                if n_vis * n_pix > 10**7:
                    logger.info(
                        f"INTERFEROMETER - This dataset uses `TransformerDFT` at "
                        f"N_vis x N_pix = {n_vis * n_pix:.1e}, above the ~1e7 crossover where "
                        f"`TransformerNUFFT` transforms faster (1.2-1.9x at 1e7-1e8); above ~1e8 "
                        f"the DFT's O(N_vis x N_pix) allocation makes it infeasible rather than "
                        f"merely slow. The two agree to ~3e-13 relative."
                    )

            nufft_precision_operator = self.psf_precision_operator_from(
                method=method,
                eps=eps,
                nufft_chunk_size=nufft_chunk_size,
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
        method: str = "nufft",
        eps: Optional[float] = None,
        nufft_chunk_size: Optional[int] = None,
    ):
        """
        Compute the NUFFT precision matrix for this interferometer dataset.

        The precision matrix encodes the response of every real-space pixel to every visibility
        baseline, weighted by the noise map. It is the core precomputed quantity required for
        efficient pixelized source reconstruction via the sparse linear algebra formalism.

        The default builder (`method="nufft"`) computes this as a type-1 (adjoint) NUFFT, which is
        `O(N_vis * nspread^2 + M log M)` for `M = 4 * Ny * Nx` — seconds even at a million
        visibilities. The brute-force builders (`method="numpy"` / `"jax"`, and the `use_jax`
        kwarg) are `O(N_vis * N_pix)` and can take minutes to hours on a CPU for a
        high-resolution mask; they are kept as the reference the NUFFT builder is pinned against.
        The result can still be saved to disk and reloaded rather than recomputed on each run —
        use `apply_sparse_operator(nufft_precision_operator=...)` to attach a cached result.

        Parameters
        ----------
        chunk_k
            The number of visibilities processed per chunk by the brute-force builders. Reducing
            this lowers peak memory usage during computation at the cost of speed.
        show_progress
            If `True`, a progress bar is shown during computation by the NumPy brute force.
        show_memory
            If `True`, memory usage statistics are printed during computation.
        use_jax
            If `True`, the JAX brute-force builder is used (equivalent to `method="jax"`).
        method
            Which builder computes the operator: `"nufft"` (default), `"numpy"` or `"jax"`.
        eps
            The requested NUFFT precision of the `"nufft"` builder. `None` takes the transformer's
            own `eps` when it is a `TransformerNUFFT`, else `1e-12`.
        nufft_chunk_size
            The visibility chunk size of the `"nufft"` builder, a memory ceiling rather than an
            optimisation. `None` takes the transformer's own `chunk_size` when it is a
            `TransformerNUFFT`, else no chunking.

        Returns
        -------
        np.ndarray
            The NUFFT precision matrix of shape (total_pixels, total_pixels) where total_pixels
            is the number of unmasked real-space pixels.
        """
        transformer = self.transformer

        # The NUFFT builder and `TransformerNUFFT` spread the same visibilities onto a mode grid
        # with the same library, so a dataset that has already chosen an accuracy and a memory
        # ceiling for its transformer should not have to repeat them here.
        if eps is None:
            eps = (
                transformer.eps
                if isinstance(transformer, TransformerNUFFT)
                else 1.0e-12
            )

        if nufft_chunk_size is None and isinstance(transformer, TransformerNUFFT):
            nufft_chunk_size = transformer.chunk_size

        return inversion_interferometer_util.nufft_precision_operator_from(
            noise_map_real=self.noise_map.array.real,
            uv_wavelengths=self.uv_wavelengths,
            shape_masked_pixels_2d=transformer.grid.mask.shape_native_masked_pixels,
            grid_radians_2d=transformer.grid.mask.derive_grid.all_false.in_radians.native.array,
            method=method,
            eps=eps,
            chunk_size=nufft_chunk_size,
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
