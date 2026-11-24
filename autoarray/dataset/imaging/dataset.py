import copy
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union

from autoarray.dataset.abstract.dataset import AbstractDataset
from autoarray.dataset.grids import GridsDataset
from autoarray.inversion.inversion.imaging.inversion_imaging_util import (
    ImagingSparseOperator,
)
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import ConvolverState
from autoarray.operators.convolver import Convolver
from autoarray.mask.mask_2d import Mask2D
from autoarray.setup.setup import Setup
from autoarray.setup.setup import setup_from
from autoarray import type as ty

from autoarray import exc

from autoarray.inversion.inversion.imaging import inversion_imaging_util

logger = logging.getLogger(__name__)


def _validate_convolve_over_sample_size(
    name: str,
    convolve_over_sample_size,
    over_sample_size,
) -> None:
    """
    Validate a `convolve_over_sample_size` input against its matching
    `over_sample_size`: it must be a plain int >= 1, and when above 1 every entry of
    the matching over sample size (int or adaptive Array2D) must be divisible by it —
    the k x s coupling, whereby values evaluated at per-pixel sizes k_i * s are
    partially binned to the uniform s the convolution requires (see
    `over_sample_util.binned_to_convolve_size_from`).
    """
    if isinstance(convolve_over_sample_size, bool) or not isinstance(
        convolve_over_sample_size, (int, np.integer)
    ):
        raise TypeError(
            f"convolve_over_sample_size_{name} must be a plain int (adaptive over "
            f"sampling is not supported for PSF convolution), but a "
            f"{type(convolve_over_sample_size).__name__} was input."
        )

    if convolve_over_sample_size < 1:
        raise exc.DatasetException(
            f"convolve_over_sample_size_{name} must be >= 1, but "
            f"{convolve_over_sample_size} was input."
        )

    if convolve_over_sample_size == 1:
        return

    if not np.all(np.mod(np.array(over_sample_size), convolve_over_sample_size) == 0):
        raise exc.DatasetException(
            f"convolve_over_sample_size_{name}={convolve_over_sample_size} requires "
            f"every over_sample_size_{name} entry to be divisible by it (the k x s "
            f"coupling: evaluate at k_i * s per pixel, partially bin to s, convolve), "
            f"but {over_sample_size} was input."
        )


_SPARSE_OPERATOR_CONVOLVE_OVER_SAMPLE_SIZE_ERROR = (
    "The sparse linear algebra formalism precomputes PSF products at "
    "image resolution and is incompatible with an oversampled PSF "
    "(convolve_over_sample_size > 1)."
)


def _warn_sparse_operator_discarded(sparse_operator, method_name: str) -> None:
    """
    Log a warning when a dataset rebuild discards an attached sparse operator.

    The sparse operator precomputes PSF products of every pair of masked noise-map
    values, so it is invalidated by any change to the mask or noise-map. Rebuilds
    which make such changes (e.g. `apply_mask`, `apply_noise_scaling`) therefore do
    not carry it, and the returned dataset falls back to the (slower) mapping matrix
    formalism unless the operator is re-applied. This warning stops that fallback
    from happening silently.
    """
    if sparse_operator is None:
        return

    logger.warning(
        f"IMAGING - {method_name} changes the mask or noise-map, which invalidates the "
        f"sparse operator attached to this dataset, so it has been discarded. Pixelized "
        f"reconstructions will use the slower mapping matrix formalism unless the sparse "
        f"operator is re-applied (via apply_sparse_operator() or "
        f"apply_sparse_operator_cpu()) as the last dataset operation."
    )


class Imaging(AbstractDataset):
    def __init__(
        self,
        data: Array2D,
        noise_map: Optional[Array2D] = None,
        psf: Optional[Convolver] = None,
        psf_setup_state: bool = False,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        over_sample_size_lp: Union[int, Array2D] = 4,
        over_sample_size_pixelization: Union[int, Array2D] = 4,
        convolve_over_sample_size_lp: int = 1,
        convolve_over_sample_size_pixelization: int = 1,
        use_normalized_psf: Optional[bool] = True,
        check_noise_map: bool = True,
        sparse_operator: Optional[ImagingSparseOperator] = None,
        setup: Optional[Union[Setup, str]] = None,
    ):
        """
        An imaging dataset, containing the image data, noise-map, PSF and associated quantities
        for calculations like the grid.

        This object is the input to the `FitImaging` object, which fits the dataset with a model image and quantifies
        the goodness-of-fit via a residual map, likelihood, chi-squared and other quantities.

        The following quantities of the imaging data are available and used for the following tasks:

        - `data`: The image data, which shows the signal that is analysed and fitted with a model image.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        - `psf`: The Point Spread Function of the data, used to perform 2D convolution on images to produce a model
        image which is compared to the data.

        The dataset also has a number of (y,x) grids of coordinates associated with it, which map to the centres
        of its image pixels. They are used for performing calculations which map directly to the data and have
        over sampling calculations built in which approximate the 2D line integral of these calculations within a
        pixel. This is explained in more detail in the `GridsDataset` class.

        Parameters
        ----------
        data
            The array of the image data containing the signal that is fitted (in PyAutoGalaxy and PyAutoLens the
            recommended units are electrons per second).
        noise_map
            An array describing the RMS standard deviation error in each pixel used for computing quantities like the
            chi-squared in a fit (in PyAutoGalaxy and PyAutoLens the recommended units are electrons per second).
        psf
            The Point Spread Function kernel of the image which accounts for diffraction due to the telescope optics
            via 2D convolution.
        psf_setup_state
            If `True`, a `ConvolverState` is precomputed from the PSF kernel and mask, storing the
            convolution pair indices required for efficient 2D convolution. This is set automatically
            to `True` when a mask is applied via `apply_mask()` and should not normally be set by hand.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        convolve_over_sample_size_lp
            The over sample size of the PSF for light-profile operations. If above 1, PSF convolution of light
            profile images is performed at this multiple of the image resolution (requiring the PSF to be supplied
            at that resolution) and binned back to image resolution, improving the accuracy of the blurring.
            Requires `over_sample_size_lp` to be uniform and equal to it. A value of 1 (default) leaves all
            behaviour unchanged.
        convolve_over_sample_size_pixelization
            The over sample size of the PSF for pixelization operations, with the same requirements as
            `convolve_over_sample_size_lp` applied to `over_sample_size_pixelization`. Incompatible with the
            sparse linear algebra formalism (`sparse_operator`), whose PSF products are precomputed at image
            resolution.
        use_normalized_psf
            If `True`, the PSF kernel values are rescaled such that they sum to 1.0. This can be important for ensuring
            the PSF kernel does not change the overall normalization of the image when it is convolved with it.
        check_noise_map
            If True, the noise-map is checked to ensure all values are above zero.
        sparse_operator
            The sparse linear algebra formalism of the linear algebra equations precomputes the convolution of every pair of masked
            noise-map values given the PSF (see `inversion.inversion_util`). Pass the `ImagingSparseOperator` object here to
            enable this linear algebra formalism for pixelized reconstructions.
        setup
            The compute setup (an `aa.Setup` or a setup_id string) this dataset is evaluated with, which is
            stored and carried by `apply_mask`, `apply_noise_scaling` and `apply_over_sampling`. Passing it here
            only records it; use `apply_setup()` to attach the operator it names.
        """

        _validate_convolve_over_sample_size(
            name="lp",
            convolve_over_sample_size=convolve_over_sample_size_lp,
            over_sample_size=over_sample_size_lp,
        )
        _validate_convolve_over_sample_size(
            name="pixelization",
            convolve_over_sample_size=convolve_over_sample_size_pixelization,
            over_sample_size=over_sample_size_pixelization,
        )

        if (
            convolve_over_sample_size_lp > 1
            and convolve_over_sample_size_pixelization > 1
            and convolve_over_sample_size_lp != convolve_over_sample_size_pixelization
        ):
            raise exc.DatasetException(
                f"Different convolve_over_sample_size values for the lp "
                f"({convolve_over_sample_size_lp}) and pixelization "
                f"({convolve_over_sample_size_pixelization}) operations are not yet "
                f"supported, because the dataset holds a single PSF kernel at one "
                f"resolution."
            )

        if (
            sparse_operator is not None
            and convolve_over_sample_size_pixelization > 1
        ):
            raise exc.DatasetException(
                "convolve_over_sample_size_pixelization > 1 is incompatible with the "
                "sparse linear algebra formalism (sparse_operator), whose PSF "
                "products are precomputed at image resolution."
            )

        self.convolve_over_sample_size_lp = int(convolve_over_sample_size_lp)
        self.convolve_over_sample_size_pixelization = int(
            convolve_over_sample_size_pixelization
        )

        super().__init__(
            data=data,
            noise_map=noise_map,
            noise_covariance_matrix=noise_covariance_matrix,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

        if self.noise_map.native is not None and check_noise_map:
            if ((self.noise_map.native <= 0.0) * np.invert(self.noise_map.mask)).any():
                zero_entries = np.argwhere(self.noise_map.native <= 0.0)

                raise exc.DatasetException(
                    f"""
                    A value in the noise-map of the dataset is {np.min(self.noise_map)}. 

                    This is less than or equal to zero, and therefore an ill-defined value which must be corrected.
                    
                    The 2D indexes of the arrays in the native noise map array are {zero_entries}.
                    """
                )

        convolve_over_sample_size = max(
            self.convolve_over_sample_size_lp,
            self.convolve_over_sample_size_pixelization,
        )

        if psf is not None:

            if use_normalized_psf:

                psf.kernel._array = np.divide(
                    psf.kernel._array, np.sum(psf.kernel._array)
                )

            if (
                convolve_over_sample_size > 1
                and psf.convolve_over_sample_size != convolve_over_sample_size
            ):
                psf = Convolver(
                    kernel=psf.kernel,
                    use_fft=psf._use_fft,
                    convolve_over_sample_size=convolve_over_sample_size,
                )

            if psf_setup_state:

                # Always rebuild for this dataset's mask — state_from would return a
                # cached state built for a previous mask (e.g. via apply_mask).
                if psf.convolve_over_sample_size > 1:
                    state = psf._fine_state_from(mask=self.data.mask)
                else:
                    state = ConvolverState(kernel=psf.kernel, mask=self.data.mask)

                psf = Convolver(
                    kernel=psf.kernel,
                    state=state,
                    normalize=use_normalized_psf,
                    use_fft=psf._use_fft,
                    convolve_over_sample_size=psf.convolve_over_sample_size,
                )

        self.psf = psf

        self.grids = GridsDataset(
            mask=self.data.mask,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
            psf=self.psf,
        )

        self.sparse_operator = sparse_operator
        self.setup = setup_from(setup)

        # The operator `sparse_operator_for` built lazily, if any; private, so it never changes
        # what an attached `sparse_operator` means.
        self._lazy_sparse_operator = None

    @classmethod
    def from_fits(
        cls,
        pixel_scales: ty.PixelScales,
        data_path: Union[Path, str],
        noise_map_path: Union[Path, str],
        data_hdu: int = 0,
        noise_map_hdu: int = 0,
        psf_path: Optional[Union[Path, str]] = None,
        psf_hdu: int = 0,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        check_noise_map: bool = True,
        over_sample_size_lp: Union[int, Array2D] = 4,
        over_sample_size_pixelization: Union[int, Array2D] = 4,
        convolve_over_sample_size_lp: int = 1,
        convolve_over_sample_size_pixelization: int = 1,
        psf_pixel_scales: Optional[ty.PixelScales] = None,
    ) -> "Imaging":
        """
        Load an imaging dataset from multiple .fits file.

        For each attribute of the imaging data (e.g. `data`, `noise_map`, `pre_cti_data`) the path to
        the .fits and the `hdu` containing the data can be specified.

        The `noise_map` assumes the noise value in each `data` value are independent, where these values are the
        the RMS standard deviation error in each pixel.

        A `noise_covariance_matrix` can be input instead, which represents the covariance between noise values in
        the data and can be used to fit the data accounting for correlations (the `noise_map` is the diagonal values
        of this matrix).

        If the dataset has a mask associated with it (e.g. in a `mask.fits` file) the file must be loaded separately
        via the `Mask2D` object and applied to the imaging after loading via fits using the `from_fits` method.

        Parameters
        ----------
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        data_path
            The path to the data .fits file containing the image data (e.g. '/path/to/image.fits').
        data_hdu
            The hdu the image data is contained in the .fits file specified by `data_path`.
        psf_path
            The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits').
        psf_hdu
            The hdu the psf is contained in the .fits file specified by `psf_path`.
        noise_map_path
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits').
        noise_map_hdu
            The hdu the noise map is contained in the .fits file specified by `noise_map_path`.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value.
        check_noise_map
            If True, the noise-map is checked to ensure all values are above zero.
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        convolve_over_sample_size_lp
            The over sample size of the PSF for light-profile operations (see `Imaging.__init__`). If above 1
            the PSF .fits file must contain the PSF sampled at that multiple of the image resolution, and its
            pixel scales are set accordingly (or via `psf_pixel_scales`).
        convolve_over_sample_size_pixelization
            The over sample size of the PSF for pixelization operations (see `Imaging.__init__`).
        psf_pixel_scales
            Optional explicit pixel scales of the PSF .fits file. Defaults to the image pixel scales divided by
            the convolve over sample size (i.e. the fine resolution when oversampled convolution is used).
        """

        from autoarray.util.dataset_util import cap_array_2d_for_small_datasets

        data = Array2D.from_fits(
            file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
        )
        data, pixel_scales = cap_array_2d_for_small_datasets(data, pixel_scales)

        noise_map = Array2D.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )
        noise_map, pixel_scales = cap_array_2d_for_small_datasets(noise_map, pixel_scales)

        if psf_path is not None:

            convolve_over_sample_size = max(
                convolve_over_sample_size_lp, convolve_over_sample_size_pixelization
            )

            if psf_pixel_scales is None:
                if isinstance(pixel_scales, (int, float)):
                    psf_pixel_scales = pixel_scales / convolve_over_sample_size
                else:
                    psf_pixel_scales = tuple(
                        ps / convolve_over_sample_size for ps in pixel_scales
                    )

            kernel = Array2D.from_fits(
                file_path=psf_path,
                hdu=psf_hdu,
                pixel_scales=psf_pixel_scales,
            )
            psf = Convolver(
                kernel=kernel,
                convolve_over_sample_size=convolve_over_sample_size,
            )

        else:
            kernel = None
            psf = None

        return Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            noise_covariance_matrix=noise_covariance_matrix,
            check_noise_map=check_noise_map,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
            convolve_over_sample_size_lp=convolve_over_sample_size_lp,
            convolve_over_sample_size_pixelization=convolve_over_sample_size_pixelization,
        )

    def apply_mask(self, mask: Mask2D) -> "Imaging":
        """
        Apply a mask to the imaging dataset, whereby the mask is applied to the image data, noise-map and other
        quantities one-by-one.

        The mask is applied to the `data`, `noise_map`, `over_sample_size_lp` and
        `over_sample_size_pixelization` arrays. If a `noise_covariance_matrix` is present, the rows
        and columns corresponding to masked pixels are removed so it stays consistent with the
        remaining unmasked pixels. The PSF `ConvolverState` is recomputed for the new mask.

        The `apply_mask` function cannot be called multiple times — a new mask cannot expand the
        unmasked region beyond what was already unmasked, as the underlying data has already been
        trimmed. An exception is raised if this is attempted. If you wish to apply a different mask,
        reload the dataset from .fits files.

        A new mask invalidates any attached sparse operator. If the dataset has a `setup` (see
        `apply_setup()`), the setup stays in effect: it is re-applied to the masked dataset, so the
        operator it names is rebuilt for the new mask. Without a setup the operator is discarded
        (with a warning).

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image.

        Returns
        -------
        Imaging
            A new `Imaging` dataset with the mask applied to all arrays.
        """
        invalid = np.logical_and(self.data.mask, np.logical_not(mask))

        if np.any(invalid):
            raise exc.DatasetException(
                "The new mask overlaps with pixels that are already unmasked in the dataset. "
                "You cannot apply a new mask on top of an existing one. "
                "If you wish to apply a different mask, please reload the dataset from .fits files."
            )

        data = Array2D(values=self.data.native, mask=mask)

        noise_map = Array2D(values=self.noise_map.native, mask=mask)

        if self.noise_covariance_matrix is not None:
            noise_covariance_matrix = self.noise_covariance_matrix

            noise_covariance_matrix = np.delete(
                noise_covariance_matrix, mask.derive_indexes.masked_slim, 0
            )
            noise_covariance_matrix = np.delete(
                noise_covariance_matrix, mask.derive_indexes.masked_slim, 1
            )

        else:
            noise_covariance_matrix = None

        over_sample_size_lp = Array2D(values=self.over_sample_size_lp.native, mask=mask)
        over_sample_size_pixelization = Array2D(
            values=self.over_sample_size_pixelization.native, mask=mask
        )

        if self.setup is None:
            _warn_sparse_operator_discarded(
                sparse_operator=self.sparse_operator, method_name="apply_mask"
            )

        dataset = Imaging(
            data=data,
            noise_map=noise_map,
            psf=self.psf,
            psf_setup_state=True,
            noise_covariance_matrix=noise_covariance_matrix,
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
            convolve_over_sample_size_lp=self.convolve_over_sample_size_lp,
            convolve_over_sample_size_pixelization=self.convolve_over_sample_size_pixelization,
            setup=self.setup,
        )

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        if self.setup is not None:
            return dataset.apply_setup(setup=self.setup)

        return dataset

    def apply_noise_scaling(
        self,
        mask: Mask2D,
        noise_value: float = 1e8,
        signal_to_noise_value: Optional[float] = None,
        should_zero_data: bool = True,
    ) -> "Imaging":
        """
        Apply a mask to the imaging dataset using noise scaling, whereby the maskmay zero the data and increase
        noise-map values to change how they enter the likelihood calculation.

        Given this data region is masked, it is likely thr data itself should not be included and therefore
        the masked data values are set to zero. This can be disabled by setting `should_zero_data=False`.

        Two forms of scaling are supported depending on whether the `signal_to_noise_value` is input:

        - `noise_value`: The noise-map values in the masked region are set to this value, typically a very large value,
        such that they are never included in the likelihood calculation.

        - `signal_to_noise_value`: The noise-map values in the masked region are set to values such that they give
        this signal-to-noise ratio. This overwrites the `noise_value` parameter.

        For certain modeling tasks, the mask defines regions of the data that are used to calculate the likelihood.
        For example, all data points in a mask may be used to create a pixel-grid, which is used in the likelihood.
        When data points are moved via `apply_mask`, they would be omitted from this grid entirely, which would
        lead to an incorrect likelihood calculation. Noise scaling retains these data points in the likelihood
        calculation, but ensures they do not contribute to the fit.

        This function can only be applied before actual masking.

        A changed noise-map invalidates any attached sparse operator. If the dataset has a `setup`
        (see `apply_setup()`), the setup is re-applied to the returned dataset so the operator it names
        is rebuilt for the new noise-map. Without a setup the operator is discarded (with a warning).

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image and noise-map, to scale the noise-map values to large values.
        noise_value
            The value that the noise-map values are set to in the masked region where noise scaling is applied.
        signal_to_noise_value
            The noise-map values are instead set to values such that they give this signal-to-noise_maps ratio.
            This overwrites the noise_value parameter.
        should_zero_data
            If True, the data values in the masked region are set to zero.
        """

        if signal_to_noise_value is None:
            noise_map = self.noise_map.native
            noise_map[mask.array == False] = noise_value
        else:
            noise_map = np.where(
                mask == False,
                np.median(self.data.native[mask.derive_mask.edge == False])
                / signal_to_noise_value,
                self.noise_map.native.array,
            )

        if should_zero_data:
            data = np.where(np.invert(mask.array), 0.0, self.data.native.array)
        else:
            data = self.data.native.array

        data = Array2D(values=data, mask=self.data.mask)

        noise_map = Array2D(values=noise_map, mask=self.data.mask)

        if self.setup is None:
            _warn_sparse_operator_discarded(
                sparse_operator=self.sparse_operator, method_name="apply_noise_scaling"
            )

        dataset = Imaging(
            data=data,
            noise_map=noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
            convolve_over_sample_size_lp=self.convolve_over_sample_size_lp,
            convolve_over_sample_size_pixelization=self.convolve_over_sample_size_pixelization,
            check_noise_map=False,
            setup=self.setup,
        )

        logger.info(
            f"IMAGING - Data noise scaling applied, a total of {mask.pixels_in_mask} pixels were scaled to large noise values."
        )

        if self.setup is not None:
            return dataset.apply_setup(setup=self.setup)

        return dataset

    def apply_over_sampling(
        self,
        over_sample_size_lp: Union[int, Array2D] = None,
        over_sample_size_pixelization: Union[int, Array2D] = None,
    ) -> "AbstractDataset":
        """
        Apply new over sampling objects to the grid and grid pixelization of the dataset.

        This method is used to change the over sampling of the grid and grid pixelization, for example when the
        user wishes to perform over sampling with a higher sub grid size or with an iterative over sampling strategy.

        The `grid` and grids.pixelization` are cached properties which after use are stored in memory for efficiency.
        This function resets the cached properties so that the new over sampling is used in the grid and grid
        pixelization.

        The `noise_covariance_matrix`, any attached `sparse_operator`, the `setup` and any lazily built sparse
        operator (see `sparse_operator_for()`) are carried over, as none depends on the over sampling (the sparse operator depends only on the noise-map, PSF kernel and mask). This
        means `apply_sparse_operator()` / `apply_sparse_operator_cpu()` / `apply_setup()` may be called before or
        after this method.

        Parameters
        ----------
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        """

        dataset = Imaging(
            data=self.data,
            noise_map=self.noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            over_sample_size_lp=over_sample_size_lp or self.over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization
            or self.over_sample_size_pixelization,
            convolve_over_sample_size_lp=self.convolve_over_sample_size_lp,
            convolve_over_sample_size_pixelization=self.convolve_over_sample_size_pixelization,
            check_noise_map=False,
            sparse_operator=self.sparse_operator,
            setup=self.setup,
        )

        dataset._lazy_sparse_operator = getattr(self, "_lazy_sparse_operator", None)

        return dataset

    def apply_sparse_operator(
        self,
        batch_size: int = 128,
    ):
        """
        Precompute the PSF precision operator for efficient pixelized source reconstruction.

        The sparse linear algebra formalism precomputes the convolution of every pair of masked
        noise-map values given the PSF (see `inversion.inversion_util`). This is the imaging
        equivalent of the interferometer NUFFT precision matrix.

        The `ImagingSparseOperator` stores these precomputed values in the imaging dataset ensuring
        they are only computed once per analysis, enabling fast repeated likelihood evaluations during
        model fitting.

        Parameters
        ----------
        batch_size
            The number of image pixels processed per batch when computing the sparse operator via
            FFT-based convolution. Reducing this lowers peak memory usage at the cost of speed.

        Returns
        -------
        Imaging
            A new `Imaging` dataset with the precomputed `ImagingSparseOperator` attached, enabling
            efficient pixelized source reconstruction via the sparse linear algebra formalism.

        Notes
        -----
        `PYAUTO_DISABLE_JAX=1` is *not* honoured here, unlike
        `Interferometer.apply_sparse_operator`, and the asymmetry is deliberate. There the
        variable overrides a `use_jax` argument that already selects between two backends
        computing the same operator. Here there is no such argument: this method is the JAX
        implementation, and the NumPy/CPU alternative is the separately named
        `apply_sparse_operator_cpu`, which returns a different operator class
        (`SparseLinAlgImagingNumba`) and requires numba. Silently returning that under an
        environment variable would change the type of the returned object based on the
        environment, which is a larger change than honouring a switch -- and an unmeasured
        one: every JIT cost the phase-8 workspace timings attribute to this variable
        (2.3-3.2 s per script) was on the interferometer path.
        """

        if self.psf is not None and self.psf.convolve_over_sample_size > 1:
            raise exc.DatasetException(
                _SPARSE_OPERATOR_CONVOLVE_OVER_SAMPLE_SIZE_ERROR
            )

        logger.info(
            "IMAGING - Setting Up Sparse Operator For low Memory Pixelizations."
        )

        sparse_operator = (
            inversion_imaging_util.ImagingSparseOperator.from_noise_map_and_psf(
                data=self.data,
                noise_map=self.noise_map,
                psf=self.psf.kernel.native,
                batch_size=batch_size,
            )
        )

        return Imaging(
            data=self.data,
            noise_map=self.noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
            convolve_over_sample_size_lp=self.convolve_over_sample_size_lp,
            convolve_over_sample_size_pixelization=self.convolve_over_sample_size_pixelization,
            check_noise_map=False,
            sparse_operator=sparse_operator,
            setup=self.setup,
        )

    def apply_sparse_operator_cpu(
        self,
    ):
        """
        Precompute the PSF precision operator using a CPU-only Numba implementation.

        This is the CPU alternative to `apply_sparse_operator()`, using Numba JIT compilation
        for the convolution loop rather than JAX. It requires `numba` to be installed; an
        `InversionException` is raised if it is not available.

        The resulting `SparseLinAlgImagingNumba` operator is stored on the returned `Imaging`
        dataset and used by `FitImaging` when performing pixelized source reconstructions.

        Returns
        -------
        Imaging
            A new `Imaging` dataset with a precomputed Numba-based sparse operator attached,
            enabling efficient pixelized source reconstruction on CPU hardware.
        """
        if self.psf is not None and self.psf.convolve_over_sample_size > 1:
            raise exc.DatasetException(
                _SPARSE_OPERATOR_CONVOLVE_OVER_SAMPLE_SIZE_ERROR
            )

        try:
            import numba
        except ModuleNotFoundError:
            raise exc.InversionException(
                "Inversion w-tilde functionality (pixelized reconstructions) is "
                "disabled if numba is not installed.\n\n"
                "This is because the run-times without numba are too slow.\n\n"
                "Please install numba, which is described at the following web page:\n\n"
                "https://pyautolens.readthedocs.io/en/latest/installation/overview.html"
            )

        from autoarray.inversion.inversion.imaging_numba import (
            inversion_imaging_numba_util,
        )

        (
            psf_precision_operator_sparse,
            indexes,
            lengths,
        ) = inversion_imaging_numba_util.psf_precision_operator_sparse_from(
            noise_map_native=np.array(self.noise_map.native.array).astype("float64"),
            kernel_native=np.array(self.psf.kernel.native.array).astype("float64"),
            native_index_for_slim_index=np.array(
                self.mask.derive_indexes.native_for_slim
            ).astype("int"),
        )

        sparse_operator = inversion_imaging_numba_util.SparseLinAlgImagingNumba(
            psf_precision_operator_sparse=psf_precision_operator_sparse,
            indexes=indexes.astype("int"),
            lengths=lengths.astype("int"),
            noise_map=self.noise_map,
            psf=self.psf,
            mask=self.mask,
        )

        return Imaging(
            data=self.data,
            noise_map=self.noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
            convolve_over_sample_size_lp=self.convolve_over_sample_size_lp,
            convolve_over_sample_size_pixelization=self.convolve_over_sample_size_pixelization,
            check_noise_map=False,
            sparse_operator=sparse_operator,
            setup=self.setup,
        )

    def sparse_operator_for(self, xp=np, needs_operator: bool = True, settings=None):
        """
        The sparse operator a fit of this dataset should use: the attached `sparse_operator` if
        there is one, otherwise (with the lazy policy on) the numba `SparseLinAlgImagingNumba`
        operator, built once on first use and cached privately on this dataset.

        This lets calculations outside a model-fit -- a plain `FitImaging(dataset, ...)`, an
        aggregator reload, `analysis.fit_from(...)` before a search -- use the sparse operator
        modeling attaches eagerly (via `apply_setup()`), instead of the slow, memory-heavy dense
        mapping-matrix route. The policy is off by default (see
        `autoarray.dataset.lazy_operator.lazy_sparse_operator_enabled`), in which case this
        returns the attached operator or `None`, exactly as before.

        Nothing is built, and `None` is returned, when `needs_operator` is False, the policy is
        off, `xp` is not NumPy or the noise-map is a JAX tracer, or the dataset has an explicit
        non-sparse (e.g. dense) setup. Inside a multiprocessing worker a cached operator is
        returned but none is built (workers inherit the parent's cache, as PyAutoFit evaluates one
        likelihood before forking). The build is also skipped when the dataset has no PSF, the PSF
        is oversampled (`convolve_over_sample_size > 1`, which `apply_sparse_operator_cpu()`
        rejects) or numba is not installed.

        The cache is carried by `apply_over_sampling()` and dropped by `apply_mask()`,
        `apply_noise_scaling()` and `apply_setup()`, which change the mask or noise-map or attach
        their own operator.

        Parameters
        ----------
        xp
            The array module of the fit (`numpy` or `jax.numpy`).
        needs_operator
            Whether the fit has a pixelization, i.e. would use a sparse operator at all (False
            for, e.g., an MGE-only fit).
        settings
            The `Settings` of the fit, whose `lazy_sparse_operator` decides the policy; `None`
            reads the config.

        Returns
        -------
        The attached or lazily built sparse operator, or `None`.
        """
        if self.sparse_operator is not None:
            return self.sparse_operator

        from autoarray.dataset import lazy_operator

        reason = lazy_operator.lazy_build_blocked_reason(
            dataset=self, xp=xp, needs_operator=needs_operator, settings=settings
        )

        if reason is not None:
            return None

        # `getattr`: a dataset unpickled from before this attribute existed has none.
        cached = getattr(self, "_lazy_sparse_operator", None)

        if cached is not None:
            return cached

        if lazy_operator.in_worker_process():
            return None

        if self.psf is None:
            logger.debug(
                "INVERSION - lazy sparse operator skipped: the dataset has no PSF."
            )
            return None

        if self.psf.convolve_over_sample_size > 1:
            logger.debug(
                "INVERSION - lazy sparse operator skipped: the PSF is oversampled "
                "(convolve_over_sample_size > 1)."
            )
            return None

        try:
            import numba  # noqa: F401
        except ModuleNotFoundError:
            logger.debug(
                "INVERSION - lazy sparse operator skipped: numba is not installed."
            )
            return None

        start = time.perf_counter()

        self._lazy_sparse_operator = self.apply_sparse_operator_cpu().sparse_operator

        logger.info(
            f"INVERSION - built numba sparse operator ({self.mask.pixels_in_mask} pixels) "
            f"in {time.perf_counter() - start:.2f} s"
        )

        return self._lazy_sparse_operator

    def _sparse_operator_matches(self, setup: Setup) -> bool:
        """
        Whether the attached `sparse_operator` is the one `setup` names: a JAX
        `ImagingSparseOperator` for a `jax` sparse setup, a numba `SparseLinAlgImagingNumba`
        for a `numba` sparse setup, and no operator for any other operator token (`dense`,
        `na`, ...).
        """
        if setup.operator != "sparse":
            return self.sparse_operator is None

        if setup.backend == "jax":
            return isinstance(self.sparse_operator, ImagingSparseOperator)

        from autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util import (
            SparseLinAlgImagingNumba,
        )

        return isinstance(self.sparse_operator, SparseLinAlgImagingNumba)

    def apply_setup(
        self,
        setup: Union[Setup, str],
        batch_size: int = 128,
    ) -> "Imaging":
        """
        Apply a compute setup to the dataset, attaching the operator it names.

        - An operator `"sparse"` setup attaches the sparse linear algebra operator of its
          backend: `apply_sparse_operator()` (an `ImagingSparseOperator`) for `backend="jax"`
          and `apply_sparse_operator_cpu()` (a `SparseLinAlgImagingNumba`) for
          `backend="numba"`.
        - Any other operator (`"dense"`, `"na"`) attaches no operator, dropping an attached
          one, so pixelized reconstructions use the mapping matrix formalism.

        The setup is stored as `dataset.setup` and carried by `apply_mask`,
        `apply_noise_scaling` and `apply_over_sampling`; the first two rebuild the operator for
        the new mask or noise-map. The inversion factory checks the operator against the setup
        and the array backend, and raises on a mismatch.

        This method is idempotent: if the attached operator already matches the setup, no
        operator is rebuilt (the dataset is returned as is, or as a shallow copy recording the
        new setup when its provenance differs).

        The setup decides only how the likelihood is computed (backend, operator), never the
        model (mesh, over-sampling, regularization).

        Parameters
        ----------
        setup
            The `aa.Setup`, or a setup_id string such as `"numba_cpu_sparse_fp64"`.
        batch_size
            The batch size of `apply_sparse_operator()`, used by a `jax` sparse setup.

        Returns
        -------
        Imaging
            The dataset with the setup applied.

        Raises
        ------
        exc.DatasetException
            If a sparse setup is applied to a dataset without a PSF, or names a backend with no
            imaging sparse operator (`numpy`).
        """
        setup = setup_from(setup)

        if self._sparse_operator_matches(setup=setup):
            if self.setup == setup:
                return self
            dataset = copy.copy(self)
            dataset.setup = setup
            dataset._lazy_sparse_operator = None
            return dataset

        if setup.operator == "sparse":
            if setup.backend not in ("jax", "numba"):
                raise exc.DatasetException(
                    f"The setup {setup.setup_id!r} asks for the imaging sparse operator on the "
                    f"{setup.backend!r} backend, which has none: the imaging sparse operator is "
                    f"built for 'jax' (apply_sparse_operator) or 'numba' "
                    f"(apply_sparse_operator_cpu). Use a jax or numba setup, or a dense one."
                )
            if self.psf is None:
                raise exc.DatasetException(
                    f"The setup {setup.setup_id!r} asks for the imaging sparse operator, which "
                    f"precomputes PSF products, but this dataset has no PSF. Use a dense setup."
                )

        dataset = Imaging(
            data=self.data,
            noise_map=self.noise_map,
            psf=self.psf,
            noise_covariance_matrix=self.noise_covariance_matrix,
            over_sample_size_lp=self.over_sample_size_lp,
            over_sample_size_pixelization=self.over_sample_size_pixelization,
            convolve_over_sample_size_lp=self.convolve_over_sample_size_lp,
            convolve_over_sample_size_pixelization=self.convolve_over_sample_size_pixelization,
            check_noise_map=False,
            setup=setup,
        )

        if setup.operator != "sparse":
            return dataset

        if setup.backend == "jax":
            return dataset.apply_sparse_operator(batch_size=batch_size)

        return dataset.apply_sparse_operator_cpu()
