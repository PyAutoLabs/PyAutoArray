from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray import Mask2D

import numpy as np
from pathlib import Path
import scipy
from typing import Optional, Tuple, Union
import warnings

from autoconf import conf
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray import exc


class ConvolverState:
    def __init__(
        self,
        kernel: Array2D,
        mask: Mask2D,
    ):
        """
        Compute and store the padded shapes and masks required for FFT-based convolution
        of masked 2D data with a kernel.

        FFT convolution operates on fully-sampled rectangular arrays, whereas scientific
        imaging data are typically defined only on a subset of pixels via a mask. This
        class determines how masked real-space data are embedded into a padded array,
        transformed to Fourier space, convolved with a kernel, and transformed back such
        that the result is equivalent to linear (not circular) convolution.

        The input mask defines which pixels contain valid data and therefore which
        regions of the image must be retained when mapping to and from FFT space. The
        kernel shape defines how far flux from unmasked pixels can spread into masked
        regions during convolution.

        This initializer inspects the mask and kernel to compute three key array shapes:

        ``mask_shape``
            The minimal rectangular bounding box enclosing all unmasked (False) pixels
            in the mask, expanded by half the kernel size in each direction. This is the
            smallest region that must be retained to ensure that convolution does not
            lose flux near the mask boundary.

        ``full_shape``
            The minimal array shape required for exact linear convolution, defined as::

                full_shape = mask_shape + kernel_shape - 1

            Padding to this size guarantees that FFT-based convolution is mathematically
            equivalent to direct spatial convolution, with no wrap-around artefacts.

        ``fft_shape``
            The FFT-efficient padded shape actually used for computation. Each dimension
            of ``full_shape`` is independently rounded up to the next fast length for
            real FFTs using ``scipy.fft.next_fast_len``. This shape defines the size of
            all arrays sent to and returned from FFT space.

            Note that even FFT sizes are currently incremented to odd sizes as a
            workaround for kernel-centering issues with even-sized kernels. This is an
            implementation detail and should be replaced by correct internal padding
            and centering logic.

        After determining ``fft_shape``, the input mask is padded accordingly and a
        *blurring mask* is derived. The blurring mask identifies pixels that are outside
        the original unmasked region but receive non-zero flux due to convolution with
        the kernel. These pixels must be retained when mapping results back to the
        masked domain to ensure correct convolution near mask boundaries.

        Parameters
        ----------
        kernel
            The 2D convolution kernel (e.g. a PSF). If a 1D kernel is provided, it is
            internally promoted to a minimal 2D kernel.
        mask
            A 2D boolean mask where False values indicate unmasked (valid) pixels and
            True values indicate masked pixels. The spatial extent of False pixels
            defines the region of the image that is embedded into FFT space.

        Attributes
        ----------
        fft_shape
            The FFT-friendly padded shape used for all Fourier transforms.
        mask
            The input mask padded to ``fft_shape``, with masked pixels set to True.
        blurring_mask
            A derived mask identifying pixels that are masked in the original input
            but receive flux due to convolution with the kernel.
        fft_kernel
            The real FFT of the padded kernel, used for efficient convolution in
            Fourier space.
        fft_kernel_mapping
            A broadcast-ready view of ``fft_kernel`` for multi-channel convolution.
        """
        if len(kernel) == 1:
            kernel = kernel.resized_from(new_shape=(3, 3))

        self.kernel = kernel

        ys, xs = np.where(~mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        (pad_y, pad_x) = self.kernel.shape_native

        mask_shape = (
            (y_max + pad_y // 2) - (y_min - pad_y // 2),
            (x_max + pad_x // 2) - (x_min - pad_x // 2),
        )

        full_shape = tuple(
            s1 + s2 - 1 for s1, s2 in zip(mask_shape, self.kernel.shape_native)
        )
        fft_shape = tuple(scipy.fft.next_fast_len(s, real=True) for s in full_shape)

        self.fft_shape = fft_shape
        self.mask = mask.resized_from(self.fft_shape, pad_value=1)
        self.blurring_mask = self.mask.derive_mask.blurring_from(
            kernel_shape_native=self.kernel.shape_native
        )

        self.fft_kernel = np.fft.rfft2(self.kernel.native.array, s=self.fft_shape)
        self.fft_kernel_mapping = np.expand_dims(self.fft_kernel, 2)


class Convolver:
    def __init__(
        self,
        kernel: Array2D,
        state: Optional[ConvolverState] = None,
        normalize: bool = False,
        use_fft: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        """
        A 2D convolution kernel paired with a mask, providing real-space and FFT-based
        convolution of images or mapping matrices.

        The ``Convolver`` is a subclass of ``Array2D`` with additional methods for
        performing point spread function (PSF) convolution. Each entry of the kernel
        corresponds to the PSF value at the centre of a pixel on a uniform 2D grid.

        Two convolution modes are supported:

        - **Real-space convolution**:
          Performed directly via sliding-window summation or
          ``jax.scipy.signal.convolve``. This mode is exact and requires no padding,
          but becomes computationally expensive for large kernels.

        - **FFT-based convolution**:
          Performed by embedding the input image and kernel into padded arrays,
          transforming them to Fourier space, multiplying, and transforming back.
          This mode is typically faster for kernels larger than approximately 5×5,
          but requires careful handling of padding, masking, and kernel centering.

        All logic related to FFT padding, mask expansion, linear (non-circular)
        convolution, and blurring-mask construction is handled by
        ``ConvolverState``. See the ``ConvolverState`` docstring for a detailed
        description of how masked real-space data are mapped to and from FFT space.

        When FFT convolution is enabled, the ``Convolver`` expects a corresponding
        ``ConvolverState`` defining the FFT geometry. The padded FFT shape is stored
        in ``state.fft_shape`` and must be consistent with the shape of any arrays
        passed for convolution. Attempting FFT convolution without a valid state
        will raise an exception to avoid silent shape or alignment errors.

        Parameters
        ----------
        kernel
            The raw 2D kernel values. These represent the PSF sampled at pixel
            centres and may be normalised to sum to unity if ``normalize=True``.
        state
            Optional ``ConvolverState`` instance defining FFT padding, mask
            expansion, and kernel Fourier transforms. Required when using FFT
            convolution.
        normalize
            If True, the kernel values are rescaled such that their sum is unity.
        use_fft
            If True, convolution is performed in Fourier space using the provided
            ``ConvolverState``.
            If False, convolution is performed in real space.
            If None, the default behaviour specified in the configuration is used.
        *args, **kwargs
            Passed to the ``Array2D`` constructor.

        Notes
        -----
        - When performing real-space convolution, the kernel must have odd dimensions
          in both axes so that it has a well-defined central pixel.
        - When performing FFT convolution, kernel centering, padding, and mask
          expansion are handled by ``ConvolverState``.
        - Blurring masks ensure that PSF flux spilling outside the main image mask
          is included correctly. Omitting them may lead to underestimated PSF wings.
        - For very small kernels, FFT and real-space convolution may differ slightly
          near mask boundaries due to padding and truncation effects.
        """
        self.kernel = kernel

        if normalize:
            self.kernel._array = np.divide(
                self.kernel._array, np.sum(self.kernel._array)
            )

        self._use_fft = use_fft

        if not self._use_fft:
            if (
                self.kernel.shape_native[0] % 2 == 0
                or self.kernel.shape_native[1] % 2 == 0
            ):
                raise exc.KernelException("Convolver Convolver must be odd")

        self._state = state

    def state_from(self, mask):

        if (
            mask.shape_native[0] != self.kernel.shape_native[0]
            or mask.shape_native[1] != self.kernel.shape_native[1]
        ):
            return ConvolverState(kernel=self.kernel, mask=mask)

        if self._state is None:
            return ConvolverState(kernel=self.kernel, mask=mask)

        return self._state

    @property
    def use_fft(self):
        if self._use_fft is None:
            return conf.instance["general"]["psf"]["use_fft_default"]

        return self._use_fft

    @property
    def normalized(self) -> "Convolver":
        """
        Normalize the Convolver such that its data_vector values sum to unity.

        A copy of the kernel is used to avoid mutating the original kernel instance,
        and no existing state is reused so that any cached FFTs are recomputed for
        the normalized kernel.
        """
        kernel_copy = self.kernel.copy()
        return Convolver(kernel=kernel_copy, state=None, normalize=True)

    @classmethod
    def no_blur(cls, pixel_scales):
        """
        Setup the Convolver as a kernel which does not convolve any signal, which is simply an array of shape (1, 1)
        with value 1.

        Parameters
        ----------
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        """

        kernel = Array2D.no_mask(
            values=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            pixel_scales=pixel_scales,
        )

        return cls(kernel=kernel)

    @classmethod
    def from_gaussian(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales,
        sigma: float,
        centre: Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        normalize: bool = False,
    ) -> "Convolver":
        """
        Setup the Convolver as a 2D symmetric elliptical Gaussian profile, according to the equation:

        (1.0 / (sigma * sqrt(2.0*pi))) * exp(-0.5 * (r/sigma)**2)


        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        sigma
            The value of sigma in the equation, describing the size and full-width half maximum of the Gaussian.
        centre
            The (y,x) central coordinates of the Gaussian.
        axis_ratio
            The axis-ratio of the elliptical Gaussian.
        angle
            The rotational angle of the Gaussian's ellipse defined counter clockwise from the positive x-axis.
        normalize
            If True, the Convolver's array values are normalized such that they sum to 1.0.
        """

        grid = Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scales)
        grid_shifted = np.subtract(grid.array, centre)
        grid_radius = np.sqrt(np.sum(grid_shifted**2.0, 1))
        theta_coordinate_to_profile = np.arctan2(
            grid_shifted[:, 0], grid_shifted[:, 1]
        ) - np.radians(angle)
        grid_transformed = np.vstack(
            (
                grid_radius * np.sin(theta_coordinate_to_profile),
                grid_radius * np.cos(theta_coordinate_to_profile),
            )
        ).T

        grid_elliptical_radii = np.sqrt(
            np.add(
                np.square(grid_transformed[:, 1]),
                np.square(np.divide(grid_transformed[:, 0], axis_ratio)),
            )
        )

        gaussian = np.multiply(
            np.divide(1.0, sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_elliptical_radii, sigma))),
        )

        gaussian = Array2D.no_mask(
            values=gaussian, pixel_scales=pixel_scales, shape_native=shape_native
        )

        return Convolver(
            kernel=gaussian,
            normalize=normalize,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        hdu: int,
        pixel_scales,
        origin=(0.0, 0.0),
        normalize: bool = False,
    ) -> "Convolver":
        """
        Loads the Convolver from a .fits file.

        Parameters
        ----------
        file_path
            The path the file is loaded from, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        hdu
            The Header-Data Unit of the .fits file the array data is loaded from.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Convolver's array values are normalized such that they sum to 1.0.
        """

        array = Array2D.from_fits(
            file_path=file_path,
            hdu=hdu,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return Convolver(
            kernel=array,
            normalize=normalize,
        )

    def mapping_matrix_native_from(
        self,
        mapping_matrix: np.ndarray,
        mask: "Mask2D",
        blurring_mapping_matrix: Optional[np.ndarray] = None,
        blurring_mask: Optional["Mask2D"] = None,
        use_mixed_precision: bool = False,
        xp=np,
    ) -> np.ndarray:
        """
        Expand a slim mapping matrix (image-plane) and optional blurring mapping matrix
        into a full native 3D cube (ny, nx, n_src).

        This is primarily used for real-space convolution, where the pixel-to-source
        mapping must be represented on the full image grid.

        Parameters
        ----------
        mapping_matrix : ndarray (N_pix, N_src)
            Slim mapping matrix for unmasked image pixels, mapping each image pixel
            to source-plane pixels.
        mask : Mask2D
            Mask defining which image pixels are unmasked. Used to expand the slim
            mapping matrix into a native grid.
        blurring_mapping_matrix : ndarray (N_blur, N_src), optional
            Mapping matrix for blurring pixels outside the main mask (e.g. light
            spilling in from outside). If provided, it is also scattered into the
            native cube.
        blurring_mask : Mask2D, optional
            Mask defining the blurring region pixels. Must be provided if
            `blurring_mapping_matrix` is given and `slim_to_native_blurring_tuple`
            is not already cached.
        use_mixed_precision
            If True, the mapping matrices are cast to single precision (float32) to
            speed up GPU computations and reduce VRAM usage. If False, double precision
            (float64) is used for maximum accuracy.

        Returns
        -------
        ndarray (ny, nx, N_src)
            Native 3D mapping matrix cube with dimensions (image_y, image_x, sources).
            Contains contributions from both the main mapping matrix and, if provided,
            the blurring mapping matrix.
        """
        dtype_native = xp.float32 if use_mixed_precision else xp.float64

        n_src = mapping_matrix.shape[1]

        mapping_matrix_native = xp.zeros(mask.shape + (n_src,), dtype=dtype_native)

        # Cast inputs to the target dtype to avoid implicit up/downcasts inside scatter
        mm = (
            mapping_matrix
            if mapping_matrix.dtype == dtype_native
            else xp.asarray(mapping_matrix, dtype=dtype_native)
        )

        if xp.__name__.startswith("jax"):
            mapping_matrix_native = mapping_matrix_native.at[
                mask.slim_to_native_tuple
            ].set(mm)
        else:
            mapping_matrix_native[mask.slim_to_native_tuple] = np.asarray(mm)

        if blurring_mapping_matrix is not None:
            bm = blurring_mapping_matrix
            if getattr(bm, "dtype", None) != dtype_native:
                bm = xp.asarray(bm, dtype=dtype_native)

            if xp.__name__.startswith("jax"):
                mapping_matrix_native = mapping_matrix_native.at[
                    blurring_mask.slim_to_native_tuple
                ].set(bm)
            else:
                mapping_matrix_native[blurring_mask.slim_to_native_tuple] = np.asarray(
                    bm
                )

        return mapping_matrix_native

    def convolved_image_from(
        self,
        image,
        blurring_image,
        jax_method="direct",
        use_mixed_precision: bool = False,
        xp=np,
    ):
        """
        Convolve an input masked image with this PSF.

        This method chooses between an FFT-based convolution (default if
        ``self.use_fft=True``) or a direct real-space convolution, depending on
        how the Convolver was configured.

        In the FFT branch:
        - The input image (and optional blurring image) are resized / padded to
          match the FFT-friendly padded shape (``fft_shape``) associated with this kernel.
        - The PSF and image are transformed to Fourier space via ``jax.numpy.fft.rfft2``.
        - Convolution is performed as elementwise multiplication.
        - The result is inverse-transformed and cropped back to the masked region.

        Padding ensures that the FFT implements *linear* convolution, not circular,
        and avoids wrap-around artefacts. The required padding is determined by
        ``fft_shape_from(mask)``. If no precomputed shapes exist, they are computed
        on the fly. For reproducible behaviour, precompute and set
        ``fft_shape` on the kernel.

        If ``use_fft=False``, convolution falls back to
        :meth:`Convolver.convolved_image_via_real_space_from`.

        Parameters
        ----------
        image
            Masked 2D image array to convolve.
        blurring_image
            Masked image containing flux from outside the mask core that blurs
            into the masked region after convolution. If ``None``, only the direct
            image is convolved, which may be numerically incorrect if the mask
            excludes PSF wings.
        jax_method : {"direct", "fft"}
            Backend passed to ``jax.scipy.signal.convolve`` when in real-space mode.
            Ignored for FFT convolutions.

        Returns
        -------
        Array2D
            The convolved image in slim (1D masked) format.
        """
        if xp is np:
            return self.convolved_image_via_real_space_np_from(
                image=image, blurring_image=blurring_image, xp=xp
            )

        if not self.use_fft:
            return self.convolved_image_via_real_space_from(
                image=image, blurring_image=blurring_image, jax_method=jax_method, xp=xp
            )

        import jax
        import jax.numpy as jnp
        from autoarray.structures.arrays.uniform_2d import Array2D

        state = self.state_from(mask=image.mask)

        # Build combined native image in the FFT dtype
        image_both_native = xp.zeros(state.fft_shape, dtype=jnp.float64)

        image_both_native = image_both_native.at[state.mask.slim_to_native_tuple].set(
            jnp.asarray(image.array, dtype=jnp.float64)
        )

        if blurring_image is not None:
            image_both_native = image_both_native.at[
                state.blurring_mask.slim_to_native_tuple
            ].set(jnp.asarray(blurring_image.array, dtype=jnp.float64))
        else:
            warnings.warn(
                "No blurring_image provided. Only the direct image will be convolved. "
                "This may change the correctness of the PSF convolution."
            )

        # FFT the combined image
        fft_image_native = xp.fft.rfft2(
            image_both_native, s=state.fft_shape, axes=(0, 1)
        )

        # Multiply by PSF in Fourier space and invert
        blurred_image_full = xp.fft.irfft2(
            state.fft_kernel * fft_image_native, s=state.fft_shape, axes=(0, 1)
        )
        ky, kx = self.kernel.shape_native  # (21, 21)
        off_y = (ky - 1) // 2
        off_x = (kx - 1) // 2

        blurred_image_full = xp.roll(
            blurred_image_full, shift=(-off_y, -off_x), axis=(0, 1)
        )

        start_indices = (off_y, off_x)

        blurred_image_native = jax.lax.dynamic_slice(
            blurred_image_full, start_indices, state.fft_shape
        )

        # Return slim form; optionally cast for downstream stability
        blurred_slim = blurred_image_native[state.mask.slim_to_native_tuple]

        blurred_image = Array2D(values=blurred_slim, mask=image.mask)

        if use_mixed_precision:
            blurred_image = blurred_image.astype(jnp.float32)

        return blurred_image

    def convolved_mapping_matrix_from(
        self,
        mapping_matrix,
        mask,
        blurring_mapping_matrix=None,
        blurring_mask: Optional[Mask2D] = None,
        jax_method="direct",
        use_mixed_precision: bool = False,
        xp=np,
    ):
        """
        Convolve a source-plane mapping matrix with this PSF.

        A mapping matrix maps image-plane unmasked pixels to source-plane pixels.
        This method performs the equivalent operation of PSF convolution on the
        mapping matrix, so that model visibilities / images can be computed via
        matrix multiplication instead of explicit convolution.

        If ``use_fft=True``, convolution is performed in Fourier space:
        - The mapping matrix is scattered into a 3D native cube
          (ny, nx, n_src).
        - An FFT of this cube is multiplied by the precomputed FFT of the PSF.
        - The inverse FFT is taken and cropped to the mask region.
        - The slim (masked 1D) representation is returned.

        If ``use_fft=False``, convolution falls back to
        :meth:`Convolver.convolved_mapping_matrix_via_real_space_from`.

        Notes
        -----
        - FFT convolution requires that ``self.fft_shape`` and related padding
          attributes are precomputed. If not, a ``ValueError`` is raised with the
          expected vs actual shapes. This ensures the mapping matrix is padded
          consistently with the PSF.
        - The optional ``blurring_mapping_matrix`` plays the same role as
          ``blurring_image`` in :meth:`convolved_image_from`, accounting for PSF flux
          that falls into the masked region from outside.

        Parameters
        ----------
        mapping_matrix : ndarray of shape (N_pix, N_src)
            Slim mapping matrix from unmasked pixels to source pixels.
        mask : Mask2D
            Associated mask defining the image grid.
        blurring_mapping_matrix : ndarray of shape (N_blur, N_src), optional
            Mapping matrix for the blurring region, outside the mask core.
        jax_method : str
            Backend passed to real-space convolution if ``use_fft=False``.
        use_mixed_precision
            If `True`, the FFT is performed using single precision, which provide significant speed up when using a
            GPU (x4), reduces VRAM use and is expected to have minimal impact on the accuracy of the results. If `False`,
            the FFT is performed using double precision, which is the default and is more accurate but slower on a GPU.

        Returns
        -------
        ndarray of shape (N_pix, N_src)
            Convolved mapping matrix in slim form.
        """
        # -------------------------------------------------------------------------
        # NumPy path unchanged
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # NumPy path unchanged
        # -------------------------------------------------------------------------
        if xp is np:
            return self.convolved_mapping_matrix_via_real_space_np_from(
                mapping_matrix=mapping_matrix,
                mask=mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
                blurring_mask=blurring_mask,
                xp=xp,
            )

        # -------------------------------------------------------------------------
        # Non-FFT JAX path unchanged
        # -------------------------------------------------------------------------
        if not self.use_fft:
            return self.convolved_mapping_matrix_via_real_space_from(
                mapping_matrix=mapping_matrix,
                mask=mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
                blurring_mask=blurring_mask,
                jax_method=jax_method,
                xp=xp,
            )

        import jax
        import jax.numpy as jnp

        state = self.state_from(mask=mask)

        # -------------------------------------------------------------------------
        # Mixed precision handling
        # -------------------------------------------------------------------------
        fft_complex_dtype = jnp.complex64 if use_mixed_precision else jnp.complex128

        # -------------------------------------------------------------------------
        # Build native cube on the *native mask grid*
        # -------------------------------------------------------------------------
        mapping_matrix_native = self.mapping_matrix_native_from(
            mapping_matrix=mapping_matrix,
            mask=state.mask,
            blurring_mapping_matrix=blurring_mapping_matrix,
            blurring_mask=state.blurring_mask,
            use_mixed_precision=use_mixed_precision,
            xp=xp,
        )
        # shape: (ny_native, nx_native, n_src)

        # -------------------------------------------------------------------------
        # FFT convolution
        # -------------------------------------------------------------------------
        fft_mapping_matrix_native = xp.fft.rfft2(
            mapping_matrix_native, s=state.fft_shape, axes=(0, 1)
        )

        blurred_mapping_matrix_full = xp.fft.irfft2(
            state.fft_kernel_mapping * fft_mapping_matrix_native,
            s=state.fft_shape,
            axes=(0, 1),
        )

        # -------------------------------------------------------------------------
        # APPLY SAME FIX AS convolved_image_from
        # -------------------------------------------------------------------------
        ky, kx = self.kernel.shape_native
        off_y = (ky - 1) // 2
        off_x = (kx - 1) // 2

        blurred_mapping_matrix_full = xp.roll(
            blurred_mapping_matrix_full,
            shift=(-off_y, -off_x),
            axis=(0, 1),
        )

        # -------------------------------------------------------------------------
        # Extract native grid (same as image path)
        # -------------------------------------------------------------------------
        start_indices = (off_y, off_x, 0)

        out_shape = state.mask.shape_native + (blurred_mapping_matrix_full.shape[2],)

        blurred_mapping_matrix_native = jax.lax.dynamic_slice(
            blurred_mapping_matrix_full,
            start_indices,
            out_shape,
        )

        # -------------------------------------------------------------------------
        # Slim using ORIGINAL mask indices (same grid)
        # -------------------------------------------------------------------------
        blurred_slim = blurred_mapping_matrix_native[state.mask.slim_to_native_tuple]

        return blurred_slim

    def convolved_image_via_real_space_from(
        self,
        image: np.ndarray,
        blurring_image: Optional[np.ndarray] = None,
        jax_method: str = "direct",
        xp=np,
    ):
        """
        Convolve an input masked image with this PSF in real space.

        This is the direct method (non-FFT) where convolution is explicitly
        performed using ``jax.scipy.signal.convolve`` with the kernel in native
        space.

        Unlike FFT convolution, this does not require padding shapes, but it is
        typically much slower for large kernels (> ~5x5).

        Parameters
        ----------
        image
            Masked image array to convolve.
        blurring_image
            Blurring contribution from outside the mask core. If None, only the
            direct image is convolved (which may be numerically incorrect).
        jax_method
            Method flag for JAX convolution backend (default "direct").

        Returns
        -------
        Array2D
            Convolved image in slim format.
        """

        if xp is np:
            return self.convolved_image_via_real_space_np_from(
                image=image, blurring_image=blurring_image, xp=xp
            )

        import jax

        state = self.state_from(mask=image.mask)

        # start with native array padded with zeros
        image_native = xp.zeros(state.fft_shape, dtype=image.array.dtype)

        # set image pixels
        image_native = image_native.at[state.mask.slim_to_native_tuple].set(image.array)

        # add blurring contribution if provided
        if blurring_image is not None:

            image_native = image_native.at[
                state.blurring_mask.slim_to_native_tuple
            ].set(blurring_image.array)

        else:
            warnings.warn(
                "No blurring_image provided. Only the direct image will be convolved. "
                "This may change the correctness of the PSF convolution."
            )

        convolve_native = jax.scipy.signal.convolve(
            image_native, self.kernel.native.array, mode="same", method=jax_method
        )

        convolved_array_1d = convolve_native[state.mask.slim_to_native_tuple]

        return Array2D(values=convolved_array_1d, mask=image.mask)

    def convolved_mapping_matrix_via_real_space_from(
        self,
        mapping_matrix: np.ndarray,
        mask,
        blurring_mapping_matrix: Optional[np.ndarray] = None,
        blurring_mask: Optional[Mask2D] = None,
        jax_method: str = "direct",
        xp=np,
    ):
        """
        Convolve a source-plane mapping matrix with this PSF in real space.

        Equivalent to :meth:`convolved_mapping_matrix_from`, but using explicit
        real-space convolution rather than FFTs. This avoids FFT padding issues
        but is slower for large kernels.

        The mapping matrix is expanded into a native cube (ny, nx, n_src),
        convolved with the kernel (broadcast along the source axis),
        and reduced back to slim form.

        Parameters
        ----------
        mapping_matrix
            Slim mapping matrix from unmasked pixels to source pixels.
        mask
            Mask defining the pixelization grid.
        blurring_mapping_matrix : ndarray (N_blur, N_src), optional
            Mapping matrix for blurring region pixels outside the mask core.
        jax_method
            Backend passed to JAX convolution.

        Returns
        -------
        ndarray (N_pix, N_src)
            Convolved mapping matrix in slim form.
        """

        if xp is np:
            return self.convolved_mapping_matrix_via_real_space_np_from(
                mapping_matrix=mapping_matrix,
                mask=mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
                blurring_mask=blurring_mask,
                xp=xp,
            )

        import jax

        state = self.state_from(mask=mask)

        mapping_matrix_native = self.mapping_matrix_native_from(
            mapping_matrix=mapping_matrix,
            mask=state.mask,
            blurring_mapping_matrix=blurring_mapping_matrix,
            blurring_mask=state.blurring_mask,
            xp=xp,
        )

        blurred_mapping_matrix_native = jax.scipy.signal.convolve(
            mapping_matrix_native,
            self.kernel.native.array[..., None],
            mode="same",
            method=jax_method,
        )

        # return slim form
        return blurred_mapping_matrix_native[state.mask.slim_to_native_tuple]

    def convolved_image_via_real_space_np_from(
        self, image: np.ndarray, blurring_image: Optional[np.ndarray] = None, xp=np
    ):
        """
        Convolve an input masked image with this PSF in real space.

        This is the direct method (non-FFT) where convolution is explicitly
        performed using ``jax.scipy.signal.convolve`` with the kernel in native
        space.

        Unlike FFT convolution, this does not require padding shapes, but it is
        typically much slower for large kernels (> ~5x5).

        Parameters
        ----------
        image
            Masked image array to convolve.
        blurring_image
            Blurring contribution from outside the mask core. If None, only the
            direct image is convolved (which may be numerically incorrect).
        jax_method
            Method flag for JAX convolution backend (default "direct").

        Returns
        -------
        Array2D
            Convolved image in slim format.
        """

        from scipy.signal import convolve as scipy_convolve

        state = self.state_from(mask=image.mask)

        # start with native array padded with zeros
        image_native = xp.zeros(state.fft_shape)

        # set image pixels
        image_native[state.mask.slim_to_native_tuple] = image.array

        # add blurring contribution if provided
        if blurring_image is not None:

            image_native[state.blurring_mask.slim_to_native_tuple] = (
                blurring_image.array
            )

        else:
            warnings.warn(
                "No blurring_image provided. Only the direct image will be convolved. "
                "This may change the correctness of the PSF convolution."
            )

        convolve_native = scipy_convolve(
            image_native, self.kernel.native.array, mode="same", method="auto"
        )

        convolved_array_1d = convolve_native[state.mask.slim_to_native_tuple]

        return Array2D(values=convolved_array_1d, mask=image.mask)

    def convolved_mapping_matrix_via_real_space_np_from(
        self,
        mapping_matrix: np.ndarray,
        mask,
        blurring_mapping_matrix: Optional[np.ndarray] = None,
        blurring_mask: Optional[Mask2D] = None,
        xp=np,
    ):
        """
        Convolve a source-plane mapping matrix with this PSF in real space.

        Equivalent to :meth:`convolved_mapping_matrix_from`, but using explicit
        real-space convolution rather than FFTs. This avoids FFT padding issues
        but is slower for large kernels.

        The mapping matrix is expanded into a native cube (ny, nx, n_src),
        convolved with the kernel (broadcast along the source axis),
        and reduced back to slim form.

        Parameters
        ----------
        mapping_matrix
            Slim mapping matrix from unmasked pixels to source pixels.
        mask
            Mask defining the pixelization grid.
        blurring_mapping_matrix : ndarray (N_blur, N_src), optional
            Mapping matrix for blurring region pixels outside the mask core.
        jax_method
            Backend passed to JAX convolution.

        Returns
        -------
        ndarray (N_pix, N_src)
            Convolved mapping matrix in slim form.
        """

        from scipy.signal import convolve as scipy_convolve

        state = self.state_from(mask=mask)

        mapping_matrix_native = self.mapping_matrix_native_from(
            mapping_matrix=mapping_matrix,
            mask=state.mask,
            blurring_mapping_matrix=blurring_mapping_matrix,
            blurring_mask=state.blurring_mask,
            xp=xp,
        )

        blurred_mapping_matrix_native = scipy_convolve(
            mapping_matrix_native,
            self.kernel.native.array[..., None],
            mode="same",
        )

        # return slim form
        return blurred_mapping_matrix_native[state.mask.slim_to_native_tuple]
