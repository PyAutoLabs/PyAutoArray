from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray import Mask2D

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

from autonerves import cached_property
from autonerves import conf
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray import exc


class ConvolverState:
    def __init__(
        self,
        kernel: Array2D,
        mask: Mask2D,
        blurring_mask: Optional["Mask2D"] = None,
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
        blurring_mask
            Optional explicit blurring mask (same shape as ``mask``, before FFT
            resizing). If omitted it is derived from the resized mask and kernel shape
            as before. Oversampled convolution passes the upscaled image-resolution
            blurring mask here, because the region a caller evaluates blurring flux on
            is defined at image resolution, not by the fine kernel's reach.

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
        import scipy.fft
        from autoarray.mask.mask_2d_util import required_shape_for_kernel

        if blurring_mask is None:
            min_blur_shape = required_shape_for_kernel(mask, self.kernel.shape_native)
        else:
            # The explicit blurring region can extend further than the kernel's own
            # reach (its footprint is defined at image resolution and upscaled), so
            # the FFT frame must be sized to keep every blurring pixel after resizing.
            combined_mask = np.array(mask) & np.array(blurring_mask)
            min_blur_shape = required_shape_for_kernel(
                combined_mask, self.kernel.shape_native
            )

        fft_shape = tuple(
            scipy.fft.next_fast_len(max(s, r), real=True)
            for s, r in zip(full_shape, min_blur_shape)
        )

        self.fft_shape = fft_shape
        self.source_mask = mask
        self.mask = mask.resized_from(self.fft_shape, pad_value=1)

        if blurring_mask is None:
            self.blurring_mask = self.mask.derive_mask.blurring_from(
                kernel_shape_native=self.kernel.shape_native
            )
        else:
            self.blurring_mask = blurring_mask.resized_from(self.fft_shape, pad_value=1)

        # Set by Convolver.state_from when convolve_over_sample_size > 1: the
        # permutations from per-pixel sub-block ordering to the fine mask's
        # row-major slim ordering, for the image and blurring regions.
        self.sub_slim_to_fine_slim = None
        self.blurring_sub_slim_to_fine_slim = None
        self.image_mask = None

        self.fft_kernel = np.fft.rfft2(self.kernel.native.array, s=self.fft_shape)
        self.fft_kernel_mapping = np.expand_dims(self.fft_kernel, 2)
        # Pre-cached complex64 view for the use_mixed_precision=True path of
        # convolved_image_from. Cast once here so the FFT branch does not
        # repeat the astype per JIT trace — it would otherwise produce a fresh
        # numpy buffer each call, which on CPU costs more than the fp32 FFT
        # saves. convolved_mapping_matrix_from intentionally does NOT use a
        # complex64 kernel — see that method's body for why.
        self.fft_kernel_c64 = self.fft_kernel.astype(np.complex64)

    def is_for_mask(self, mask) -> bool:
        """
        Whether this state was built from the input mask, and can therefore be reused
        instead of rebuilt (its padded FFT geometry is only valid for that mask).
        """
        return (
            self.source_mask.pixel_scales == mask.pixel_scales
            and self.source_mask.shape_native == mask.shape_native
            and np.array_equal(np.array(self.source_mask), np.array(mask))
        )


class Convolver:
    def __init__(
        self,
        kernel: Array2D,
        state: Optional[ConvolverState] = None,
        normalize: bool = False,
        use_fft: Optional[bool] = None,
        convolve_over_sample_size: int = 1,
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
        convolve_over_sample_size
            The integer over sample size of the PSF. If above 1, the ``kernel`` is the
            PSF sampled at ``over_sample_size`` times the image resolution (e.g. a
            value of 2 means the PSF has a resolution 2x higher than the image, with
            pixel scales half the image's). Convolution is then performed on a grid
            upscaled by this factor and the result binned back to image resolution by
            the mean of each block, improving the accuracy of the blurring. The
            convolution methods then expect over-sampled (sub-gridded) inputs rather
            than image-resolution arrays. A value of 1 (default) leaves all behaviour
            unchanged.
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

        if isinstance(convolve_over_sample_size, bool) or not isinstance(
            convolve_over_sample_size, (int, np.integer)
        ):
            raise TypeError(
                f"convolve_over_sample_size must be a plain int (adaptive over "
                f"sampling is not supported for PSF convolution), but a "
                f"{type(convolve_over_sample_size).__name__} was input."
            )

        if convolve_over_sample_size < 1:
            raise exc.KernelException(
                f"convolve_over_sample_size must be >= 1, but "
                f"{convolve_over_sample_size} was input."
            )

        self.convolve_over_sample_size = int(convolve_over_sample_size)

        self._state = state

    @property
    def kernel_shape_image_resolution(self) -> Tuple[int, int]:
        """
        The shape of the kernel's footprint in image-resolution pixels.

        For ``convolve_over_sample_size=1`` this is the kernel's native shape. For an
        oversampled kernel it is the (odd) number of image pixels the fine kernel
        reaches, used e.g. to derive the image-resolution blurring mask.
        """
        s = self.convolve_over_sample_size

        if s == 1:
            return self.kernel.shape_native

        return tuple(
            2 * int(np.ceil((k // 2) / s)) + 1 for k in self.kernel.shape_native
        )

    @cached_property
    def reversed_kernel(self) -> "Convolver":
        """
        This convolver with its kernel reversed along both axes.

        Convolving with the reversed kernel is exactly *correlating* with this convolver's
        kernel, because reversing both axes of one operand converts a convolution into a
        correlation::

            (x * flip(k))[i] = sum_d k[d] x[i + d - c] = correlate(x, k)[i]

        Callers whose operator is defined as a sliding-window correlation (for example the
        mapper x linear-func block of the imaging curvature matrix, which sums
        ``psf[dy, dx] * image[y + dy - cy, x + dx - cx]``) can therefore route through the
        batched convolution machinery by convolving with this convolver instead of hand
        rolling the correlation.

        The reversed convolver inherits this one's ``use_fft`` policy and
        ``convolve_over_sample_size``, and reuses this convolver's ``ConvolverState``
        geometry (rebuilt for the reversed kernel, whose Fourier transform differs) when one
        was preloaded, so the FFT geometry is built once rather than once per call.

        Cached, so a `Convolver` that outlives the objects using it (a dataset's PSF outlives
        the `Inversion` rebuilt for every likelihood evaluation) builds its reversed kernel and
        that kernel's FFT geometry only once.
        """
        kernel = Array2D.no_mask(
            values=np.asarray(self.kernel.native.array)[::-1, ::-1].copy(),
            pixel_scales=self.kernel.pixel_scales,
            origin=self.kernel.origin,
        )

        # An oversampled state carries sub-pixel permutations that only `state_from` can
        # attach, so it is left to rebuild that case rather than preloading a partial state.
        state = (
            ConvolverState(kernel=kernel, mask=self._state.source_mask)
            if self._state is not None and self.convolve_over_sample_size == 1
            else None
        )

        return Convolver(
            kernel=kernel,
            state=state,
            use_fft=self._use_fft,
            convolve_over_sample_size=self.convolve_over_sample_size,
        )

    def state_from(self, mask):

        if self.convolve_over_sample_size > 1:

            if self._state is not None:
                return self._state

            return self._fine_state_from(mask=mask)

        if self._state is not None and self._state.is_for_mask(mask=mask):
            return self._state

        return ConvolverState(kernel=self.kernel, mask=mask)

    def _fine_state_from(self, mask) -> ConvolverState:
        """
        Build the ``ConvolverState`` for oversampled convolution: the input
        image-resolution mask is upscaled by ``convolve_over_sample_size`` and the
        existing state machinery runs on the fine mask, with the sub-block <-> fine
        slim permutations cached on the state.

        Parameters
        ----------
        mask
            The image-resolution mask the over-sampled inputs are defined on.
        """
        from autoarray.mask.mask_2d import Mask2D
        from autoarray.operators.over_sampling.over_sample_util import (
            mask_2d_upscaled_from,
            sub_slim_to_fine_slim_from,
        )

        s = self.convolve_over_sample_size

        expected_pixel_scales = (
            mask.pixel_scales[0] / s,
            mask.pixel_scales[1] / s,
        )

        if not np.allclose(
            self.kernel.pixel_scales, expected_pixel_scales, rtol=1.0e-4
        ):
            raise exc.KernelException(
                f"The kernel's pixel scales {self.kernel.pixel_scales} do not match "
                f"the mask's pixel scales divided by convolve_over_sample_size="
                f"{s} ({expected_pixel_scales}). An oversampled Convolver requires "
                f"the PSF sampled at the fine resolution."
            )

        blurring_mask = mask.derive_mask.blurring_from(
            kernel_shape_native=self.kernel_shape_image_resolution,
            allow_padding=True,
        )

        # When the mask sits close to the image edge, blurring_from pads its output
        # to a larger frame (symmetric, parity-preserving). The fine geometry needs
        # the image mask and blurring mask on one common frame, so the image mask is
        # embedded with the same padding arithmetic; symmetric padding preserves the
        # row-major slim ordering of the unmasked pixels, so the permutations and the
        # original image mask (used to wrap outputs) remain valid.
        if blurring_mask.shape_native != mask.shape_native:
            dy = blurring_mask.shape_native[0] - mask.shape_native[0]
            dx = blurring_mask.shape_native[1] - mask.shape_native[1]

            mask_frame = Mask2D(
                mask=np.pad(
                    np.array(mask),
                    ((dy // 2, dy - dy // 2), (dx // 2, dx - dx // 2)),
                    constant_values=True,
                ),
                pixel_scales=mask.pixel_scales,
                origin=mask.origin,
            )
        else:
            mask_frame = mask

        mask_fine = mask_2d_upscaled_from(mask_2d=mask_frame, over_sample_size=s)

        blurring_mask_fine = mask_2d_upscaled_from(
            mask_2d=blurring_mask, over_sample_size=s
        )

        state = ConvolverState(
            kernel=self.kernel, mask=mask_fine, blurring_mask=blurring_mask_fine
        )

        state.sub_slim_to_fine_slim = sub_slim_to_fine_slim_from(
            mask_2d=mask_frame, over_sample_size=s
        )
        state.blurring_sub_slim_to_fine_slim = sub_slim_to_fine_slim_from(
            mask_2d=blurring_mask, over_sample_size=s
        )
        state.image_mask = mask

        return state

    def _over_sampled_state_from(self, mask=None) -> ConvolverState:
        """
        Resolve the fine ``ConvolverState`` for oversampled convolution: the
        precomputed state if one was supplied (e.g. via ``Imaging(psf_setup_state=True)``),
        else built from an explicit image-resolution mask. Over-sampled inputs cannot
        carry the image mask themselves, so having neither is an error.
        """
        if self._state is not None:
            return self._state

        if mask is not None:
            return self._fine_state_from(mask=mask)

        raise exc.KernelException(
            "Oversampled convolution (convolve_over_sample_size > 1) requires either "
            "a precomputed ConvolverState or an explicit image-resolution mask, "
            "because over-sampled input arrays do not carry the mask."
        )

    def _check_over_sampled_length(self, n: int, perm: np.ndarray) -> None:
        """
        Validate the length of an over-sampled input against the cached permutation;
        a binned (image-resolution) input is a distinct, explicit error.
        """
        if n == perm.size:
            return

        s = self.convolve_over_sample_size

        if n * s**2 == perm.size:
            raise exc.KernelException(
                f"An image-resolution (binned) array of length {n} was input to an "
                f"oversampled Convolver (convolve_over_sample_size={s}), which "
                f"requires the over-sampled values of length {perm.size} in "
                f"per-pixel sub-block order (evaluate on the over-sampled grid "
                f"without binning)."
            )

        raise exc.KernelException(
            f"The input array length {n} does not match the expected over-sampled "
            f"length {perm.size} (convolve_over_sample_size="
            f"{self.convolve_over_sample_size})."
        )

    @staticmethod
    def _values_from(values):
        return values.array if hasattr(values, "array") else values

    def _over_sampled_binned_from(self, fine_slim, perm, trailing: tuple):
        """
        Bin a fine-mask slim array (row-major order) back to image resolution:
        reorder to per-pixel sub-block order via the permutation and take the mean
        of each sub-block. ``trailing`` is the shape of any extra axes (e.g. the
        source axis of a mapping matrix).
        """
        s = self.convolve_over_sample_size
        return fine_slim[perm].reshape((-1, s**2) + trailing).mean(axis=1)

    def _convolved_over_sampled_np_from(self, values, blurring_values, state):
        """
        Real-space numpy convolution of over-sampled inputs (a 1D image or a 2D
        mapping matrix, per-pixel sub-block order): scatter onto the fine FFT frame
        via the cached permutations, convolve with the fine kernel, and bin the
        result back to image resolution by the mean of each sub-block.
        """
        from scipy.signal import convolve as scipy_convolve

        perm = state.sub_slim_to_fine_slim
        self._check_over_sampled_length(n=values.shape[0], perm=perm)

        trailing = values.shape[1:]
        rows, cols = state.mask.slim_to_native_tuple

        native = np.zeros(state.fft_shape + trailing)
        native[rows[perm], cols[perm]] = values

        if blurring_values is not None:
            bperm = state.blurring_sub_slim_to_fine_slim
            self._check_over_sampled_length(n=blurring_values.shape[0], perm=bperm)
            brows, bcols = state.blurring_mask.slim_to_native_tuple
            native[brows[bperm], bcols[bperm]] = blurring_values

        kernel = self.kernel.native.array
        if trailing:
            kernel = kernel[..., None]

        convolved_native = scipy_convolve(native, kernel, mode="same", method="auto")

        fine_slim = convolved_native[state.mask.slim_to_native_tuple]
        return self._over_sampled_binned_from(fine_slim, perm, trailing)

    def _convolved_over_sampled_jax_from(
        self, values, blurring_values, state, fft_kernel, dtype, xp
    ):
        """
        FFT (JAX) convolution of over-sampled inputs (a 1D image or a 2D mapping
        matrix): the body mirrors the s=1 FFT paths on the fine-mask state, with
        the scatter indices permuted from sub-block order and a mean bin-down
        appended. ``fft_kernel`` and ``dtype`` carry the caller's mixed-precision
        semantics (the image path casts the kernel, the mapping path keeps it
        complex128 — see ``convolved_mapping_matrix_from``).
        """
        import jax
        import jax.numpy as jnp

        perm = state.sub_slim_to_fine_slim
        self._check_over_sampled_length(n=values.shape[0], perm=perm)

        trailing = values.shape[1:]
        rows, cols = state.mask.slim_to_native_tuple

        native = xp.zeros(state.fft_shape + trailing, dtype=dtype)
        native = native.at[rows[perm], cols[perm]].set(jnp.asarray(values, dtype=dtype))

        if blurring_values is not None:
            bperm = state.blurring_sub_slim_to_fine_slim
            self._check_over_sampled_length(n=blurring_values.shape[0], perm=bperm)
            brows, bcols = state.blurring_mask.slim_to_native_tuple
            native = native.at[brows[bperm], bcols[bperm]].set(
                jnp.asarray(blurring_values, dtype=dtype)
            )

        fft_native = xp.fft.rfft2(native, s=state.fft_shape, axes=(0, 1))

        blurred_full = xp.fft.irfft2(
            fft_kernel * fft_native, s=state.fft_shape, axes=(0, 1)
        )

        ky, kx = self.kernel.shape_native
        off_y = (ky - 1) // 2
        off_x = (kx - 1) // 2

        blurred_full = xp.roll(blurred_full, shift=(-off_y, -off_x), axis=(0, 1))

        blurred_native = jax.lax.dynamic_slice(
            blurred_full,
            (off_y, off_x) + (0,) * len(trailing),
            state.fft_shape + trailing,
        )

        fine_slim = blurred_native[state.mask.slim_to_native_tuple]
        return self._over_sampled_binned_from(fine_slim, perm, trailing)

    @staticmethod
    def _warn_no_blurring_image():
        warnings.warn(
            "No blurring_image provided. Only the direct image will be convolved. "
            "This may change the correctness of the PSF convolution."
        )

    def _convolved_image_over_sampled_np_from(self, image, blurring_image, mask=None):
        state = self._over_sampled_state_from(mask=mask)

        if blurring_image is None:
            self._warn_no_blurring_image()

        binned = self._convolved_over_sampled_np_from(
            values=np.asarray(self._values_from(image)),
            blurring_values=(
                np.asarray(self._values_from(blurring_image))
                if blurring_image is not None
                else None
            ),
            state=state,
        )

        return Array2D(values=binned, mask=state.image_mask)

    def _convolved_mapping_matrix_over_sampled_np_from(
        self, mapping_matrix, mask, blurring_mapping_matrix=None
    ):
        state = self._over_sampled_state_from(mask=mask)

        return self._convolved_over_sampled_np_from(
            values=mapping_matrix,
            blurring_values=blurring_mapping_matrix,
            state=state,
        )

    def _convolved_image_over_sampled_jax_from(
        self, image, blurring_image, mask=None, use_mixed_precision: bool = False, xp=np
    ):
        import jax.numpy as jnp

        state = self._over_sampled_state_from(mask=mask)

        if blurring_image is None:
            self._warn_no_blurring_image()

        binned = self._convolved_over_sampled_jax_from(
            values=self._values_from(image),
            blurring_values=(
                self._values_from(blurring_image)
                if blurring_image is not None
                else None
            ),
            state=state,
            fft_kernel=(
                state.fft_kernel_c64 if use_mixed_precision else state.fft_kernel
            ),
            dtype=jnp.float32 if use_mixed_precision else jnp.float64,
            xp=xp,
        )

        return Array2D(values=binned, mask=state.image_mask)

    def _convolved_mapping_matrix_over_sampled_jax_from(
        self,
        mapping_matrix,
        mask,
        blurring_mapping_matrix=None,
        use_mixed_precision: bool = False,
        xp=np,
    ):
        import jax.numpy as jnp

        state = self._over_sampled_state_from(mask=mask)

        return self._convolved_over_sampled_jax_from(
            values=mapping_matrix,
            blurring_values=blurring_mapping_matrix,
            state=state,
            fft_kernel=state.fft_kernel_mapping,
            dtype=jnp.float32 if use_mixed_precision else jnp.float64,
            xp=xp,
        )

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
        return Convolver(
            kernel=kernel_copy,
            state=None,
            normalize=True,
            convolve_over_sample_size=self.convolve_over_sample_size,
        )

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
        convolve_over_sample_size: int = 1,
    ) -> "Convolver":
        """
        Setup the Convolver as a 2D symmetric elliptical Gaussian profile, according to the equation:

        (1.0 / (sigma * sqrt(2.0*pi))) * exp(-0.5 * (r/sigma)**2)


        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with. The kernel is always built at this
            size, including when ``PYAUTO_SMALL_DATASETS=1`` is set — a kernel's shape is intrinsic
            to the convolution operator, not a dataset size, so the fast-mode dataset cap does not
            apply to it.
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
        convolve_over_sample_size
            The over sample size of the PSF (see ``Convolver.__init__``). When above 1 the
            ``pixel_scales`` input should be the fine resolution (image pixel scale divided by this size).
        """

        grid = Grid2D.uniform(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            # The kernel is wrapped in an `Array2D` at `shape_native` below, so letting the
            # `PYAUTO_SMALL_DATASETS` cap shrink this grid would leave the two inconsistent.
            respect_small_datasets=False,
        )
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
            convolve_over_sample_size=convolve_over_sample_size,
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
        mask: Optional["Mask2D"] = None,
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
        mask
            The image-resolution mask, required when ``convolve_over_sample_size > 1``
            and no precomputed state exists, because over-sampled inputs do not carry
            the mask. Ignored otherwise.

        Returns
        -------
        Array2D
            The convolved image in slim (1D masked) format.

        Notes
        -----
        When ``convolve_over_sample_size > 1`` the ``image`` and ``blurring_image``
        must be the over-sampled (sub-gridded, per-pixel sub-block ordered) values —
        evaluate on the over-sampled grid without binning. The returned image is at
        image resolution. The oversampled JAX path always uses the FFT formalism.
        """
        if self.convolve_over_sample_size > 1:
            if xp is np:
                return self._convolved_image_over_sampled_np_from(
                    image=image, blurring_image=blurring_image, mask=mask
                )
            return self._convolved_image_over_sampled_jax_from(
                image=image,
                blurring_image=blurring_image,
                mask=mask,
                use_mixed_precision=use_mixed_precision,
                xp=xp,
            )

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

        # When use_mixed_precision is on, the FFT runs in complex64 end-to-end:
        # the input cube is allocated as float32, rfft2 emits complex64, the
        # precomputed (complex128) kernel is cast on the fly, and irfft2
        # returns float32 natively. No trailing astype is needed.
        real_dtype = jnp.float32 if use_mixed_precision else jnp.float64

        # Build combined native image in the FFT dtype
        image_both_native = xp.zeros(state.fft_shape, dtype=real_dtype)

        image_both_native = image_both_native.at[state.mask.slim_to_native_tuple].set(
            jnp.asarray(image.array, dtype=real_dtype)
        )

        if blurring_image is not None:
            image_both_native = image_both_native.at[
                state.blurring_mask.slim_to_native_tuple
            ].set(jnp.asarray(blurring_image.array, dtype=real_dtype))
        else:
            warnings.warn(
                "No blurring_image provided. Only the direct image will be convolved. "
                "This may change the correctness of the PSF convolution."
            )

        # FFT the combined image
        fft_image_native = xp.fft.rfft2(
            image_both_native, s=state.fft_shape, axes=(0, 1)
        )

        # Pick the precomputed kernel matching the FFT dtype. ConvolverState
        # caches both complex128 (default) and complex64 (mixed precision) at
        # init time, so this is a constant lookup rather than a per-call cast.
        fft_kernel = state.fft_kernel_c64 if use_mixed_precision else state.fft_kernel

        # Multiply by PSF in Fourier space and invert
        blurred_image_full = xp.fft.irfft2(
            fft_kernel * fft_image_native, s=state.fft_shape, axes=(0, 1)
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

        # Return slim form; dtype already matches use_mixed_precision via the
        # FFT path, so no explicit downcast.
        blurred_slim = blurred_image_native[state.mask.slim_to_native_tuple]

        return Array2D(values=blurred_slim, mask=image.mask)

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
        if self.convolve_over_sample_size > 1:
            if xp is np:
                return self._convolved_mapping_matrix_over_sampled_np_from(
                    mapping_matrix=mapping_matrix,
                    mask=mask,
                    blurring_mapping_matrix=blurring_mapping_matrix,
                )
            return self._convolved_mapping_matrix_over_sampled_jax_from(
                mapping_matrix=mapping_matrix,
                mask=mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
                use_mixed_precision=use_mixed_precision,
                xp=xp,
            )

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
        # mapping_matrix_native_from honors use_mixed_precision and produces a
        # fp32 native cube. rfft2 of that cube emits complex64. We deliberately
        # multiply by the complex128 precomputed kernel below, which upcasts
        # the product back to complex128 so the irfft2 returns float64. This
        # asymmetry is intentional: pixelization meshes with K >> 40 source
        # pixels accumulate enough fp32 round-off through the NNLS active-set
        # / log-determinant that the figure_of_merit drifts by O(1) units
        # (verified on the delaunay_mge regression). The fp32 input cube and
        # complex64 forward FFT still buy us a faster scatter and slightly
        # cheaper rfft2; keeping the kernel multiply in complex128 preserves
        # the precision the downstream linear algebra needs.
        # convolved_image_from (used by light profiles) takes the full fp32
        # path because its 40-column linear systems are well-conditioned.

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
        if self.convolve_over_sample_size > 1:
            if xp is np:
                return self._convolved_image_over_sampled_np_from(
                    image=image, blurring_image=blurring_image
                )
            return self._convolved_image_over_sampled_jax_from(
                image=image, blurring_image=blurring_image, xp=xp
            )

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
        if self.convolve_over_sample_size > 1:
            if xp is np:
                return self._convolved_mapping_matrix_over_sampled_np_from(
                    mapping_matrix=mapping_matrix,
                    mask=mask,
                    blurring_mapping_matrix=blurring_mapping_matrix,
                )
            return self._convolved_mapping_matrix_over_sampled_jax_from(
                mapping_matrix=mapping_matrix,
                mask=mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
                xp=xp,
            )

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
        self,
        image: np.ndarray,
        blurring_image: Optional[np.ndarray] = None,
        mask: Optional["Mask2D"] = None,
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
        if self.convolve_over_sample_size > 1:
            return self._convolved_image_over_sampled_np_from(
                image=image, blurring_image=blurring_image, mask=mask
            )

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
        if self.convolve_over_sample_size > 1:
            return self._convolved_mapping_matrix_over_sampled_np_from(
                mapping_matrix=mapping_matrix,
                mask=mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
            )

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
