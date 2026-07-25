import copy
import numpy as np
import warnings
from typing import Optional, Tuple


class NUFFTPlaceholder:
    pass


try:
    from pynufft.linalg.nufft_cpu import NUFFT_cpu
except ModuleNotFoundError:
    NUFFT_cpu = NUFFTPlaceholder


from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.visibilities import Visibilities

from autoarray.structures.arrays import array_2d_util
from autoarray.operators import transformer_util


try:
    import nufftax as _nufftax
except ModuleNotFoundError:
    _nufftax = None


def pynufft_exception():
    raise ModuleNotFoundError(
        "\n--------------------\n"
        "You are attempting to perform interferometer analysis with the legacy "
        "pynufft-backed `TransformerNUFFTPyNUFFT`.\n\n"
        "However, the optional library PyNUFFT (https://github.com/jyhmiinlin/pynufft) is not installed.\n\n"
        "Install it via the command `pip install pynufft==2022.2.2`.\n\n"
        "----------------------"
    )


def nufftax_exception():
    raise ModuleNotFoundError(
        "\n--------------------\n"
        "You are attempting to perform interferometer analysis with the default "
        "JAX-native `TransformerNUFFT`.\n\n"
        "However, the optional library nufftax (https://github.com/GragasLab/nufftax) is not installed.\n\n"
        "Install it via the command `pip install nufftax`.\n\n"
        "If you want to use the legacy pynufft backend instead, pass "
        "`transformer_class=TransformerNUFFTPyNUFFT` and install pynufft.\n\n"
        "----------------------"
    )


class TransformerDFT:
    def __init__(
        self,
        uv_wavelengths: np.ndarray,
        real_space_mask: Mask2D,
    ):
        """
        A direct Fourier transform (DFT) operator for radio interferometric imaging.

        This class performs the forward and inverse mapping between real-space images and
        complex visibilities measured by an interferometer. It uses a direct implementation
        of the Fourier transform (not FFT-based), making it suitable for irregular uv-coverage.

        Optionally, it precomputes and stores the sine and cosine terms used in the transform,
        which can significantly improve performance for repeated operations but at the cost of memory.

        Parameters
        ----------
        uv_wavelengths
            The (u, v) coordinates in wavelengths of the measured visibilities.
        real_space_mask
            The real-space mask that defines the image grid and which pixels are valid.

        Attributes
        ----------
        grid : ndarray
            The unmasked real-space grid in radians.
        total_visibilities : int
            The number of measured visibilities.
        total_image_pixels : int
            The number of unmasked pixels in the real-space image grid.
        preload_real_transforms : ndarray, optional
            The precomputed cosine terms used in the real part of the DFT.
        preload_imag_transforms : ndarray, optional
            The precomputed sine terms used in the imaginary part of the DFT.
        real_space_pixels : int
            Alias for `total_image_pixels`.
        adjoint_scaling : float
            Scaling factor applied to the adjoint operator to normalize the inverse transform.
        """
        super().__init__()

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.real_space_mask = real_space_mask
        self.grid = self.real_space_mask.derive_grid.unmasked.in_radians

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = self.real_space_mask.pixels_in_mask

        # NOTE: This is the scaling factor that needs to be applied to the adjoint operator
        self.adjoint_scaling = (2.0 * self.grid.shape_native[0]) * (
            2.0 * self.grid.shape_native[1]
        )

    def visibilities_from(self, image: Array2D, xp=np) -> Visibilities:
        """
        Computes the visibilities from a real-space image using the direct Fourier transform (DFT).

        This method transforms the input image into the uv-plane (Fourier space), simulating the
        measurements made by an interferometer at specified uv-wavelengths.

        Parameters
        ----------
        image
            The real-space image to be transformed to the uv-plane. Must be defined on the
            same grid and mask as this transformer's `real_space_mask`.

        Returns
        -------
        The complex visibilities resulting from the Fourier transform of the input image.
        """

        visibilities = transformer_util.visibilities_from(
            image_1d=image.slim.array,
            grid_radians=self.grid.array,
            uv_wavelengths=self.uv_wavelengths,
            xp=xp,
        )

        return Visibilities(visibilities=visibilities)

    def image_from(
        self, visibilities: Visibilities, use_adjoint_scaling: bool = False, xp=np
    ) -> Array2D:
        """
        Computes the real-space image from a set of visibilities using the adjoint of the DFT.

        This is not a true inverse Fourier transform, but rather the adjoint operation, which maps
        complex visibilities back into image space. This is typically used as the first step
        in inverse imaging algorithms like CLEAN or regularized reconstruction.

        Parameters
        ----------
        visibilities
            The complex visibilities to be transformed into a real-space image.
        use_adjoint_scaling
            If True, the result is scaled by a normalization factor. Currently unused.

        Returns
        -------
        The real-space image resulting from the adjoint DFT operation, defined on the same
        mask as this transformer's `real_space_mask`.
        """
        image_slim = transformer_util.image_direct_from(
            visibilities=visibilities.in_array,
            grid_radians=self.grid.array,
            uv_wavelengths=self.uv_wavelengths,
        )

        image_native = array_2d_util.array_2d_native_from(
            array_2d_slim=image_slim, mask_2d=self.real_space_mask, xp=xp
        )

        return Array2D(values=image_native, mask=self.real_space_mask)

    def transform_mapping_matrix(self, mapping_matrix: np.ndarray, xp=np) -> np.ndarray:
        """
        Applies the DFT to a mapping matrix that maps source pixels to image pixels.

        This is used in linear inversion frameworks, where the transform of each source basis function
        (represented by a column of the mapping matrix) is computed individually. The result is a matrix
        mapping source pixels directly to visibilities.

        Parameters
        ----------
        mapping_matrix
            A 2D array of shape (n_image_pixels, n_source_pixels) that maps source pixels to image-plane pixels.

        Returns
        -------
        A 2D complex-valued array of shape (n_visibilities, n_source_pixels) that maps source-plane basis
        functions directly to the visibilities.
        """

        return transformer_util.transformed_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            grid_radians=self.grid.array,
            uv_wavelengths=self.uv_wavelengths,
            xp=xp,
        )


class TransformerNUFFTPyNUFFT(NUFFT_cpu):
    def __init__(
        self, uv_wavelengths: np.ndarray, real_space_mask: Mask2D, xp=np, **kwargs
    ):
        """
        Performs the Non-Uniform Fast Fourier Transform (NUFFT) for interferometric image reconstruction.

        Legacy pynufft-backed transformer. The default `TransformerNUFFT` is now backed by `nufftax`
        (JAX-native, differentiable, ~zero gridding error) — this class is retained so users who depend
        on pynufft's specific gridding behaviour can opt in by passing
        `transformer_class=TransformerNUFFTPyNUFFT`.

        This transformer uses the PyNUFFT library to efficiently compute the Fourier transform
        of an image defined on a regular real-space grid to a set of non-uniform uv-plane (Fourier space)
        coordinates, as is typical in radio interferometry.

        It is initialized with the interferometer uv-wavelengths and a real-space mask, which defines
        the pixelized image domain.

        Parameters
        ----------
        uv_wavelengths
            The uv-coordinates (Fourier-space sampling points) corresponding to the measured visibilities.
            Should be an array of shape (n_vis, 2), where the two columns represent u and v coordinates in wavelengths.

        real_space_mask
            The 2D mask defining the real-space pixel grid on which the image is defined. Used to create the
            unmasked grid required for NUFFT planning.

        Notes
        -----
        - The `initialize_plan()` method builds the internal NUFFT plan based on the input grid and uv sampling.
        - A complex exponential `shift` factor is applied to align the center of the Fourier transform correctly,
          accounting for the pixel-center offset in the real-space grid.
        - The adjoint operation (used in inverse imaging) must be scaled by `adjoint_scaling` to normalize its output.
        - This transformer inherits directly from PyNUFFT's `NUFFT_cpu` base class.
        - If `NUFFTPlaceholder` is detected (indicating PyNUFFT is not available), an exception is raised.

        Attributes
        ----------
        grid : Grid2D
            The real-space pixel grid derived from the mask, in radians.
        native_index_for_slim_index : np.ndarray
            Index map converting from slim (1D) grid to native (2D) indexing, for image reshaping.
        shift : np.ndarray
            Complex exponential phase shift applied to account for real-space pixel centering.
        total_visibilities : int
            Total number of visibilities across all uv-wavelength components.
        adjoint_scaling : float
            Scaling factor for adjoint operations to normalize reconstructed images.
        """
        from astropy import units

        if isinstance(self, NUFFTPlaceholder):
            pynufft_exception()

        super(TransformerNUFFTPyNUFFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths
        self.real_space_mask = real_space_mask
        #        self.grid = self.real_space_mask.unmasked_grid.in_radians
        self.grid = Grid2D.from_mask(mask=self.real_space_mask).in_radians
        self.native_index_for_slim_index = copy.copy(
            real_space_mask.derive_indexes.native_for_slim.astype("int")
        )

        # NOTE: The plan need only be initialized once
        self.initialize_plan()

        # ...
        self.shift = np.exp(
            -2.0
            * np.pi
            * 1j
            * (
                self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * self.uv_wavelengths[:, 1]
                + self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * self.uv_wavelengths[:, 0]
            )
        )

        # NOTE: If reshaped the shape of the operator is (2 x Nvis, Np) else it is (Nvis, Np)
        self.total_visibilities = int(uv_wavelengths.shape[0] * uv_wavelengths.shape[1])

        # NOTE: This is the scaling factor that needs to be applied to the adjoint operator
        self.adjoint_scaling = (2.0 * self.grid.shape_native[0]) * (
            2.0 * self.grid.shape_native[1]
        )

    def initialize_plan(self, ratio: int = 2, interp_kernel: Tuple[int, int] = (6, 6)):
        """
        Initializes the PyNUFFT plan for performing the NUFFT operation.

        This method precomputes the interpolation structure and gridding
        needed by the NUFFT algorithm to map between the regular real-space
        image grid and the non-uniform uv-plane sampling defined by the
        interferometric visibilities.

        Parameters
        ----------
        ratio
            The oversampling ratio used to pad the Fourier grid before interpolation.
            A higher value improves accuracy at the cost of increased memory and computation.
            Default is 2 (i.e., the Fourier grid is twice the size of the image grid).

        interp_kernel
            The interpolation kernel size along each axis, given as (Jy, Jx).
            This determines how many neighboring Fourier grid points are used
            to interpolate each uv-point.
            Default is (6, 6), a good trade-off between accuracy and performance.

        Notes
        -----
        - The uv-coordinates are normalized and rescaled into the range expected by PyNUFFT
          using the real-space grid’s pixel scale and the Nyquist frequency limit.
        - The plan must be initialized before performing any NUFFT operations (e.g., forward or adjoint).
        - This method modifies the internal state of the NUFFT object by calling `self.plan(...)`.
        """
        from astropy import units

        if not isinstance(ratio, int):
            ratio = int(ratio)

        # ... NOTE : The u,v coordinated should be given in the order ...
        visibilities_normalized = np.array(
            [
                self.uv_wavelengths[:, 1]
                / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad)))
                * np.pi,
                self.uv_wavelengths[:, 0]
                / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad)))
                * np.pi,
            ]
        ).T

        # NOTE:
        self.plan(
            om=visibilities_normalized,
            Nd=self.grid.shape_native,
            Kd=(ratio * self.grid.shape_native[0], ratio * self.grid.shape_native[1]),
            Jd=interp_kernel,
        )

    def _pynufft_forward_numpy(self, image_np: np.ndarray) -> np.ndarray:
        """
        NumPy-only forward NUFFT. Runs on host.
        """
        warnings.filterwarnings("ignore")

        # Flip vertically (PyNUFFT internal convention)
        image_np = image_np[::-1, :]

        # PyNUFFT forward
        vis = self.forward(image_np)

        return vis

    def visibilities_from_jax(self, image: np.ndarray) -> np.ndarray:
        """
        JAX-compatible wrapper around PyNUFFT forward.
        Can be used inside jax.jit.
        """

        import jax
        import jax.numpy as jnp
        from jax import ShapeDtypeStruct

        # You MUST tell JAX the output shape & dtype

        out_shape = (self.total_visibilities // 2,)  # example
        out_dtype = jnp.complex128

        result_shape = ShapeDtypeStruct(
            shape=out_shape,
            dtype=out_dtype,
        )

        return jax.pure_callback(
            lambda img: self._pynufft_forward_numpy(img),
            result_shape,
            image,
            vmap_method="sequential",
        )

    def visibilities_from(self, image, xp=np):

        # start with native image padded with zeros
        image_native = xp.zeros(image.mask.shape, dtype=image.dtype)

        if xp.__name__.startswith("jax"):

            image_native = image_native.at[image.mask.slim_to_native_tuple].set(
                image.array
            )

        else:

            image_native = image.native.array

        if xp is np:
            warnings.filterwarnings("ignore")
            return Visibilities(visibilities=self.forward(image_native[::-1, :]))

        else:

            vis = self.visibilities_from_jax(image_native)

            return Visibilities(visibilities=vis)

    def image_from(
        self, visibilities: Visibilities, use_adjoint_scaling: bool = False, xp=np
    ) -> Array2D:
        """
        Reconstructs a real-space image from visibilities using the NUFFT adjoint transform.

        Parameters
        ----------
        visibilities
            The complex visibilities in the uv-plane to be inverted.
        use_adjoint_scaling
            If True, apply a scaling factor to the adjoint result to improve accuracy.
            Default is False.

        Returns
        -------
        The reconstructed real-space image after applying the NUFFT adjoint transform.

        Notes
        -----
        - The output image is flipped vertically to align with the input image orientation.
        - Warnings during the adjoint operation are suppressed.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = np.real(self.adjoint(visibilities.array))[::-1, :]

        if use_adjoint_scaling:
            image *= self.adjoint_scaling

        return Array2D(values=image, mask=self.real_space_mask)

    def transform_mapping_matrix(self, mapping_matrix: np.ndarray, xp=np) -> np.ndarray:
        """
        Applies the NUFFT forward transform to each column of a mapping matrix, producing transformed visibilities.

        Parameters
        ----------
        mapping_matrix
            A 2D array where each column corresponds to a source-plane pixel intensity distribution flattened into image space.

        Returns
        -------
        A complex-valued 2D array where each column contains the visibilities corresponding to the respective column
        in the input mapping matrix.

        Notes
        -----
        - Each column of the input mapping matrix is reshaped into the native 2D image grid before transformation.
        - This method repeatedly calls `visibilities_from` for each column, which may be computationally intensive.
        """
        transformed_mapping_matrix = 0 + 0j * xp.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )

        for source_pixel_1d_index in range(mapping_matrix.shape[1]):

            image_2d = xp.zeros(self.grid.shape_native, dtype=mapping_matrix.dtype)

            if xp.__name__.startswith("jax"):

                image_2d = image_2d.at[self.grid.mask.slim_to_native_tuple].set(
                    mapping_matrix[:, source_pixel_1d_index]
                )

            else:

                image_2d[self.grid.mask.slim_to_native_tuple] = mapping_matrix[
                    :, source_pixel_1d_index
                ]

            image = Array2D(values=image_2d, mask=self.grid.mask)

            visibilities = self.visibilities_from(image=image, xp=xp)

            if xp.__name__.startswith("jax"):
                transformed_mapping_matrix = transformed_mapping_matrix.at[
                    :, source_pixel_1d_index
                ].set(visibilities.array)
            else:
                transformed_mapping_matrix[:, source_pixel_1d_index] = (
                    visibilities.array
                )

        return transformed_mapping_matrix


class TransformerNUFFT:
    def __init__(
        self,
        uv_wavelengths: np.ndarray,
        real_space_mask: Mask2D,
        eps: float = 1e-12,
        chunk_size: Optional[int] = None,
        xp=np,
        **kwargs,
    ):
        """
        JAX-native Non-Uniform FFT for image -> visibilities, backed by `nufftax`.

        This is the default `TransformerNUFFT` in PyAutoArray. It uses the
        `nufftax` library (https://github.com/GragasLab/nufftax), a pure-JAX
        NUFFT implementation that supports `jax.jit`, `jax.grad`, and
        `jax.vmap`. It replaces the legacy `TransformerNUFFTPyNUFFT` (which
        wraps the non-differentiable `pynufft` library) as the default backend.

        Convention recipe (matches `TransformerDFT` to ~1e-13 relative across
        odd/even/non-square image sizes):

            image_flipped = image[::-1, :]
            x = 2 * pi * u_lambda * pixel_scale_rad
            y = 2 * pi * v_lambda * pixel_scale_rad
            offset_x = 0.5 if N_x is even else 0.0
            offset_y = 0.5 if N_y is even else 0.0
            shift = exp(-i * (offset_x * x + offset_y * y))
            visibilities = nufftax.nufft2d2(x, y, image_flipped, eps, -1) * shift

        The `shift` factor is the half-pixel correction between autoarray's
        grid centre at index `(N - 1) / 2` and nufftax's mode-0 at index
        `N // 2`; pynufft applies this internally, nufftax does not.

        Parameters
        ----------
        uv_wavelengths
            The (u, v) coordinates of the measured visibilities in wavelengths,
            shape `(n_vis, 2)`.
        real_space_mask
            The 2D mask defining the real-space image grid.
        eps
            Requested NUFFT precision passed to nufftax. Defaults to `1e-12`
            (effectively machine precision); relax to `1e-9` or `1e-6` for
            faster execution if marginal accuracy is acceptable.
        chunk_size
            If set to a positive integer, the forward and adjoint NUFFT
            calls split the visibility axis into chunks of this size and
            iterate (via ``jax.lax.scan`` on the JAX path, a Python loop
            on the numpy path). This caps the nufftax gather-buffer
            allocation (~``2 * chunk_size * nspread^2 * dtype_size``) at
            the cost of per-chunk overhead. Required for visibility counts
            above ~5M on a 40-80 GB GPU. If ``None`` (default), a single
            one-shot call is used — preserves existing behaviour for
            small-N callers (sma-class datasets).
        xp
            Accepted for signature compatibility with the legacy class; not
            stored. The active backend is selected per-call via the `xp`
            argument to `visibilities_from` / `image_from`.

        Attributes
        ----------
        grid
            The real-space pixel grid in radians (computed from the mask).
        total_visibilities
            Number of measured visibilities.
        total_image_pixels
            Number of unmasked pixels in the image grid.
        adjoint_scaling
            Scaling factor available for callers who want to apply an
            optional normalisation to the adjoint output. Provided for
            parity with the legacy class.
        """
        from astropy import units

        if _nufftax is None:
            nufftax_exception()

        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer or None, got {chunk_size}"
            )

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.real_space_mask = real_space_mask
        self.grid = Grid2D.from_mask(mask=self.real_space_mask).in_radians
        self.eps = eps
        self.chunk_size = chunk_size
        self.native_index_for_slim_index = copy.copy(
            real_space_mask.derive_indexes.native_for_slim.astype("int")
        )

        pixel_scale_rad = self.grid.pixel_scales[0] * units.arcsec.to(units.rad)
        # nufft2d2 frequency arguments:
        #   x is paired with the column-axis mode (image x)
        #   y is paired with the row-axis    mode (image y)
        # Both must lie in [-pi, pi); the 2*pi*Δ_rad scaling makes uv_lambda
        # land in that range for any sane uv-coverage.
        self._x = 2.0 * np.pi * self.uv_wavelengths[:, 0] * pixel_scale_rad
        self._y = 2.0 * np.pi * self.uv_wavelengths[:, 1] * pixel_scale_rad

        n_y, n_x = self.real_space_mask.shape_native
        offset_x = 0.5 if n_x % 2 == 0 else 0.0
        offset_y = 0.5 if n_y % 2 == 0 else 0.0
        self._shift = np.exp(-1j * (offset_x * self._x + offset_y * self._y))

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = real_space_mask.pixels_in_mask
        self.adjoint_scaling = (2.0 * n_y) * (2.0 * n_x)

    def _forward_native(self, image_native_2d, xp=np):
        """Run nufft2d2 on a 2D native-shape image array, returning visibilities.

        When ``self.chunk_size`` is set, the visibility axis is processed in
        fixed-size chunks via ``jax.lax.scan`` (JAX path) or a Python loop
        (numpy path) — caps the nufftax gather-buffer allocation per call.
        """
        K = int(self._x.shape[0])

        if xp.__name__.startswith("jax"):
            import jax
            import jax.numpy as jnp

            img = jnp.asarray(image_native_2d)[::-1, :].astype(jnp.complex128)
            x_all = jnp.asarray(self._x)
            y_all = jnp.asarray(self._y)
            shift_all = jnp.asarray(self._shift)

            if self.chunk_size is None or self.chunk_size >= K:
                return _nufftax.nufft2d2(x_all, y_all, img, self.eps, -1) * shift_all

            cs = int(self.chunk_size)
            n_chunks = (K + cs - 1) // cs
            K_pad = n_chunks * cs
            x_pad = jnp.pad(x_all, (0, K_pad - K))
            y_pad = jnp.pad(y_all, (0, K_pad - K))
            shift_pad = jnp.pad(shift_all, (0, K_pad - K))
            eps = self.eps

            def body(carry, i):
                k0 = i * cs
                x_s = jax.lax.dynamic_slice(x_pad, (k0,), (cs,))
                y_s = jax.lax.dynamic_slice(y_pad, (k0,), (cs,))
                shift_s = jax.lax.dynamic_slice(shift_pad, (k0,), (cs,))
                vis = _nufftax.nufft2d2(x_s, y_s, img, eps, -1) * shift_s
                return carry, vis

            _, vis_chunks = jax.lax.scan(body, None, jnp.arange(n_chunks))
            return vis_chunks.reshape(K_pad)[:K]

        img = image_native_2d[::-1, :].astype(np.complex128)

        if self.chunk_size is None or self.chunk_size >= K:
            out = _nufftax.nufft2d2(self._x, self._y, img, self.eps, -1) * self._shift
            return np.array(np.asarray(out))

        cs = int(self.chunk_size)
        parts = []
        for k0 in range(0, K, cs):
            k1 = min(k0 + cs, K)
            vis = (
                _nufftax.nufft2d2(
                    self._x[k0:k1], self._y[k0:k1], img, self.eps, -1
                )
                * self._shift[k0:k1]
            )
            parts.append(np.asarray(vis))
        return np.concatenate(parts, axis=0)

    def visibilities_from(self, image, xp=np) -> Visibilities:
        """
        Forward NUFFT: real-space image -> visibilities at the configured uv points.

        For numpy callers (`xp=np`) the result is materialised back to numpy
        before being wrapped in `Visibilities`. For JAX callers (`xp=jnp`)
        the result stays as a `jax.Array` so it can flow through `jax.jit`
        / `jax.grad` / `jax.vmap` without device round-trips.
        """
        if xp.__name__.startswith("jax"):
            import jax.numpy as jnp

            image_native = jnp.zeros(image.mask.shape, dtype=image.dtype)
            image_native = image_native.at[image.mask.slim_to_native_tuple].set(
                image.array
            )
        else:
            image_native = image.native.array

        return Visibilities(visibilities=self._forward_native(image_native, xp=xp))

    def image_from(
        self,
        visibilities: Visibilities,
        use_adjoint_scaling: bool = False,
        xp=np,
    ) -> Array2D:
        """
        Adjoint NUFFT: visibilities -> real-space (dirty) image.

        Implemented as `nufftax.nufft2d1` with `conj(shift)` applied to the
        visibilities and a final row-flip to return to autoarray's native
        orientation. The real part is taken to discard imaginary residue,
        matching the legacy class' behaviour.

        Note that this is the **mathematical adjoint** of `visibilities_from`,
        with no kernel deconvolution applied. The dirty image therefore
        differs in absolute scale from the legacy `TransformerNUFFTPyNUFFT`
        adjoint (which applies pynufft's internal IFFT and kernel
        deconvolution). The structure of the dirty image is the same, and
        the values match `TransformerDFT.image_from` exactly.

        `use_adjoint_scaling` is accepted for API compatibility with the
        legacy class and is otherwise unused (the nufftax adjoint is already
        the mathematical adjoint; no extra normalisation is needed). This
        matches `TransformerDFT.image_from` semantics so the sparse-operator
        path is scale-consistent across both transformers.
        """
        n_y, n_x = self.real_space_mask.shape_native
        n_modes = (n_x, n_y)  # nufftax wants (n1, n2) = (N_x, N_y)
        K = int(self._x.shape[0])

        if xp.__name__.startswith("jax"):
            import jax
            import jax.numpy as jnp

            x_all = jnp.asarray(self._x)
            y_all = jnp.asarray(self._y)
            shift_conj_all = jnp.asarray(np.conj(self._shift))
            c_all = jnp.asarray(visibilities.array) * shift_conj_all

            if self.chunk_size is None or self.chunk_size >= K:
                f = _nufftax.nufft2d1(x_all, y_all, c_all, n_modes, self.eps, +1)
                image = jnp.real(f)[::-1, :]
                return Array2D(values=image, mask=self.real_space_mask)

            cs = int(self.chunk_size)
            n_chunks = (K + cs - 1) // cs
            K_pad = n_chunks * cs
            x_pad = jnp.pad(x_all, (0, K_pad - K))
            y_pad = jnp.pad(y_all, (0, K_pad - K))
            c_pad = jnp.pad(c_all, (0, K_pad - K))
            eps = self.eps
            idx = jnp.arange(cs)

            def body(f_accum, i):
                k0 = i * cs
                x_s = jax.lax.dynamic_slice(x_pad, (k0,), (cs,))
                y_s = jax.lax.dynamic_slice(y_pad, (k0,), (cs,))
                c_s = jax.lax.dynamic_slice(c_pad, (k0,), (cs,))
                valid = (idx + k0) < K
                c_s = jnp.where(valid, c_s, jnp.complex128(0))
                f_chunk = _nufftax.nufft2d1(x_s, y_s, c_s, n_modes, eps, +1)
                return f_accum + f_chunk, None

            f_init = jnp.zeros((n_y, n_x), dtype=jnp.complex128)
            f_total, _ = jax.lax.scan(body, f_init, jnp.arange(n_chunks))
            image = jnp.real(f_total)[::-1, :]
            return Array2D(values=image, mask=self.real_space_mask)

        c_all = visibilities.array * np.conj(self._shift)

        if self.chunk_size is None or self.chunk_size >= K:
            f = _nufftax.nufft2d1(self._x, self._y, c_all, n_modes, self.eps, +1)
            image = np.array(np.asarray(f)[::-1, :].real)
            return Array2D(values=image, mask=self.real_space_mask)

        cs = int(self.chunk_size)
        f_total = np.zeros((n_y, n_x), dtype=np.complex128)
        for k0 in range(0, K, cs):
            k1 = min(k0 + cs, K)
            f_chunk = _nufftax.nufft2d1(
                self._x[k0:k1], self._y[k0:k1], c_all[k0:k1], n_modes, self.eps, +1
            )
            f_total = f_total + np.asarray(f_chunk)
        image = np.array(f_total[::-1, :].real)
        return Array2D(values=image, mask=self.real_space_mask)

    def transform_mapping_matrix(self, mapping_matrix, xp=np):
        """
        Apply the forward NUFFT to each column of a mapping matrix.

        All columns are scattered into a single batched native-shape image
        of shape ``(n_src, N_y, N_x)`` and passed through nufft2d2 in one
        call (nufft2d2 supports batched ``f``). This avoids the
        per-column Python loop that, under ``jax.jit``, would unroll into
        ``n_src`` separate NUFFT invocations and blow up the JIT graph
        for pixelization-heavy fits (notably double-source-plane).
        """
        n_src = mapping_matrix.shape[1]
        rows, cols = self.real_space_mask.slim_to_native_tuple
        n_y, n_x = self.real_space_mask.shape_native

        if xp.__name__.startswith("jax"):
            import jax.numpy as jnp

            mm_T = jnp.asarray(mapping_matrix).T.astype(jnp.complex128)
            source_images = jnp.zeros((n_src, n_y, n_x), dtype=jnp.complex128)
            source_images = source_images.at[
                jnp.arange(n_src)[:, None],
                jnp.asarray(rows)[None, :],
                jnp.asarray(cols)[None, :],
            ].set(mm_T)
            flipped = source_images[:, ::-1, :]
            x = jnp.asarray(self._x)
            y = jnp.asarray(self._y)
            shift = jnp.asarray(self._shift)
            # nufft2d2 returns shape (n_trans, M); transpose to (M, n_src).
            vis_batched = (
                _nufftax.nufft2d2(x, y, flipped, self.eps, -1) * shift[None, :]
            )
            return vis_batched.T

        mm_T = np.asarray(mapping_matrix).T.astype(np.complex128)
        source_images = np.zeros((n_src, n_y, n_x), dtype=np.complex128)
        source_images[np.arange(n_src)[:, None], rows[None, :], cols[None, :]] = mm_T
        flipped = source_images[:, ::-1, :]
        vis_batched = (
            _nufftax.nufft2d2(self._x, self._y, flipped, self.eps, -1)
            * self._shift[None, :]
        )
        return np.array(np.asarray(vis_batched).T)
