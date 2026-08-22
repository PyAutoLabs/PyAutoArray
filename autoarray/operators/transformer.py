import copy
import numpy as np
import warnings
from typing import Optional, Tuple


from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.visibilities import Visibilities

from autoarray.structures.arrays import array_2d_util
from autoarray.operators import transformer_util

# nufftax pulls in jax at import (~0.7s), which sessions that never touch an
# interferometer transformer should not pay for — deferred to _load_nufftax(),
# called from TransformerNUFFT's entry points (not only __init__, because
# unpickled instances in multiprocessing workers never re-run __init__).
_nufftax = None
_nufftax_loaded = False


def _load_nufftax():
    global _nufftax, _nufftax_loaded
    if _nufftax_loaded:
        return _nufftax
    _nufftax_loaded = True
    try:
        import nufftax
    except ModuleNotFoundError:
        return None
    _nufftax = nufftax
    _version = tuple(int(v) for v in _nufftax.__version__.split(".")[:2])
    # Only the 0.6.x series both has the primitives module and needs the shim.
    if (0, 6) <= _version < (0, 7):
        _patch_nufftax_batchers()
    return _nufftax


def _patch_nufftax_batchers():
    """Rank-guard nufftax's vmap batching rules (shim for nufftax <0.7).

    nufftax's batching fast path re-binds a primitive unchanged whenever only
    the source argument is vmapped, assuming the impl's single native batch
    axis can absorb it. Under nested batching — e.g. `jax.vmap(jax.value_and_grad(...))`
    over a fit whose `transform_mapping_matrix` already passes a batched stack —
    the dims accumulate past what the impl accepts and tracing crashes
    ("too many values to unpack"). Until that is fixed upstream
    (github.com/GragasLab/nufftax), re-register every primitive's batcher with
    the missing guard: within native rank, bind as before; beyond it, collapse
    the stacked batch axes into the one native axis and unflatten the output.
    """
    import jax
    from jax.interpreters import batching

    P = _nufftax.transforms.primitives

    # (primitive, impl, index of the source argument, unbatched source rank)
    table = [
        (P.nufft1d1_p, P._impl_1d1, 1, 1),
        (P.nufft1d2_p, P._impl_1d2, 1, 1),
        (P.nufft2d1_p, P._impl_2d1, 2, 1),
        (P.nufft2d2_p, P._impl_2d2, 2, 2),
        (P.nufft3d1_p, P._impl_3d1, 3, 1),
        (P.nufft3d2_p, P._impl_3d2, 3, 3),
        (P.nufft1d3_p, P._impl_1d3, 1, 1),
        (P.nufft2d3_p, P._impl_2d3, 2, 1),
        (P.nufft3d3_p, P._impl_3d3, 3, 1),
    ]

    def guarded_batcher(prim, impl_fn, source_idx, base_rank):
        def batcher(args, dims, **kwargs):
            batched = [i for i, d in enumerate(dims) if d is not None]
            if batched == [source_idx] and dims[source_idx] == 0:
                src = args[source_idx]
                extra = src.ndim - (base_rank + 1)
                if extra <= 0:
                    return prim.bind(*args, **kwargs), 0
                lead = src.shape[: extra + 1]
                flat = src.reshape((-1,) + src.shape[extra + 1 :])
                new_args = list(args)
                new_args[source_idx] = flat
                out = prim.bind(*new_args, **kwargs)
                return out.reshape(lead + out.shape[1:]), 0
            out = jax.vmap(lambda *a: impl_fn(*a, **kwargs), in_axes=tuple(dims))(*args)
            return out, 0

        return batcher

    for prim, impl, source_idx, base_rank in table:
        batching.primitive_batchers[prim] = guarded_batcher(
            prim, impl, source_idx, base_rank
        )


def nufftax_exception():
    raise ModuleNotFoundError(
        "\n--------------------\n"
        "You are attempting to perform interferometer analysis with the default "
        "JAX-native `TransformerNUFFT`.\n\n"
        "However, the optional library nufftax (https://github.com/GragasLab/nufftax) is not installed.\n\n"
        "Install it via the command `pip install nufftax`.\n\n"
        "Note that nufftax requires JAX, which has no wheels for Intel macOS; on\n"
        "that platform use `transformer_class=TransformerDFT` instead.\n\n"
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
            If True, normalise the adjoint output onto the common scale shared by
            every transformer (that of the plain mathematical adjoint). Both
            remaining transformers already return the plain mathematical
            adjoint, so this is a no-op for each of them; it is retained as a
            stable part of the transformer interface. See `Interferometer.
            apply_sparse_operator`, which passes `True` so the sparse-operator
            dirty image is scale-consistent across both transformers.

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
        `jax.vmap`. Note that nufftax requires JAX; on platforms with no JAX
        wheels (notably Intel macOS) use `TransformerDFT` instead.

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
        `N // 2`; nufftax does not apply it internally.

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

        if _load_nufftax() is None:
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
        _load_nufftax()

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
                _nufftax.nufft2d2(self._x[k0:k1], self._y[k0:k1], img, self.eps, -1)
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
        orientation. The real part is taken to discard imaginary residue.

        Note that this is the **mathematical adjoint** of `visibilities_from`,
        with no kernel deconvolution applied. The values match
        `TransformerDFT.image_from` exactly.

        `use_adjoint_scaling` normalises the adjoint onto the common scale
        shared by every transformer. It is a no-op here (and for
        `TransformerDFT`) because both remaining adjoints are already the plain
        mathematical adjoint; it is retained as a stable part of the
        transformer interface. `Interferometer.apply_sparse_operator` passes
        `True` so the sparse-operator dirty image is scale-consistent across
        both transformers.
        """
        _load_nufftax()

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
        _load_nufftax()

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
