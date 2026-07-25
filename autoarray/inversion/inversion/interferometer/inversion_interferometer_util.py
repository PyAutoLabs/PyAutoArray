from dataclasses import dataclass
import logging
import numpy as np
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def data_vector_via_transformed_mapping_matrix_from(
    transformed_mapping_matrix: np.ndarray,
    visibilities: np.ndarray,
    noise_map: np.ndarray,
) -> np.ndarray:
    """
    Returns the data vector `D` from a transformed mapping matrix `f` and the 1D image `d` and 1D noise-map `sigma`
    (see Warren & Dye 2003).

    Parameters
    ----------
    transformed_mapping_matrix
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    visibilities
        The complex 1D array of observed visibilities the inversion is fitting, with real and imaginary components.
    noise_map
        The complex 1D array of the noise-map used by the inversion during the fit.
    """
    # Extract components
    vis_real = visibilities.real
    vis_imag = visibilities.imag
    f_real = transformed_mapping_matrix.real
    f_imag = transformed_mapping_matrix.imag
    noise_real = noise_map.real
    noise_imag = noise_map.imag

    # Square noise components
    inv_var_real = 1.0 / (noise_real**2)
    inv_var_imag = 1.0 / (noise_imag**2)

    # Real and imaginary contributions
    weighted_real = (vis_real * inv_var_real)[:, None] * f_real
    weighted_imag = (vis_imag * inv_var_imag)[:, None] * f_imag

    # Sum over visibilities
    return np.sum(weighted_real + weighted_imag, axis=0)


def mapped_reconstructed_visibilities_from(
    transformed_mapping_matrix: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurrred mapping matrix `f` and solution vector *S*.

    Parameters
    ----------
    transformed_mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    return transformed_mapping_matrix @ reconstruction


def _report_memory(arr):
    """
    Report array memory + process RSS (best-effort).
    Safe to call inside a tqdm loop.
    """
    try:
        import resource

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        arr_mb = arr.nbytes / 1024**2
        from tqdm import tqdm

        tqdm.write(f"    Memory: array={arr_mb:.1f} MB, RSS≈{rss_mb:.1f} MB")
    except Exception:
        pass


def nufft_precision_operator_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d,
    grid_radians_2d: np.ndarray,
    *,
    chunk_k: int = 2048,
    show_progress: bool = False,
    show_memory: bool = False,
    use_jax: bool = False,
) -> np.ndarray:
    """
     Compute the interferometer W-tilde curvature preload on a rectangular offset grid,
     exploiting translational symmetry of the NUFFT kernel.

     This function computes a compact 2D preload array that depends only on the relative
     (dy, dx) offsets between image pixels, avoiding construction of the dense
     W-tilde matrix of shape [N_image_pixels, N_image_pixels].

     The result can be used to rapidly assemble or apply W-tilde during curvature
     matrix construction without performing a NUFFT per source pixel.

     -------------------------------------------------------------------------------
     Backend behaviour
     -------------------------------------------------------------------------------
     - NumPy backend (use_jax=False, default):
         * CPU execution
         * Explicit Python loop over visibility chunks
         * Supports progress bars and optional memory reporting
         * Numerically closest to the original reference implementation

     - JAX backend (use_jax=True):
         * JIT-compilable and GPU/TPU capable
         * Uses fixed-size chunking and lax.fori_loop
         * No Python-side loops during execution
         * Progress bars and memory reporting are disabled
         * Floating-point results are numerically stable but not guaranteed to be
           bitwise-identical to NumPy due to parallel reduction order

     -------------------------------------------------------------------------------
     Numerical notes
     -------------------------------------------------------------------------------
     The preload values are computed as:

         sum_k w_k * cos(dx * ku_k + dy * kv_k)

     where ku_k = 2π u_k and kv_k = 2π v_k. This corresponds to the real part of the
     adjoint NUFFT evaluated on a uniform real-space offset grid.

     The chunking strategy controls temporary memory usage and GPU occupancy. Changing
     `chunk_k` in JAX mode triggers recompilation.

     -------------------------------------------------------------------------------
     Full Description (Original Documentation)
     -------------------------------------------------------------------------------
     The matrix `translation_invariant_nufft` a matrix of dimensions [unmasked_image_pixels, unmasked_image_pixels]
     that encodes the NUFFT of every pair of image pixels given the noise map. This can be used to efficiently compute
     the curvature matrix via the mapping matrix, in a way that omits having to perform the NUFFT on every individual
     source pixel. This provides a significant speed up for inversions of interferometer datasets with large number of
     visibilities.

     The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
     making it impossible to store in memory and its use in linear algebra calculations extremely. This methods creates
     a preload matrix that can compute the matrix via an efficient preloading scheme which exploits the
     symmetries in the NUFFT.

     To compute `translation_invariant_nufft`, one first defines a real space mask where every False entry is an
     unmasked pixel which is used in the calculation, for example:

         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI     This is an imaging.Mask2D, where:
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI     x = `True` (Pixel is masked and excluded from lens)
         IxIxIxIoIoIoIxIxIxIxI     o = `False` (Pixel is not masked and included in lens)
         IxIxIxIoIoIoIxIxIxIxI
         IxIxIxIoIoIoIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI

     Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and
     downwards, therefore:

         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxI0I1I2IxIxIxIxI
         IxIxIxI3I4I5IxIxIxIxI
         IxIxIxI6I7I8IxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI
         IxIxIxIxIxIxIxIxIxIxI

     In the standard calculation of `translation_invariant_nufft` it is a matrix of
     dimensions [unmasked_image_pixels, unmasked_pixel_images], therefore for the example mask above it would be
     dimensions [9, 9]. One performs a double for loop over `unmasked_image_pixels`, using the (y,x) spatial offset
     between every possible pair of unmasked image pixels to precompute values that depend on the properties of the NUFFT.

     This calculation has a lot of redundancy, because it uses the (y,x) *spatial offset* between the image pixels. For
     example, if two image pixel are next to one another by the same spacing the same value will be computed via the
     NUFFT. For the example mask above:

     - The value precomputed for pixel pair [0,1] is the same as pixel pairs [1,2], [3,4], [4,5], [6,7] and [7,9].
     - The value precomputed for pixel pair [0,3] is the same as pixel pairs [1,4], [2,5], [3,6], [4,7] and [5,8].
     - The values of pixels paired with themselves are also computed repeatedly for the standard calculation (e.g. 9
       times using the mask above).

     The `nufft_precision_operator` method instead only computes each value once. To do this, it stores the preload values in a
     matrix of dimensions [shape_masked_pixels_y, shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x)
     size of the vertical and horizontal extent of unmasked pixels, e.g. the spatial extent over which the real space
     grid extends.

     Each entry in the matrix `nufft_precision_operator[:,:,0]` provides the the precomputed NUFFT value mapping an image pixel
     to a pixel offset by that much in the y and x directions, for example:

     - nufft_precision_operator[0,0,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
       in the x direction by 0 - the values of pixels paired with themselves.
     - nufft_precision_operator[1,0,0] gives the precomputed values of image pixels that are offset in the y direction by 1 and
       in the x direction by 0 - the values of pixel pairs [0,3], [1,4], [2,5], [3,6], [4,7] and [5,8]
     - nufft_precision_operator[0,1,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
       in the x direction by 1 - the values of pixel pairs [0,1], [1,2], [3,4], [4,5], [6,7] and [7,9].

     Flipped pairs:

     The above preloaded values pair all image pixel NUFFT values when a pixel is to the right and / or down of the
     first image pixel. However, one must also precompute pairs where the paired pixel is to the left of the host
     pixels. These pairings are stored in `nufft_precision_operator[:,:,1]`, and the ordering of these pairings is flipped in the
     x direction to make it straight forward to use this matrix when computing the nufft weighted noise.

    Notes
    -----
    - If use_jax=True, the JAX implementation is used (requires JAX installed).
    - If use_jax=False, the NumPy implementation is used.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    shape_masked_pixels_2d
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    """
    if use_jax:
        return nufft_precision_operator_via_jax_from(
            noise_map_real=noise_map_real,
            uv_wavelengths=uv_wavelengths,
            shape_masked_pixels_2d=shape_masked_pixels_2d,
            grid_radians_2d=grid_radians_2d,
            chunk_k=chunk_k,
        )

    return nufft_precision_operator_via_np_from(
        noise_map_real=noise_map_real,
        uv_wavelengths=uv_wavelengths,
        shape_masked_pixels_2d=shape_masked_pixels_2d,
        grid_radians_2d=grid_radians_2d,
        chunk_k=chunk_k,
        show_progress=show_progress,
        show_memory=show_memory,
    )


def nufft_precision_operator_via_np_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d,
    grid_radians_2d: np.ndarray,
    *,
    chunk_k: int = 2048,
    show_progress: bool = False,
    show_memory: bool = False,
) -> np.ndarray:
    """
    NumPy/CPU implementation of the interferometer W-tilde curvature preload.

    See `nufft_precision_operator_from` for full description.
    """
    if chunk_k <= 0:
        raise ValueError("chunk_k must be a positive integer")

    noise_map_real = np.asarray(noise_map_real, dtype=np.float64)
    uv_wavelengths = np.asarray(uv_wavelengths, dtype=np.float64)
    grid_radians_2d = np.asarray(grid_radians_2d, dtype=np.float64)

    y_shape, x_shape = shape_masked_pixels_2d
    grid = grid_radians_2d[:y_shape, :x_shape]
    gy = grid[..., 0]
    gx = grid[..., 1]

    K = uv_wavelengths.shape[0]
    n_chunks = (K + chunk_k - 1) // chunk_k

    w = 1.0 / (noise_map_real**2)
    ku = 2.0 * np.pi * uv_wavelengths[:, 0]
    kv = 2.0 * np.pi * uv_wavelengths[:, 1]

    translation_invariant_kernel = np.zeros(
        (2 * y_shape, 2 * x_shape), dtype=np.float64
    )

    # Corner coordinates
    y00, x00 = gy[0, 0], gx[0, 0]
    y0m, x0m = gy[0, x_shape - 1], gx[0, x_shape - 1]
    ym0, xm0 = gy[y_shape - 1, 0], gx[y_shape - 1, 0]
    ymm, xmm = gy[y_shape - 1, x_shape - 1], gx[y_shape - 1, x_shape - 1]

    # -------------------------------------------------
    # Set up a single global progress bar
    # -------------------------------------------------
    pbar = None
    if show_progress:

        from tqdm import tqdm  # type: ignore

        n_quadrants = 1
        if x_shape > 1:
            n_quadrants += 1
        if y_shape > 1:
            n_quadrants += 1
        if (y_shape > 1) and (x_shape > 1):
            n_quadrants += 1

        pbar = tqdm(
            total=n_chunks * n_quadrants,
            desc="Accumulating visibilities (W-tilde preload)",
        )

    def accum_from_corner_np(y_ref, x_ref, gy_block, gx_block):
        dy = y_ref - gy_block
        dx = x_ref - gx_block

        acc = np.zeros(gy_block.shape, dtype=np.float64)

        for k0 in range(0, K, chunk_k):
            k1 = min(K, k0 + chunk_k)

            phase = dx[..., None] * ku[k0:k1] + dy[..., None] * kv[k0:k1]
            acc += np.sum(np.cos(phase) * w[k0:k1], axis=2)

            if pbar is not None:
                pbar.update(1)

            if show_memory and show_progress and "_report_memory" in globals():
                globals()["_report_memory"](acc)

        return acc

    # -----------------------------
    # Main quadrant (+,+)
    # -----------------------------
    translation_invariant_kernel[:y_shape, :x_shape] = accum_from_corner_np(
        y00, x00, gy, gx
    )

    # -----------------------------
    # Flip in x (+,-)
    # -----------------------------
    if x_shape > 1:
        block = accum_from_corner_np(y0m, x0m, gy[:, ::-1], gx[:, ::-1])
        translation_invariant_kernel[:y_shape, -1:-(x_shape):-1] = block[:, 1:]

    # -----------------------------
    # Flip in y (-,+)
    # -----------------------------
    if y_shape > 1:
        block = accum_from_corner_np(ym0, xm0, gy[::-1, :], gx[::-1, :])
        translation_invariant_kernel[-1:-(y_shape):-1, :x_shape] = block[1:, :]

    # -----------------------------
    # Flip in x and y (-,-)
    # -----------------------------
    if (y_shape > 1) and (x_shape > 1):
        block = accum_from_corner_np(ymm, xmm, gy[::-1, ::-1], gx[::-1, ::-1])
        translation_invariant_kernel[-1:-(y_shape):-1, -1:-(x_shape):-1] = block[1:, 1:]

    if pbar is not None:
        pbar.close()

    return translation_invariant_kernel


def nufft_precision_operator_via_jax_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d,
    grid_radians_2d: np.ndarray,
    *,
    chunk_k: int = 2048,
) -> np.ndarray:
    """
    JAX implementation of the interferometer W-tilde curvature preload.

    This version is intended for performance (CPU/GPU/TPU) and therefore:
      - uses JIT compilation internally
      - uses a compiled for-loop (lax.fori_loop) over fixed-size visibility chunks
      - does not support progress bars or memory reporting (those require Python loops)

    See `nufft_precision_operator_from` for full description.
    """
    import jax
    import jax.numpy as jnp

    if chunk_k <= 0:
        raise ValueError("chunk_k must be a positive integer")

    y_shape, x_shape = shape_masked_pixels_2d

    # Device arrays; keep float64 to match NumPy path as closely as possible.
    noise_map_real_x = jnp.asarray(noise_map_real, dtype=jnp.float64)
    uv_wavelengths_x = jnp.asarray(uv_wavelengths, dtype=jnp.float64)
    grid_radians_2d_x = jnp.asarray(grid_radians_2d, dtype=jnp.float64)

    # Precompute weights and angular frequencies on device
    w_x = 1.0 / (noise_map_real_x**2)
    ku_x = 2.0 * jnp.pi * uv_wavelengths_x[:, 0]
    kv_x = 2.0 * jnp.pi * uv_wavelengths_x[:, 1]

    grid = grid_radians_2d_x[:y_shape, :x_shape]
    gy = grid[..., 0]
    gx = grid[..., 1]

    # -----------------------------
    # IMPORTANT: pad so dynamic_slice(chunk_k) is always legal
    # -----------------------------
    K = int(uv_wavelengths_x.shape[0])  # known at trace/compile time
    n_chunks = (K + chunk_k - 1) // chunk_k
    K_pad = n_chunks * chunk_k
    pad_len = K_pad - K

    if pad_len > 0:
        ku_x = jnp.pad(ku_x, (0, pad_len))
        kv_x = jnp.pad(kv_x, (0, pad_len))
        w_x = jnp.pad(w_x, (0, pad_len))

    # A fixed [chunk_k] index vector used to mask the padded tail (last chunk).
    idx = jnp.arange(chunk_k)

    def _compute_all_quadrants(gy, gx, *, chunk_k: int):
        # Corner coordinates
        y00, x00 = gy[0, 0], gx[0, 0]
        y0m, x0m = gy[0, x_shape - 1], gx[0, x_shape - 1]
        ym0, xm0 = gy[y_shape - 1, 0], gx[y_shape - 1, 0]
        ymm, xmm = gy[y_shape - 1, x_shape - 1], gx[y_shape - 1, x_shape - 1]

        def accum_from_corner_jax(y_ref, x_ref, gy_block, gx_block):
            dy = y_ref - gy_block
            dx = x_ref - gx_block

            acc = jnp.zeros(gy_block.shape, dtype=jnp.float64)

            def body(i, acc_):
                k0 = i * chunk_k

                # Always legal because ku_x/kv_x/w_x were padded to length K_pad.
                ku_s = jax.lax.dynamic_slice(ku_x, (k0,), (chunk_k,))
                kv_s = jax.lax.dynamic_slice(kv_x, (k0,), (chunk_k,))
                w_s = jax.lax.dynamic_slice(w_x, (k0,), (chunk_k,))

                # Mask the padded tail (only the first K entries are real).
                valid = (idx + k0) < K
                w_s = jnp.where(valid, w_s, 0.0)

                phase = (
                    dx[..., None] * ku_s[None, None, :]
                    + dy[..., None] * kv_s[None, None, :]
                )
                return acc_ + jnp.sum(jnp.cos(phase) * w_s[None, None, :], axis=2)

            return jax.lax.fori_loop(0, n_chunks, body, acc)

        out = jnp.zeros((2 * y_shape, 2 * x_shape), dtype=jnp.float64)

        # (+,+)
        out = out.at[:y_shape, :x_shape].set(accum_from_corner_jax(y00, x00, gy, gx))

        # (+,-) x-flip
        if x_shape > 1:
            block = accum_from_corner_jax(y0m, x0m, gy[:, ::-1], gx[:, ::-1])
            out = out.at[:y_shape, -1:-(x_shape):-1].set(block[:, 1:])

        # (-,+) y-flip
        if y_shape > 1:
            block = accum_from_corner_jax(ym0, xm0, gy[::-1, :], gx[::-1, :])
            out = out.at[-1:-(y_shape):-1, :x_shape].set(block[1:, :])

        # (-,-) x- and y-flip
        if (y_shape > 1) and (x_shape > 1):
            block = accum_from_corner_jax(ymm, xmm, gy[::-1, ::-1], gx[::-1, ::-1])
            out = out.at[-1:-(y_shape):-1, -1:-(x_shape):-1].set(block[1:, 1:])

        return out

    _compute_all_quadrants_jit = jax.jit(
        _compute_all_quadrants, static_argnames=("chunk_k",)
    )

    t0 = time.time()
    translation_invariant_kernel = _compute_all_quadrants_jit(gy, gx, chunk_k=chunk_k)
    translation_invariant_kernel.block_until_ready()  # ensure timing includes actual device execution
    t1 = time.time()

    logger.info("INTERFEROMETER - Finished W-Tilde (JAX) in %.3f seconds", (t1 - t0))

    return np.asarray(translation_invariant_kernel)


def nufft_weighted_noise_via_sparse_operator_from(
    translation_invariant_kernel, native_index_for_slim_index
):
    """
    Use the `translation_invariant_kernel` (see `nufft_precision_operator_from`) to compute
    the `nufft_weighted_noise` efficiently.

    Parameters
    ----------
    translation_invariant_kernel
        The preloaded translation invariant values of the NUFFT that enable efficient computation of the
        NUFFT weighted noise matrix.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    slim_size = len(native_index_for_slim_index)

    nufft_weighted_noise = np.zeros((slim_size, slim_size))

    for i in range(slim_size):
        i_y, i_x = native_index_for_slim_index[i]

        for j in range(i, slim_size):
            j_y, j_x = native_index_for_slim_index[j]

            y_diff = j_y - i_y
            x_diff = j_x - i_x

            nufft_weighted_noise[i, j] = translation_invariant_kernel[y_diff, x_diff]

    for i in range(slim_size):
        for j in range(i, slim_size):
            nufft_weighted_noise[j, i] = nufft_weighted_noise[i, j]

    return nufft_weighted_noise


@dataclass(frozen=True)
class InterferometerSparseOperator:
    """
    Fully static FFT / geometry state for W~ curvature.

    Safe to cache as long as:
      - nufft_precision_operator is fixed
      - mask / rectangle definition is fixed
      - dtype is fixed
      - batch_size is fixed
    """

    dirty_image: np.ndarray
    y_shape: int
    x_shape: int
    M: int
    batch_size: int
    w_dtype: "jax.numpy.dtype"
    Khat: "jax.Array"  # (2y, 2x), complex
    col_offsets: "jax.Array"  # (batch_size,) int32
    """
    Cached FFT operator state for fast interferometer curvature-matrix assembly.

    This class packages *static* quantities needed to apply the interferometer
    W~ operator efficiently using FFTs, so that repeated likelihood evaluations
    do not redo expensive precomputation.

    Conceptually, the interferometer W~ operator is a translationally-invariant
    linear operator on a rectangular real-space grid, constructed from the
    `nufft_precision_operator` (a 2D array of correlation values on pixel offsets).
    By taking an FFT of this preload, the operator can be applied to batches of
    images via elementwise multiplication in Fourier space:

        apply_W(F) = IFFT( FFT(F_pad) * Khat )

    where `F_pad` is a (2y, 2x) padded version of `F` and `Khat = FFT(nufft_precision_operator)`.

    The curvature matrix for a pixelization (mapper) is then assembled from sparse
    mapping triplets without forming dense mapping matrices:

        C = A^T W A

    where A is the sparse mapping from source pixels to image pixels.

    Caching / validity
    ------------------
    Instances are safe to cache and reuse as long as all of the following remain fixed:

    - `nufft_precision_operator` (hence `Khat`)
    - the definition of the rectangular FFT grid (y_shape, x_shape)
    - dtype / precision (float32 vs float64)
    - `batch_size`

    Parameters stored
    -----------------
    dirty_image
        Convenience field for associated dirty image data (not used directly in
        curvature assembly in this method). Stored as a NumPy array to match
        upstream interfaces.
    y_shape, x_shape
        Shape of the *rectangular* real-space grid (not the masked slim grid).
    M
        Number of rectangular pixels, M = y_shape * x_shape.
    batch_size
        Number of source-pixel columns assembled and operated on per block.
        Larger batch sizes improve throughput on GPU but increase memory usage.
    w_dtype
        Floating-point dtype for weights and accumulations (e.g. float64).
    Khat
        FFT of the curvature preload, shape (2y_shape, 2x_shape), complex.
        This is the frequency-domain representation of the W~ operator kernel.
    """

    @classmethod
    def from_nufft_precision_operator(
        cls,
        nufft_precision_operator: np.ndarray,
        dirty_image: np.ndarray,
        *,
        batch_size: int = 128,
    ):
        """
        Construct an `InterferometerSparseOperator` from a curvature-preload array.

        This is the standard factory used in interferometer inversions.

        The curvature preload is assumed to be defined on a (2y, 2x) rectangular
        grid of pixel offsets, where y and x correspond to the *unmasked extent*
        of the real-space grid. The preload is FFT'd once to obtain `Khat`, which
        is then reused for every subsequent curvature matrix build.

        Parameters
        ----------
        nufft_precision_operator
            Real-valued array of shape (2y, 2x) encoding the W~ operator in real
            space as a function of pixel offsets. The shape must be even in both
            axes so that y_shape = H2//2 and x_shape = W2//2 are integers.
        dirty_image
            The dirty image associated with the dataset (or any convenient
            reference image). Not required for curvature computation itself,
            but commonly stored alongside the state for debugging / plotting.
        batch_size
            Number of source-pixel columns processed per block when assembling
            the curvature matrix. Higher values typically improve GPU efficiency
            but increase intermediate memory usage.

        Returns
        -------
        InterferometerSparseOperator
            Immutable cached state object containing shapes and FFT kernel `Khat`.

        Raises
        ------
        ValueError
            If `nufft_precision_operator` does not have even shape in both dimensions.
        """
        import jax.numpy as jnp

        H2, W2 = nufft_precision_operator.shape
        if (H2 % 2) != 0 or (W2 % 2) != 0:
            raise ValueError(
                f"nufft_precision_operator must have even shape (2y,2x). Got {nufft_precision_operator.shape}."
            )

        y_shape = H2 // 2
        x_shape = W2 // 2
        M = y_shape * x_shape

        Khat = jnp.fft.fft2(nufft_precision_operator)

        return InterferometerSparseOperator(
            dirty_image=dirty_image,
            y_shape=y_shape,
            x_shape=x_shape,
            M=M,
            batch_size=int(batch_size),
            w_dtype=nufft_precision_operator.dtype,
            Khat=Khat,
            col_offsets=jnp.arange(int(batch_size), dtype=jnp.int32),
        )

    def apply_operator(self, Fbatch_flat):
        """
        Apply the interferometer W~ operator to a batch of vectors.

        Given an input matrix of shape (M, B) on the rectangular real-space
        grid (M = y_shape * x_shape), this method computes

            G = W~ Fbatch_flat

        via FFT-based convolution with the cached `Khat` kernel:

            apply_W(F) = Re( IFFT( FFT(F_pad) * Khat ) )[:y, :x]

        where `F_pad` is the (2y, 2x) zero-padded version of `F`.

        Parameters
        ----------
        Fbatch_flat
            Array of shape (M, B) representing B vectors on the rectangular grid.

        Returns
        -------
        ndarray
            Array of shape (M, B) equal to W~ applied to the batch.
        """
        import jax.numpy as jnp

        y_shape, x_shape = self.y_shape, self.x_shape
        M = y_shape * x_shape
        Khat = self.Khat

        B = Fbatch_flat.shape[1]
        F_img = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        F_pad = jnp.pad(F_img, ((0, 0), (0, y_shape), (0, x_shape)))
        Fhat = jnp.fft.fft2(F_pad)
        Ghat = Fhat * Khat[None, :, :]
        G_pad = jnp.fft.ifft2(Ghat)
        G = jnp.real(G_pad[:, :y_shape, :x_shape])
        return G.reshape((B, M)).T

    def curvature_matrix_diag_from(self, rows, cols, vals, *, S: int):
        """
        Compute the diagonal (mapper-mapper) curvature matrix block F = Aᵀ W~ A.

        This method mirrors `ImagingSparseOperator.curvature_matrix_diag_from`
        and is the structural counterpart for the interferometer W~ operator.

        Given a sparse mapping operator A in COO triplet form (rows, cols, vals)
        with `S` source pixels, it computes

            F = Aᵀ W~ A

        in column blocks of width `batch_size`:

        1) Assemble Fbatch = A[:, start:start+B] on the rectangular grid via scatter-add.
        2) Apply W~ to the block via FFT: Gbatch = W~(Fbatch).
        3) Project back with Aᵀ via segment_sum over `cols`.

        Parameters
        ----------
        rows, cols, vals
            COO triplets encoding the sparse mapping operator A.
            - `rows`: rectangular-grid pixel indices (flat) in [0, M), shape (nnz,)
            - `cols`: source pixel indices in [0, S), shape (nnz,)
            - `vals`: mapping weights (interpolation + any sub-fraction normalisation),
              shape (nnz,)
            These should already be produced by `mapper.sparse_triplets_curvature`.
        S
            Number of source pixels / parameters for this mapper.

        Returns
        -------
        ndarray
            Curvature matrix of shape (S, S), symmetric.
        """
        import jax.numpy as jnp
        from jax import lax
        from jax.ops import segment_sum

        rows = jnp.asarray(rows, dtype=jnp.int32)
        cols = jnp.asarray(cols, dtype=jnp.int32)
        vals = jnp.asarray(vals, dtype=jnp.float64)

        M = self.M
        B = self.batch_size

        n_blocks = (S + B - 1) // B
        S_pad = n_blocks * B

        C0 = jnp.zeros((S, S_pad), dtype=jnp.float64)

        def body(block_i, C):
            start = block_i * B

            in_block = (cols >= start) & (cols < (start + B))
            bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
            v = jnp.where(in_block, vals, 0.0)

            F = jnp.zeros((M, B), dtype=jnp.float64)
            F = F.at[rows, bc].add(v)

            G = self.apply_operator(F)  # (M, B)

            contrib = vals[:, None] * G[rows, :]
            Cblock = segment_sum(contrib, cols, num_segments=S)  # (S, B)

            width = jnp.minimum(B, jnp.maximum(0, S - start))
            Cblock = Cblock * (self.col_offsets < width)[None, :]

            return lax.dynamic_update_slice(C, Cblock, (0, start))

        C_pad = lax.fori_loop(0, n_blocks, body, C0)
        C = C_pad[:, :S]
        return 0.5 * (C + C.T)
