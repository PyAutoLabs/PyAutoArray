from dataclasses import dataclass
from functools import partial
import numpy as np
from typing import Optional, List, Tuple


def psf_weighted_data_from(
    weight_map_native: np.ndarray,
    kernel_native: np.ndarray,
    native_index_for_slim_index,
    xp=np,
) -> np.ndarray:
    """
    The sparse linear algebra uses a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When it is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `psf_weighted_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix psf_weighted_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    Parameters
    ----------
    weight_map_native
        The two dimensional masked weight-map of values the PSF convolution is computed from, which is the data
        divided by the noise-map squared.
    kernel_native
        The two dimensional PSF kernel that `psf_weighted_data` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """
    Ky, Kx = kernel_native.shape
    ph, pw = Ky // 2, Kx // 2

    # 2) pad so neighbourhood gathers never go OOB
    padded = xp.pad(
        weight_map_native, ((ph, ph), (pw, pw)), mode="constant", constant_values=0.0
    )

    # 3) build broadcasted neighbourhood indices for all requested pixels
    # shift pixel coords into the padded frame
    ys = native_index_for_slim_index[:, 0] + ph  # (N,)
    xs = native_index_for_slim_index[:, 1] + pw  # (N,)

    # kernel-relative offsets
    dy = xp.arange(Ky) - ph  # (Ky,)
    dx = xp.arange(Kx) - pw  # (Kx,)

    # broadcast to (N, Ky, Kx)
    Y = ys[:, None, None] + dy[None, :, None]
    X = xs[:, None, None] + dx[None, None, :]

    # 4) gather patches and correlate (no kernel flip)
    patches = padded[Y, X]  # (N, Ky, Kx)
    return xp.sum(patches * kernel_native[None, :, :], axis=(1, 2))  # (N,)


def data_vector_via_psf_weighted_data_from(
    psf_weighted_data: np.ndarray,  # (M_pix,) float64
    rows: np.ndarray,  # (nnz,) int32  each triplet's data pixel (slim index)
    cols: np.ndarray,  # (nnz,) int32  source pixel index
    vals: np.ndarray,  # (nnz,) float64 mapping weights incl sub_fraction
    S: int,  # number of source pixels
) -> np.ndarray:
    r"""
    Returns the data vector `D` from the `psf_weighted_data` matrix (see `psf_weighted_data_from`), which encodes the
    the 1D image `d` and 1D noise-map values `\sigma` (see Warren & Dye 2003).

    This uses the sparse matrix triplet representation of the mapping matrix to efficiently compute the data vector
    without having to compute the full mapping matrix.

    Computes:
        D[p] = sum_{triplets t with col_t=p} vals[t] * weighted_data_slim[slim_rows[t]]

    Parameters
    ----------
    psf_weighted_data
        The matrix representing the PSF convolution of the imaging data divided by the noise-map squared.
    rows
        The row indices of the sparse mapping matrix triplet representation, which map to data pixels.
    cols
        The column indices of the sparse mapping matrix triplet representation, which map to source pixels.
    vals
        The values of the sparse mapping matrix triplet representation, which map the image pixels to source pixels.
    S
        The number of source pixels.
    """
    from jax.ops import segment_sum

    w = psf_weighted_data[rows]  # (nnz,)
    contrib = vals * w  # (nnz,)
    return segment_sum(contrib, cols, num_segments=S)  # (S,)


def data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix: np.ndarray, image: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    r"""
    Returns the data vector `D` from a blurred mapping matrix `f` and the 1D image `d` and 1D noise-map $\sigma$`
    (see Warren & Dye 2003).

    Parameters
    ----------
    blurred_mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    image
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    return (image / noise_map**2.0) @ blurred_mapping_matrix


def data_linear_func_matrix_from(
    curvature_weights_matrix: np.ndarray, kernel_native, mask
) -> np.ndarray:
    """
    Returns a matrix that for each data pixel, maps it to the sum of the values of a linear object function convolved
    with the PSF kernel at the data pixel.

    If a linear function in an inversion is fixed, its values can be evaluated and preloaded beforehand. For every
    data pixel, the PSF convolution with this preloaded linear function can also be preloaded, in a matrix of
    shape [data_pixels, 1].

    Given that multiple linear functions can be used and fixed in an inversion, this matrix is extended to have
    dimensions [data_pixels, total_fixed_linear_functions].

    When mapper objects and linear functions are used simultaneously in an inversion, this preloaded matrix
    significantly speed up the computation of their off-diagonal terms in the curvature matrix.

    This is similar to the preloading performed via the w-tilde formalism, except that there it is the PSF convolved
    values of each noise-map value pair that are preloaded.

    In **PyAutoGalaxy** and **PyAutoLens**, this preload is used when linear light profiles are fixed in the model.
    For example, when using a multi Gaussian expansion, the values defining how those Gaussians are evaluated
    (e.g. `centre`, `ell_comps` and `sigma`) are often fixed in a model, meaning this matrix can be preloaded and
    used for speed up.

    Parameters
    ----------
    curvature_weights_matrix
        The operated values of each linear function divided by the noise-map squared, in a matrix of shape
        [data_pixels, total_fixed_linear_functions].
    kernel_native
        The 2D PSF kernel.

    Returns
    -------
    ndarray
        A matrix of shape [data_pixels, total_fixed_linear_functions] that for each data pixel, maps it to the sum of
        the values of a linear object function convolved with the PSF kernel at the data pixel.
    """

    ny, nx = mask.shape_native
    n_unmasked, n_funcs = curvature_weights_matrix.shape

    # Expand masked -> native grid
    native = np.zeros((ny, nx, n_funcs))
    native[~mask] = curvature_weights_matrix  # put values into unmasked positions

    # Convolve each function with PSF kernel
    from scipy.signal import fftconvolve

    blurred_list = []
    for i in range(n_funcs):
        blurred = fftconvolve(native[..., i], kernel_native, mode="same")
        # Re-mask: only keep unmasked pixels
        blurred_list.append(blurred[~mask])

    return np.stack(blurred_list, axis=1)  # shape (n_unmasked, n_funcs)


def curvature_matrix_mirrored_from(
    curvature_matrix,
    *,
    xp=np,
):
    """
    Mirror a curvature matrix so that any non-zero entry C[i,j]
    is copied to C[j,i].

    Supports:
      - NumPy (xp=np)
      - JAX  (xp=jax.numpy)

    Parameters
    ----------
    curvature_matrix : (N, N) array
        Possibly triangular / partially-filled curvature matrix.

    xp : module
        np or jax.numpy

    Returns
    -------
    curvature_matrix_mirrored : (N, N) array
        Symmetric curvature matrix.
    """

    # Ensure array type
    C = curvature_matrix

    # Boolean mask of where entries exist
    mask = C != 0

    # Copy entries symmetrically
    C_T = xp.swapaxes(C, 0, 1)

    # Prefer non-zero values from either side
    curvature_matrix_mirrored = xp.where(mask, C, C_T)

    return curvature_matrix_mirrored


def curvature_matrix_with_added_to_diag_from(
    curvature_matrix,
    value: float,
    no_regularization_index_list: Optional[List[int]] = None,
    *,
    xp=np,
):
    """
    Add a small stabilizing value to the diagonal entries of the curvature matrix.

    Supports:
      - NumPy (xp=np): in-place update
      - JAX  (xp=jax.numpy): functional `.at[].add()`

    Parameters
    ----------
    curvature_matrix : (N, N) array
        Curvature matrix to modify.

    value : float
        Value added to selected diagonal entries.

    no_regularization_index_list : list of int
        Indices where diagonal should be boosted.

    xp : module
        np or jax.numpy

    Returns
    -------
    curvature_matrix : array
        Updated matrix (new array in JAX, modified in NumPy).
    """

    if no_regularization_index_list is None:
        return curvature_matrix

    inds = xp.asarray(no_regularization_index_list, dtype=xp.int32)

    if xp is np:
        # -----------------------
        # NumPy: in-place update
        # -----------------------
        curvature_matrix[inds, inds] += value
        return curvature_matrix

    else:
        # -----------------------
        # JAX: functional update
        # -----------------------
        return curvature_matrix.at[inds, inds].add(value)


def mapped_reconstructed_operated_data_via_sparse_operator_from(
    reconstruction,  # (S,)
    rows,
    cols,
    vals,  # (nnz,)
    fft_index_for_masked_pixel,
    data_shape: int,  # y_shape * x_shape
):
    import jax.numpy as jnp
    from jax.ops import segment_sum

    reconstruction = jnp.asarray(reconstruction, dtype=jnp.float64)
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    contrib = vals * reconstruction[cols]  # (nnz,)
    image_rect = segment_sum(
        contrib, rows, num_segments=data_shape[0] * data_shape[1]
    )  # (M_rect,)

    image_slim = image_rect[fft_index_for_masked_pixel]  # (M_pix,)
    return image_slim


@dataclass(frozen=True)
class ImagingSparseOperator:

    data_native: np.ndarray
    noise_map_native: np.ndarray
    weight_map: np.ndarray
    inverse_variances_native: "jax.Array"  # (y, x) float64
    y_shape: int
    x_shape: int
    Ky: int
    Kx: int
    fft_shape: Tuple[int, int]
    batch_size: int
    col_offsets: "jax.Array"  # (batch_size,) int32
    Khat_r: "jax.Array"  # (Fy, Fx//2+1) complex
    Khat_flip_r: "jax.Array"  # (Fy, Fx//2+1) complex

    @classmethod
    def from_noise_map_and_psf(
        cls,
        data,
        noise_map,
        psf,
        *,
        batch_size: int = 128,
        dtype=None,
    ) -> "ImagingSparseOperator":
        """
        Construct an `ImagingSparseOperator` from imaging arrays and a PSF.

        This factory method builds all static FFT state required to apply

            W = Hᵀ N⁻¹ H

        repeatedly during curvature matrix construction:

        - `inverse_variances_native` is computed from `noise_map` as 1/noise^2 (masked-safe).
        - `weight_map` is computed from `data` as data/noise^2 (masked-safe).
        - `Khat_r` and `Khat_flip_r` are precomputed as rFFT transforms of the padded PSF
          and its flipped version on the required FFT shape.

        Parameters
        ----------
        data
            Imaging data object (e.g. `Array2D`) providing `.array` (slim) and `.native`.
        noise_map
            Imaging noise-map object (e.g. `Array2D`) providing `.array` (slim),
            `.native`, `.mask`, and `.shape_native`.
        psf
            PSF kernel in native 2D form. Can be NumPy or JAX convertible.
        batch_size
            Number of curvature columns processed per FFT batch. Larger values can improve
            GPU utilization but increase VRAM usage.
        dtype
            JAX dtype for PSF / FFT precompute. Defaults to `jax.numpy.float64`.

        Returns
        -------
        ImagingSparseOperator
            Fully initialized operator containing all precomputed FFT state.

        Notes
        -----
        - This method assumes the FFT shape is `(Hy+Ky-1, Hx+Kx-1)` which is sufficient
          for linear convolution followed by cropping to “same”.
        - Mask handling: the `Array2D(..., mask=...)` wrapper ensures masked pixels are
          correctly represented in native arrays; you may still explicitly zero masked
          pixels if your downstream code expects it.
        """
        import jax.numpy as jnp

        from autoarray.structures.arrays.uniform_2d import Array2D

        if dtype is None:
            dtype = jnp.float64

        # ----------------------------
        # Shapes
        # ----------------------------
        y_shape = int(noise_map.shape_native[0])
        x_shape = int(noise_map.shape_native[1])

        # ----------------------------
        # inverse_variances_native (native 2D)
        # Make safe (0 where invalid)
        # ----------------------------
        # Try to get a plain native ndarray from your Array2D-like object:
        inverse_variances_native = 1.0 / noise_map**2
        inverse_variances_native = Array2D(
            values=inverse_variances_native, mask=noise_map.mask
        )
        inverse_variances_native = inverse_variances_native.native

        weight_map = data.array / (noise_map.array**2)
        weight_map = Array2D(values=weight_map, mask=noise_map.mask)

        # If you *also* want to zero masked pixels explicitly:
        # mask_native = noise_map.mask  (depends on your API; might be bool native)
        # inverse_variances_native = inverse_variances_native.at[mask_native].set(0.0)

        # ----------------------------
        # PSF + FFT precompute
        # ----------------------------
        psf = jnp.asarray(psf, dtype=dtype)
        Ky, Kx = map(int, psf.shape)

        fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)
        Fy, Fx = fft_shape

        def precompute(psf2d):
            psf_pad = jnp.pad(psf2d, ((0, Fy - Ky), (0, Fx - Kx)))
            return jnp.fft.rfft2(psf_pad, s=(Fy, Fx))

        Khat_r = precompute(psf)
        Khat_flip_r = precompute(jnp.flip(psf, axis=(0, 1)))

        return cls(
            data_native=data.native,
            noise_map_native=noise_map.native,
            weight_map=weight_map.native,
            inverse_variances_native=inverse_variances_native,
            y_shape=y_shape,
            x_shape=x_shape,
            Ky=Ky,
            Kx=Kx,
            fft_shape=(int(Fy), int(Fx)),
            batch_size=int(batch_size),
            col_offsets=jnp.arange(batch_size, dtype=jnp.int32),
            Khat_r=Khat_r,
            Khat_flip_r=Khat_flip_r,
        )

    def apply_operator(self, Fbatch_flat):
        """
        Apply the imaging precision operator W = Hᵀ N⁻¹ H to a batch of vectors.

        This is the fundamental linear operator application used throughout curvature
        assembly. Given an input matrix `Fbatch_flat` with shape:

            (M_rect, B)

        where:
        - `M_rect = y_shape * x_shape` is the number of pixels on the rectangular FFT grid,
        - `B` is the number of right-hand-side vectors (typically `batch_size`),

        this method computes:

            G = W Fbatch_flat

        using two FFT-based convolutions:
        1) Forward blur:   H (correlation with PSF)
        2) Weighting:      multiply by inverse variances (N⁻¹)
        3) Backproject:    Hᵀ (correlation with flipped PSF)

        Parameters
        ----------
        Fbatch_flat
            Array of shape (M_rect, B) representing B vectors on the rectangular grid.

        Returns
        -------
        ndarray
            Array of shape (M_rect, B) equal to W applied to the batch.

        Notes
        -----
        - The PSF is applied via rFFT on a padded grid of shape `fft_shape`.
        - The output is cropped back to the native `(y_shape, x_shape)` “same” region.
        - This method expects rectangular-grid indexing (flat). If you have slim masked
          indexing you must scatter/gather appropriately outside this function.
        """
        import jax.numpy as jnp

        y_shape, x_shape = self.y_shape, self.x_shape
        Ky, Kx = self.Ky, self.Kx
        Fy, Fx = self.fft_shape
        M = y_shape * x_shape

        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))

        # forward blur
        Fpad = jnp.pad(Fimg, ((0, 0), (0, Fy - y_shape), (0, Fx - x_shape)))
        Fhat = jnp.fft.rfft2(Fpad, s=(Fy, Fx))
        blurred_pad = jnp.fft.irfft2(Fhat * self.Khat_r[None, :, :], s=(Fy, Fx))

        cy, cx = Ky // 2, Kx // 2
        blurred = blurred_pad[:, cy : cy + y_shape, cx : cx + x_shape]

        weighted = blurred * self.inverse_variances_native[None, :, :]

        # backprojection
        Wpad = jnp.pad(weighted, ((0, 0), (0, Fy - y_shape), (0, Fx - x_shape)))
        What = jnp.fft.rfft2(Wpad, s=(Fy, Fx))
        back_pad = jnp.fft.irfft2(What * self.Khat_flip_r[None, :, :], s=(Fy, Fx))
        back = back_pad[:, cy : cy + y_shape, cx : cx + x_shape]

        return back.reshape((B, M)).T  # (M, B)

    def curvature_matrix_diag_from(self, rows, cols, vals, *, S: int):
        """
        Compute the diagonal (mapper–mapper) curvature matrix block F = Aᵀ W A.

        This method computes the curvature matrix for a single mapper/operator `A`
        represented in COO triplets (rows, cols, vals), using the FFT-backed W operator.

        Conceptually, if A is the (M_rect × S) mapping matrix, then:

            F = Aᵀ W A

        where:
        - `S` is the number of source pixels / parameters for this mapper.
        - `M_rect = y_shape * x_shape` is the number of rectangular grid pixels.

        The computation proceeds in column blocks of width `batch_size`:
        1) Assemble Fbatch = A[:, start:start+B] on the rectangular grid via scatter-add.
        2) Apply W to the block: Gbatch = W(Fbatch).
        3) Project back with Aᵀ via a segment_sum over `cols`.

        Parameters
        ----------
        rows, cols, vals
            COO triplets encoding the sparse mapping operator A.
            Expected conventions:
            - `rows`: rectangular-grid pixel indices (flat), shape (nnz,)
            - `cols`: source pixel indices in [0, S), shape (nnz,)
            - `vals`: mapping weights (incl sub-fraction / interpolation), shape (nnz,)
        S
            Number of source pixels / parameters for this mapper.

        Returns
        -------
        ndarray
            Curvature matrix of shape (S, S), symmetric.

        Notes
        -----
        - The output is symmetrized as 0.5*(F + Fᵀ) to mitigate numerical asymmetry
          introduced by floating-point reductions.
        - Padding to `S_pad = ceil(S/B)*B` ensures `dynamic_update_slice` is always legal
          even when S is not a multiple of batch_size.
        """
        import jax.numpy as jnp
        from jax import lax
        from jax.ops import segment_sum

        rows = jnp.asarray(rows, dtype=jnp.int32)
        cols = jnp.asarray(cols, dtype=jnp.int32)
        vals = jnp.asarray(vals, dtype=jnp.float64)

        y_shape, x_shape = self.y_shape, self.x_shape
        M = y_shape * x_shape
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

    def curvature_matrix_off_diag_from(
        self, rows0, cols0, vals0, rows1, cols1, vals1, *, S0: int, S1: int
    ):
        """
        Compute the off-diagonal curvature block F01 = A0ᵀ W A1.

        Given two sparse mapping operators:
        - A0 : (M_rect × S0)
        - A1 : (M_rect × S1)

        this method computes:

            F01 = A0ᵀ W A1

        using the same FFT-backed W operator as in the diagonal computation.

        The computation proceeds in column blocks over A1:
        1) Assemble Fbatch = A1[:, start:start+B] on the rectangular grid.
        2) Apply W: Gbatch = W(Fbatch).
        3) Project with A0ᵀ via segment_sum over cols0.

        Parameters
        ----------
        rows0, cols0, vals0
            COO triplets for A0. `rows0` are rectangular-grid indices.
        rows1, cols1, vals1
            COO triplets for A1. `rows1` are rectangular-grid indices.
        S0
            Number of source parameters for mapper/operator 0.
        S1
            Number of source parameters for mapper/operator 1.

        Returns
        -------
        ndarray
            Off-diagonal curvature block of shape (S0, S1).

        Notes
        -----
        - The result is *not* symmetrized here because it is not square in general.
          If you need the symmetric counterpart, use F10 = F01ᵀ when A0/A1 share W.
        - Padding to `S1_pad = ceil(S1/B)*B` ensures `dynamic_update_slice` is always legal.
        """
        import jax.numpy as jnp
        from jax import lax
        from jax.ops import segment_sum

        rows0 = jnp.asarray(rows0, dtype=jnp.int32)
        cols0 = jnp.asarray(cols0, dtype=jnp.int32)
        vals0 = jnp.asarray(vals0, dtype=jnp.float64)

        rows1 = jnp.asarray(rows1, dtype=jnp.int32)
        cols1 = jnp.asarray(cols1, dtype=jnp.int32)
        vals1 = jnp.asarray(vals1, dtype=jnp.float64)

        y_shape, x_shape = self.y_shape, self.x_shape
        M = y_shape * x_shape
        B = self.batch_size

        n_blocks = (S1 + B - 1) // B
        S1_pad = n_blocks * B

        F01_0 = jnp.zeros((S0, S1_pad), dtype=jnp.float64)

        def body(block_i, F01):
            start = block_i * B

            in_block = (cols1 >= start) & (cols1 < (start + B))
            bc = jnp.where(in_block, cols1 - start, 0).astype(jnp.int32)
            v = jnp.where(in_block, vals1, 0.0)

            F = jnp.zeros((M, B), dtype=jnp.float64)
            F = F.at[rows1, bc].add(v)

            G = self.apply_operator(F)  # (M, B)

            contrib = vals0[:, None] * G[rows0, :]
            block = segment_sum(contrib, cols0, num_segments=S0)

            width = jnp.minimum(B, jnp.maximum(0, S1 - start))
            block = block * (self.col_offsets < width)[None, :]

            return lax.dynamic_update_slice(F01, block, (0, start))

        F01_pad = lax.fori_loop(0, n_blocks, body, F01_0)
        return F01_pad[:, :S1]

    def curvature_matrix_off_diag_func_list_from(
        self,
        curvature_weights,  # (M_pix, n_funcs)
        fft_index_for_masked_pixel,  # (M_pix,)  slim -> rect(flat)
        rows,
        cols,
        vals,  # triplets where rows are RECT indices
        *,
        S: int,
    ):
        """
        Compute the mapper–linear-function off-diagonal block: Aᵀ Hᵀ (curvature_weights).

        This method computes the off-diagonal block between a sparse mapper/operator A
        and a set of fixed linear functions (e.g. linear light profiles) whose operated
        images have already been PSF-convolved and noise-weighted.

        The input `curvature_weights` is assumed to already contain:

            curvature_weights = (H B) / noise^2

        evaluated on the *slim masked* grid, for each linear function column.

        The returned matrix is:

            off_diag = Aᵀ [ Hᵀ (curvature_weights_native) ]

        which has shape (S, n_funcs).

        Parameters
        ----------
        curvature_weights
            Array of shape (M_pix, n_funcs) on the *slim masked* grid.
            Each column corresponds to one linear function already convolved by H and
            weighted by inverse variance.
        fft_index_for_masked_pixel
            Array of shape (M_pix,) mapping slim masked pixel indices to rectangular
            FFT-grid flat indices. Used to scatter values onto the rectangular grid.
        rows, cols, vals
            COO triplets for the mapper A, where:
            - `rows` are rectangular-grid indices (flat), shape (nnz,)
            - `cols` are source pixel indices, shape (nnz,)
            - `vals` are mapping weights, shape (nnz,)
        S
            Number of source pixels / parameters in the mapper.

        Returns
        -------
        ndarray
            Off-diagonal block of shape (S, n_funcs).

        Notes
        -----
        - This method performs only one convolution per function column (batched),
          using the flipped PSF (Hᵀ), because `curvature_weights` is assumed to already
          include the forward blur H.
        - No batch_size parameter is required here because we are convolving `n_funcs`
          columns (often small) rather than sweeping over source pixels S.
        """
        import jax.numpy as jnp
        from jax.ops import segment_sum

        curvature_weights = jnp.asarray(curvature_weights, dtype=jnp.float64)
        fft_index_for_masked_pixel = jnp.asarray(
            fft_index_for_masked_pixel, dtype=jnp.int32
        )

        rows = jnp.asarray(rows, dtype=jnp.int32)
        cols = jnp.asarray(cols, dtype=jnp.int32)
        vals = jnp.asarray(vals, dtype=jnp.float64)

        y_shape, x_shape = self.y_shape, self.x_shape
        Ky, Kx = self.Ky, self.Kx
        Fy, Fx = self.fft_shape
        M_rect = y_shape * x_shape

        M_pix, n_funcs = curvature_weights.shape

        # 1) scatter slim -> rect(flat)
        grid_flat = jnp.zeros((M_rect, n_funcs), dtype=jnp.float64)
        grid_flat = grid_flat.at[fft_index_for_masked_pixel, :].set(curvature_weights)

        # 2) apply H^T: conv with flipped PSF
        images = grid_flat.T.reshape((n_funcs, y_shape, x_shape))  # (B=n_funcs, Hy, Hx)

        # --- rfft conv (same as your helper) ---
        images_pad = jnp.pad(images, ((0, 0), (0, Fy - y_shape), (0, Fx - x_shape)))
        Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))
        out_pad = jnp.fft.irfft2(Fhat * self.Khat_flip_r[None, :, :], s=(Fy, Fx))

        cy, cx = Ky // 2, Kx // 2
        back_native = out_pad[
            :, cy : cy + y_shape, cx : cx + x_shape
        ]  # (n_funcs, Hy, Hx)

        # 3) gather at mapper rows (rect coords)
        back_flat = back_native.reshape((n_funcs, M_rect)).T  # (M_rect, n_funcs)
        back_at_rows = back_flat[rows, :]  # (nnz, n_funcs)

        # 4) accumulate to source pixels
        contrib = vals[:, None] * back_at_rows
        return segment_sum(contrib, cols, num_segments=S)  # (S, n_funcs)
