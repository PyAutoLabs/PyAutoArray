from dataclasses import dataclass
import logging
import numpy as np
import time
from pathlib import Path
from typing import Optional

from autoarray.operators.transformer import _load_nufftax, nufftax_exception

try:
    from autonerves.test_mode import disable_jax
except ImportError:
    # Mirrors the fallback in `autoarray/dataset/interferometer/dataset.py`:
    # `disable_jax()` arrives in the autonerves release that closes
    # PyAutoNerves#159, and a `--no-deps` install, an editable checkout or a
    # hand-built virtualenv can all put an older autonerves on the path
    # regardless of the floor in `pyproject.toml`. Degrade to the predicate's
    # own one-line body rather than fail at module load; delete the fallback
    # when the floor names a release carrying the predicate.
    import os

    def disable_jax():
        return os.environ.get("PYAUTO_DISABLE_JAX") == "1"


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
    method: str = "nufft",
    eps: float = 1.0e-12,
    chunk_size: Optional[int] = None,
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
     Backend behaviour (the two brute-force builders)
     -------------------------------------------------------------------------------
     - NumPy backend (`method="numpy"`):
         * CPU execution
         * Explicit Python loop over visibility chunks
         * Supports progress bars and optional memory reporting
         * Numerically closest to the original reference implementation

     - JAX backend (`method="jax"`, or `method="numpy"` with `use_jax=True`):
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
    Three builders compute the same array; `method` selects which:

    - `"nufft"` (default) -- `nufft_precision_operator_via_nufft_from`, the type-1
      (adjoint) NUFFT. `O(K*nspread^2 + M log M)` for `M = 4*Ny*Nx`, i.e. seconds
      where the brute-force builders take minutes to hours. Needs `nufftax`.
    - `"numpy"` -- `nufft_precision_operator_via_np_from`, the brute-force
      `O(N_pix*K)` reference builder. Kept as the reference the NUFFT builder is
      pinned against, and used as the fallback below.
    - `"jax"` -- `nufft_precision_operator_via_jax_from`, the same brute force on
      JAX. `method="jax"` is the explicit way to ask for it.

    `use_jax` is only honoured when a brute-force method is selected: it upgrades
    `method="numpy"` to `method="jax"` and is otherwise ignored. Under the default
    `method="nufft"` it does nothing, because the NUFFT already runs on JAX -- an
    existing `use_jax=True` caller therefore keeps the fast path rather than being
    demoted to the `O(N_pix*K)` brute force.

    Two fallbacks to `"numpy"` are taken, both logged loudly (never silently),
    because they cost `O(N_pix*K)` where the NUFFT is `O(K*nspread^2 + M log M)`:

    1. `disable_jax()` is true (`PYAUTO_DISABLE_JAX=1`, the test-mode kill switch).
       Both `"nufft"` and `"jax"` run on JAX, so both are demoted.
    2. `nufftax` is not importable, so `"nufft"` cannot run.

    Any other `method` raises `ValueError`.

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
    method
        Which builder computes the operator: `"nufft"` (default), `"numpy"` or `"jax"`.
    eps
        The requested NUFFT precision, used by `method="nufft"` only.
    chunk_size
        The visibility chunk size of the NUFFT builder (a memory ceiling, not an
        optimisation), used by `method="nufft"` only. `None` is one shot.
    chunk_k
        The visibility chunk size of the two brute-force builders.
    use_jax
        Only honoured when a brute-force method is selected: `method="numpy"` with
        `use_jax=True` runs the JAX brute force (equivalent to `method="jax"`). It is
        ignored under the default `method="nufft"`, which already runs on JAX.
    """
    if method == "numpy" and use_jax:
        method = "jax"

    if method not in ("nufft", "numpy", "jax"):
        raise ValueError(
            f"nufft_precision_operator_from: unknown method {method!r}. "
            'Use "nufft" (the default type-1 NUFFT builder), "numpy" or "jax" (the '
            "brute-force reference builders)."
        )

    if method in ("nufft", "jax") and disable_jax():
        logger.warning(
            f"INTERFEROMETER - `PYAUTO_DISABLE_JAX=1` is set, so the NUFFT precision "
            f"operator cannot be built with method={method!r} (both the NUFFT and the "
            f"JAX brute force run on JAX). Falling back to the NumPy brute force, which "
            f"is O(N_pix * K) rather than O(K * nspread^2 + M log M) and can take "
            f"minutes to hours on a real dataset."
        )
        method = "numpy"

    if method == "nufft" and _load_nufftax() is None:
        logger.warning(
            "INTERFEROMETER - `nufftax` is not installed, so the NUFFT precision "
            "operator cannot be built with the type-1 NUFFT. Falling back to the NumPy "
            "brute force, which is O(N_pix * K) rather than O(K * nspread^2 + M log M) "
            "and can take minutes to hours on a real dataset. Install it via "
            "`pip install nufftax`."
        )
        method = "numpy"

    if method == "nufft":
        return nufft_precision_operator_via_nufft_from(
            noise_map_real=noise_map_real,
            uv_wavelengths=uv_wavelengths,
            shape_masked_pixels_2d=shape_masked_pixels_2d,
            grid_radians_2d=grid_radians_2d,
            eps=eps,
            chunk_size=chunk_size,
        )

    if method == "jax":
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


def _pixel_scale_radians_from(grid_radians_2d: np.ndarray) -> float:
    """
    Returns the pixel scale in radians, `delta_rad`, read off the radian grid as an
    adjacent-pixel difference.

    It is taken from the grid rather than from `mask.pixel_scales` because the grid is what
    the brute-force builders difference to get their `dx` / `dy`: deriving it any other way
    would let a unit or half-pixel convention drift in between the two implementations that
    have to agree exactly.

    Square pixels are asserted rather than handled. Every interferometer preset is square,
    the `[2Ny, 2Nx]` offset grid has a single mode spacing per axis by construction, and a
    rectangular-pixel dataset would silently produce a *plausible* wrong operator.

    Parameters
    ----------
    grid_radians_2d
        The 2D (y,x) native grid of coordinates in radians, shape `[ny, nx, 2]`.
    """
    grid_radians_2d = np.asarray(grid_radians_2d, dtype=np.float64)

    if grid_radians_2d.ndim != 3 or grid_radians_2d.shape[-1] != 2:
        raise ValueError(
            f"grid_radians_2d must be [ny, nx, 2] native; got {grid_radians_2d.shape}."
        )

    n_y, n_x = grid_radians_2d.shape[:2]

    if n_y < 2 or n_x < 2:
        raise ValueError(
            "grid_radians_2d must be at least 2x2 for the pixel scale to be read off as an "
            f"adjacent-pixel difference; got {(n_y, n_x)}."
        )

    # Native y decreases down the rows, x increases along the columns.
    delta_y = float(grid_radians_2d[0, 0, 0] - grid_radians_2d[1, 0, 0])
    delta_x = float(grid_radians_2d[0, 1, 1] - grid_radians_2d[0, 0, 1])

    if not np.isclose(delta_y, delta_x, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "The NUFFT precision operator requires square pixels: the radian grid's row "
            f"spacing {delta_y!r} and column spacing {delta_x!r} differ."
        )

    return delta_x


def nufft_precision_operator_via_nufft_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d,
    grid_radians_2d: np.ndarray,
    *,
    eps: float = 1.0e-12,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """
    Returns the `W~` precision operator built as the real part of a **type-1 (adjoint)
    NUFFT**.

    Same signature family and same return value as the brute-force builders
    `nufft_precision_operator_via_np_from` / `..._via_jax_from`, but `O(K * nspread^2 +
    M log M)` instead of `O(N_pix * K)`, where `M = 4 * Ny * Nx` is the doubled offset
    grid. At ALMA scale (`K = 1e6`, `N_pix = 15380`) that is the difference between
    ~35 minutes and ~7 seconds.

    The construction
    ----------------
    The brute-force builders compute, over the mask's bounding extent
    `(Ny, Nx) = shape_native_masked_pixels`:

        P[i, j] = sum_k w_k cos(2 pi (dx * u_k + dy * v_k)),   w_k = 1 / sigma_k^2

    with `dx = -j * delta_rad` and `dy = +i * delta_rad` on autoarray's radian grid
    (native `y` *decreases* down the rows, `x` increases along the columns), the four
    quadrants filled from the four corners so that offset `0` sits at `[0, 0]` and
    negative offsets sit at negative indices (wraparound / FFT ordering), with the middle
    row `Ny` and column `Nx` left zero as padding.

    Writing `x_k = 2 pi u_k delta_rad` and `y_k = 2 pi v_k delta_rad` -- the transformer's
    own scaled frequencies -- that is exactly

        P[i, j] = Re sum_k w_k exp(i(-j * x_k + i * y_k))

    i.e. the real part of a type-1 NUFFT of the weights onto the `(2Ny, 2Nx)` mode grid.
    `nufftax.nufft2d1(x, y, c, n_modes=(N1, N2), eps, isign)` returns
    `f[m2, m1] = sum_k c_k exp(isign * i(m1 * x_k + m2 * y_k))` on the **centred** mode
    grid, shape `(N2, N1)`, so the mapping is

        f = nufft2d1(-x, y, w, n_modes=(2Nx, 2Ny), eps, isign=+1)
        P = ifftshift(Re f);  P[Ny, :] = 0;  P[:, Nx] = 0

    Why this mapping and not one of the other seven
    -----------------------------------------------
    Pinned empirically against `nufft_precision_operator_via_np_from`. Of the eight
    candidates (axis swap x sign of `x` x sign of `y`) exactly two agree with the brute
    force -- `(-x, +y)` above and `(+x, -y)` -- at `max|delta| = 1.7e-17`, i.e. `8.7e-14`
    of the peak `P[0, 0]`. The other six are wrong by `2.1e-1` of the peak, so the
    identification is not marginal: the discrimination is thirteen orders of magnitude.

    The two survivors are the *same* construction: `w` is real, so `f(-x, +y)` and
    `f(+x, -y)` are complex conjugates and their real parts are identical. `(-x, +y)` is
    kept because it reads off the formula above term by term. That degeneracy is also why
    `P[i, j] == P[-i, -j]` (cosine evenness) holds -- pinned separately.

    `ifftshift` vs `fftshift` is likewise not a choice here: both axes have even length
    `2N`, and for even `N` the two shifts are the same permutation. The canonical
    `ifftshift` (centred -> wraparound) is used because that is the direction the transform
    actually goes.

    The padding row / column is at index `Ny` / `Nx`, not `Ny - 1` / `Nx - 1`: after
    `ifftshift`, index `Ny` carries mode `-Ny`, the Nyquist mode, which the brute force
    never evaluates (its quadrants span offsets `-(Ny - 1) ... Ny - 1`). The NUFFT *does*
    return a value there, so it is zeroed explicitly.

    Accuracy
    --------
    `eps` is the NUFFT's requested precision and its error bound is **peak-scaled**, not
    elementwise-relative: a type-1 NUFFT bounds `max|delta|` against `sum_k |c_k|`, so the
    near-zero entries of `P` -- five orders below its peak -- carry no relative accuracy
    guarantee at all. `eps = 1e-12` saturates fp64 at every instrument profiled: the
    measured `max|delta| = 1.7e-17` is already the round-off floor (`eps = 1e-14` only
    reaches `1.4e-17`), yet the worst *elementwise* relative error is `6.0e-10` on 32 of
    19600 entries. A pin against this builder must therefore be **mixed** --
    `rtol = 1e-10` with `atol = 1e-10 * P[0, 0]` -- never `rtol` with `atol = 0`.

    Chunking is mandatory at scale
    ------------------------------
    `chunk_size` is not an optimisation, it is a memory ceiling -- the same one
    `TransformerNUFFT` carries as its own `chunk_size`. The spreader's gather buffer is
    `K * nspread^2` complex128; at `eps = 1e-12`, `nspread ~ 14`, so `K = 1e6` needs ~3 GB
    and `K = 5e6` needs ~15 GB, which is where an unchunked call on a 15 GB machine is
    killed by the OOM reaper rather than returning slowly. Use the instrument's own
    transformer chunk size. The transform is linear in the weights, so the chunks'
    transforms are summed and the result is the same array (to summation order).

    A caveat on the timings quoted above: they are **CPU** seconds. The saving is in the
    algorithm, not the backend -- the brute force is `O(N_pix * K)` whatever it runs on.

    Parameters
    ----------
    noise_map_real
        `[K]` real noise-map values of the interferometer data; `w = 1 / sigma^2`.
    uv_wavelengths
        `[K, 2]` `(u, v)` baselines in wavelengths.
    shape_masked_pixels_2d
        `(Ny, Nx)`, the mask's bounding extent (`mask.shape_native_masked_pixels`).
    grid_radians_2d
        `[ny, nx, 2]` native `(y, x)` grid in radians. Only its pixel spacing is used, so
        the full native grid and the extent sub-grid give the same answer.
    eps
        The requested NUFFT precision.
    chunk_size
        Cap on the visibilities passed to `nufft2d1` in one call, or `None` for one shot.

    Returns
    -------
    np.ndarray
        `[2Ny, 2Nx]` float64, wraparound-ordered, with the padding row / column zero.
    """
    nufftax = _load_nufftax()

    if nufftax is None:
        nufftax_exception()

    import jax.numpy as jnp

    noise_map_real = np.asarray(noise_map_real, dtype=np.float64)
    uv_wavelengths = np.asarray(uv_wavelengths, dtype=np.float64)
    grid_radians_2d = np.asarray(grid_radians_2d, dtype=np.float64)

    y_shape, x_shape = (int(s) for s in shape_masked_pixels_2d)

    pixel_scale_radians = _pixel_scale_radians_from(grid_radians_2d)

    # The transformer's own scaled frequencies.
    x = 2.0 * np.pi * uv_wavelengths[:, 0] * pixel_scale_radians
    y = 2.0 * np.pi * uv_wavelengths[:, 1] * pixel_scale_radians

    w = 1.0 / (noise_map_real**2)

    n_modes = (2 * x_shape, 2 * y_shape)
    total_visibilities = int(x.shape[0])

    if chunk_size is None or chunk_size >= total_visibilities:
        chunk_size = total_visibilities

    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer or None, got {chunk_size}."
        )

    # Only Re(f) is ever used, so each chunk's real part is accumulated in float64 and the
    # complex block is released before the next one is spread.
    real_modes = np.zeros((2 * y_shape, 2 * x_shape), dtype=np.float64)

    for k0 in range(0, total_visibilities, chunk_size):
        k1 = min(total_visibilities, k0 + chunk_size)

        f = nufftax.nufft2d1(
            jnp.asarray(-x[k0:k1]),
            jnp.asarray(y[k0:k1]),
            jnp.asarray(w[k0:k1], dtype=jnp.complex128),
            n_modes,
            eps,
            1,
        )

        real_modes += np.asarray(np.real(f), dtype=np.float64)

        del f

    nufft_precision_operator = np.ascontiguousarray(np.fft.ifftshift(real_modes))

    nufft_precision_operator[y_shape, :] = 0.0
    nufft_precision_operator[:, x_shape] = 0.0

    return nufft_precision_operator


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

    Precondition
    ------------
    The `nufft_precision_operator` this state is built from encodes `W~ = Re(F^H W F)` for a
    single real-valued noise weighting, computed from the real-part noise sigma alone. It is
    therefore exact only for datasets where every visibility has equal real and imaginary
    noise sigma (`sigma_real == sigma_imag`); with unequal sigmas the curvature matrix
    assembled here silently disagrees with the dense `InversionInterferometerMapping` path.
    `Interferometer.apply_sparse_operator` enforces this precondition and raises a
    `DatasetException` when it is violated.
    """

    dirty_image: np.ndarray
    y_shape: int
    x_shape: int
    M: int
    batch_size: int
    w_dtype: "jax.numpy.dtype"
    Khat: "jax.Array"  # (2y, x+1), rfft2 of the real preload
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

        apply_W(F) = IRFFT( RFFT(F_pad) * Khat )

    where `F_pad` is a (2y, 2x) padded version of `F` and
    `Khat = rfft2(nufft_precision_operator)`.

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
        Real FFT of the curvature preload, shape (2y_shape, x_shape + 1), complex.
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
        of the real-space grid. The preload is real, so it is transformed once with
        a real FFT (`rfft2`) to obtain `Khat` of shape (2y, x + 1), which is then
        reused for every subsequent curvature matrix build.

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
            Immutable cached state object containing shapes and FFT kernel `Khat`,
            of shape (2y, x + 1) and complex dtype.

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

        Khat = jnp.fft.rfft2(nufft_precision_operator)

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

            apply_W(F) = IRFFT( RFFT(F_pad) * Khat )[:y, :x]

        where `F_pad` is the (2y, 2x) zero-padded version of `F`.

        Both the preload and the batch are real-valued, so the real-transform pair
        (`rfft2` / `irfft2`) is exact here rather than an approximation: the discarded
        half of the spectrum is the conjugate mirror of the half that is kept, and the
        inverse real transform reconstructs it, so the product is identical to the
        complex `fft2` / `ifft2` route to floating-point round-off (the old code took
        `Re(...)` of an already-real result). It halves the transform work and the size
        of `Khat`, measured at 1.27-1.61x faster on every backend (autolens_profiling
        #226, `results/notes/numba_interferometer_verdict.md`).

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
        Fhat = jnp.fft.rfft2(F_pad)
        Ghat = Fhat * Khat[None, :, :]
        G_pad = jnp.fft.irfft2(Ghat, s=(2 * y_shape, 2 * x_shape))
        G = G_pad[:, :y_shape, :x_shape]
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

    def curvature_matrix_off_diag_from(
        self, rows0, cols0, vals0, rows1, cols1, vals1, *, S0: int, S1: int
    ):
        """
        Compute the off-diagonal (mapper-mapper) curvature block F01 = A0ᵀ W~ A1.

        This method mirrors `ImagingSparseOperator.curvature_matrix_off_diag_from` and is the
        structural counterpart for the interferometer W~ operator. The difference between the two
        is the operator itself: for imaging W = Hᵀ N⁻¹ H is a PSF correlation, whereas here
        W~ = Re(Fᴴ W F) is the (translationally invariant) real-space operator of the non-uniform
        Fourier transform `F`, applied via `apply_operator` on the *unmasked-extent* rectangular
        grid (M = y_shape * x_shape).

        Given two sparse mapping operators:

        - A0 : (M × S0)
        - A1 : (M × S1)

        this method computes F01 = A0ᵀ W~ A1 in column blocks of width `batch_size`:

        1) Assemble Fbatch = A1[:, start:start+B] on the rectangular grid via scatter-add.
        2) Apply W~ to the block via FFT: Gbatch = W~(Fbatch).
        3) Project back with A0ᵀ via segment_sum over `cols0`.

        Parameters
        ----------
        rows0, cols0, vals0
            COO triplets for A0, where `rows0` are extent-grid (flat) indices in [0, M).
        rows1, cols1, vals1
            COO triplets for A1, where `rows1` are extent-grid (flat) indices in [0, M).
        S0
            Number of source pixels / parameters for mapper 0.
        S1
            Number of source pixels / parameters for mapper 1.

        Returns
        -------
        ndarray
            Off-diagonal curvature block of shape (S0, S1).

        Notes
        -----
        - The result is *not* symmetrized here because it is not square in general. The symmetric
          counterpart is F10 = F01ᵀ, because A0 and A1 share the same W~.
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

        M = self.M
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

    def operated_matrix_slim_from(self, matrix_slim, extent_index_for_masked_pixel):
        """
        Apply the interferometer W~ operator to columns defined on the *slim masked* grid.

        The input columns are scattered from the slim masked grid onto the unmasked-extent
        rectangular grid (on which W~ is defined), operated on with `apply_operator`, and gathered
        back onto the slim masked grid.

        Parameters
        ----------
        matrix_slim
            Array of shape (M_pix, n_cols) on the slim masked grid (e.g. the real-space
            `mapping_matrix` of an `AbstractLinearObjFuncList`).
        extent_index_for_masked_pixel
            Array of shape (M_pix,) mapping slim masked pixel indices to extent-grid flat indices.

        Returns
        -------
        ndarray
            Array of shape (M_pix, n_cols) equal to W~ applied to each column.
        """
        import jax.numpy as jnp

        matrix_slim = jnp.asarray(matrix_slim, dtype=jnp.float64)
        extent_index_for_masked_pixel = jnp.asarray(
            extent_index_for_masked_pixel, dtype=jnp.int32
        )

        grid_flat = jnp.zeros((self.M, matrix_slim.shape[1]), dtype=jnp.float64)
        grid_flat = grid_flat.at[extent_index_for_masked_pixel, :].set(matrix_slim)

        return self.apply_operator(grid_flat)[extent_index_for_masked_pixel, :]

    def curvature_matrix_off_diag_func_list_from(
        self,
        curvature_weights,  # (M_pix, n_funcs)
        extent_index_for_masked_pixel,  # (M_pix,) slim -> extent(flat)
        rows,
        cols,
        vals,  # triplets where rows are EXTENT indices
        *,
        S: int,
    ):
        """
        Compute the mapper–linear-function off-diagonal block Aᵀ W~ B.

        This is the interferometer counterpart of
        `ImagingSparseOperator.curvature_matrix_off_diag_func_list_from`, but with one important
        difference in what `curvature_weights` must contain.

        For imaging the operator is split as W = Hᵀ N⁻¹ H, so the imaging method is passed
        `curvature_weights = (H B) / noise²` (the forward blur and the inverse variance are folded
        into the input) and only applies Hᵀ internally.

        For an interferometer the whole operator W~ = Re(Fᴴ W F) is applied by `apply_operator`,
        with the inverse-variance weighting *already inside* W~. Therefore `curvature_weights` is
        the plain real-space `mapping_matrix` of the linear function list on the slim masked grid,
        with **no** noise weighting and **no** forward operator applied.

        The returned matrix is:

            off_diag = Aᵀ W~ B

        which has shape (S, n_funcs).

        Parameters
        ----------
        curvature_weights
            Array of shape (M_pix, n_funcs) on the *slim masked* grid: the un-operated,
            un-weighted real-space mapping matrix of the linear function list.
        extent_index_for_masked_pixel
            Array of shape (M_pix,) mapping slim masked pixel indices to extent-grid flat indices.
            Used to scatter values onto the rectangular grid W~ is defined on.
        rows, cols, vals
            COO triplets for the mapper A, where:
            - `rows` are extent-grid indices (flat), shape (nnz,)
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
        - No `batch_size` sweep is required because the operator is applied to `n_funcs` columns
          (typically a handful) rather than to all S source pixels.
        """
        import jax.numpy as jnp
        from jax.ops import segment_sum

        curvature_weights = jnp.asarray(curvature_weights, dtype=jnp.float64)
        extent_index_for_masked_pixel = jnp.asarray(
            extent_index_for_masked_pixel, dtype=jnp.int32
        )

        rows = jnp.asarray(rows, dtype=jnp.int32)
        cols = jnp.asarray(cols, dtype=jnp.int32)
        vals = jnp.asarray(vals, dtype=jnp.float64)

        n_funcs = curvature_weights.shape[1]

        # 1) scatter slim -> extent(flat)
        grid_flat = jnp.zeros((self.M, n_funcs), dtype=jnp.float64)
        grid_flat = grid_flat.at[extent_index_for_masked_pixel, :].set(
            curvature_weights
        )

        # 2) apply W~ on the extent grid
        operated = self.apply_operator(grid_flat)  # (M, n_funcs)

        # 3) gather at the mapper's rows (extent coords) and accumulate to source pixels
        contrib = vals[:, None] * operated[rows, :]
        return segment_sum(contrib, cols, num_segments=S)  # (S, n_funcs)

    def curvature_matrix_func_list_from(
        self,
        curvature_weights_0,  # (M_pix, n_funcs_0)
        curvature_weights_1,  # (M_pix, n_funcs_1)
        extent_index_for_masked_pixel,  # (M_pix,) slim -> extent(flat)
    ):
        """
        Compute a linear-function–linear-function curvature block B0ᵀ W~ B1.

        The imaging sparse inversion forms this block as a plain dot product of noise-weighted,
        PSF-convolved mapping matrices, because for imaging those matrices are already in the
        data frame. For an interferometer the equivalent dense construction would require the
        (expensive) visibility-space transformed mapping matrix, which the sparse formalism exists
        to avoid. Because W~ = Re(Fᴴ W F) is exact and translationally invariant on the extent
        grid, the block is instead formed directly through the same operator used by every other
        block, which is both cheaper and keeps every block of `F` self-consistent.

        Parameters
        ----------
        curvature_weights_0, curvature_weights_1
            The un-operated, un-weighted real-space `mapping_matrix` of each linear function list,
            on the slim masked grid, of shape (M_pix, n_funcs).
        extent_index_for_masked_pixel
            Array of shape (M_pix,) mapping slim masked pixel indices to extent-grid flat indices.

        Returns
        -------
        ndarray
            Curvature block of shape (n_funcs_0, n_funcs_1).
        """
        import jax.numpy as jnp

        curvature_weights_0 = jnp.asarray(curvature_weights_0, dtype=jnp.float64)

        operated = self.operated_matrix_slim_from(
            matrix_slim=curvature_weights_1,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
        )

        return curvature_weights_0.T @ operated
