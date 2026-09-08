import numpy as np
from typing import Tuple

from autoarray import exc

from autoarray.inversion.regularization.adapt import (
    adapt_regularization_weights_from,
)
from autoarray.inversion.regularization.adapt import (
    weighted_regularization_matrix_from,
)
from autoarray.inversion.regularization.adapt import (
    weighted_regularization_matrix_single_scatter_from,
)
from autoarray.inversion.regularization.brightness_zeroth import (
    brightness_zeroth_regularization_matrix_from,
)
from autoarray.inversion.regularization.brightness_zeroth import (
    brightness_zeroth_regularization_weights_from,
)
from autoarray.inversion.regularization.constant import (
    constant_regularization_matrix_from,
)
from autoarray.inversion.regularization.constant_zeroth import (
    constant_zeroth_regularization_matrix_from,
)
from autoarray.inversion.regularization.exponential_kernel import exp_cov_matrix_from
from autoarray.inversion.regularization.gaussian_kernel import gauss_cov_matrix_from
from autoarray.inversion.regularization.matern_kernel import matern_kernel
from autoarray.inversion.regularization.zeroth import zeroth_regularization_matrix_from


# ---------------------------------------------------------------------------
# Split-regularization stencil compaction (JAX path of
# `pixel_splitted_regularization_matrix_from`)
# ---------------------------------------------------------------------------
#
# The split stencil tables are fixed-shape `(4P, K)` arrays: `K = 4` for the `Delaunay` mesh, but
# `K = 33` for the natural-neighbor `DelaunayNN` mesh (`SIBSON_MAX_NEIGHBORS` 32 + 1 spare column
# for the self insertion). On the real HST `DelaunayNN` cell the occupied post-`reg_split_from`
# width is min 1 / median 5 / p99 9 / max 11, so a full `(4P, K, K)` outer-product scatter spends
# ~97% of its 6.5M entries on padding, at a cost quadratic in the padded width:
#
#   compact width     |  12    16    20    24    28    32    33 (= today, no compaction)
#   A100 fp64 ms/call |  0.58  1.47  2.78  4.53  6.71  9.31  10.03      (vmap 16, real HST tables)
#   CPU  fp64 ms/call |  12.6                                 72.8
#
# so width 12 is a 17x GPU and 5.8x CPU improvement over the uncompacted scatter, with no backend
# gate needed. Width 12 covers the production stencil with margin, but the cap audit over 101
# ensemble geometries (`autolens_profiling/results/notes/delaunay_nn_cap_audit.md`) saw rare tail
# geometries reach 21 natural neighbors (99.9th pct 11, 99.99th pct 15, max 21), so rows wider than
# the compact width are supplemented exactly rather than dropped: the `SPLIT_REG_WIDE_ROW_BUDGET`
# widest rows get a full-width supplementary scatter. The budget of 256 rows is ~4x the count of
# above-width rows in the worst audited geometry and costs `W * (K**2 - kc**2)` ~ 0.24M entries,
# an order of magnitude below the 6.5M it replaces. Beyond the budget the matrix is NaN (see the
# function docstring), never silently truncated.
SPLIT_REG_COMPACT_WIDTH = 12
SPLIT_REG_WIDE_ROW_BUDGET = 256


def split_points_from(points, area_weights, xp=np):
    """
    points : (N, 2)
    areas  : (N,)
    xp     : np or jnp

    Returns (4*N, 2)
    """

    N = points.shape[0]
    offsets = area_weights

    x = points[:, 0]
    y = points[:, 1]

    # Allocate output (N, 4, 2)
    out = xp.zeros((N, 4, 2), dtype=points.dtype)

    if xp.__name__.startswith("jax"):
        # ----------------------------
        # JAX → use .at[] updates
        # ----------------------------
        out = out.at[:, 0, 0].set(x + offsets)
        out = out.at[:, 0, 1].set(y)

        out = out.at[:, 1, 0].set(x - offsets)
        out = out.at[:, 1, 1].set(y)

        out = out.at[:, 2, 0].set(x)
        out = out.at[:, 2, 1].set(y + offsets)

        out = out.at[:, 3, 0].set(x)
        out = out.at[:, 3, 1].set(y - offsets)

    else:

        # ----------------------------
        # NumPy → direct assignment OK
        # ----------------------------
        out[:, 0, 0] = x + offsets
        out[:, 0, 1] = y

        out[:, 1, 0] = x - offsets
        out[:, 1, 1] = y

        out[:, 2, 0] = x
        out[:, 2, 1] = y + offsets

        out[:, 3, 0] = x
        out[:, 3, 1] = y - offsets

    return out.reshape((N * 4, 2))


def reg_split_np_from(
    splitted_mappings: np.ndarray,
    splitted_sizes: np.ndarray,
    splitted_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    When creating the regularization matrix of a source pixelization, this function assumes each source pixel has been
    split into a cross of four points (the size of which is based on the area of the source pixel). This cross of
    points represents points which together can evaluate the gradient of the pixelization's reconstructed values.

    This function takes each cross of points and determines the regularization weights of every point on the cross,
    to construct a regulariaztion matrix based on the gradient of each pixel.

    The size of each cross depends on the Delaunay pixel area, thus this regularization scheme and its weights depend
    on the pixel area (there are larger weights for bigger pixels). This ensures that bigger pixels are regularized
    more.

    The number of pixel neighbors over which regularization is 4 * the total number of source pixels. This contrasts
    other regularization schemes, where the number of neighbors changes depending on, for example, the Delaunay mesh
    geometry. By having a fixed number of neighbors this removes stochasticty in the regularization that is applied
    to a solution.

    There are cases where a grid has over 100 neighbors, corresponding to very coordinate transformations. In such
    extreme cases, we raise a `exc.FitException`.

    Parameters
    ----------
    splitted_mappings
    splitted_sizes
    splitted_weights

    Returns
    -------

    """
    splitted_weights = -1.0 * splitted_weights

    for i in range(len(splitted_mappings)):

        pixel_index = i // 4

        flag = 0

        for j in range(splitted_sizes[i]):
            if splitted_mappings[i][j] == pixel_index:
                splitted_weights[i][j] += 1.0
                flag = 1

        if flag == 0:
            splitted_mappings[i][j + 1] = pixel_index
            splitted_sizes[i] += 1
            splitted_weights[i][j + 1] = 1.0

    return splitted_mappings, splitted_sizes, splitted_weights


def reg_split_from(
    splitted_mappings: np.ndarray,
    splitted_sizes: np.ndarray,
    splitted_weights: np.ndarray,
    xp=np,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    When creating the regularization matrix of a source pixelization, this function assumes each source pixel has been
    split into a cross of four points (the size of which is based on the area of the source pixel). This cross of
    points represents points which together can evaluate the gradient of the pixelization's reconstructed values.

    This function takes each cross of points and determines the regularization weights of every point on the cross,
    to construct a regulariaztion matrix based on the gradient of each pixel.

    The size of each cross depends on the Delaunay pixel area, thus this regularization scheme and its weights depend
    on the pixel area (there are larger weights for bigger pixels). This ensures that bigger pixels are regularized
    more.

    The number of pixel neighbors over which regularization is 4 * the total number of source pixels. This contrasts
    other regularization schemes, where the number of neighbors changes depending on, for example, the Delaunay mesh
    geometry. By having a fixed number of neighbors this removes stochasticty in the regularization that is applied
    to a solution.

    There are cases where a grid has over 100 neighbors, corresponding to very coordinate transformations. In such
    extreme cases, we raise a `exc.FitException`.

    Parameters
    ----------
    splitted_mappings
    splitted_sizes
    splitted_weights

    Returns
    -------

    """
    if xp == np:
        return reg_split_np_from(
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )

    import jax.numpy as jnp
    import jax.nn as jnn

    mappings = jnp.asarray(splitted_mappings)
    sizes = jnp.asarray(splitted_sizes)
    weights = jnp.asarray(splitted_weights)

    N, K = mappings.shape

    # -------------------------------------------------------------
    # 1. Negate all weights (same as Python: splitted_weights *= -1)
    # -------------------------------------------------------------
    weights = -weights

    # -------------------------------------------------------------
    # 2. Pixel index for each row: i // 4
    # -------------------------------------------------------------
    pixel_index = (jnp.arange(N) // 4).astype(mappings.dtype)  # (N,)
    pix_b = pixel_index[:, None]  # (N,1)

    # -------------------------------------------------------------
    # 3. Mask of valid columns j < size[i]
    # -------------------------------------------------------------
    cols = jnp.arange(K)[None, :]  # (1,4)
    valid_mask = cols < sizes[:, None]  # (N,4)

    # -------------------------------------------------------------
    # 4. Self match: mapping[i,j] == pixel_index AND j is valid
    # -------------------------------------------------------------
    self_mask = (mappings == pix_b) & valid_mask  # (N,4)
    row_has_self = jnp.any(self_mask, axis=1)  # (N,)

    # Position of self per row
    self_pos = jnp.argmax(self_mask, axis=1)  # (N,)

    # -------------------------------------------------------------
    # 5. Add +1 weight at self_pos where row_has_self == True
    # -------------------------------------------------------------
    one_hot = jnn.one_hot(self_pos, K, dtype=weights.dtype)  # (N,4)
    weights = weights + one_hot * row_has_self[:, None]

    # -------------------------------------------------------------
    # 6. Handle rows where pixel_index must be inserted
    # -------------------------------------------------------------
    no_self = ~row_has_self

    # Insert position = sizes[i]
    insert_pos = sizes  # (N,)
    insert_mask = no_self[:, None] & (cols == sizes[:, None])

    # New mappings and weights
    mappings = jnp.where(insert_mask, pix_b, mappings)
    weights = jnp.where(insert_mask, jnp.array(1.0, weights.dtype), weights)

    # Updated sizes: +1 if no self detected
    sizes_new = sizes + no_self.astype(sizes.dtype)

    return mappings, sizes_new, weights


def pixel_splitted_regularization_matrix_np_from(
    regularization_weights: np.ndarray,
    splitted_mappings: np.ndarray,
    splitted_sizes: np.ndarray,
    splitted_weights: np.ndarray,
) -> np.ndarray:
    # I'm not sure what is the best way to add surface brightness weight to the regularization scheme here.
    # Currently, I simply mulitply the i-th weight to the i-th source pixel, but there should be different ways.
    # Need to keep an eye here.

    parameters = int(len(splitted_mappings) / 4)

    regularization_matrix = np.zeros(shape=(parameters, parameters))

    regularization_weight = regularization_weights**2.0

    for i in range(parameters):
        regularization_matrix[i, i] += 2e-8

        for j in range(4):
            k = i * 4 + j

            size = splitted_sizes[k]
            mapping = splitted_mappings[k]
            weight = splitted_weights[k]

            for l in range(size):
                for m in range(size - l):
                    regularization_matrix[mapping[l], mapping[l + m]] += (
                        weight[l] * weight[l + m] * regularization_weight[i]
                    )
                    regularization_matrix[mapping[l + m], mapping[l]] += (
                        weight[l] * weight[l + m] * regularization_weight[i]
                    )

    for i in range(parameters):
        regularization_matrix[i, i] /= 2.0

    return regularization_matrix


def pixel_splitted_regularization_matrix_from(
    regularization_weights: np.ndarray,  # (P,)
    splitted_mappings: np.ndarray,  # (4P, K)
    splitted_sizes: np.ndarray,  # (4P,)
    splitted_weights: np.ndarray,  # (4P, K)
    xp=np,
    compact_width: int = SPLIT_REG_COMPACT_WIDTH,
    wide_row_budget: int = SPLIT_REG_WIDE_ROW_BUDGET,
):
    """
    Returns the regularization matrix for the adaptive split-pixel regularization scheme.

    This scheme splits each source pixel into a cross of four regularization points and interpolates
    to those points to smooth the inversion solution. It is designed to mitigate stochasticity in
    the regularization that can arise when the number of neighboring pixels varies across a
    mesh (e.g., in a Delaunay tessellation).

    A visual description and further details are provided in the appendix of He et al. (2024):
    https://arxiv.org/abs/2403.16253

    JAX path: compact main scatter plus a wide-row supplement
    ---------------------------------------------------------
    The stencil tables are fixed-shape ``(4P, K)`` arrays whose columns beyond each row's
    ``splitted_sizes`` entry are padding (mapping ``-1``, weight ``0``). For the natural-neighbor
    (``DelaunayNN``) mesh ``K = 33`` while the real occupied width is at most ~11, so scattering the
    full ``(4P, K, K)`` outer product spends ~97% of its traffic on padding and costs
    ``O(K**2)``. This function therefore scatters only the first ``kc = min(K, compact_width)``
    columns and supplements the ``wide_row_budget`` widest rows with the blocks the compact pass
    did not cover (see :data:`SPLIT_REG_COMPACT_WIDTH`). The result is exact: padded columns
    contribute mapping ``0`` / weight ``0``, so a row whose size is ``<= kc`` is reproduced
    bit-for-bit by the compact pass alone.

    If more rows exceed ``kc`` than the supplement budget holds, the matrix is poisoned with NaN
    rather than silently truncated. This is the same NaN-on-overflow contract the Sibson
    natural-neighbor caps already use (``mesh/interpolator/sibson.py``, where ``neighbor_overflow``
    / ``failed`` set the interpolation weights to NaN): the likelihood of an out-of-budget geometry
    evaluates to NaN and is discarded by the sampler, never returning a silently wrong ``H``.

    When ``K <= compact_width`` (the ``Delaunay`` mesh's ``K = 4``, and the adapt-split family) the
    compaction is a no-op: the function performs today's single scatter with no supplement and no
    overflow guard.

    Parameters
    ----------
    regularization_weights
        The regularization weight per pixel, adaptively controlling the strength of regularization
        applied to each inversion parameter.
    splitted_mappings
        The image pixel index mappings for each of the four regularization points into which each source pixel is split.
    splitted_sizes
        The number of neighbors or interpolation terms associated with each regularization point.
    splitted_weights
        The interpolation weights corresponding to each mapping entry, used to apply regularization
        between split points.
    xp
        The array module used, `numpy` or `jax.numpy`.
    compact_width
        The number of stencil columns scattered for every row on the JAX path (see
        :data:`SPLIT_REG_COMPACT_WIDTH`). Ignored on the numpy path.
    wide_row_budget
        The number of widest rows given a full-width supplementary scatter on the JAX path (see
        :data:`SPLIT_REG_WIDE_ROW_BUDGET`). Ignored on the numpy path.

    Returns
    -------
    The regularization matrix of shape [source_pixels, source_pixels].
    """

    if xp == np:
        return pixel_splitted_regularization_matrix_np_from(
            regularization_weights=regularization_weights,
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )

    import jax
    import jax.numpy as jnp

    # How many real pixels?
    P = splitted_mappings.shape[0] // 4
    K = splitted_mappings.shape[1]

    # Square, positive regularization weights
    reg_w = regularization_weights**2.0  # (P,)

    # Add diagonal jitter (2e-8)
    reg_mat = jnp.eye(P) * 2e-8  # (P, P)

    # ----- Build all 4P contributions at once -----

    # Mask away padded entries (where mapping = -1)
    valid = splitted_mappings != -1  # (4P, K)

    # Extract valid mapping rows and weights
    # BUT keep fixed shape (K) and just zero out invalid ones
    map_fixed = jnp.where(valid, splitted_mappings, 0)  # (4P, K)
    w_fixed = jnp.where(valid, splitted_weights, 0.0)  # (4P, K)

    # Each block is scaled by its pixel's regularization weight.
    # Rows 0-3 belong to pixel 0, rows 4-7 to pixel 1, etc.
    pixel_index = jnp.arange(4 * P) // 4  # (4P,)
    block_scale = reg_w[pixel_index]  # (4P,)

    # ----- Compact main scatter over the first kc columns -----

    kc = min(K, int(compact_width))

    map_head = map_fixed[:, :kc]  # (4P, kc)
    w_head = w_fixed[:, :kc]  # (4P, kc)

    outer = w_head[:, :, None] * w_head[:, None, :]  # (4P, kc, kc)
    outer_scaled = outer * block_scale[:, None, None]

    rows = map_head[:, :, None]  # (4P, kc, 1)
    cols = map_head[:, None, :]  # (4P, 1, kc)

    reg_mat = reg_mat.at[rows, cols].add(outer_scaled)

    # ----- Wide-row supplement (only when the tables are padded wider than kc) -----

    if K > kc:
        W = min(4 * P, int(wide_row_budget))

        # The widest rows, by their post-split occupied size. `top_k` on the integer sizes is not
        # differentiated through, so the weights stay fully differentiable.
        _, wide_rows = jax.lax.top_k(splitted_sizes, W)  # (W,)

        map_wide = map_fixed[wide_rows]  # (W, K)
        w_wide = w_fixed[wide_rows]  # (W, K)
        scale_wide = block_scale[wide_rows]  # (W,)

        outer_wide = w_wide[:, :, None] * w_wide[:, None, :]  # (W, K, K)
        outer_wide = outer_wide * scale_wide[:, None, None]

        # Zero the head x head block, which the compact pass above already scattered, leaving the
        # head x tail, tail x head and tail x tail blocks. Scattering the masked (W, K, K) block in
        # one `.at[].add` costs a single kernel launch, which on GPU beats three block scatters.
        col_index = jnp.arange(K)
        head_block = (col_index[:, None] < kc) & (col_index[None, :] < kc)  # (K, K)
        outer_wide = jnp.where(head_block[None, :, :], 0.0, outer_wide)

        rows_wide = map_wide[:, :, None]  # (W, K, 1)
        cols_wide = map_wide[:, None, :]  # (W, 1, K)

        reg_mat = reg_mat.at[rows_wide, cols_wide].add(outer_wide)

        # Overflow guard: more rows wider than the compact width than the supplement budget holds.
        overflow = jnp.sum(splitted_sizes > kc) > W
    else:
        overflow = None

    # Divide diagonal by 2
    reg_mat = reg_mat.at[jnp.diag_indices(reg_mat.shape[0])].add(-1e-8)

    if overflow is not None:
        # Poison the matrix on overflow, matching the Sibson cap convention (NaN weights -> NaN
        # likelihood -> the sample is discarded), instead of returning a silently truncated matrix.
        reg_mat = jnp.where(overflow, jnp.nan, reg_mat)

    return reg_mat
