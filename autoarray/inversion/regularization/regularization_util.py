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
    splitted_mappings: np.ndarray,  # (4P, 4)
    splitted_sizes: np.ndarray,  # (4P,)
    splitted_weights: np.ndarray,  # (4P, 4)
    xp=np,
):
    """
    Returns the regularization matrix for the adaptive split-pixel regularization scheme.

    This scheme splits each source pixel into a cross of four regularization points and interpolates
    to those points to smooth the inversion solution. It is designed to mitigate stochasticity in
    the regularization that can arise when the number of neighboring pixels varies across a
    mesh (e.g., in a Delaunay tessellation).

    A visual description and further details are provided in the appendix of He et al. (2024):
    https://arxiv.org/abs/2403.16253

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

    import jax.numpy as jnp

    # How many real pixels?
    P = splitted_mappings.shape[0] // 4

    # Square, positive regularization weights
    reg_w = regularization_weights**2.0  # (P,)

    # Add diagonal jitter (2e-8)
    reg_mat = jnp.eye(P) * 2e-8  # (P, P)

    # ----- Build all 4P contributions at once -----

    # Mask away padded entries (where mapping = -1)
    valid = splitted_mappings != -1  # (4P, 4)

    # Extract valid mapping rows and weights
    # BUT keep fixed shape (4) and just zero out invalid ones
    map_fixed = jnp.where(valid, splitted_mappings, 0)  # (4P, 4)
    w_fixed = jnp.where(valid, splitted_weights, 0.0)  # (4P, 4)

    # Compute all outer products of weights
    # w_fixed[:, :, None] * w_fixed[:, None, :]  → (4P, 4, 4)
    outer = w_fixed[:, :, None] * w_fixed[:, None, :]  # (4P, 4, 4)

    # Build corresponding row and col index grids
    rows = map_fixed[:, :, None]  # (4P, 4, 1)
    cols = map_fixed[:, None, :]  # (4P, 1, 4)

    # Multiply each 4x4 block by its pixel’s regularization weight
    # Rows 0–3 belong to pixel 0, rows 4–7 to pixel 1, etc.
    pixel_index = jnp.arange(4 * P) // 4  # (4P,)
    block_scale = reg_w[pixel_index]  # (4P,)
    outer_scaled = outer * block_scale[:, None, None]

    # Now scatter-add all entries into the (P,P) matrix
    reg_mat = reg_mat.at[rows, cols].add(outer_scaled)

    # Divide diagonal by 2
    reg_mat = reg_mat.at[jnp.diag_indices(reg_mat.shape[0])].add(-1e-8)

    return reg_mat
