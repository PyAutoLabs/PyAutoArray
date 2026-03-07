import numpy as np
from functools import partial

from autoconf import cached_property

from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator


def forward_interp(xp, yp, x):

    import jax
    import jax.numpy as jnp

    return jax.vmap(jnp.interp, in_axes=(1, 1, 1, None, None), out_axes=(1))(
        x, xp, yp, 0, 1
    )


def reverse_interp(xp, yp, x):
    import jax
    import jax.numpy as jnp

    return jax.vmap(jnp.interp, in_axes=(1, 1, 1), out_axes=(1))(x, xp, yp)


def forward_interp_np(xp, yp, x):
    """
    xp: (N, M)
    yp: (N, M)
    x : (M,)  ← one x per column
    """

    if yp.ndim == 1 and xp.ndim == 2:
        yp = np.broadcast_to(yp[:, None], xp.shape)

    K, M = x.shape

    out = np.empty((K, 2), dtype=xp.dtype)

    for j in range(2):
        out[:, j] = np.interp(x[:, j], xp[:, j], yp[:, j], left=0, right=1)

    return out


def reverse_interp_np(xp, yp, x):
    """
    xp : (N,) or (N, M)
    yp : (N, M)
    x  : (K, M)   query points per column
    """

    # Ensure xp is 2D: (N, M)
    if xp.ndim == 1 and yp.ndim == 2:  # (N, 1)
        xp = np.broadcast_to(xp[:, None], yp.shape)

    # Shapes
    K, M = x.shape

    # Output
    out = np.empty((K, 2), dtype=yp.dtype)

    # Column-wise interpolation (cannot avoid this loop in pure NumPy)
    for j in range(2):
        out[:, j] = np.interp(x[:, j], xp[:, j], yp[:, j])

    return out


def create_transforms(traced_points, mesh_weight_map=None, xp=np):

    N = traced_points.shape[0]  # // 2

    if mesh_weight_map is None:
        t = xp.arange(1, N + 1) / (N + 1)
        t = xp.stack([t, t], axis=1)
        sort_points = xp.sort(traced_points, axis=0)  # [::2]
    else:
        sdx = xp.argsort(traced_points, axis=0)
        sort_points = xp.take_along_axis(traced_points, sdx, axis=0)
        t = xp.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = xp.take_along_axis(t, sdx, axis=0)
        t = xp.cumsum(t, axis=0)

    if xp.__name__.startswith("jax"):
        transform = partial(forward_interp, sort_points, t)
        inv_transform = partial(reverse_interp, t, sort_points)
        return transform, inv_transform

    transform = partial(forward_interp_np, sort_points, t)
    inv_transform = partial(reverse_interp_np, t, sort_points)
    return transform, inv_transform


def adaptive_rectangular_transformed_grid_from(
    data_grid, grid, mesh_weight_map=None, xp=np
):

    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    return inv_full(grid)


def adaptive_rectangular_mappings_weights_via_interpolation_from(
    source_grid_size: int,
    data_grid,
    data_grid_over_sampled,
    mesh_weight_map=None,
    xp=np,
):
    """
    Compute bilinear interpolation indices and weights for mapping an oversampled
    source-plane grid onto a regular rectangular pixelization.

    This function takes a set of irregularly-sampled source-plane coordinates and
    builds an adaptive mapping onto a `source_grid_size x source_grid_size` rectangular
    pixelization using bilinear interpolation. The interpolation is expressed as:

        f(x, y) ≈ w_bl * f(ix_down, iy_down) +
                  w_br * f(ix_up,   iy_down) +
                  w_tl * f(ix_down, iy_up) +
                  w_tr * f(ix_up,   iy_up)

    where `(ix_down, ix_up, iy_down, iy_up)` are the integer grid coordinates
    surrounding the continuous position `(x, y)`.

    Steps performed:
      1. Normalize the source-plane grid by subtracting its mean and dividing by
         the minimum axis standard deviation (to balance scaling).
      2. Construct forward/inverse transforms which map the grid into the unit square [0,1]^2.
      3. Transform the oversampled source-plane grid into [0,1]^2, then scale it
         to index space `[0, source_grid_size)`.
      4. Compute floor/ceil along x and y axes to find the enclosing rectangular cell.
      5. Build the four corner indices: bottom-left (bl), bottom-right (br),
         top-left (tl), and top-right (tr).
      6. Flatten the 2D indices into 1D indices suitable for scatter operations,
         with a flipped row-major convention: row = source_grid_size - i, col = j.
      7. Compute bilinear interpolation weights (`w_bl, w_br, w_tl, w_tr`).
      8. Return arrays of flattened indices and weights of shape `(N, 4)`, where
         `N` is the number of oversampled coordinates.

    Parameters
    ----------
    source_grid_size : int
        The number of pixels along one dimension of the rectangular pixelization.
        The grid is square: (source_grid_size x source_grid_size).
    data_grid : (M, 2) ndarray
        The base source-plane coordinates, used to define normalization and transforms.
    data_grid_over_sampled : (N, 2) ndarray
        Oversampled source-plane coordinates to be interpolated onto the rectangular grid.
    mesh_weight_map
        The weight map used to weight the creation of the rectangular mesh grid, which is used for the
        `RectangularBrightness` mesh which adapts the size of its pixels to where the source is reconstructed.

    Returns
    -------
    flat_indices : (N, 4) int ndarray
        The flattened indices of the four neighboring pixel corners for each oversampled point.
        Order: [bl, br, tl, tr].
    weights : (N, 4) float ndarray
        The bilinear interpolation weights for each of the four neighboring pixels.
        Order: [w_bl, w_br, w_tl, w_tr].
    """

    # --- Step 1. Normalize grid ---
    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    # --- Step 2. Build transforms ---
    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    # --- Step 3. Transform oversampled grid into index space ---
    grid_over_sampled_scaled = (data_grid_over_sampled - mu) / scale
    grid_over_sampled_transformed = transform(grid_over_sampled_scaled)
    grid_over_index = (source_grid_size - 3) * grid_over_sampled_transformed + 1

    # --- Step 4. Floor/ceil indices ---
    ix_down = xp.floor(grid_over_index[:, 0])
    ix_up = xp.ceil(grid_over_index[:, 0])
    iy_down = xp.floor(grid_over_index[:, 1])
    iy_up = xp.ceil(grid_over_index[:, 1])

    # --- Step 5. Four corners ---
    idx_tl = xp.stack([ix_up, iy_down], axis=1)
    idx_tr = xp.stack([ix_up, iy_up], axis=1)
    idx_br = xp.stack([ix_down, iy_up], axis=1)
    idx_bl = xp.stack([ix_down, iy_down], axis=1)

    # --- Step 6. Flatten indices ---
    def flatten(idx, n):
        row = n - idx[:, 0]
        col = idx[:, 1]
        return row * n + col

    flat_tl = flatten(idx_tl, source_grid_size)
    flat_tr = flatten(idx_tr, source_grid_size)
    flat_bl = flatten(idx_bl, source_grid_size)
    flat_br = flatten(idx_br, source_grid_size)

    flat_indices = xp.stack([flat_tl, flat_tr, flat_bl, flat_br], axis=1).astype(
        "int64"
    )

    # --- Step 7. Bilinear interpolation weights ---
    t_row = (grid_over_index[:, 0] - ix_down) / (ix_up - ix_down + 1e-12)
    t_col = (grid_over_index[:, 1] - iy_down) / (iy_up - iy_down + 1e-12)

    # Weights
    w_tl = (1 - t_row) * (1 - t_col)
    w_tr = (1 - t_row) * t_col
    w_bl = t_row * (1 - t_col)
    w_br = t_row * t_col
    weights = xp.stack([w_tl, w_tr, w_bl, w_br], axis=1)

    return flat_indices, weights


class InterpolatorRectangular(AbstractInterpolator):

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """
        A grid of (y,x) coordinates which represent a uniform rectangular pixelization.

        A `InterpolatorRectangular` is ordered such pixels begin from the top-row and go rightwards and then downwards.
        It is an ndarray of shape [total_pixels, 2], where the first dimension of the ndarray corresponds to the
        pixelization's pixel index and second element whether it is a y or x arc-second coordinate.

        For example:

        - grid[3,0] = the y-coordinate of the 4th pixel in the rectangular pixelization.
        - grid[6,1] = the x-coordinate of the 7th pixel in the rectangular pixelization.

        This class is used in conjuction with the `inversion/pixelizations` package to create rectangular pixelizations
        and mappers that perform an `Inversion`.

        Parameters
        ----------
        values
            The grid of (y,x) coordinates corresponding to the centres of each pixel in the rectangular pixelization.
        shape_native
            The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) origin of the pixelization.
        """
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            adapt_data=adapt_data,
            xp=xp,
        )
        self.mesh_weight_map = mesh_weight_map

    @cached_property
    def mesh_geometry(self):

        from autoarray.inversion.mesh.mesh_geometry.rectangular import (
            MeshGeometryRectangular,
        )

        return MeshGeometryRectangular(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=self.data_grid,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )

    @cached_property
    def _mappings_sizes_weights(self):

        mappings, weights = (
            adaptive_rectangular_mappings_weights_via_interpolation_from(
                source_grid_size=self.mesh.shape[0],
                data_grid=self.data_grid.array,
                data_grid_over_sampled=self.data_grid.over_sampled.array,
                mesh_weight_map=self.mesh_weight_map,
                xp=self._xp,
            )
        )

        sizes = 4 * self._xp.ones(len(mappings), dtype="int")

        return mappings, sizes, weights
