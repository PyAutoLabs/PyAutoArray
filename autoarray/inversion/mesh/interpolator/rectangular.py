"""
The adaptive rectangular interpolator, built on a per-axis CDF transform of
the traced source-plane coordinates so that the uniform mesh pixels adapt to
the density (or weight) of the traced points. Two transforms are provided,
selected by the ``transform`` argument threaded through this module:

**Empirical rank CDF** (``transform="rank"`` — the ``RectangularBilinear*``
meshes). The per-axis CDF is the empirical (weighted) rank CDF of the traced
points: a sort plus a cumulative sum, linearly interpolated between points.
O(N log N) per likelihood evaluation with no kernel hyperparameters — the
fast, conceptually simple CPU default. The trade-off is smoothness: the
transform is piecewise-linear in the queries and its dependence on the
traced-point *positions* passes through ranks, so with pixelization
over-sampling 1 (and on the interferometer sparse path, which has no
over-sampling) the likelihood is exactly piecewise-constant in mass/shear —
gradients are identically zero. Gradient (JAX sampler) users need
``over_sample_size_pixelization >= 4`` on imaging, or the kernel-CDF meshes.

**Kernel-density CDF** (``transform="kernel"`` — the ``RectangularRTU*``
meshes, the RTU formulation of Enzi et al., arXiv:2606.30620): per axis

    F(x) = sum_i w_i * Phi((x - x_i) / h)

where ``Phi`` is the standard normal CDF, ``x_i`` the traced points, ``w_i``
uniform weights (density adaption) or the normalized adapt-image weights
(image adaption), and ``h`` a bandwidth tied to the mesh resolution.

Properties of the kernel CDF, by construction:

- strictly monotone;
- C-infinity in the queries AND the traced-point positions — no ranks, no
  sorts, no ``argsort`` anywhere, so there is nothing to swap and the
  likelihood carries smooth mass/shear gradients in every configuration
  (including pixelization over-sampling 1 and the interferometer sparse
  path, where the empirical rank CDF is exactly piecewise-constant);
- duplicate-safe: coincident traced points simply stack their weights.

The kernel forward transform is evaluated exactly at the query points
(keeping the C-infinity guarantee) and rescaled so the data bounding box
maps onto the unit square exactly, clamping to [0, 1] outside the data
range.  The inverse — only needed at fixed unit-square grid values (mesh
pixel centres/edges) — is a linear-interp lookup on a small fixed table of
``n_knots`` knots spanning the data range; because the inverse queries are
constants, gradients flow smoothly through the table values. The exact
kernel sum is the O(M_sub x N_data) erf evaluation that dominates the CPU
likelihood at production scale — the reason the rank CDF is the CPU default.

Both transforms warp the same lattice; the 4-pixel bilinear interpolation of
``adaptive_rectangular_mappings_weights_via_interpolation_from`` is shared.
"""

import math

import numpy as np
from functools import partial
from typing import Optional

from autonerves import cached_property

from autoarray import numba_util

from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator


def forward_interp(xp, yp, x):
    import jax
    import jax.numpy as jnp

    return jax.vmap(jnp.interp, in_axes=(1, 1, 1, None, None), out_axes=(1))(
        x, xp, yp, 0, 1
    )


def forward_interp_np(xp, yp, x):
    """
    xp: (N, M)
    yp: (N, M)
    x : (K, M)  ← query points per column
    """

    if yp.ndim == 1 and xp.ndim == 2:
        yp = np.broadcast_to(yp[:, None], xp.shape)

    K, M = x.shape

    out = np.empty((K, 2), dtype=xp.dtype)

    for j in range(2):
        out[:, j] = np.interp(x[:, j], xp[:, j], yp[:, j], left=0, right=1)

    return out


def reverse_interp(xp, yp, x):
    import jax
    import jax.numpy as jnp

    return jax.vmap(jnp.interp, in_axes=(1, 1, 1), out_axes=(1))(x, xp, yp)


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


KERNEL_CDF_DEFAULT_BANDWIDTH: float = 1.0
KERNEL_CDF_DEFAULT_KNOTS: int = 64

# Queries per block of the forward-transform evaluation. The exact kernel sum
# broadcasts an (M, N, 2) array; blocking the query dimension caps the peak at
# (block, N, 2) — 512 queries × 15.4k points × 2 axes × 8 B ≈ 126 MB at
# production imaging scale (previously ~60 GB unblocked at M ≈ 246k). Values
# are identical to float precision: the sum over points is unchanged, blocks
# only tile the query axis.
KERNEL_FORWARD_BLOCK: int = 512

_SQRT2 = np.sqrt(2.0)

# Phi(t) saturates to exactly-representable 0/1 contributions in fp64 well
# inside |t| = 9: the dropped tail terms are < 1e-19 of the weight sum, far
# below the ~1e-13 accumulation noise of the blocked numpy sum the windowed
# numba kernel replaces (measured max deviation 1e-13 on the hst fiducial).
_KERNEL_CDF_SATURATION_T = 9.0


@numba_util.jit()
def _kernel_cdf_dim_windowed(p_sorted, w_sorted, w_prefix, h_d, q, T):
    """
    Exact 1D weighted kernel CDF ``F(q) = sum_i w_i Phi((q - p_i) / h)`` for
    one dimension, evaluated with a saturation window over sorted points.

    Points below ``q - T h`` contribute exactly their weight (prefix sum);
    points above ``q + T h`` contribute zero; only the window is summed with
    ``erfc``. Replaces the O(M x N) blocked numpy broadcast on the numpy path
    — same values to ~1e-13 (see ``_KERNEL_CDF_SATURATION_T``) at ~3x the
    speed and none of the ~126 MB per-block temporaries; the blocked numpy
    implementation remains the JAX-path/differentiable reference.

    Sorting note: the module docstring's "no sorts anywhere" invariant is a
    JAX-differentiability guarantee. This kernel runs only on the ``xp is
    np`` branch, which carries no gradients — the sort is an internal
    evaluation order and the returned VALUES are those of the sort-free sum.
    """
    out = np.empty(q.shape[0])
    inv = 1.0 / (h_d * 1.4142135623730951)
    for m in range(q.shape[0]):
        qm = q[m]
        a = np.searchsorted(p_sorted, qm - T * h_d)
        b = np.searchsorted(p_sorted, qm + T * h_d)
        acc = w_prefix[a]
        for i in range(a, b):
            acc += w_sorted[i] * 0.5 * math.erfc((p_sorted[i] - qm) * inv)
        out[m] = acc
    return out


def _norm_cdf(t, xp):
    """Standard normal CDF, xp-aware (scipy erf on numpy, jax.scipy under jax)."""
    if xp is np:
        from scipy.special import erf
    else:
        from jax.scipy.special import erf

    return 0.5 * (1.0 + erf(t / _SQRT2))


def create_transforms_rank(traced_points, mesh_weight_map=None, xp=np):
    """
    Build the per-axis empirical rank-CDF transform pair (the ``Bilinear``
    meshes' lattice transform).

    The CDF is the empirical (weighted) rank CDF of the traced points — a
    sort plus a cumulative sum, linearly interpolated between points — so a
    likelihood evaluation costs O(N log N) with no kernel hyperparameters.
    Recovered from the pre-consolidation implementation (PR #402 deleted it
    when the kernel CDF took the plain class names); see the module
    docstring for the gradient trade-off versus the kernel CDF.

    The returned ``transform`` maps (scaled) source-plane coordinates into
    the unit square and ``inv_transform`` maps unit-square coordinates back
    to the (scaled) source plane.

    Parameters
    ----------
    traced_points
        The (N, 2) scaled source-plane coordinates the CDF adapts to.
    mesh_weight_map
        Optional (N,) weights from the adapt image (image adaption). ``None``
        gives uniform weights (density adaption).
    xp
        The array library to use (numpy or jax.numpy).
    """
    N = traced_points.shape[0]

    if mesh_weight_map is None:
        t = xp.arange(1, N + 1) / (N + 1)
        t = xp.stack([t, t], axis=1)
        sort_points = xp.sort(traced_points, axis=0)
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


def create_transforms(
    traced_points,
    mesh_pixels: int,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    xp=np,
):
    """
    Build the per-axis kernel-density CDF transform pair.

    The returned ``transform`` maps (scaled) source-plane coordinates into the
    unit square and ``inv_transform`` maps unit-square coordinates back to the
    (scaled) source plane.

    Parameters
    ----------
    traced_points
        The (N, 2) scaled source-plane coordinates the CDF adapts to.
    mesh_pixels
        Mesh pixels per axis; sets the default bandwidth scale
        ``h = bandwidth * data_span / mesh_pixels`` per axis.
    mesh_weight_map
        Optional (N,) weights from the adapt image (image adaption). ``None``
        gives uniform weights (density adaption).
    bandwidth
        Bandwidth in units of the mesh pixel scale (data span / mesh_pixels).
        Smaller values track the point density more sharply (approaching the
        empirical CDF and its staircase); larger values smooth the mesh
        geometry towards uniform.
    n_knots
        Size of the fixed knot table used to invert the CDF.
    xp
        The array library to use (numpy or jax.numpy).
    """
    points = traced_points
    N = points.shape[0]

    if mesh_weight_map is None:
        w = xp.full((N,), 1.0 / N)
    else:
        w = mesh_weight_map / xp.sum(mesh_weight_map)

    lo = xp.min(points, axis=0)
    hi = xp.max(points, axis=0)
    span = hi - lo
    h = bandwidth * span / mesh_pixels

    def F_raw_block(q_block):
        t = (q_block[:, None, :] - points[None, :, :]) / h[None, None, :]
        return xp.sum(w[None, :, None] * _norm_cdf(t, xp), axis=1)

    if xp.__name__.startswith("jax"):

        def F_raw(q):
            import jax

            M = q.shape[0]
            n_blocks = -(-M // KERNEL_FORWARD_BLOCK)
            pad = n_blocks * KERNEL_FORWARD_BLOCK - M
            q_padded = xp.pad(q, ((0, pad), (0, 0)))
            blocks = q_padded.reshape(n_blocks, KERNEL_FORWARD_BLOCK, 2)
            out = jax.lax.map(F_raw_block, blocks)
            return out.reshape(n_blocks * KERNEL_FORWARD_BLOCK, 2)[:M]

    else:
        # numpy fast path: per-dimension sorted points + weight prefix sums,
        # evaluated by the windowed numba kernel. Same values as the blocked
        # broadcast above to ~1e-13; that implementation stays as the JAX
        # branch and the differentiable reference.
        _p_sorted = []
        _w_sorted = []
        _w_prefix = []
        for _d in range(2):
            _order = np.argsort(points[:, _d], kind="stable")
            _p_sorted.append(np.ascontiguousarray(np.asarray(points)[_order, _d]))
            _w_sorted.append(np.ascontiguousarray(np.asarray(w)[_order]))
            _w_prefix.append(np.concatenate([[0.0], np.cumsum(_w_sorted[_d])]))

        def F_raw(q):
            q = np.asarray(q)
            out = np.empty_like(q)
            for d in range(2):
                out[:, d] = _kernel_cdf_dim_windowed(
                    _p_sorted[d],
                    _w_sorted[d],
                    _w_prefix[d],
                    float(h[d]),
                    np.ascontiguousarray(q[:, d]),
                    _KERNEL_CDF_SATURATION_T,
                )
            return out

    # The unit square maps onto the data bounding box exactly (the kernel
    # tails outside [lo, hi] are absorbed by the rescale).
    u = xp.linspace(0.0, 1.0, n_knots)
    knots = lo[None, :] + u[:, None] * span[None, :]

    F_knots_raw = F_raw(knots)
    F_lo = F_knots_raw[0]
    F_hi = F_knots_raw[-1]
    denom = F_hi - F_lo

    F_knots = (F_knots_raw - F_lo[None, :]) / denom[None, :]

    def transform(q):
        F_q = (F_raw(q) - F_lo[None, :]) / denom[None, :]
        return xp.clip(F_q, 0.0, 1.0)

    if xp.__name__.startswith("jax"):
        inv_transform = partial(reverse_interp, F_knots, knots)
        return transform, inv_transform

    inv_transform = partial(reverse_interp_np, F_knots, knots)
    return transform, inv_transform


def _transforms_from(
    source_grid_scaled,
    mesh_pixels: int,
    mesh_weight_map,
    bandwidth: float,
    n_knots: int,
    transform: str,
    xp,
):
    """
    Dispatch to the requested lattice-transform pair: the empirical rank CDF
    (``"rank"``, the Bilinear meshes — ``bandwidth`` / ``n_knots`` unused) or
    the kernel-density CDF (``"kernel"``, the RTU meshes).
    """
    if transform == "rank":
        return create_transforms_rank(
            source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
        )
    if transform != "kernel":
        raise ValueError(f"transform must be 'rank' or 'kernel', got {transform!r}.")
    return create_transforms(
        source_grid_scaled,
        mesh_pixels=mesh_pixels,
        mesh_weight_map=mesh_weight_map,
        bandwidth=bandwidth,
        n_knots=n_knots,
        xp=xp,
    )


def adaptive_rectangular_transformed_grid_from(
    data_grid,
    grid,
    mesh_pixels: int,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    transform: str = "kernel",
    xp=np,
):
    """Map unit-square coordinates back to the source plane via the adaptive CDF."""
    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    _, inv_transform = _transforms_from(
        source_grid_scaled,
        mesh_pixels=mesh_pixels,
        mesh_weight_map=mesh_weight_map,
        bandwidth=bandwidth,
        n_knots=n_knots,
        transform=transform,
        xp=xp,
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    return inv_full(grid)


def adaptive_rectangular_areas_from(
    source_grid_shape,
    data_grid,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    transform: str = "kernel",
    xp=np,
):
    """The source-plane area of every mesh pixel under the adaptive CDF transform."""
    edges_y = xp.linspace(1, 0, source_grid_shape[0] + 1)
    edges_x = xp.linspace(0, 1, source_grid_shape[1] + 1)

    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    _, inv_transform = _transforms_from(
        source_grid_scaled,
        mesh_pixels=source_grid_shape[0],
        mesh_weight_map=mesh_weight_map,
        bandwidth=bandwidth,
        n_knots=n_knots,
        transform=transform,
        xp=xp,
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    pixel_edges = inv_full(xp.stack([edges_y, edges_x]).T)
    pixel_lengths = xp.diff(pixel_edges, axis=0).squeeze()

    dy = pixel_lengths[:, 0]
    dx = pixel_lengths[:, 1]

    return xp.abs(xp.outer(dy, dx).flatten())


def adaptive_rectangular_mappings_weights_via_interpolation_from(
    source_grid_size: int,
    data_grid,
    data_grid_over_sampled,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    transform: str = "kernel",
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
      2. Construct the adaptive-CDF forward/inverse transforms (rank or
         kernel, per ``transform``) which map the grid into the unit square
         [0,1]^2.
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
        The weight map used to weight the creation of the rectangular mesh grid, which
        adapts the size of the mesh pixels to where the source is reconstructed.
    bandwidth
        Bandwidth in units of the mesh pixel scale (see ``create_transforms``;
        kernel transform only).
    n_knots
        Size of the fixed knot table used to invert the CDF (kernel transform
        only).
    transform
        The lattice transform: ``"rank"`` (empirical rank CDF, the Bilinear
        meshes) or ``"kernel"`` (kernel-density CDF, the RTU meshes).

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
    transform_func, _ = _transforms_from(
        source_grid_scaled,
        mesh_pixels=source_grid_size,
        mesh_weight_map=mesh_weight_map,
        bandwidth=bandwidth,
        n_knots=n_knots,
        transform=transform,
        xp=xp,
    )

    # --- Step 3. Transform oversampled grid into index space ---
    grid_over_sampled_scaled = (data_grid_over_sampled - mu) / scale
    grid_over_sampled_transformed = transform_func(grid_over_sampled_scaled)
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
        return (n - idx[:, 0]) * n + idx[:, 1]

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

    w_tl = (1 - t_row) * (1 - t_col)
    w_tr = (1 - t_row) * t_col
    w_bl = t_row * (1 - t_col)
    w_br = t_row * t_col
    weights = xp.stack([w_tl, w_tr, w_bl, w_br], axis=1)

    return flat_indices, weights


class InterpolatorRectangular(AbstractInterpolator):
    """Adaptive rectangular interpolator (rank or kernel-density CDF).

    A grid of (y,x) coordinates which represent an adaptive rectangular
    pixelization, ordered such that pixels begin from the top-row and go
    rightwards and then downwards. The mesh pixels adapt to the density (or
    adapt-image weights) of the traced points through the per-axis CDF
    transform selected by ``transform``: the empirical rank CDF of
    ``create_transforms_rank`` (the Bilinear meshes) or the smooth kernel
    CDF of ``create_transforms`` (the RTU meshes).

    This class is used in conjunction with the `inversion/pixelizations`
    package to create rectangular pixelizations and mappers that perform an
    `Inversion`.
    """

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map,
        adapt_data: Optional[np.ndarray] = None,
        bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
        n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
        transform: str = "kernel",
        xp=np,
    ):
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            adapt_data=adapt_data,
            xp=xp,
        )
        self.mesh_weight_map = mesh_weight_map
        self.bandwidth = bandwidth
        self.n_knots = n_knots
        self.transform = transform

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
            kernel_bandwidth=self.bandwidth,
            kernel_knots=self.n_knots,
            transform=self.transform,
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
                bandwidth=self.bandwidth,
                n_knots=self.n_knots,
                transform=self.transform,
                xp=self._xp,
            )
        )

        sizes = 4 * self._xp.ones(len(mappings), dtype="int")

        return mappings, sizes, weights

    # NOTE: `_mappings_sizes_weights_split` is deliberately NOT implemented here.
    #
    # It previously returned `self._mappings_sizes_weights` unchanged, on the reasoning that
    # bilinear interpolation already factors in the 4-corner neighbourhood. That was not correct:
    # `reg_split_from` expects the split-cross structure, so the pass-through raised
    # `IndexError: index 4 is out of bounds for axis 0 with size 4` one frame later.
    #
    # Rectangular meshes do not support split regularization. The combination is now rejected at
    # `Pixelization` construction via `AbstractMesh.supports_split_regularization`, which names
    # both the mesh and the regularization instead of failing deep inside the inversion.
