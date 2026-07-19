import numpy as np

from autonerves import cached_property

from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay


def wendland_c4(r, h):

    import jax.numpy as jnp

    """
    Wendland C4: (1 - r/h)^6 * (35*(r/h)^2 + 18*r/h + 3)
    C4 continuous, smoother, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s) ** 6 * (35.0 * s**2 + 18.0 * s + 3.0), 0.0)
    return w


def get_interpolation_weights(
    points, query_points, k_neighbors, radius_scale, point_block=128
):
    import jax
    import jax.numpy as jnp

    points = jnp.asarray(points)
    query_points = jnp.asarray(query_points)

    M = int(query_points.shape[0])
    N = int(points.shape[0])

    if N == 0:
        raise ValueError("points has zero length; cannot compute kNN weights.")

    # Clamp k so top_k is valid even when N < k_neighbors
    k = int(k_neighbors)
    if k > N:
        k = N

    # Clamp block so dynamic_slice is always valid even when N < point_block
    B = int(point_block)
    if B > N:
        B = N

    # Precompute ||q||^2 once (M, 1)
    q2 = jnp.sum(query_points * query_points, axis=1, keepdims=True)

    # Running best: store NEGATIVE squared distances so we can use lax.top_k (largest)
    best_vals = -jnp.inf * jnp.ones((M, k), dtype=query_points.dtype)
    best_idx = jnp.zeros((M, k), dtype=jnp.int32)

    # How many blocks
    n_blocks = (N + B - 1) // B

    def body_fun(bi, carry):
        best_vals, best_idx = carry
        start = bi * B

        # How many valid points in this block (<= B)
        block_n = jnp.minimum(B, N - start)

        # Safe because B <= N by construction
        p_block = jax.lax.dynamic_slice(points, (start, 0), (B, points.shape[1]))

        # Mask out padded rows (only matters for last block)
        mask = jnp.arange(B) < block_n  # (B,)

        # dist_sq = ||q||^2 + ||p||^2 - 2 q·p
        p2 = jnp.sum(p_block * p_block, axis=1, keepdims=True).T  # (1, B)
        qp = query_points @ p_block.T  # (M, B)
        dist_sq = q2 + p2 - 2.0 * qp  # (M, B)
        dist_sq = jnp.maximum(dist_sq, 0.0)

        # Invalidate padded points
        dist_sq = jnp.where(mask[None, :], dist_sq, jnp.inf)

        vals = -dist_sq  # (M, B)

        # Indices for this block (M, B)
        idx_block = (start + jnp.arange(B, dtype=jnp.int32))[None, :]
        idx_block = jnp.broadcast_to(idx_block, (M, B))

        # Merge + top-k
        merged_vals = jnp.concatenate([best_vals, vals], axis=1)  # (M, k+B)
        merged_idx = jnp.concatenate([best_idx, idx_block], axis=1)

        new_vals, new_pos = jax.lax.top_k(merged_vals, k)
        new_idx = jnp.take_along_axis(merged_idx, new_pos, axis=1)

        return new_vals, new_idx

    best_vals, best_idx = jax.lax.fori_loop(
        0, n_blocks, body_fun, (best_vals, best_idx)
    )

    # Distances for selected k
    knn_dist_sq = -best_vals
    knn_distances = jnp.sqrt(knn_dist_sq + 1e-20)

    # Radius per query
    h = jnp.max(knn_distances, axis=1, keepdims=True) * radius_scale

    # Wendland weights + normalize
    weights = wendland_c4(knn_distances, h)
    weights_sum = jnp.sum(weights, axis=1, keepdims=True) + 1e-10
    weights_normalized = weights / weights_sum

    return best_idx, weights_normalized, knn_distances


def kernel_interpolate_points(points, query_chunk, values, k, radius_scale):
    """
    Compute kernel interpolation for a chunk of query points using K nearest neighbors.

    Args:
        query_chunk: (M, 2) query points
        points: (N, 2) source points
        values: (N,) values at source points
        k: number of nearest neighbors
        radius_scale: multiplier for radius

    Returns:
        (M,) interpolated values
    """

    import jax.numpy as jnp

    # Compute weights using the intermediate function
    top_k_indices, weights_normalized, _ = get_interpolation_weights(
        points,
        query_chunk,
        k,
        radius_scale,
    )

    # Get neighbor values
    neighbor_values = values[top_k_indices]  # (M, k)

    # Interpolate: weighted sum
    interpolated = jnp.sum(weights_normalized * neighbor_values, axis=1)  # (M,)

    return interpolated


class InterpolatorKNearestNeighbor(InterpolatorDelaunay):

    @cached_property
    def _mappings_sizes_weights(self):

        try:
            query_points = self.data_grid.over_sampled.array
        except AttributeError:
            try:
                query_points = self.data_grid.array
            except AttributeError:
                query_points = self.data_grid

        mappings, weights, _ = get_interpolation_weights(
            points=self.mesh_grid_xy,
            query_points=query_points,
            k_neighbors=self.mesh.k_neighbors,
            radius_scale=self.mesh.radius_scale,
        )

        mappings = self._xp.asarray(mappings)
        weights = self._xp.asarray(weights)

        sizes = self._xp.full(
            (mappings.shape[0],),
            mappings.shape[1],
        )

        return mappings, sizes, weights

    @cached_property
    def distance_to_self(self):

        _, _, distance_to_self = get_interpolation_weights(
            points=self.mesh_grid_xy,
            query_points=self.mesh_grid_xy,
            k_neighbors=self.mesh.k_neighbors,
            radius_scale=self.mesh.radius_scale,
        )

        return distance_to_self

    @cached_property
    def _mappings_sizes_weights_split(self):
        """
        kNN mappings + kernel weights computed at split points (for split regularization schemes),
        with split-point step sizes derived from kNN local spacing (no Delaunay / simplices).
        """
        from autoarray.inversion.regularization.regularization_util import (
            split_points_from,
        )

        neighbor_index = int(self.mesh.k_neighbors) // self.mesh.split_neighbor_division
        # e.g. k=10, division=2 -> neighbor_index=5

        distance_to_self = self.distance_to_self  # (N, k_neighbors), col 0 is self

        others = distance_to_self[:, 1:]  # (N, k_neighbors-1)

        # Clamp to valid range (0-based indexing into `others`)
        idx = int(neighbor_index) - 1
        idx = max(0, min(idx, others.shape[1] - 1))

        r_k = others[:, idx]  # (N,)

        # Split cross step size (length): sqrt(area) ~ r_k
        split_step = self.mesh.areas_factor * r_k  # (N,)

        # Split points (xp-native)
        split_points = split_points_from(
            points=self.mesh_grid.array,
            area_weights=split_step,
            xp=self._xp,
        )

        interpolator = InterpolatorKNearestNeighbor(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=split_points,
            xp=self._xp,
        )

        mappings = interpolator.mappings
        weights = interpolator.weights

        sizes = self._xp.full(
            (mappings.shape[0],),
            mappings.shape[1],
        )

        return mappings, sizes, weights

    # def interpolate(self, query_points, points, values):
    #     return kernel_interpolate_points(
    #         points=self.mesh_grid_xy,
    #         query_points=self.data_grid.over_sampled,
    #         values,
    #         k=self.mesh.k_neighbors,
    #         radius_scale=self.mesh.radius_scale,
    #     )


def barycentric_weights_from_3_nearest(
    query_points,
    mesh_points,
    nearest_3_indices,
    xp,
):
    """
    Compute barycentric weights for each query point on the triangle formed by its
    3 nearest mesh vertices.

    Signed barycentric coordinates are computed, then clipped to be non-negative
    and renormalized so each row sums to 1. Queries inside the triangle return
    the exact Delaunay weights; queries outside return a clipped approximation
    (a convex combination of the 3 nearest, biased toward whichever vertices are
    on the same side of the triangle as the query).

    Degenerate triangles (collinear vertices) get zero weights to avoid NaN.

    Parameters
    ----------
    query_points : (Q, 2)
        Query point (x, y) coordinates.
    mesh_points : (N, 2)
        Mesh vertex (x, y) coordinates.
    nearest_3_indices : (Q, 3)
        Indices into mesh_points of the 3 nearest vertices for each query.
    xp : module
        numpy or jax.numpy.

    Returns
    -------
    weights : (Q, 3)
        Barycentric weights, clipped non-negative and row-normalized.
    """
    vertices = mesh_points[nearest_3_indices]  # (Q, 3, 2)
    p0 = vertices[:, 0]
    p1 = vertices[:, 1]
    p2 = vertices[:, 2]
    q = query_points

    def signed_cross(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
            b[..., 1] - a[..., 1]
        ) * (c[..., 0] - a[..., 0])

    total = signed_cross(p0, p1, p2)
    w0 = signed_cross(q, p1, p2)
    w1 = signed_cross(p0, q, p2)
    w2 = signed_cross(p0, p1, q)

    eps = xp.asarray(1e-12, dtype=total.dtype)
    safe_total = xp.where(xp.abs(total) > eps, total, 1.0)

    bary = xp.stack([w0, w1, w2], axis=1) / safe_total[:, None]

    clipped = xp.maximum(bary, 0.0)
    row_sum = xp.sum(clipped, axis=1, keepdims=True)
    safe_sum = xp.where(row_sum > eps, row_sum, 1.0)
    weights = clipped / safe_sum

    # Degenerate triangles fall back to nearest-neighbor (weight 1 on column 0,
    # which `get_interpolation_weights` orders as the closest mesh vertex).
    # Same fallback policy as `pix_indexes_for_sub_slim_index_delaunay_from`
    # for outside-simplex points.
    nearest_only = xp.asarray([1.0, 0.0, 0.0], dtype=weights.dtype)

    degenerate = xp.abs(total) <= eps
    weights = xp.where(degenerate[:, None], nearest_only[None, :], weights)

    return weights


class InterpolatorKNNBarycentric(InterpolatorKNearestNeighbor):
    """
    Interpolator that picks the 3 nearest mesh vertices in the source plane and
    computes locally-exact barycentric weights on the triangle they form.

    Approximates :class:`InterpolatorDelaunay` without the scipy.spatial.Delaunay
    callback: when the 3 nearest are the containing Delaunay triangle's vertices,
    the weights are bit-identical to Delaunay; otherwise they are clipped-and-
    renormalized barycentric weights on whichever triangle the 3 nearest form.

    The kNN connectivity knobs (``k_neighbors``, ``radius_scale``,
    ``split_neighbor_division``) on the parent :class:`KNearestNeighbor` mesh are
    inherited and still control the regularization-spacing computation via
    ``distance_to_self``. Interpolation always uses k=3, irrespective of
    ``mesh.k_neighbors``.
    """

    @cached_property
    def _mappings_sizes_weights(self):

        try:
            query_points = self.data_grid.over_sampled.array
        except AttributeError:
            try:
                query_points = self.data_grid.array
            except AttributeError:
                query_points = self.data_grid

        mappings, _, _ = get_interpolation_weights(
            points=self.mesh_grid_xy,
            query_points=query_points,
            k_neighbors=3,
            radius_scale=1.0,
        )

        weights = barycentric_weights_from_3_nearest(
            query_points=query_points,
            mesh_points=self.mesh_grid_xy,
            nearest_3_indices=mappings,
            xp=self._xp,
        )

        # On the numpy path, materialize with `np.array(...)` so the regularization
        # code (which uses in-place assignment, e.g. `reg_split_np_from`) gets a
        # writable buffer rather than a read-only view of a jax.Array. On the jax
        # path, asarray is the right cast (no copy in a JIT trace).
        if self._xp is np:
            mappings = np.array(mappings)
            weights = np.array(weights)
        else:
            mappings = self._xp.asarray(mappings)
            weights = self._xp.asarray(weights)

        sizes = self._xp.full(
            (mappings.shape[0],),
            mappings.shape[1],
        )

        return mappings, sizes, weights

    @cached_property
    def _mappings_sizes_weights_split(self):
        """
        Same spacing scheme as :class:`InterpolatorKNearestNeighbor` but the
        split-point interpolator is :class:`InterpolatorKNNBarycentric` so the
        split-regularization weights are also barycentric rather than Wendland.
        """
        from autoarray.inversion.regularization.regularization_util import (
            split_points_from,
        )

        neighbor_index = int(self.mesh.k_neighbors) // self.mesh.split_neighbor_division

        distance_to_self = self.distance_to_self
        others = distance_to_self[:, 1:]
        idx = int(neighbor_index) - 1
        idx = max(0, min(idx, others.shape[1] - 1))
        r_k = others[:, idx]

        split_step = self.mesh.areas_factor * r_k

        split_points = split_points_from(
            points=self.mesh_grid.array,
            area_weights=split_step,
            xp=self._xp,
        )

        interpolator = InterpolatorKNNBarycentric(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=split_points,
            xp=self._xp,
        )

        mappings = interpolator.mappings
        weights = interpolator.weights

        # `reg_split_np_from` writes `splitted_mappings[i][j+1] = pixel_index`
        # for the "flag-zero" insertion of the central pixel, so the buffer
        # must have an extra column reserved past the k=3 mappings — matching
        # `InterpolatorDelaunay._mappings_sizes_weights_split`'s hstack-append.
        # `sizes` reports 3 (the actual mappings); `reg_split_np_from` grows it
        # to 4 in-place when it inserts.
        sizes = self._xp.full(
            (mappings.shape[0],),
            mappings.shape[1],
        )

        pad_int = self._xp.full((mappings.shape[0], 1), -1, dtype=mappings.dtype)
        pad_float = self._xp.zeros((weights.shape[0], 1), dtype=weights.dtype)
        mappings = self._xp.hstack((mappings, pad_int))
        weights = self._xp.hstack((weights, pad_float))

        return mappings, sizes, weights
