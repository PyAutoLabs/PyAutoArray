from autoconf import cached_property

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
