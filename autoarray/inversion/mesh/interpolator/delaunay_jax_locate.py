"""
JAX-native point location for Delaunay triangulation.

Replaces scipy.spatial.Delaunay.find_simplex (and the surrounding numpy
gather + kNN fallback in
``pix_indexes_for_sub_slim_index_delaunay_from``) with a pure-JAX
implementation that vmaps cleanly across the batch dimension.

Motivation
----------
At production fiducial (1231 mesh vertices, 15361 over-sampled image
pixels), scipy's ``Delaunay.find_simplex`` accounts for roughly half of
the scipy-callback wall time in ``jax_delaunay``. Under
``vmap_method="sequential"`` the callback runs serially per batch
element, which becomes the dominant Delaunay cost at production batch
sizes (~16.87 ms per element at batch=20 on A100, ~340 ms wall per
batched likelihood).

By keeping only the triangulation in scipy (small, fast) and doing
point location in JAX, the per-element work amortises across the
batch on GPU and the sequential callback shrinks dramatically.

Memory
------
The key risk is the (B, M, 2) intermediate when computing barycentric
coordinates of B queries against M triangles. For production
M = 2 * 1231 = 2462 and B = full query count of 15361, the naive
intermediate is ~900 MB per call, ~18 GB under vmap=20 — would OOM
even on A100.

This module chunks the query stream into blocks of ``chunk_size``
(default 128) and iterates via ``lax.scan``. Peak intermediate at
chunk_size=128, M=2462 is ~50 MB per chunk, ~1 GB under vmap=20 —
well within A100 80 GB and comfortable on RTX 2060 6 GB.
"""

import numpy as np


DEFAULT_CHUNK_SIZE = 128
DEFAULT_BARY_TOLERANCE = 1e-9


def jax_find_simplex_and_gather(
    simplices_padded,
    mesh_points,
    queries,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    bary_tolerance: float = DEFAULT_BARY_TOLERANCE,
):
    """JAX equivalent of scipy.Delaunay.find_simplex + vertex gather.

    Parameters
    ----------
    simplices_padded : (M, 3) int32 jax.Array
        Triangulation. Rows beyond the active triangle count are padded
        with -1 entries (matching the existing ``scipy_delaunay``
        contract).
    mesh_points : (N, 2) float jax.Array
        Mesh vertex positions, in the same coordinate convention used
        to compute the triangulation (i.e. (x, y) since scipy/Qhull
        defaults to that).
    queries : (Q, 2) float jax.Array
        Query points to locate, in the same convention as
        ``mesh_points``.
    chunk_size : int
        Number of queries processed per ``lax.scan`` step. Trades
        kernel launch count for peak memory.
    bary_tolerance : float
        A query is considered inside a triangle if all three barycentric
        coordinates are >= ``-bary_tolerance``. Matches scipy's
        edge-tolerance behaviour.

    Returns
    -------
    mappings : (Q, 3) int32 jax.Array
        For each query, the three mesh vertex indices of its containing
        triangle. For queries outside the convex hull, the first column
        is the nearest mesh vertex index (kNN-1 fallback) and the
        remaining two columns are -1, matching
        ``pix_indexes_for_sub_slim_index_delaunay_from``'s contract.
    """
    import jax
    import jax.numpy as jnp

    M = simplices_padded.shape[0]
    N = mesh_points.shape[0]
    Q = queries.shape[0]

    # Padded rows have -1 in column 0; mark them so we don't accept
    # bogus barycentric values from a zeroed-out denominator below.
    valid_mask = simplices_padded[:, 0] >= 0  # (M,)

    # Safe gather: clamp -1 indices to 0 so jnp.take doesn't fault. We
    # zero out their contribution via valid_mask later.
    safe_simplices = jnp.where(simplices_padded >= 0, simplices_padded, 0)
    tri_verts = mesh_points[safe_simplices]  # (M, 3, 2)
    v0 = tri_verts[:, 0, :]
    v1 = tri_verts[:, 1, :]
    v2 = tri_verts[:, 2, :]
    e1 = v1 - v0  # (M, 2)
    e2 = v2 - v0  # (M, 2)

    d00 = jnp.sum(e1 * e1, axis=-1)  # (M,)
    d01 = jnp.sum(e1 * e2, axis=-1)
    d11 = jnp.sum(e2 * e2, axis=-1)
    denom = d00 * d11 - d01 * d01  # (M,)
    # Avoid /0 on padded (zero-area) triangles. The result for those is
    # then masked out via ``valid_mask``.
    safe_denom = jnp.where(jnp.abs(denom) > 0, denom, 1.0)

    inv_denom = 1.0 / safe_denom

    def locate_chunk(chunk):
        # chunk: (B, 2)
        # (B, M, 2): the only big intermediate. JAX/XLA should fuse the
        # subsequent reductions so it doesn't have to materialise all of
        # them at once, but we still bound chunk_size to control peak.
        eq = chunk[:, None, :] - v0[None, :, :]
        d20 = jnp.sum(eq * e1[None, :, :], axis=-1)  # (B, M)
        d21 = jnp.sum(eq * e2[None, :, :], axis=-1)  # (B, M)
        # Cramer's rule for barycentric coordinates with origin at v0
        w1 = (d11[None, :] * d20 - d01[None, :] * d21) * inv_denom[None, :]
        w2 = (d00[None, :] * d21 - d01[None, :] * d20) * inv_denom[None, :]
        w0 = 1.0 - w1 - w2
        w_min = jnp.minimum(jnp.minimum(w0, w1), w2)  # (B, M)
        # Mark padded triangles as never-inside
        w_min = jnp.where(valid_mask[None, :], w_min, -jnp.inf)
        inside = w_min >= -bary_tolerance  # (B, M)
        has_inside = jnp.any(inside, axis=-1)  # (B,)
        # argmax on the bool->int cast returns the FIRST True index.
        # Matches scipy's deterministic "first matching simplex" behaviour
        # well enough for science (ties on edge are resolved consistently).
        first_inside = jnp.argmax(inside.astype(jnp.int32), axis=-1)  # (B,)
        chosen = simplices_padded[first_inside]  # (B, 3)

        # Outside-hull fallback: nearest mesh vertex via brute-force
        # squared-distance argmin. (B, N) intermediate at N=1231 is tiny
        # (~256 KB at chunk_size=128 fp64).
        diffs = chunk[:, None, :] - mesh_points[None, :, :]  # (B, N, 2)
        dists = jnp.sum(diffs * diffs, axis=-1)  # (B, N)
        nn_idx = jnp.argmin(dists, axis=-1).astype(jnp.int32)  # (B,)
        neg_one = jnp.full_like(nn_idx, -1)
        fallback = jnp.stack([nn_idx, neg_one, neg_one], axis=-1)  # (B, 3)

        return jnp.where(has_inside[:, None], chosen, fallback).astype(jnp.int32)

    # Pad Q up to a multiple of chunk_size so dynamic_slice has a static
    # window into a fixed-size buffer. The padding rows produce garbage
    # we slice off at the end.
    n_chunks = (Q + chunk_size - 1) // chunk_size
    padded_Q = n_chunks * chunk_size
    pad_amount = padded_Q - Q
    queries_padded = jnp.pad(queries, ((0, pad_amount), (0, 0)))

    def scan_body(_, chunk_idx):
        start = chunk_idx * chunk_size
        chunk = jax.lax.dynamic_slice(
            queries_padded, (start, 0), (chunk_size, 2)
        )
        return None, locate_chunk(chunk)

    _, chunks_out = jax.lax.scan(scan_body, None, jnp.arange(n_chunks))
    # chunks_out shape: (n_chunks, chunk_size, 3)
    mappings = chunks_out.reshape(padded_Q, 3)[:Q]
    return mappings
