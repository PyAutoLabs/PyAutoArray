"""NumPy and JAX implementations of Sibson natural-neighbour interpolation.

The Delaunay connectivity still comes from the small qhull callback used by
``delaunay.py``.  Everything depending continuously on the mesh and query
coordinates -- circumcircles, cavity traversal, Watson areas, normalization
and stencil compaction -- is array code and therefore differentiable by JAX.

``InterpolatorDelaunayNN`` exposes the implementation through the public
``mesh.DelaunayNN`` mesh.  Overflow and degeneracy diagnostics remain part of
the internal interface: a fixed-shape limit failure produces NaN weights so a
model sample is rejected instead of silently using a truncated stencil.
"""

import os

import numpy as np
from autonerves import cached_property

from autoarray.inversion.mesh.interpolator.delaunay import (
    DelaunayInterface,
    InterpolatorDelaunay,
    _jax_delaunay_tables,
    barycentric_dual_area_from,
    pix_indexes_delaunay_walk_from,
    pix_indexes_for_sub_slim_index_delaunay_from,
)
from autoarray.inversion.regularization.regularization_util import (
    split_points_from,
)

# Fixed-shape production caps. The workspace mass-model audit (101 traced
# Hilbert meshes, including split points) observed maxima of 25 cavity
# triangles and 27 natural neighbours. Caps 16 and 24 overflowed; 32 was the
# smallest tested safe shape. See
# autolens_workspace_test/scripts/misc/jax_assertions/delaunay_nn_caps.py.
SIBSON_MAX_CAVITY_TRIANGLES = 32
SIBSON_MAX_NEIGHBORS = 32


def _positive_int_env(name, raw):
    """Parse a positive-integer environment override; ``None`` when unset."""
    if raw is None:
        return None
    message = f"{name} must be a positive integer, got {raw!r}"
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(message) from error
    if value <= 0:
        raise ValueError(message)
    return value


def _bool_env(name, raw):
    """Parse a ``"0"``/``"1"`` environment override; ``None`` when unset."""
    if raw is None:
        return None
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {raw!r}")
    return raw == "1"


# ``jax.lax.map`` block size for the JAX Sibson query loop, bound onto the mesh
# as ``DelaunayNN.query_chunk``.  It is a MEMORY GUARD and nothing else: it
# bounds the ``(C, 3, 2)`` per-query intermediates of the cavity -- the
# inserted Watson circumcircle centres and their contributions, of order
# 15-25 kB per query per lane at cap 32 -- so the live footprint stays bounded
# when the likelihood is vmapped over many live points.
#
# It is emphatically not a speed knob to be kept small.  The walk and the two
# cavity loops inside a chunk are latency-bound, so each extra chunk serialises
# another complete set of kernel launches: halving the chunk roughly doubles
# the launch count of the pass (issue #532).  Raise it as far as device memory
# allows for the mesh and vmap batch in use.
#
# The default is the A100 sweep of issue #532 (2026-09-08, RAL
# ``euclid-ral-gpu-2``, NVIDIA A100 80GB PCIe, fp64, jobs 342321/342322 and the
# array 342323_[0-3]; autolens_profiling
# ``results/notes/delaunay_nn_launch_latency.md``).  Cell: the HST imaging
# DelaunayNN likelihood breakdown, Hilbert-1500 mesh, MGE-60 lens light,
# ConstantSplit regularization, 17,980 over-sampled data queries + 6,000 split
# points, ``--split-setup --vmap-batch 16``.  Measured params->H prefix and the
# peak ``nvidia-smi`` memory sampled through the whole run at vmap 16:
#
#   chunk | params->H unbatched | params->H per call @vmap 16 | peak VRAM
#     256 |            88.29 ms |                    23.07 ms | 41,495 MiB
#     512 |            52.21 ms |                    18.24 ms | 41,503 MiB
#    1024 |            35.32 ms |                    19.57 ms | 41,503 MiB
#    2048 |            27.22 ms |                    19.13 ms | 41,503 MiB
#    4096 |            24.66 ms |                    16.45 ms | 41,503 MiB
#
# (control, ``main`` at ``d7c96762``, chunk 256: 143.90 ms / 24.32 ms /
# 41,495 MiB.  The eager ``EXPECTED_LOG_EVIDENCE_HST = 29144.581944`` pin held
# on every row, so the chunk is bit-neutral as designed.)
#
# 4096 is fastest on both readings and the VRAM clause of the sweep's decision
# rule turned out uninformative: the ~41.5 GiB plateau is identical at every
# chunk *and* on the control, because it is the vmap-16 dense inversion block,
# not the cavity intermediates.  The guard arithmetic says why there is room --
# ``(C, 3, 2)`` fp64 intermediates at ~25 kB per query per lane over 4096
# queries x 16 lanes is ~1.6 GB, ~2 % of an 80 GB card.
#
# Set ``PYAUTO_SIBSON_QUERY_CHUNK`` (a positive integer) to override the
# default at import time.  That is both how the sweep above was run without
# editing source and the escape hatch for a smaller GPU or a much larger cell:
# lower it until the ``(C, 3, 2)`` intermediates fit, at a proportional cost in
# sequential ``lax.map`` trips.
SIBSON_QUERY_CHUNK = 4096

_QUERY_CHUNK_OVERRIDE = _positive_int_env(
    "PYAUTO_SIBSON_QUERY_CHUNK", os.environ.get("PYAUTO_SIBSON_QUERY_CHUNK")
)
if _QUERY_CHUNK_OVERRIDE is not None:
    SIBSON_QUERY_CHUNK = _QUERY_CHUNK_OVERRIDE

# Candidate-edge loop strategy inside the cavity walk
# (:func:`_cavity_triangle_indexes_jax`).  The three edges of a cavity triangle
# are either unrolled at trace time or run through a 3-trip ``fori_loop``.  The
# two are BIT-IDENTICAL -- the same ``add_candidate`` calls for edges 0, 1, 2 in
# that order, so the same cavity insertion order and therefore the same stencil
# column order and the same floating-point summation order downstream (verified
# against a frozen reference of every ``jax_delaunay_nn`` output).  They differ
# only in how the program is emitted, and which is faster is a property of the
# backend, so it is decided at trace time:
#
#   * accelerator -- the cavity walk is launch-latency bound, and unrolling the
#     inner loop removes ~28% of the kernel launches per chunk (1,244 -> 892 on
#     the A100 HST / Hilbert-1500 cell, issue #532), so unroll.
#   * CPU -- there is no launch cost to remove, and the unrolled body is three
#     times the code inside a 32-trip loop.  Measured with an interleaved
#     in-process paired A/B of ``jax_delaunay_nn`` (N=1500, Q=17,980 data +
#     6,000 split, fp64, warm median of 25-30 alternating rounds): rolled
#     566.0 / 567.8 ms vs unrolled 614.4 / 614.7 ms, i.e. the rolled loop is
#     ~8-9% faster.  The same harness reads 0.998 between two copies of
#     identical code, so that gap is real, and the standing no-CPU-slowdown
#     constraint keeps the loop rolled here.
#
# ``PYAUTO_SIBSON_UNROLL_CANDIDATES`` ("1" or "0") forces one strategy for
# benchmarking; unset means decide from ``jax.default_backend()``.
SIBSON_UNROLL_CANDIDATES = _bool_env(
    "PYAUTO_SIBSON_UNROLL_CANDIDATES",
    os.environ.get("PYAUTO_SIBSON_UNROLL_CANDIDATES"),
)


def _sibson_unroll_candidates():
    """Whether to unroll the cavity walk's three candidate edges.

    ``SIBSON_UNROLL_CANDIDATES`` (the environment override) wins when set;
    otherwise every backend but CPU unrolls.  See the comment above.
    """
    if SIBSON_UNROLL_CANDIDATES is not None:
        return SIBSON_UNROLL_CANDIDATES

    import jax

    return jax.default_backend() != "cpu"


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def circumcircles_from(a, b, c, xp=np):
    """Return circumcentres, squared radii and validity for point triples.

    The leading dimensions of ``a``, ``b`` and ``c`` are broadcast normally;
    the final dimension is the two Cartesian coordinates.
    """
    ba = b - a
    ca = c - a
    denominator = 2.0 * _cross(ba, ca)
    scale = xp.maximum(
        xp.maximum(xp.sum(ba * ba, axis=-1), xp.sum(ca * ca, axis=-1)),
        xp.asarray(1.0, dtype=a.dtype),
    )
    tolerance = 32.0 * xp.finfo(a.dtype).eps * scale
    valid = xp.abs(denominator) > tolerance
    safe_denominator = xp.where(valid, denominator, 1.0)

    ba2 = xp.sum(ba * ba, axis=-1)
    ca2 = xp.sum(ca * ca, axis=-1)
    offset = (
        xp.stack(
            [
                ba2 * ca[..., 1] - ca2 * ba[..., 1],
                ba[..., 0] * ca2 - ca[..., 0] * ba2,
            ],
            axis=-1,
        )
        / safe_denominator[..., None]
    )
    centre = a + offset
    radius_squared = xp.sum(offset * offset, axis=-1)

    centre = xp.where(valid[..., None], centre, 0.0)
    radius_squared = xp.where(valid, radius_squared, 0.0)
    return centre, radius_squared, valid


def delaunay_circumcircles_from(points, simplices_padded, xp=np):
    """Circumcircles for fixed-shape, -1-padded Delaunay simplices."""
    simplex_valid = (simplices_padded >= 0).all(axis=1)
    safe_simplices = simplices_padded.clip(min=0)
    vertices = points[safe_simplices]
    centres, radii_squared, circle_valid = circumcircles_from(
        vertices[:, 0], vertices[:, 1], vertices[:, 2], xp=xp
    )
    valid = simplex_valid & circle_valid
    centres = xp.where(valid[:, None], centres, 0.0)
    radii_squared = xp.where(valid, radii_squared, 0.0)
    return centres, radii_squared, valid


def _contains_query(
    triangle_index,
    query,
    circumcentres,
    circumradii_squared,
    circumcircle_valid,
    xp,
):
    safe_index = xp.maximum(triangle_index, 0)
    delta = circumcentres[safe_index] - query
    distance_squared = xp.sum(delta * delta)
    radius_squared = circumradii_squared[safe_index]
    tolerance = 64.0 * xp.finfo(query.dtype).eps * xp.maximum(radius_squared, 1.0)
    return (
        (triangle_index >= 0)
        & circumcircle_valid[safe_index]
        & (distance_squared <= radius_squared + tolerance)
    )


def _cavity_triangle_indexes_numpy(
    query,
    seed_simplex,
    simplex_neighbors,
    circumcentres,
    circumradii_squared,
    circumcircle_valid,
    max_cavity_triangles,
):
    cavity = np.full(max_cavity_triangles, -1, dtype=np.int32)
    if seed_simplex < 0:
        return cavity, np.int32(0), np.bool_(False)

    cavity[0] = seed_simplex
    count = 1
    overflow = False

    for position in range(max_cavity_triangles):
        if position >= count:
            break
        triangle_index = cavity[position]
        for candidate in simplex_neighbors[triangle_index]:
            if candidate < 0 or candidate in cavity[:count]:
                continue
            if not _contains_query(
                candidate,
                query,
                circumcentres,
                circumradii_squared,
                circumcircle_valid,
                np,
            ):
                continue
            if count == max_cavity_triangles:
                overflow = True
            else:
                cavity[count] = candidate
                count += 1

    return cavity, np.int32(count), np.bool_(overflow)


def _cavity_triangle_indexes_jax(
    query,
    seed_simplex,
    simplex_neighbors,
    circumcentres,
    circumradii_squared,
    circumcircle_valid,
    max_cavity_triangles,
):
    import jax
    import jax.numpy as jnp

    unroll = _sibson_unroll_candidates()

    safe_seed = jnp.maximum(seed_simplex, 0)
    cavity = -jnp.ones((max_cavity_triangles,), dtype=jnp.int32)
    cavity = cavity.at[0].set(safe_seed)
    initial_count = jnp.where(seed_simplex >= 0, 1, 0).astype(jnp.int32)

    def process_triangle(position, carry):
        cavity, count, overflow = carry
        active = position < count
        triangle_index = cavity[position].clip(min=0)
        candidates = simplex_neighbors[triangle_index]

        def add_candidate(edge, inner_carry):
            cavity, count, overflow = inner_carry
            candidate = candidates[edge]
            unseen = ~jnp.any(cavity == candidate)
            contained = _contains_query(
                candidate,
                query,
                circumcentres,
                circumradii_squared,
                circumcircle_valid,
                jnp,
            )
            accepted = active & unseen & contained
            has_space = count < max_cavity_triangles
            add = accepted & has_space
            write_index = jnp.minimum(count, max_cavity_triangles - 1)
            old_value = cavity[write_index]
            cavity = cavity.at[write_index].set(jnp.where(add, candidate, old_value))
            count = count + add.astype(jnp.int32)
            overflow = overflow | (accepted & ~has_space)
            return cavity, count, overflow

        # Unrolled or rolled by ``_sibson_unroll_candidates()``; the two are
        # bit-identical by construction (same calls, edges 0, 1, 2, same
        # order), and the choice is a backend performance question only --
        # see the comment on ``SIBSON_UNROLL_CANDIDATES``.
        carry = (cavity, count, overflow)
        if unroll:
            for edge in range(3):
                carry = add_candidate(edge, carry)
            return carry
        return jax.lax.fori_loop(0, 3, add_candidate, carry)

    return jax.lax.fori_loop(
        0,
        max_cavity_triangles,
        process_triangle,
        (cavity, initial_count, jnp.asarray(False)),
    )


def _sibson_single_from_tables(
    query,
    seed_simplex,
    outside_fallback_index,
    points,
    simplices_padded,
    simplex_neighbors,
    circumcentres,
    circumradii_squared,
    circumcircle_valid,
    max_cavity_triangles,
    max_neighbors,
    xp,
):
    if xp is np:
        cavity, cavity_size, overflow = _cavity_triangle_indexes_numpy(
            query=query,
            seed_simplex=int(seed_simplex),
            simplex_neighbors=simplex_neighbors,
            circumcentres=circumcentres,
            circumradii_squared=circumradii_squared,
            circumcircle_valid=circumcircle_valid,
            max_cavity_triangles=max_cavity_triangles,
        )
    else:
        cavity, cavity_size, overflow = _cavity_triangle_indexes_jax(
            query=query,
            seed_simplex=seed_simplex,
            simplex_neighbors=simplex_neighbors,
            circumcentres=circumcentres,
            circumradii_squared=circumradii_squared,
            circumcircle_valid=circumcircle_valid,
            max_cavity_triangles=max_cavity_triangles,
        )

    cavity_valid = cavity >= 0
    safe_cavity = cavity.clip(min=0)
    triangle_vertices = simplices_padded[safe_cavity].clip(min=0)
    vertices = points[triangle_vertices]

    # Circle j passes through the query and the edge opposite triangle vertex j.
    edge_a = vertices[:, [1, 2, 0], :]
    edge_b = vertices[:, [2, 0, 1], :]
    query_for_edges = xp.broadcast_to(query, edge_a.shape)
    inserted_centres, _, inserted_valid = circumcircles_from(
        edge_a, edge_b, query_for_edges, xp=xp
    )

    original_centres = circumcentres[safe_cavity][:, None, :]
    relative = inserted_centres - original_centres
    contributions = xp.stack(
        [
            _cross(relative[:, 1], relative[:, 2]),
            _cross(relative[:, 2], relative[:, 0]),
            _cross(relative[:, 0], relative[:, 1]),
        ],
        axis=1,
    )
    contribution_valid = cavity_valid[:, None] & xp.stack(
        [
            inserted_valid[:, 1] & inserted_valid[:, 2],
            inserted_valid[:, 2] & inserted_valid[:, 0],
            inserted_valid[:, 0] & inserted_valid[:, 1],
        ],
        axis=1,
    )
    contributions = xp.where(contribution_valid, contributions, 0.0)

    flat_vertices = triangle_vertices.reshape(-1)
    flat_contributions = contributions.reshape(-1)
    flat_valid = xp.broadcast_to(cavity_valid[:, None], contributions.shape).reshape(-1)
    sentinel = points.shape[0]
    sortable_vertices = xp.where(flat_valid, flat_vertices, sentinel)
    sortable_contributions = xp.where(flat_valid, flat_contributions, 0.0)

    # Compact the at-most 3*C Watson contributions without allocating an
    # N-vertex vector per query.  Equal vertex ids become adjacent after the
    # small fixed-size sort and are scatter-added into consecutive groups.
    # This is substantially cheaper than a global N-way top-k for production
    # meshes (N ~= 1200, while 3*C <= 96).
    order = xp.argsort(sortable_vertices)
    sorted_vertices = sortable_vertices[order]
    sorted_contributions = sortable_contributions[order]
    sorted_valid = sorted_vertices < sentinel
    group_start = sorted_valid & xp.concatenate(
        [
            xp.ones((1,), dtype=bool),
            sorted_vertices[1:] != sorted_vertices[:-1],
        ]
    )
    group = xp.cumsum(group_start.astype(xp.int32)) - 1
    unique_count = xp.sum(group_start).astype(xp.int32)

    compact_weights = xp.zeros(flat_vertices.shape[0], dtype=points.dtype)
    if xp is np:
        np.add.at(
            compact_weights,
            group,
            np.where(sorted_valid, sorted_contributions, 0.0),
        )
        start_positions = np.flatnonzero(group_start)
        compact_indexes = np.full(flat_vertices.shape[0], -1, dtype=np.int32)
        compact_indexes[: len(start_positions)] = sorted_vertices[start_positions]
    else:
        compact_weights = compact_weights.at[group].add(
            xp.where(sorted_valid, sorted_contributions, 0.0)
        )
        start_positions = xp.nonzero(
            group_start, size=flat_vertices.shape[0], fill_value=0
        )[0]
        compact_indexes = xp.where(
            xp.arange(flat_vertices.shape[0]) < unique_count,
            sorted_vertices[start_positions],
            -1,
        ).astype(xp.int32)

    inside = seed_simplex >= 0
    seed_vertices = simplices_padded[xp.maximum(seed_simplex, 0)].clip(min=0)
    seed_points = points[seed_vertices]
    seed_distances_squared = xp.sum((seed_points - query[None, :]) ** 2, axis=1)
    local_vertex = seed_vertices[xp.argmin(seed_distances_squared)]
    coordinate_scale = xp.maximum(
        xp.maximum(xp.max(xp.abs(query)), xp.max(xp.abs(points[seed_vertices]))),
        1.0,
    )
    vertex_tolerance = (64.0 * xp.finfo(points.dtype).eps * coordinate_scale) ** 2
    on_vertex = inside & (xp.min(seed_distances_squared) <= vertex_tolerance)

    seed_denominator = _cross(
        seed_points[1] - seed_points[0], seed_points[2] - seed_points[0]
    )
    safe_seed_denominator = xp.where(seed_denominator != 0.0, seed_denominator, 1.0)
    seed_barycentric = (
        xp.stack(
            [
                _cross(seed_points[1] - query, seed_points[2] - query),
                _cross(seed_points[2] - query, seed_points[0] - query),
                _cross(seed_points[0] - query, seed_points[1] - query),
            ]
        )
        / safe_seed_denominator
    )
    edge_tolerance = 128.0 * xp.finfo(points.dtype).eps
    on_edge = (
        inside & (~on_vertex) & (xp.min(xp.abs(seed_barycentric)) <= edge_tolerance)
    )

    fallback_index = xp.where(on_vertex, local_vertex, outside_fallback_index)
    use_fallback = (~inside) | on_vertex

    weight_sum = xp.sum(compact_weights)
    degenerate = (
        inside
        & (~on_vertex)
        & (~on_edge)
        & (
            xp.any(cavity_valid[:, None] & ~inserted_valid)
            | (~xp.isfinite(weight_sum))
            | (xp.abs(weight_sum) <= xp.finfo(points.dtype).tiny)
        )
    )
    safe_sum = xp.where(
        xp.abs(weight_sum) > xp.finfo(points.dtype).tiny, weight_sum, 1.0
    )
    compact_weights = compact_weights / safe_sum

    compact_length = flat_vertices.shape[0]
    if max_neighbors <= compact_length:
        mappings = compact_indexes[:max_neighbors]
        weights = compact_weights[:max_neighbors]
    else:
        padding = max_neighbors - compact_length
        mappings = xp.pad(compact_indexes, (0, padding), constant_values=-1)
        weights = xp.pad(compact_weights, (0, padding))

    selected_valid = (xp.arange(max_neighbors) < unique_count) & (weights > 0.0)
    mappings = xp.where(selected_valid, mappings, -1).astype(xp.int32)
    weights = xp.where(selected_valid, weights, 0.0)

    fallback_mappings = -xp.ones((max_neighbors,), dtype=xp.int32)
    fallback_weights = xp.zeros((max_neighbors,), dtype=points.dtype)
    if xp is np:
        fallback_mappings[0] = int(fallback_index)
        fallback_weights[0] = 1.0
    else:
        fallback_mappings = fallback_mappings.at[0].set(fallback_index)
        fallback_weights = fallback_weights.at[0].set(1.0)
    mappings = xp.where(use_fallback, fallback_mappings, mappings)
    weights = xp.where(use_fallback, fallback_weights, weights)
    size = xp.where(use_fallback, 1, xp.sum(selected_valid)).astype(xp.int32)

    # Inserting a query exactly on an existing edge makes one of the Watson
    # circumcircles collinear. Sibson coordinates there are the ordinary
    # linear coordinates of the two edge endpoints, which the containing
    # triangle's barycentric row represents exactly (the third weight is 0).
    edge_mappings = -xp.ones((max_neighbors,), dtype=xp.int32)
    edge_weights = xp.zeros((max_neighbors,), dtype=points.dtype)
    if xp is np:
        edge_mappings[:3] = seed_vertices
        edge_weights[:3] = seed_barycentric
    else:
        edge_mappings = edge_mappings.at[:3].set(seed_vertices)
        edge_weights = edge_weights.at[:3].set(seed_barycentric)
    mappings = xp.where(on_edge, edge_mappings, mappings)
    weights = xp.where(on_edge, edge_weights, weights)
    size = xp.where(on_edge, 3, size).astype(xp.int32)

    neighbor_overflow = (
        inside & (~on_vertex) & (~on_edge) & (unique_count > max_neighbors)
    )
    overflow = overflow | neighbor_overflow
    failed = overflow | degenerate
    weights = xp.where(failed, xp.nan, weights)
    return mappings, size, weights, cavity_size, overflow, degenerate


def sibson_mappings_weights_from_tables(
    query_points,
    points,
    simplices_padded,
    simplex_neighbors,
    simplex_indexes,
    outside_fallback_indexes,
    max_cavity_triangles=SIBSON_MAX_CAVITY_TRIANGLES,
    max_neighbors=SIBSON_MAX_NEIGHBORS,
    query_chunk=SIBSON_QUERY_CHUNK,
    circumcircles=None,
    xp=np,
):
    """Calculate fixed-shape Sibson mappings and weights from Delaunay tables.

    Loop structure (and why the chunk exists)
    -----------------------------------------
    On the JAX path this is three nested loops: a ``jax.lax.map`` over
    ``query_chunk``-sized blocks of queries, a ``fori_loop`` of
    ``max_cavity_triangles`` trips walking the insertion cavity
    (:func:`_cavity_triangle_indexes_jax`), and the three candidate edges of
    each cavity triangle -- the last of which is unrolled.  Only the outer
    ``lax.map`` is sequential *in the queries*: every chunk re-runs the whole
    cavity walk, so the kernel-launch count of one call is roughly
    ``ceil(Q / query_chunk)`` times the launches of a single chunk.  With the
    cavity loop dominating that per-chunk count, a small chunk multiplies a
    latency-bound program rather than saving time, which is why
    ``query_chunk`` is documented as a memory guard on the ``(C, 3, 2)``
    per-cavity intermediates and nothing else (issue #532).  The NumPy path
    ignores ``query_chunk`` entirely and loops over queries in Python.

    Parameters
    ----------
    circumcircles
        Optional precomputed ``(centres, radii_squared, valid)`` triple for
        ``simplices_padded``, as returned by
        :func:`delaunay_circumcircles_from`.  They depend only on the frozen
        simplex table, so a caller interpolating several query sets against
        one mesh (``jax_delaunay_nn``) computes them once and passes them in.
        ``None`` computes them here, which is what every other caller does.

    Returns
    -------
    mappings, sizes, weights
        Mapper-compatible arrays with at most ``max_neighbors`` unique source
        vertices per query.
    cavity_sizes, cavity_overflow, degenerate
        Prototype diagnostics.  Overflow or a Watson edge degeneracy produces
        NaN weights rather than silently returning an approximate stencil.
    """
    if circumcircles is None:
        circumcircles = delaunay_circumcircles_from(points, simplices_padded, xp=xp)
    circumcentres, circumradii_squared, circumcircle_valid = circumcircles

    def single(query, seed_simplex, outside_fallback_index):
        return _sibson_single_from_tables(
            query=query,
            seed_simplex=seed_simplex,
            outside_fallback_index=outside_fallback_index,
            points=points,
            simplices_padded=simplices_padded,
            simplex_neighbors=simplex_neighbors,
            circumcentres=circumcentres,
            circumradii_squared=circumradii_squared,
            circumcircle_valid=circumcircle_valid,
            max_cavity_triangles=max_cavity_triangles,
            max_neighbors=max_neighbors,
            xp=xp,
        )

    if xp is np:
        rows = [
            single(query, seed, fallback)
            for query, seed, fallback in zip(
                query_points, simplex_indexes, outside_fallback_indexes
            )
        ]
        return tuple(np.stack(values) for values in zip(*rows))

    import jax

    query_count = query_points.shape[0]
    pad = (-query_count) % query_chunk
    padded_queries = xp.concatenate(
        [query_points, xp.zeros((pad, 2), dtype=query_points.dtype)]
    )
    padded_simplex_indexes = xp.concatenate(
        [simplex_indexes, -xp.ones((pad,), dtype=xp.int32)]
    )
    padded_fallback_indexes = xp.concatenate(
        [outside_fallback_indexes, xp.zeros((pad,), dtype=xp.int32)]
    )

    chunk_inputs = (
        padded_queries.reshape(-1, query_chunk, 2),
        padded_simplex_indexes.reshape(-1, query_chunk),
        padded_fallback_indexes.reshape(-1, query_chunk),
    )
    batched_single = jax.vmap(single)
    outputs = jax.lax.map(lambda args: batched_single(*args), chunk_inputs)
    return tuple(
        output.reshape((-1,) + output.shape[2:])[:query_count] for output in outputs
    )


def jax_sibson(
    points,
    query_points,
    max_cavity_triangles=SIBSON_MAX_CAVITY_TRIANGLES,
    max_neighbors=SIBSON_MAX_NEIGHBORS,
    query_chunk=SIBSON_QUERY_CHUNK,
):
    """Prototype end-to-end JAX Sibson interpolation-table construction.

    Qhull supplies only integer connectivity through ``pure_callback``.  The
    returned floating-point weights retain gradients with respect to both
    ``points`` and ``query_points`` everywhere the natural-neighbour geometry
    is differentiable.
    """
    import jax.numpy as jnp

    simplices_padded, simplex_neighbors, vertex_simplex = _jax_delaunay_tables(points)
    delaunay_mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=query_points,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=jnp,
        return_simplex_indexes=True,
    )
    outputs = sibson_mappings_weights_from_tables(
        query_points=query_points,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=delaunay_mappings[:, 0],
        max_cavity_triangles=max_cavity_triangles,
        max_neighbors=max_neighbors,
        query_chunk=query_chunk,
        xp=jnp,
    )
    return (points, simplices_padded) + outputs


def scipy_delaunay_nn(
    points_np,
    query_points_np,
    areas_factor=0.5,
    max_cavity_triangles=SIBSON_MAX_CAVITY_TRIANGLES,
    max_neighbors=SIBSON_MAX_NEIGHBORS,
    query_chunk=SIBSON_QUERY_CHUNK,
):
    """Build all NumPy tables required by ``InterpolatorDelaunayNN``."""
    from scipy.spatial import Delaunay

    triangle = Delaunay(points_np)
    points = triangle.points.astype(points_np.dtype)
    simplex_count = triangle.simplices.shape[0]

    simplices_padded = -np.ones((2 * points.shape[0], 3), dtype=np.int32)
    neighbors_padded = -np.ones_like(simplices_padded)
    simplices_padded[:simplex_count] = triangle.simplices.astype(np.int32)
    neighbors_padded[:simplex_count] = triangle.neighbors.astype(np.int32)

    def mappings_weights_for(query_points):
        simplex_indexes = triangle.find_simplex(query_points).astype(np.int32)
        delaunay_mappings = pix_indexes_for_sub_slim_index_delaunay_from(
            data_grid=query_points,
            simplex_index_for_sub_slim_index=simplex_indexes,
            pix_indexes_for_simplex_index=triangle.simplices,
            delaunay_points=points,
        )
        return sibson_mappings_weights_from_tables(
            query_points=query_points,
            points=points,
            simplices_padded=simplices_padded,
            simplex_neighbors=neighbors_padded,
            simplex_indexes=simplex_indexes,
            outside_fallback_indexes=delaunay_mappings[:, 0],
            max_cavity_triangles=max_cavity_triangles,
            max_neighbors=max_neighbors,
            query_chunk=query_chunk,
            xp=np,
        )

    mappings, sizes, weights, cavity_sizes, overflow, degenerate = mappings_weights_for(
        query_points_np
    )

    areas = barycentric_dual_area_from(
        mesh_grid=points,
        simplices=triangle.simplices,
        xp=np,
    )
    split_points = split_points_from(
        points=points,
        area_weights=areas_factor * np.sqrt(areas),
        xp=np,
    )
    (
        splitted_mappings,
        splitted_sizes,
        splitted_weights,
        split_cavity_sizes,
        split_overflow,
        split_degenerate,
    ) = mappings_weights_for(split_points)

    return (
        points,
        simplices_padded,
        mappings,
        sizes,
        weights,
        split_points,
        splitted_mappings,
        splitted_sizes,
        splitted_weights,
        cavity_sizes,
        overflow,
        degenerate,
        split_cavity_sizes,
        split_overflow,
        split_degenerate,
    )


def jax_delaunay_nn(
    points,
    query_points,
    areas_factor=0.5,
    max_cavity_triangles=SIBSON_MAX_CAVITY_TRIANGLES,
    max_neighbors=SIBSON_MAX_NEIGHBORS,
    query_chunk=SIBSON_QUERY_CHUNK,
):
    """Build all JAX tables required by ``InterpolatorDelaunayNN``.

    Qhull returns only fixed-shape integer connectivity through a stopped
    ``pure_callback``.  Point location, circumcircles, Sibson weights, dual
    areas, and split-cross coordinates all remain in the JAX graph.

    The data grid and the ``4N`` split-cross points are located *and*
    interpolated in ONE concatenated pass, the same pattern
    :func:`jax_delaunay` uses for its walk.  Both the visibility walk and the
    Sibson cavity loops are latency-bound, so two passes pay two full sets of
    kernel launches while one concatenated pass pays roughly one (issue #532).
    The circumcircles of the frozen simplex table do not depend on the
    queries, so they are computed once here and handed to
    :func:`sibson_mappings_weights_from_tables` instead of being recomputed
    per pass.  Split points are still seeded at their own nearest vertex and
    keep their own outside-hull fallback: concatenation changes only how many
    programs run, never a per-query result.
    """
    import jax.numpy as jnp

    simplices_padded, simplex_neighbors, vertex_simplex = _jax_delaunay_tables(points)

    valid = simplices_padded[:, 0] >= 0
    simplices = simplices_padded.clip(min=0)
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    triangle_cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    contribution = jnp.where(valid, 0.5 * jnp.abs(triangle_cross) / 3.0, 0.0)
    areas = jnp.zeros(points.shape[0], dtype=points.dtype)
    for vertex_index in range(3):
        areas = areas.at[simplices[:, vertex_index]].add(contribution)

    split_points = split_points_from(
        points=points,
        area_weights=areas_factor * jnp.sqrt(areas),
        xp=jnp,
    )

    n_query = query_points.shape[0]
    all_query_points = jnp.concatenate([query_points, split_points])

    delaunay_mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=all_query_points,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=jnp,
        return_simplex_indexes=True,
    )

    circumcircles = delaunay_circumcircles_from(points, simplices_padded, xp=jnp)

    outputs = sibson_mappings_weights_from_tables(
        query_points=all_query_points,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=delaunay_mappings[:, 0],
        max_cavity_triangles=max_cavity_triangles,
        max_neighbors=max_neighbors,
        query_chunk=query_chunk,
        circumcircles=circumcircles,
        xp=jnp,
    )

    (
        mappings,
        sizes,
        weights,
        cavity_sizes,
        overflow,
        degenerate,
    ) = (output[:n_query] for output in outputs)
    (
        splitted_mappings,
        splitted_sizes,
        splitted_weights,
        split_cavity_sizes,
        split_overflow,
        split_degenerate,
    ) = (output[n_query:] for output in outputs)

    return (
        points,
        simplices_padded,
        mappings,
        sizes,
        weights,
        split_points,
        splitted_mappings,
        splitted_sizes,
        splitted_weights,
        cavity_sizes,
        overflow,
        degenerate,
        split_cavity_sizes,
        split_overflow,
        split_degenerate,
    )


class DelaunayNNInterface(DelaunayInterface):
    """Fixed-shape natural-neighbour interpolation tables and diagnostics."""

    def __init__(
        self,
        points,
        simplices,
        mappings,
        sizes,
        weights,
        split_points,
        splitted_mappings,
        splitted_sizes,
        splitted_weights,
        cavity_sizes,
        overflow,
        degenerate,
        split_cavity_sizes,
        split_overflow,
        split_degenerate,
        xp=np,
    ):
        super().__init__(
            points=points,
            simplices=simplices,
            mappings=mappings,
            split_points=split_points,
            splitted_mappings=splitted_mappings,
            xp=xp,
        )
        self._sizes = sizes
        self.weights = weights
        self._splitted_sizes = splitted_sizes
        self.splitted_weights = splitted_weights
        self.cavity_sizes = cavity_sizes
        self.overflow = overflow
        self.degenerate = degenerate
        self.split_cavity_sizes = split_cavity_sizes
        self.split_overflow = split_overflow
        self.split_degenerate = split_degenerate

    @cached_property
    def sizes(self):
        return self._sizes.astype(np.int32)

    @cached_property
    def splitted_sizes(self):
        return self._splitted_sizes.astype(np.int32)


class InterpolatorDelaunayNN(InterpolatorDelaunay):
    """Delaunay-mesh interpolator using Sibson natural-neighbour weights.

    Connectivity is obtained from the same qhull callback as
    ``InterpolatorDelaunay``.  In JAX mode all floating-point geometry is
    autodifferentiable with respect to both mesh and query coordinates while
    the piecewise-constant connectivity is frozen.  Unlike triangle-local
    barycentric interpolation, Sibson coordinates agree through a Delaunay
    diagonal flip, removing the finite sign/jump seen when the mass model
    changes the triangulation.
    """

    @cached_property
    def delaunay(self):
        kwargs = {
            "areas_factor": self.mesh.areas_factor,
            "max_cavity_triangles": self.mesh.max_cavity_triangles,
            "max_neighbors": self.mesh.max_neighbors,
            "query_chunk": self.mesh.query_chunk,
        }
        if self._xp.__name__.startswith("jax"):
            outputs = jax_delaunay_nn(
                points=self.mesh_grid_xy,
                query_points=self.data_grid.over_sampled.array,
                **kwargs,
            )
        else:
            outputs = scipy_delaunay_nn(
                points_np=self.mesh_grid_xy,
                query_points_np=self.data_grid.over_sampled.array,
                **kwargs,
            )

        return DelaunayNNInterface(*outputs, xp=self._xp)

    @cached_property
    def _mappings_sizes_weights(self):
        return (
            self.delaunay.mappings.astype(self._xp.int32),
            self.delaunay.sizes.astype(self._xp.int32),
            self.delaunay.weights,
        )

    @cached_property
    def _mappings_sizes_weights_split(self):
        # ``reg_split_from`` may need to insert the centre pixel when it is not
        # already in a split-point stencil, so retain one spare padded column.
        row_count = self.delaunay.splitted_mappings.shape[0]
        mappings = self._xp.hstack(
            (
                self.delaunay.splitted_mappings.astype(self._xp.int32),
                -self._xp.ones((row_count, 1), dtype=self._xp.int32),
            )
        )
        weights = self._xp.hstack(
            (
                self.delaunay.splitted_weights,
                self._xp.zeros((row_count, 1), dtype=self.delaunay.weights.dtype),
            )
        )
        return mappings, self.delaunay.splitted_sizes.astype(self._xp.int32), weights
