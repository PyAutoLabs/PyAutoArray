"""numba kernel for the NumPy Sibson path; see ``sibson.sibson_numba_kernel``.

Kept in its own module, imported only on first use, so ``import numba`` stays
off the library import path while every jitted function is a plain
module-level function: numba's on-disk cache (``cache=True``) keys functions
by module file and qualified name, which closures built inside a factory do
not have stably, so they recompiled in every new process.
"""

import numba
import numpy as np
from numba import prange

# float64 machine constants as module globals, which numba freezes into the
# compiled code (and the on-disk cache) as literals.
EPS = 2.220446049250313e-16
TINY = 2.2250738585072014e-308


@numba.njit(cache=True, nogil=True)
def _np_maximum(a, b):
    # np.maximum semantics: NaN propagates.
    if a != a or b != b:
        return np.nan
    return a if a >= b else b


@numba.njit(cache=True, nogil=True)
def _block_sum(values, start, n):
    # NumPy ``pairwise_sum`` leaf (n <= 128): plain loop below 8, otherwise
    # eight interleaved accumulators combined as a fixed tree.
    if n < 8:
        result = -0.0
        for i in range(n):
            result += values[start + i]
        return result
    r0 = values[start + 0]
    r1 = values[start + 1]
    r2 = values[start + 2]
    r3 = values[start + 3]
    r4 = values[start + 4]
    r5 = values[start + 5]
    r6 = values[start + 6]
    r7 = values[start + 7]
    i = 8
    stop = n - (n % 8)
    while i < stop:
        r0 += values[start + i + 0]
        r1 += values[start + i + 1]
        r2 += values[start + i + 2]
        r3 += values[start + i + 3]
        r4 += values[start + i + 4]
        r5 += values[start + i + 5]
        r6 += values[start + i + 6]
        r7 += values[start + i + 7]
        i += 8
    result = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        result += values[start + i]
        i += 1
    return result


@numba.njit(cache=True, nogil=True)
def _pairwise_sum(values):
    # Explicit-stack form of NumPy's recursive ``pairwise_sum``: blocks
    # above 128 split at ``n2 = n // 2 - (n // 2) % 8`` and the two halves
    # are added left + right.
    node_start = np.empty(64, dtype=np.int64)
    node_n = np.empty(64, dtype=np.int64)
    node_state = np.empty(64, dtype=np.int64)
    partials = np.empty(64, dtype=np.float64)
    top = 0
    partial_top = 0
    node_start[0] = 0
    node_n[0] = values.shape[0]
    node_state[0] = 0
    top = 1
    while top > 0:
        top -= 1
        start = node_start[top]
        n = node_n[top]
        state = node_state[top]
        if n <= 128:
            partials[partial_top] = _block_sum(values, start, n)
            partial_top += 1
        elif state == 0:
            n2 = n // 2
            n2 -= n2 % 8
            node_start[top] = start
            node_n[top] = n
            node_state[top] = 1
            node_start[top + 1] = start + n2
            node_n[top + 1] = n - n2
            node_state[top + 1] = 0
            node_start[top + 2] = start
            node_n[top + 2] = n2
            node_state[top + 2] = 0
            top += 3
        else:
            right = partials[partial_top - 1]
            left = partials[partial_top - 2]
            partial_top -= 2
            partials[partial_top] = left + right
            partial_top += 1
    return partials[0]


@numba.njit(cache=True, parallel=True, nogil=True)
def _sibson_from_tables_parallel(
    query_points,
    seeds,
    fallbacks,
    points,
    simplices_padded,
    simplex_neighbors,
    circumcentres,
    circumradii_squared,
    circumcircle_valid,
    max_cavity_triangles,
    max_neighbors,
):
    query_count = query_points.shape[0]
    cavity_cap = max_cavity_triangles
    compact_length = 3 * cavity_cap

    mappings = np.full((query_count, max_neighbors), -1, dtype=np.int32)
    sizes = np.zeros(query_count, dtype=np.int32)
    weights = np.zeros((query_count, max_neighbors), dtype=np.float64)
    cavity_sizes = np.zeros(query_count, dtype=np.int32)
    overflow_out = np.zeros(query_count, dtype=np.bool_)
    degenerate_out = np.zeros(query_count, dtype=np.bool_)

    for q in prange(query_count):
        qx = query_points[q, 0]
        qy = query_points[q, 1]
        seed = seeds[q]

        # Outside the mesh: nearest-vertex fallback row, no diagnostics.
        if seed < 0:
            mappings[q, 0] = fallbacks[q]
            weights[q, 0] = 1.0
            sizes[q] = 1
            continue

        # Cavity walk (``_cavity_triangle_indexes_numpy``).
        cavity = np.full(cavity_cap, -1, dtype=np.int64)
        cavity[0] = seed
        count = 1
        cavity_overflow = False
        for position in range(cavity_cap):
            if position >= count:
                break
            triangle_index = cavity[position]
            for edge in range(3):
                candidate = simplex_neighbors[triangle_index, edge]
                if candidate < 0:
                    continue
                seen = False
                for k in range(count):
                    if cavity[k] == candidate:
                        seen = True
                        break
                if seen:
                    continue
                dx = circumcentres[candidate, 0] - qx
                dy = circumcentres[candidate, 1] - qy
                distance_squared = dx * dx + dy * dy
                radius_squared = circumradii_squared[candidate]
                tolerance = 64.0 * EPS * _np_maximum(radius_squared, 1.0)
                if not (
                    circumcircle_valid[candidate]
                    and distance_squared <= radius_squared + tolerance
                ):
                    continue
                if count == cavity_cap:
                    cavity_overflow = True
                else:
                    cavity[count] = candidate
                    count += 1

        # Watson contributions, grouped by vertex id into a sorted list.
        unique_ids = np.full(compact_length, -1, dtype=np.int64)
        compact_weights = np.zeros(compact_length, dtype=np.float64)
        unique_count = 0
        inserted_invalid = False
        vx = np.empty(3, dtype=np.float64)
        vy = np.empty(3, dtype=np.float64)
        vid = np.empty(3, dtype=np.int64)
        rx = np.empty(3, dtype=np.float64)
        ry = np.empty(3, dtype=np.float64)
        ivalid = np.empty(3, dtype=np.bool_)
        contribution = np.empty(3, dtype=np.float64)
        for slot in range(count):
            triangle_index = cavity[slot]
            for j in range(3):
                vertex = simplices_padded[triangle_index, j]
                if vertex < 0:
                    vertex = 0
                vid[j] = vertex
                vx[j] = points[vertex, 0]
                vy[j] = points[vertex, 1]
            ox = circumcentres[triangle_index, 0]
            oy = circumcentres[triangle_index, 1]
            # Circle j passes through the query and the edge opposite j.
            for j in range(3):
                ax = vx[(j + 1) % 3]
                ay = vy[(j + 1) % 3]
                bax = vx[(j + 2) % 3] - ax
                bay = vy[(j + 2) % 3] - ay
                cax = qx - ax
                cay = qy - ay
                denominator = 2.0 * (bax * cay - bay * cax)
                ba2 = bax * bax + bay * bay
                ca2 = cax * cax + cay * cay
                scale = _np_maximum(_np_maximum(ba2, ca2), 1.0)
                valid = abs(denominator) > 32.0 * EPS * scale
                safe_denominator = denominator if valid else 1.0
                offset_x = (ba2 * cay - ca2 * bay) / safe_denominator
                offset_y = (bax * ca2 - cax * ba2) / safe_denominator
                if valid:
                    centre_x = ax + offset_x
                    centre_y = ay + offset_y
                else:
                    centre_x = 0.0
                    centre_y = 0.0
                    inserted_invalid = True
                ivalid[j] = valid
                rx[j] = centre_x - ox
                ry[j] = centre_y - oy
            for j in range(3):
                j1 = (j + 1) % 3
                j2 = (j + 2) % 3
                if ivalid[j1] and ivalid[j2]:
                    contribution[j] = rx[j1] * ry[j2] - ry[j1] * rx[j2]
                else:
                    contribution[j] = 0.0
            for j in range(3):
                vertex = vid[j]
                position = 0
                while position < unique_count and unique_ids[position] < vertex:
                    position += 1
                if position < unique_count and unique_ids[position] == vertex:
                    compact_weights[position] += contribution[j]
                else:
                    for k in range(unique_count, position, -1):
                        unique_ids[k] = unique_ids[k - 1]
                        compact_weights[k] = compact_weights[k - 1]
                    unique_ids[position] = vertex
                    compact_weights[position] = 0.0 + contribution[j]
                    unique_count += 1

        # On-vertex / on-edge tests against the seed triangle.
        s0 = simplices_padded[seed, 0]
        s1 = simplices_padded[seed, 1]
        s2 = simplices_padded[seed, 2]
        if s0 < 0:
            s0 = 0
        if s1 < 0:
            s1 = 0
        if s2 < 0:
            s2 = 0
        p0x = points[s0, 0]
        p0y = points[s0, 1]
        p1x = points[s1, 0]
        p1y = points[s1, 1]
        p2x = points[s2, 0]
        p2y = points[s2, 1]

        d0x = p0x - qx
        d0y = p0y - qy
        d1x = p1x - qx
        d1y = p1y - qy
        d2x = p2x - qx
        d2y = p2y - qy
        dist0 = d0x * d0x + d0y * d0y
        dist1 = d1x * d1x + d1y * d1y
        dist2 = d2x * d2x + d2y * d2y
        local_vertex = s0
        min_distance = dist0
        if dist1 < min_distance:
            local_vertex = s1
            min_distance = dist1
        if dist2 < min_distance:
            local_vertex = s2
            min_distance = dist2

        coordinate_scale = _np_maximum(abs(qx), abs(qy))
        point_scale = _np_maximum(abs(p0x), abs(p0y))
        point_scale = _np_maximum(point_scale, abs(p1x))
        point_scale = _np_maximum(point_scale, abs(p1y))
        point_scale = _np_maximum(point_scale, abs(p2x))
        point_scale = _np_maximum(point_scale, abs(p2y))
        coordinate_scale = _np_maximum(_np_maximum(coordinate_scale, point_scale), 1.0)
        vertex_tolerance = 64.0 * EPS * coordinate_scale
        vertex_tolerance = vertex_tolerance * vertex_tolerance
        on_vertex = min_distance <= vertex_tolerance

        seed_denominator = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
        if seed_denominator == 0.0:
            seed_denominator = 1.0
        bary0 = (d1x * d2y - d1y * d2x) / seed_denominator
        bary1 = (d2x * d0y - d2y * d0x) / seed_denominator
        bary2 = (d0x * d1y - d0y * d1x) / seed_denominator
        edge_tolerance = 128.0 * EPS
        min_bary = _np_maximum(-abs(bary0), -abs(bary1))
        min_bary = -_np_maximum(min_bary, -abs(bary2))
        on_edge = (not on_vertex) and (min_bary <= edge_tolerance)

        weight_sum = _pairwise_sum(compact_weights)
        degenerate = (
            (not on_vertex)
            and (not on_edge)
            and (
                inserted_invalid
                or (not np.isfinite(weight_sum))
                or abs(weight_sum) <= TINY
            )
        )
        safe_sum = weight_sum if abs(weight_sum) > TINY else 1.0

        if on_vertex:
            mappings[q, 0] = local_vertex
            weights[q, 0] = 1.0
            size = 1
        elif on_edge:
            # Barycentric row of the containing triangle (third weight 0).
            if max_neighbors > 0:
                mappings[q, 0] = s0
                weights[q, 0] = bary0
            if max_neighbors > 1:
                mappings[q, 1] = s1
                weights[q, 1] = bary1
            if max_neighbors > 2:
                mappings[q, 2] = s2
                weights[q, 2] = bary2
            size = 3
        else:
            size = 0
            for k in range(max_neighbors):
                if k >= compact_length or k >= unique_count:
                    break
                weight = compact_weights[k] / safe_sum
                if weight > 0.0:
                    mappings[q, k] = unique_ids[k]
                    weights[q, k] = weight
                    size += 1

        neighbor_overflow = (
            (not on_vertex) and (not on_edge) and unique_count > max_neighbors
        )
        failed = cavity_overflow or neighbor_overflow or degenerate
        if failed:
            for k in range(max_neighbors):
                weights[q, k] = np.nan
            # Keep a failed row at size >= 1 so its NaN weight propagates (see the reference
            # ``_sibson_single_from_tables``, PyAutoArray#629).
            if size == 0 and max_neighbors > 0:
                mappings[q, 0] = local_vertex
                size = 1

        sizes[q] = size
        cavity_sizes[q] = count
        overflow_out[q] = cavity_overflow or neighbor_overflow
        degenerate_out[q] = degenerate

    return mappings, sizes, weights, cavity_sizes, overflow_out, degenerate_out
