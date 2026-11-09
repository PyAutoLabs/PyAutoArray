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

from autoarray import numba_util

# float64 machine constants as module globals, which numba freezes into the
# compiled code (and the on-disk cache) as literals.
EPS = 2.220446049250313e-16
TINY = 2.2250738585072014e-308
# Same values as ``sibson.SIBSON_ZERO_AREA_TOLERANCE`` / ``SIBSON_NEGATIVE_AREA_TOLERANCE`` (PyAutoArray#610).
ZERO_AREA_TOLERANCE = 1.0e-12
NEGATIVE_AREA_TOLERANCE = 1.0e-10


@numba.njit(cache=True, nogil=True)
def _np_maximum(a, b):
    # np.maximum semantics: NaN propagates.
    if a != a or b != b:
        return np.nan
    return a if a >= b else b


@numba.njit(cache=True, nogil=True)
def _credit(unique_ids, compact_weights, unique_count, vertex, value):
    """Add ``value`` to ``vertex``'s entry of the vertex-sorted compact list; return the new count."""
    position = 0
    while position < unique_count and unique_ids[position] < vertex:
        position += 1
    if position < unique_count and unique_ids[position] == vertex:
        compact_weights[position] += value
        return unique_count
    for k in range(unique_count, position, -1):
        unique_ids[k] = unique_ids[k - 1]
        compact_weights[k] = compact_weights[k - 1]
    unique_ids[position] = vertex
    compact_weights[position] = 0.0 + value
    return unique_count + 1


def _sibson_from_tables(
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

        # Natural-neighbour areas from the cavity BOUNDARY, credited per cavity-triangle edge and grouped by
        # vertex id into a sorted list (``_sibson_single_from_tables`` documents the decomposition,
        # PyAutoArray#610). Slot-major, edges j = 0, 1, 2: the same order as the reference.
        unique_ids = np.full(compact_length, -1, dtype=np.int64)
        compact_weights = np.zeros(compact_length, dtype=np.float64)
        unique_count = 0
        interior_collinear = False
        hull_found = False
        hull_a = 0
        hull_b = 0
        vx = np.empty(3, dtype=np.float64)
        vy = np.empty(3, dtype=np.float64)
        vid = np.empty(3, dtype=np.int64)
        vertex_value = np.empty(3, dtype=np.float64)
        for slot in range(count):
            triangle_index = cavity[slot]
            for j in range(3):
                vertex = simplices_padded[triangle_index, j]
                if vertex < 0:
                    vertex = 0
                vid[j] = vertex
                vx[j] = points[vertex, 0]
                vy[j] = points[vertex, 1]
            orientation = (vx[1] - vx[0]) * (vy[2] - vy[0]) - (vy[1] - vy[0]) * (
                vx[2] - vx[0]
            )
            sign = -1.0 if orientation < 0.0 else 1.0
            rtx = circumcentres[triangle_index, 0] - qx
            rty = circumcentres[triangle_index, 1] - qy
            vertex_value[0] = 0.0
            vertex_value[1] = 0.0
            vertex_value[2] = 0.0
            for j in range(3):
                ja = (j + 1) % 3
                jb = (j + 2) % 3
                a_vertex = vid[ja]
                b_vertex = vid[jb]
                other = simplex_neighbors[triangle_index, j]
                # The cavity walk's own circumcircle test (see the reference body).
                other_in_cavity = False
                if other >= 0 and circumcircle_valid[other]:
                    odx = circumcentres[other, 0] - qx
                    ody = circumcentres[other, 1] - qy
                    other_radius_squared = circumradii_squared[other]
                    other_in_cavity = odx * odx + ody * ody <= other_radius_squared + (
                        64.0 * EPS * _np_maximum(other_radius_squared, 1.0)
                    )
                value_b = 0.0
                value_a = 0.0
                if other_in_cavity:
                    if triangle_index < other:
                        rox = circumcentres[other, 0] - qx
                        roy = circumcentres[other, 1] - qy
                        term = 0.5 * (rtx * roy - rty * rox)
                        value_b = term
                        value_a = -term
                else:
                    # New circumcentre of (a, b, query), as ``circumcircles_from(a, b, query)``.
                    ax = vx[ja]
                    ay = vy[ja]
                    bax = vx[jb] - ax
                    bay = vy[jb] - ay
                    cax = qx - ax
                    cay = qy - ay
                    denominator = 2.0 * (bax * cay - bay * cax)
                    ba2 = bax * bax + bay * bay
                    ca2 = cax * cax + cay * cay
                    scale = _np_maximum(_np_maximum(ba2, ca2), 1.0)
                    if abs(denominator) > 32.0 * EPS * scale:
                        rnx = ax + (ba2 * cay - ca2 * bay) / denominator - qx
                        rny = ay + (bax * ca2 - cax * ba2) / denominator - qy
                        boundary_term = 0.5 * (rtx * rny - rty * rnx)
                        mbx = 0.5 * (vx[jb] - qx)
                        mby = 0.5 * (vy[jb] - qy)
                        max_ = 0.5 * (vx[ja] - qx)
                        may = 0.5 * (vy[ja] - qy)
                        closure_b = 0.5 * (rnx * mby - rny * mbx)
                        closure_a = 0.5 * (max_ * rny - may * rnx)
                        value_b = boundary_term + closure_b
                        value_a = -boundary_term + closure_a
                    elif other < 0:
                        if not hull_found:
                            hull_found = True
                            hull_a = a_vertex
                            hull_b = b_vertex
                    else:
                        interior_collinear = True
                vertex_value[jb] += value_b
                vertex_value[ja] += value_a
            for j in range(3):
                unique_count = _credit(
                    unique_ids, compact_weights, unique_count, vid[j], sign * vertex_value[j]
                )

        # On-vertex test against the seed triangle.
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
        on_hull_edge = (not on_vertex) and hull_found

        # Snap round-off-sized areas to 0 before normalising; a clearly negative one is degenerate.
        # One pass over the live entries only (the rest of the compact list is padding); parity with the
        # reference's pairwise sums is to round-off, which the tolerances absorb.
        total = 0.0
        for k in range(unique_count):
            total += compact_weights[k]
        magnitude = abs(total)
        significant_negative = False
        weight_sum = 0.0
        for k in range(unique_count):
            if compact_weights[k] < -NEGATIVE_AREA_TOLERANCE * magnitude:
                significant_negative = True
            if compact_weights[k] <= ZERO_AREA_TOLERANCE * magnitude:
                compact_weights[k] = 0.0
            else:
                weight_sum += compact_weights[k]
        degenerate = (
            (not on_vertex)
            and (not on_hull_edge)
            and (
                interior_collinear
                or significant_negative
                or (not np.isfinite(total))
                or abs(weight_sum) <= TINY
            )
        )
        safe_sum = weight_sum if abs(weight_sum) > TINY else 1.0

        if on_vertex:
            mappings[q, 0] = local_vertex
            weights[q, 0] = 1.0
            size = 1
        elif on_hull_edge:
            # Linear interpolation between the hull edge's endpoints (the Sibson limit on the hull).
            ex = points[hull_b, 0] - points[hull_a, 0]
            ey = points[hull_b, 1] - points[hull_a, 1]
            edge_length_squared = ex * ex + ey * ey
            if edge_length_squared <= 0.0:
                edge_length_squared = 1.0
            along = (
                (qx - points[hull_a, 0]) * ex + (qy - points[hull_a, 1]) * ey
            ) / edge_length_squared
            if along < 0.0:
                along = 0.0
            if along > 1.0:
                along = 1.0
            if max_neighbors > 0:
                mappings[q, 0] = hull_a
                weights[q, 0] = 1.0 - along
            if max_neighbors > 1:
                mappings[q, 1] = hull_b
                weights[q, 1] = along
            size = 2
        else:
            # Selected entries are written to the front, in vertex order, with no -1 holes.
            size = 0
            for k in range(max_neighbors):
                if k >= compact_length or k >= unique_count:
                    break
                weight = compact_weights[k] / safe_sum
                if weight > 0.0:
                    mappings[q, size] = unique_ids[k]
                    weights[q, size] = weight
                    size += 1

        neighbor_overflow = (
            (not on_vertex) and (not on_hull_edge) and unique_count > max_neighbors
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


# One body, compiled twice: ``prange`` runs as ``range`` under ``parallel=False``, so the two
# kernels compute identical results. The serial one is a renamed copy so both hit numba's
# on-disk cache (see ``numba_util.function_copy``).
_sibson_from_tables_parallel = numba.njit(cache=True, parallel=True, nogil=True)(
    _sibson_from_tables
)
_sibson_from_tables_serial = numba.njit(cache=True, nogil=True)(
    numba_util.function_copy(_sibson_from_tables, "_sibson_from_tables_serial")
)
