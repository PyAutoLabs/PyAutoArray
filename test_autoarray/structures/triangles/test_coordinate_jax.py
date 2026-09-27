"""
Tests of the JAX `CoordinateArrayTriangles` vertex table and its containment path.

`CoordinateArrayTriangles._vertices_and_indices` returns the flat, non-deduplicated ``(3N, 2)``
vertex table with an ``arange`` index map (PyAutoArray#568, autolens_profiling#297): under ``jit`` a
static-size ``jnp.unique`` produced 3N rows anyway and only added a lexicographic sort. These tests
pin that the table round-trips to the triangles exactly, that NaN padding stays NaN and is never
returned by containment, that the kept triangles match the deduplicating NumPy sibling, and that
the traced containment carries no sort.

The static initial lattice (`for_limits_and_scale(..., static_vertices=True)`, point-source CPU
phase 3) instead carries the cached `static_vertex_table` of geometrically unique vertices: the
tests below pin its size for the PointSolver lattice (11 859 of 69 849 slots), that it gathers back
to the triangles within a few ulp, that containment matches the flat table and the NumPy sibling,
that derived lattices drop it, and that the cache is keyed on geometry and read-only.
"""

import importlib
import re

import numpy as np
import pytest


# Plain data, defined outside the jax guard: the module-level @parametrize decorators read them
# at collection time, including on the NumPy-only matrix env.
LIMITS = dict(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0, scale=0.5)

# The PointSolver's default image-plane extent for a 100x100, 0.2" grid.
SOLVER_LIMITS = dict(y_min=-9.9, y_max=9.9, x_min=-9.9, x_max=9.9, scale=0.2)

# Step-0 containment routes (`autoarray.structures.triangles.array._STEP0_CONTAINMENT`,
# PyAutoArray#579). "gather" is the general path every other route must reproduce bit-for-bit.
STEP0_ROUTES = ("gather", "nopad", "components", "structured")

# Fuzz geometries for the step-0 routes, as (name, limits, deflect, n_extra): the solver lattice
# (199 x 117 triangles, odd key-row count -> padded grid), the same lattice with an SIS-like
# deflection of its traced vertex table, and three asymmetric lattices -- alternating key-row
# widths 14/15 (even column count), alternating widths 10/11 with an odd key-row count (padded),
# and an even row count at a non-round scale. ``n_extra`` random points are added to every
# vertex, edge midpoint and centroid of the (possibly deflected) lattice; the solver lattices
# instead sample a subset of those (see `_fuzz_points`).
STEP0_FUZZ_GEOMETRIES = (
    ("solver", SOLVER_LIMITS, False, 2048),
    ("solver_deflected", SOLVER_LIMITS, True, 1024),
    ("alternating", dict(y_min=-3.1, y_max=2.2, x_min=-1.7, x_max=2.9, scale=0.2), False, 8192),
    ("alternating_padded", dict(y_min=-2.1, y_max=1.1, x_min=-0.6, x_max=2.5, scale=0.2), True, 8192),
    ("even_rows", dict(y_min=-0.9, y_max=1.7, x_min=-2.2, x_max=0.4, scale=0.13), False, 8192),
)

# jax is an `[optional]` extra and is absent on the NumPy-only matrix env: every test in this module
# skips there.
if importlib.util.find_spec("jax") is None:
    pytestmark = pytest.mark.skip(reason="requires jax (the [optional] extras)")

    def test__placeholder_requires_jax():  # pragma: no cover
        pass

else:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from autoarray.structures.triangles import array as triangles_array
    from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE
    from autoarray.structures.triangles.coordinate_array import (
        CoordinateArrayTriangles,
        static_vertex_table,
    )
    from autoarray.structures.triangles.coordinate_array_np import (
        CoordinateArrayTrianglesNp,
    )
    from autoarray.structures.triangles.shape import Point

    def _lattice():
        return CoordinateArrayTriangles.for_limits_and_scale(**LIMITS)

    def _static_lattice(limits=LIMITS):
        return CoordinateArrayTriangles.for_limits_and_scale(
            **limits, static_vertices=True
        )

    def _max_ulps(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.max(np.abs(a - b) / np.spacing(np.maximum(np.abs(a), 1.0))))

    def _lattice_np():
        return CoordinateArrayTrianglesNp.for_limits_and_scale(**LIMITS)

    def _round_trip(coordinates):
        triangles = CoordinateArrayTriangles(coordinates=coordinates, side_length=0.5)
        return (
            triangles.triangles,
            triangles.vertices.reshape(-1, 3, 2),
            triangles.with_vertices(triangles.vertices).triangles,
        )

    def _containing(coordinates):
        triangles = CoordinateArrayTriangles(coordinates=coordinates, side_length=0.5)
        return triangles.with_vertices(triangles.vertices).containing_indices(
            Point(0.1, 0.2)
        )

    def _source_points(triangles_np):
        """
        Five source points on the lattice: two interior points, one on a lattice vertex, one on the
        midpoint of an edge shared by two triangles, and one centroid. The edge is the horizontal one
        (constant element 1), whose midpoint the barycentric test keeps on both sides; the midpoints
        of the slanted edges round outside both triangles on NumPy and JAX alike.
        """
        tri = np.asarray(triangles_np.triangles)
        interior = int(np.argmin(np.linalg.norm(tri.mean(axis=1), axis=1)))
        vertex = tri[interior, 0]
        edge_midpoint = 0.5 * (tri[interior, 1] + tri[interior, 2])
        centroid = tri[interior].mean(axis=0)
        return [
            (0.1, 0.2),
            (-0.37, 0.41),
            tuple(vertex),
            tuple(edge_midpoint),
            tuple(centroid),
        ]

    def _kept_rows(triangles, point):
        kept = triangles.for_indexes(
            triangles.with_vertices(triangles.vertices).containing_indices(
                Point(*point)
            )
        )
        rows = np.asarray(kept.triangles).reshape(-1, 6)
        rows = rows[np.all(np.isfinite(rows), axis=1)]
        return {tuple(np.round(row, 8)) for row in rows}


def test__vertices_round_trip_to_triangles():
    triangles = _lattice()

    assert triangles.vertices.shape == (3 * triangles.coordinates.shape[0], 2)
    assert np.array_equal(
        np.asarray(triangles.indices),
        np.arange(3 * triangles.coordinates.shape[0]).reshape(-1, 3),
    )
    assert np.array_equal(
        np.asarray(triangles.vertices.reshape(-1, 3, 2)),
        np.asarray(triangles.triangles),
    )
    assert np.array_equal(
        np.asarray(triangles.with_vertices(triangles.vertices).triangles),
        np.asarray(triangles.triangles),
    )


def test__vertices_round_trip_to_triangles__jit():
    """
    Compared inside one jitted computation: XLA may fuse the eager and jitted triangle arithmetic
    differently (the two differ by up to 1 ulp), so the exact round trip is asserted between the
    outputs of the same compiled program.
    """
    triangles, vertices, with_vertices = jax.jit(_round_trip)(_lattice().coordinates)

    assert np.array_equal(np.asarray(vertices), np.asarray(triangles))
    assert np.array_equal(np.asarray(with_vertices), np.asarray(triangles))


def test__nan_padding_stays_nan_and_is_never_contained():
    lattice = _lattice()
    point = tuple(np.asarray(lattice.triangles[0]).mean(axis=0))

    padded = lattice.for_indexes(jnp.array([0, 2, -1, -1]))

    coordinates = np.asarray(padded.coordinates)
    assert np.all(np.isfinite(coordinates[:2]))
    assert np.all(np.isnan(coordinates[2:]))

    triangles = np.asarray(padded.with_vertices(padded.vertices).triangles)
    assert np.all(np.isfinite(triangles[:2]))
    assert np.all(np.isnan(triangles[2:]))

    containing = np.asarray(
        padded.with_vertices(padded.vertices).containing_indices(Point(*point))
    )
    assert containing[0] == 0
    assert np.all(containing[1:] == -1)


@pytest.mark.parametrize("point_index", range(5))
def test__kept_triangles_match_numpy(point_index):
    triangles_np = _lattice_np()
    triangles = _lattice()
    point = _source_points(triangles_np)[point_index]

    kept_np = _kept_rows(triangles_np, point)
    kept = _kept_rows(triangles, point)

    assert 0 < len(kept_np) < MAX_CONTAINING_SIZE
    assert kept == kept_np


def test__numpy_vertices_are_deduplicated():
    triangles_np = _lattice_np()
    n = triangles_np.coordinates.shape[0]

    assert triangles_np.vertices.shape[0] < 3 * n
    assert np.unique(triangles_np.vertices, axis=0).shape[0] == (
        triangles_np.vertices.shape[0]
    )
    assert np.array_equal(
        triangles_np.vertices[triangles_np.indices], triangles_np.triangles
    )


def test_no_sort():
    coordinates = _lattice().coordinates
    compiled = jax.jit(_containing).lower(coordinates).compile().as_text()

    sorts = re.findall(r"\bsort\b", compiled, flags=re.IGNORECASE)
    assert not sorts, f"traced containment carries {len(sorts)} sort op(s)"


def test__default_lattice_has_no_vertex_table():
    assert _lattice().vertex_table is None


def test__static_vertex_table__solver_lattice_counts():
    """
    The +-9.9" / 0.2" lattice has 23 283 triangles, 69 849 vertex slots, 28 665 exact-float distinct
    rows and 11 859 geometrically distinct points -- the same count the deduplicating NumPy sibling
    gives once its ulp-level duplicates are rounded together.
    """
    lattice = _static_lattice(SOLVER_LIMITS)
    vertices = np.asarray(lattice.vertices)
    indices = np.asarray(lattice.indices)

    assert lattice.coordinates.shape[0] == 23283
    assert vertices.shape == (11859, 2)
    assert indices.shape == (23283, 3)
    assert indices.min() == 0 and indices.max() == 11858
    assert np.unique(np.round(vertices, 9), axis=0).shape[0] == 11859

    triangles_np = CoordinateArrayTrianglesNp.for_limits_and_scale(**SOLVER_LIMITS)
    flat_np = np.asarray(triangles_np.triangles).reshape(-1, 2)
    assert np.unique(flat_np, axis=0).shape[0] == 28665
    assert np.unique(np.round(flat_np, 9), axis=0).shape[0] == 11859


@pytest.mark.parametrize("limits", [LIMITS, SOLVER_LIMITS])
def test__static_vertex_table_round_trips_to_triangles(limits):
    static = _static_lattice(limits)
    default = CoordinateArrayTriangles.for_limits_and_scale(**limits)

    assert np.array_equal(
        np.asarray(static.coordinates), np.asarray(default.coordinates)
    )

    gathered = np.asarray(static.vertices)[np.asarray(static.indices)]
    assert gathered.shape == np.asarray(default.triangles).shape
    assert _max_ulps(gathered, default.triangles) <= 4

    via_with_vertices = np.asarray(static.with_vertices(static.vertices).triangles)
    assert np.array_equal(via_with_vertices, gathered)


@pytest.mark.parametrize("point_index", range(5))
def test__static_vertices_kept_triangles_match_numpy(point_index):
    triangles_np = _lattice_np()
    point = _source_points(triangles_np)[point_index]

    kept_np = _kept_rows(triangles_np, point)
    kept = _kept_rows(_static_lattice(), point)

    assert 0 < len(kept_np) < MAX_CONTAINING_SIZE
    assert kept == kept_np


@pytest.mark.parametrize("point_index", range(5))
def test__static_vertices_containing_indices_match_default(point_index):
    point = Point(*_source_points(_lattice_np())[point_index])
    static = _static_lattice()
    default = _lattice()

    static_indices = np.asarray(
        static.with_vertices(static.vertices).containing_indices(point)
    )
    default_indices = np.asarray(
        default.with_vertices(default.vertices).containing_indices(point)
    )

    assert set(static_indices[static_indices >= 0]) == set(
        default_indices[default_indices >= 0]
    )


def test__static_vertices_containing_indices__jit_matches_eager():
    static = _static_lattice(SOLVER_LIMITS)
    point = Point(0.13, -0.27)

    def containing():
        return static.with_vertices(static.vertices).containing_indices(point)

    assert np.array_equal(np.asarray(jax.jit(containing)()), np.asarray(containing()))


def test__derived_lattices_drop_the_vertex_table():
    static = _static_lattice()
    n = static.coordinates.shape[0]

    for derived in (
        static.for_indexes(jnp.arange(4)),
        static.up_sample(),
        static.neighborhood(),
    ):
        assert derived.vertex_table is None
        assert derived.vertices.shape == (3 * derived.coordinates.shape[0], 2)

    assert static.vertices.shape[0] < 3 * n


def test__static_vertex_table_is_cached_per_geometry():
    first = static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.5)

    assert static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.5) is first
    assert _static_lattice().vertex_table is first

    finer = static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.25)
    assert finer is not first
    assert finer[0].shape[0] > first[0].shape[0]


def test__static_vertex_table_is_read_only():
    vertices, indices = static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.5)

    assert not vertices.flags.writeable
    assert not indices.flags.writeable
    with pytest.raises(ValueError):
        vertices[0, 0] = 1.0
    with pytest.raises(ValueError):
        indices[0, 0] = 1


def test_no_sort__static_vertices():
    static = _static_lattice(SOLVER_LIMITS)

    def containing():
        return static.with_vertices(static.vertices).containing_indices(Point(0.1, 0.2))

    compiled = jax.jit(containing).lower().compile().as_text()

    sorts = re.findall(r"\bsort\b", compiled, flags=re.IGNORECASE)
    assert not sorts, f"traced containment carries {len(sorts)} sort op(s)"



# ---------------------------------------------------------------------------------------------
# Step-0 containment routes (point-source CPU phase 4b, PyAutoArray#579)
# ---------------------------------------------------------------------------------------------

# Points per jitted, vmapped call; the last chunk is padded by repeating its final point.
_FUZZ_CHUNK = 1024

# Set PYAUTO_STEP0_FUZZ_SCALE (an integer >= 1) to multiply the random and sampled fuzz points,
# e.g. 4 for ~1.5e5 points per route. The default (~5.7e4 points per route, every vertex of the
# solver lattice included) keeps this block to ~30 s on a laptop.
_FUZZ_SCALE = int(__import__("os").environ.get("PYAUTO_STEP0_FUZZ_SCALE", "1"))


def _deflected(vertices):
    """An SIS-like (theta_E = 1.6) deflection of a vertex table: an irregular traced table."""
    vertices = np.asarray(vertices)
    radius = np.sqrt(np.sum(vertices**2, axis=1, keepdims=True) + 1e-2)
    return vertices - 1.6 * vertices / radius


def _fuzz_setup(limits, deflect):
    lattice = _static_lattice(limits)
    vertices = np.asarray(lattice.vertices)
    if deflect:
        vertices = _deflected(vertices)
    return lattice, vertices


def _fuzz_points(limits, deflect, n_extra):
    """
    Every vertex of the (possibly deflected) table -- the exact step-0 ties -- plus edge
    midpoints, centroids and uniform random points over the table's bounding box. On the
    23 283-triangle solver lattices the midpoints and centroids are a random subset.
    """
    lattice, vertices = _fuzz_setup(limits, deflect)
    triangles = vertices[np.asarray(lattice.indices)]
    rng = np.random.default_rng(579)

    midpoints = np.concatenate(
        [0.5 * (triangles[:, a] + triangles[:, b]) for a, b in ((0, 1), (1, 2), (2, 0))]
    )
    centroids = triangles.mean(axis=1)
    n_extra = n_extra * _FUZZ_SCALE
    if triangles.shape[0] > 5000:
        midpoints = midpoints[rng.choice(midpoints.shape[0], n_extra // 2, replace=False)]
        centroids = centroids[rng.choice(centroids.shape[0], n_extra // 4, replace=False)]
        if deflect:
            vertices = vertices[rng.choice(vertices.shape[0], n_extra, replace=False)]

    low, high = triangles.reshape(-1, 2).min(axis=0), triangles.reshape(-1, 2).max(axis=0)
    random = rng.uniform(low, high, size=(n_extra, 2))

    return np.concatenate((vertices, midpoints, centroids, random))


def _containing_by_route(route, lattice, vertices, points, monkeypatch):
    """
    `containing_indices` of every point under ``route``, with the vertex table a traced argument
    (as on the solver path) under ``jit`` + ``vmap``. A fresh closure and ``jax.clear_caches()``
    force a re-trace, since the route is read at trace time.
    """
    monkeypatch.setattr(triangles_array, "_STEP0_CONTAINMENT", route)
    jax.clear_caches()

    def containing(vertices, point):
        point = Point(point[0], point[1])
        triangles = lattice.with_vertices(vertices)
        # The routed mask itself, kept to _FUZZ_WIDE entries: `containing_indices` truncates at
        # MAX_CONTAINING_SIZE, which a folded (deflected) table can exceed.
        inside = triangles._step0_point_mask(point)
        if inside is None:
            inside = point.mask(triangles.triangles)
        wide = jnp.where(inside, size=_FUZZ_WIDE, fill_value=-1)[0]
        return jnp.concatenate(
            (triangles.containing_indices(point), wide, jnp.sum(inside)[None])
        )

    batched = jax.jit(jax.vmap(containing, in_axes=(None, 0)))
    vertices = jnp.asarray(vertices)

    n = points.shape[0]
    padded = np.concatenate(
        (points, np.repeat(points[-1:], (-n) % _FUZZ_CHUNK, axis=0))
    )
    out = [
        np.asarray(batched(vertices, jnp.asarray(padded[i : i + _FUZZ_CHUNK])))
        for i in range(0, padded.shape[0], _FUZZ_CHUNK)
    ]
    return np.concatenate(out)[:n]


_FUZZ_REFERENCE = {}

# Width of the routed-mask comparison (see `_containing_by_route`).
_FUZZ_WIDE = 64


def _fuzz_reference(name, limits, deflect, n_extra, monkeypatch):
    if name not in _FUZZ_REFERENCE:
        lattice, vertices = _fuzz_setup(limits, deflect)
        points = _fuzz_points(limits, deflect, n_extra)
        _FUZZ_REFERENCE[name] = (
            points,
            _containing_by_route("gather", lattice, vertices, points, monkeypatch),
        )
    return _FUZZ_REFERENCE[name]


@pytest.mark.parametrize("name, limits, deflect, n_extra", STEP0_FUZZ_GEOMETRIES)
def test__step0_fuzz_geometries_have_a_structured_layout(name, limits, deflect, n_extra):
    """
    Every fuzz geometry takes the structured route for real (no silent fall-back), and the
    layout is hashable int-only aux data cached per geometry.
    """
    from autoarray.structures.triangles.coordinate_array import static_lattice_layout

    lattice = _static_lattice(limits)
    layout = lattice.with_vertices(lattice.vertices).step0_layout

    assert layout is not None and layout.grid is not None
    assert layout.n_rows * layout.n_cols == lattice.coordinates.shape[0]
    assert hash(layout) == hash(static_lattice_layout(*[float(limits[k]) for k in (
        "y_min", "y_max", "x_min", "x_max", "scale")]))
    assert static_lattice_layout(
        *[float(limits[k]) for k in ("y_min", "y_max", "x_min", "x_max", "scale")]
    ) is layout


@pytest.mark.parametrize("route", [r for r in STEP0_ROUTES if r != "gather"])
@pytest.mark.parametrize("name, limits, deflect, n_extra", STEP0_FUZZ_GEOMETRIES)
def test__step0_route_is_bit_identical_to_gather(
    name, limits, deflect, n_extra, route, monkeypatch
):
    """
    The bit-identity fuzz: each route keeps exactly the triangles the general gather path keeps,
    in the same order, for every vertex (exact ties), edge midpoint, centroid and random point
    of each geometry, with the vertex table traced under ``jit`` + ``vmap``.
    """
    points, reference = _fuzz_reference(name, limits, deflect, n_extra, monkeypatch)
    lattice, vertices = _fuzz_setup(limits, deflect)

    result = _containing_by_route(route, lattice, vertices, points, monkeypatch)

    mismatched = np.flatnonzero(np.any(result != reference, axis=1))
    assert mismatched.size == 0, (
        f"{route} on {name}: {mismatched.size} of {points.shape[0]} points differ from gather, "
        f"first at {points[mismatched[0]]}: {result[mismatched[0]]} vs {reference[mismatched[0]]}"
    )

    # The fuzz really exercises ties (vertices kept by several triangles) and misses, and the
    # wide comparison covers every contained triangle.
    contained = reference[:, -1]
    assert np.any(contained >= 2)
    assert np.any(contained == 0)
    assert np.all(contained < _FUZZ_WIDE)


@pytest.mark.parametrize("route", STEP0_ROUTES)
def test__step0_refinement_path_is_unchanged(route, monkeypatch):
    """
    Only the static step-0 lattice carries a layout. Every derived lattice of a solver-style
    refinement (kept -> neighbourhood -> up-sample) drops it, so its containment takes the
    general path on every route and matches the gather path exactly; a non-`Point` shape on the
    static lattice takes the general path too.
    """
    from autoarray.structures.triangles.shape import Circle

    lattice = _static_lattice(SOLVER_LIMITS)
    vertices = _deflected(lattice.vertices)
    source = tuple(vertices[4321])

    def refine():
        step0 = lattice.with_vertices(jnp.asarray(vertices))
        kept = lattice.for_indexes(step0.containing_indices(Point(*source)))
        up_sampled = kept.neighborhood().up_sample()
        refined = up_sampled.with_vertices(jnp.asarray(_deflected(up_sampled.vertices)))
        circle = step0.containing_indices(Circle(source[0], source[1], radius=0.3))
        return (
            [kept, kept.neighborhood(), up_sampled],
            refined,
            (
                np.asarray(step0.containing_indices(Point(*source))),
                np.asarray(refined.containing_indices(Point(*source))),
                np.asarray(circle),
            ),
        )

    monkeypatch.setattr(triangles_array, "_STEP0_CONTAINMENT", route)
    derived, refined, outputs = refine()

    for triangles in derived:
        assert triangles.step0_layout is None
        assert triangles.vertex_table is None
    assert refined.step0_layout is None

    monkeypatch.setattr(triangles_array, "_STEP0_CONTAINMENT", "gather")
    _, _, expected = refine()

    for output, reference in zip(outputs, expected):
        assert np.array_equal(output, reference)
    assert np.sum(outputs[0] >= 0) >= 2


def test__step0_default_route_has_no_triangle_gather():
    """
    The HLO guard: the optimised HLO of the default step-0 containment on the solver lattice
    carries no gather that materialises the ``(23 283, 3, 2)`` triangle array (XLA emits it as a
    ``f64[69849,1,2]`` gather on the general path). The vertex table is a traced argument -- a
    closure would let XLA constant-fold the gather away. Red on main (the general path).
    """
    lattice = _static_lattice(SOLVER_LIMITS)
    n = lattice.coordinates.shape[0]

    def containing(vertices, point):
        return lattice.with_vertices(vertices).containing_indices(
            Point(point[0], point[1])
        )

    jax.clear_caches()
    compiled = (
        jax.jit(containing)
        .lower(lattice.vertices, jnp.array([0.1, 0.2]))
        .compile()
        .as_text()
    )

    gathered = []
    for shape in re.findall(r"= \w+\[([\d,]*)\]\S* gather\(", compiled):
        gathered.append(int(np.prod([int(d) for d in shape.split(",") if d])))

    assert f"f64[{n},3,2]" not in compiled
    assert all(size < 3 * n * 2 for size in gathered), (
        f"step-0 containment gathers {gathered} elements; the triangle array is {3 * n * 2}"
    )


# A 16- and a 17-sheet fold: the static lattice's vertices are traced through z -> z**k (z = x + iy),
# so a source point at |w| = 1 has k preimages on the unit circle, each in the interior of exactly one
# traced triangle -- a k-member containing set with no boundary ties. The PointSolver's uncapped
# step-0 maximum on 200 prior draws was 17 (prior draw 12, point-source CPU phase 4c,
# PyAutoArray#583), which the old cap of 15 truncated silently.
FOLD_LIMITS = dict(y_min=-1.3, y_max=1.3, x_min=-1.3, x_max=1.3, scale=0.05)


@pytest.mark.parametrize("sheets", [16, 17])
def test__containing_set_above_the_old_cap_is_not_truncated(sheets):
    """
    Under ``jit``, a 16- or 17-triangle containing set keeps every member at the default
    `MAX_CONTAINING_SIZE` (20), padded to ``(20,)``; the old cap of 15, passed explicitly, drops
    some. Red at ``MAX_CONTAINING_SIZE = 15``.
    """
    lattice = _static_lattice(FOLD_LIMITS)
    z = lattice.vertices[:, 1] + 1j * lattice.vertices[:, 0]
    w = z**sheets
    traced = jnp.stack([w.imag, w.real], axis=1)
    point = (float(np.sin(0.3)), float(np.cos(0.3)))

    uncapped = np.flatnonzero(
        np.asarray(Point(*point).mask(lattice.with_vertices(traced).triangles))
    )
    assert len(uncapped) == sheets

    def containing(vertices, max_containing_size=None):
        kwargs = (
            {}
            if max_containing_size is None
            else {"max_containing_size": max_containing_size}
        )
        triangles = triangles_array.ArrayTriangles(
            indices=lattice.indices,
            vertices=vertices,
            step0_layout=lattice.step0_layout,
            **kwargs,
        )
        return triangles.containing_indices(Point(*point))

    assert lattice.with_vertices(traced).max_containing_size == MAX_CONTAINING_SIZE

    kept = np.asarray(jax.jit(containing)(traced))
    assert np.array_equal(np.sort(kept[kept >= 0]), uncapped)
    assert kept.shape == (20,)

    kept_via_lattice = np.asarray(
        jax.jit(lambda v: lattice.with_vertices(v).containing_indices(Point(*point)))(
            traced
        )
    )
    assert np.array_equal(kept_via_lattice, kept)

    old_cap = np.asarray(jax.jit(lambda v: containing(v, 15))(traced))
    assert old_cap.shape == (15,)
    assert np.sum(old_cap >= 0) == 15 < sheets
