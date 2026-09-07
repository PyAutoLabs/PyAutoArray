import numpy as np
from scipy.spatial import Delaunay, cKDTree

from autoarray.inversion.mesh.interpolator.delaunay import (
    _nearest_vertex_seed_from,
    _walk_from_seed,
    pix_indexes_delaunay_walk_from,
    scipy_delaunay_tri_only,
)


def _blob_ring_mesh(n, rng):
    """Adaptive-density mesh (compact blob + annulus) mimicking a lensed
    source's Hilbert mesh, so triangle sizes span a large dynamic range."""
    n_blob = n // 2
    blob = rng.normal(size=(n_blob, 2)) * 0.15
    ang = rng.uniform(0, 2 * np.pi, size=n - n_blob)
    rad = 1.0 + rng.normal(size=n - n_blob) * 0.12
    ring = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    return np.concatenate([blob, ring])


def _locate(query, points):
    simplices_padded, simplex_neighbors, vertex_simplex = scipy_delaunay_tri_only(
        points
    )
    return pix_indexes_delaunay_walk_from(
        query_points=query,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
    )


def test__tri_only_callback__tables_match_triangulation():
    rng = np.random.default_rng(1)
    points = rng.uniform(size=(60, 2))

    simplices_padded, simplex_neighbors, vertex_simplex = scipy_delaunay_tri_only(
        points
    )
    tri = Delaunay(points)
    T = tri.simplices.shape[0]

    assert (simplices_padded[:T] == tri.simplices).all()
    assert (simplices_padded[T:] == -1).all()
    assert (simplex_neighbors[:T] == tri.neighbors).all()
    assert (simplex_neighbors[T:] == -1).all()

    # vertex_simplex: any simplex containing the vertex
    for v in range(points.shape[0]):
        assert v in tri.simplices[vertex_simplex[v]]


def test__tri_only_callback__non_finite_points_return_sentinel_tables(monkeypatch):
    def qhull_must_not_run(_):
        raise AssertionError("Delaunay called for non-finite points")

    monkeypatch.setattr("scipy.spatial.Delaunay", qhull_must_not_run)

    rng = np.random.default_rng(5)
    finite_points = rng.uniform(size=(40, 2))
    poisoned_points = []

    for poison_index in (7, 39):
        points = finite_points.copy()
        points[poison_index, 0] = np.nan
        poisoned_points.append(points)

    poisoned_points.append(np.full_like(finite_points, np.nan))

    for points in poisoned_points:
        simplices, neighbors, vertex_simplex = scipy_delaunay_tri_only(points)

        assert simplices.shape == (80, 3)
        assert neighbors.shape == (80, 3)
        assert vertex_simplex.shape == (40,)
        assert simplices.dtype == np.int32
        assert neighbors.dtype == np.int32
        assert vertex_simplex.dtype == np.int32
        assert (simplices == -1).all()
        assert (neighbors == -1).all()
        assert (vertex_simplex == -1).all()


def test__walk_locator__sentinel_tables_keep_nearest_vertex_fallback_live():
    rng = np.random.default_rng(6)
    finite_points = rng.uniform(size=(40, 2))
    query = rng.uniform(size=(8, 2))

    poisoned_points = []
    for poison_index in (7, 39):
        points = finite_points.copy()
        points[poison_index, 0] = np.nan
        poisoned_points.append(points)
    poisoned_points.append(np.full_like(finite_points, np.nan))

    simplices = -np.ones((80, 3), dtype=np.int32)
    neighbors = -np.ones((80, 3), dtype=np.int32)
    vertex_simplex = -np.ones(40, dtype=np.int32)

    for points in poisoned_points:
        mappings = pix_indexes_delaunay_walk_from(
            query_points=query,
            points=points,
            simplices_padded=simplices,
            simplex_neighbors=neighbors,
            vertex_simplex=vertex_simplex,
            xp=np,
        )

        assert (mappings[:, 0] >= 0).all()
        assert (mappings[:, 0] < points.shape[0]).all()
        assert (mappings[:, 1:] == -1).all()


def _assert_matches_find_simplex(points, query):
    """The walk must reproduce find_simplex + KDTree-fallback semantics: rows
    identical, except at most fp edge ties where the returned simplex still
    genuinely contains the query (interpolation-identical)."""
    mappings = _locate(query, points)

    tri = Delaunay(points)
    simplex_idx = tri.find_simplex(query)
    inside = simplex_idx >= 0
    _, nearest = cKDTree(points).query(query, k=1)

    expected = np.full_like(mappings, -1)
    expected[inside] = tri.simplices[simplex_idx[inside]]
    expected[~inside, 0] = nearest[~inside]

    same = (np.sort(mappings, axis=1) == np.sort(expected, axis=1)).all(axis=1)

    # outside-hull rows must agree with the KDTree assignment exactly
    assert (mappings[~inside, 0] == nearest[~inside]).all()
    assert (mappings[~inside, 1:] == -1).all()

    # inside-hull rows: identical, or (fp tie on a shared edge) a different
    # simplex that also contains the query to fp precision
    ties = np.where(~same & inside)[0]
    if len(ties):
        v = mappings[ties]
        assert (v[:, 1:] >= 0).all()  # a real simplex, not a fallback
        a, b, c = points[v[:, 0]], points[v[:, 1]], points[v[:, 2]]
        q = query[ties]

        def cr(u, w):
            return u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]

        den = cr(b - a, c - a)
        w = (
            np.stack([cr(b - q, c - q), cr(c - q, a - q), cr(a - q, b - q)], 1)
            / den[:, None]
        )
        assert (w.min(axis=1) >= -1e-9).all()

    assert same.mean() >= 0.999


def test__walk_locator__uniform_mesh():
    rng = np.random.default_rng(2)
    points = rng.uniform(-1, 1, size=(400, 2))
    query = rng.uniform(-1.2, 1.2, size=(3000, 2))  # some outside hull
    _assert_matches_find_simplex(points, query)


def test__walk_locator__adaptive_blob_ring_mesh():
    rng = np.random.default_rng(4)
    points = _blob_ring_mesh(400, rng)
    query = np.concatenate(
        [_blob_ring_mesh(2800, rng), rng.normal(size=(200, 2)) * 1.6]
    )
    _assert_matches_find_simplex(points, query)


def test__walk_locator__points_on_vertices_and_edges():
    rng = np.random.default_rng(3)
    points = rng.uniform(size=(100, 2))
    tri = Delaunay(points)

    # querying the mesh vertices themselves: every returned row must contain
    # the vertex (weight concentrates there downstream)
    mappings = _locate(points, points)
    assert (mappings == np.arange(100)[:, None]).any(axis=1).all()

    # edge midpoints: contained in one of the (at most two) simplices sharing
    # the edge, so both vertices of that edge must appear in the row
    e0, e1 = tri.simplices[:, 0], tri.simplices[:, 1]
    mid = 0.5 * (points[e0] + points[e1])
    mappings = _locate(mid, points)
    has_e0 = (mappings == e0[:, None]).any(axis=1)
    has_e1 = (mappings == e1[:, None]).any(axis=1)
    assert (has_e0 & has_e1).all()


def test__nearest_vertex_seed__matches_kdtree_query():
    """The seed of the walk is the nearest mesh vertex — the same vertex
    ``cKDTree.query`` returns, which is what ``scipy_delaunay``'s outside-hull
    fallback assigns. Exact distance ties are excluded: argmin and the KD-tree
    are both free to pick either vertex there."""
    rng = np.random.default_rng(11)
    points = _blob_ring_mesh(400, rng)
    query = np.concatenate(
        [_blob_ring_mesh(2800, rng), rng.normal(size=(200, 2)) * 1.6]
    )

    seed = _nearest_vertex_seed_from(query, points, xp=np)

    assert seed.shape == (query.shape[0],)
    assert seed.dtype == np.int32

    d, nearest = cKDTree(points).query(query, k=2)
    ties = np.isclose(d[:, 0], d[:, 1], rtol=0.0, atol=1e-12)

    assert (seed[~ties] == nearest[~ties, 0]).all()


def test__walk_from_seed__reaches_find_simplex_answer_from_a_far_seed():
    """The walk converges from ANY start simplex, not just the nearest-vertex
    one — the property the JAX path relies on when the seed argmin and the walk
    are split apart. Seeding every query at vertex 0 (far from most of them)
    must still reproduce ``find_simplex`` within the step cap."""
    rng = np.random.default_rng(12)
    points = _blob_ring_mesh(400, rng)
    query = _blob_ring_mesh(1500, rng)

    simplices_padded, simplex_neighbors, vertex_simplex = scipy_delaunay_tri_only(
        points
    )

    far_seed = np.zeros(query.shape[0], dtype=np.int32)

    cur, done, outside = _walk_from_seed(
        query_points=query,
        seed=far_seed,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
    )

    # every query resolved within the cap
    assert (done | outside).all()

    tri = Delaunay(points)
    simplex_idx = tri.find_simplex(query)
    inside = simplex_idx >= 0

    assert (done == inside).all()

    verts = np.sort(simplices_padded[cur[inside]], axis=1)
    expected = np.sort(tri.simplices[simplex_idx[inside]], axis=1)
    same = (verts == expected).all(axis=1)

    # fp edge ties may land on the neighbouring simplex; it must still contain
    # the query point
    if not same.all():
        v = simplices_padded[cur[inside]][~same]
        q = query[inside][~same]
        a, b, c = points[v[:, 0]], points[v[:, 1]], points[v[:, 2]]

        def cr(u, w):
            return u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]

        den = cr(b - a, c - a)
        w = (
            np.stack([cr(b - q, c - q), cr(c - q, a - q), cr(a - q, b - q)], 1)
            / den[:, None]
        )
        assert (w.min(axis=1) >= -1e-9).all()

    assert same.mean() >= 0.999


def test__walk_locator__shapes_and_dtypes_on_an_odd_query_count():
    """The wrapper's public contract: (Q, 3) int32 mappings and (Q,) int32
    simplex indexes for a query count that is not a round multiple of
    ``DELAUNAY_LOCATE_CHUNK``."""
    rng = np.random.default_rng(13)
    points = _blob_ring_mesh(300, rng)
    query = np.concatenate([_blob_ring_mesh(1500, rng), rng.normal(size=(37, 2)) * 1.7])
    assert query.shape[0] == 1537

    simplices_padded, simplex_neighbors, vertex_simplex = scipy_delaunay_tri_only(
        points
    )

    mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=query,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
        return_simplex_indexes=True,
    )

    assert mappings.shape == (1537, 3)
    assert mappings.dtype == np.int32
    assert simplex_indexes.shape == (1537,)
    assert simplex_indexes.dtype == np.int32

    # inside-hull rows carry a real simplex; outside-hull rows are the
    # nearest-vertex fallback with simplex index -1
    inside = simplex_indexes >= 0
    assert (mappings[inside] >= 0).all()
    assert (mappings[~inside, 1:] == -1).all()
    assert (mappings[~inside, 0] >= 0).all()

    # the simplex index and the mapping row must describe the same triangle
    assert (
        np.sort(simplices_padded[simplex_indexes[inside]], axis=1)
        == np.sort(mappings[inside], axis=1)
    ).all()
