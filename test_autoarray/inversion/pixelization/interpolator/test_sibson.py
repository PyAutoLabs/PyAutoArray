import numpy as np

from autoarray.inversion.mesh.interpolator.delaunay import (
    pix_indexes_delaunay_walk_from,
    scipy_delaunay_tri_only,
)
from autoarray.inversion.mesh.interpolator.sibson import (
    delaunay_circumcircles_from,
    sibson_mappings_weights_from_tables,
)


def _sibson_numpy(points, query, max_cavity_triangles=32, max_neighbors=32):
    simplices, neighbors, vertex_simplex = scipy_delaunay_tri_only(points)
    delaunay_mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
        return_simplex_indexes=True,
    )
    return sibson_mappings_weights_from_tables(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=delaunay_mappings[:, 0],
        max_cavity_triangles=max_cavity_triangles,
        max_neighbors=max_neighbors,
        xp=np,
    )


def test__circumcircles__known_right_triangle_and_padding():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    simplices = np.array([[0, 1, 2], [-1, -1, -1]], dtype=np.int32)

    centres, radii_squared, valid = delaunay_circumcircles_from(
        points, simplices, xp=np
    )

    np.testing.assert_allclose(centres[0], [1.0, 1.0])
    np.testing.assert_allclose(radii_squared[0], 2.0)
    assert valid.tolist() == [True, False]


def test__single_triangle__matches_barycentric_and_outside_fallback():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    query = np.array([[0.4, 0.6], [0.2, 0.2], [3.0, 3.0]])

    mappings, sizes, weights, cavity_sizes, overflow, degenerate = _sibson_numpy(
        points, query, max_cavity_triangles=8, max_neighbors=4
    )

    dense_weights = np.zeros((query.shape[0], points.shape[0]))
    for row in range(query.shape[0]):
        valid = mappings[row] >= 0
        np.add.at(dense_weights[row], mappings[row, valid], weights[row, valid])

    np.testing.assert_allclose(dense_weights[0], [0.5, 0.2, 0.3])
    np.testing.assert_allclose(dense_weights[1], [0.8, 0.1, 0.1])
    np.testing.assert_allclose(dense_weights[2], [0.0, 1.0, 0.0])
    assert sizes.tolist() == [3, 3, 1]
    assert cavity_sizes.tolist() == [1, 1, 0]
    assert not overflow.any()
    assert not degenerate.any()


def test__watson_weights__match_historical_c_natural_neighbor_reference():
    points = np.array(
        [
            [-1.0, -0.8],
            [-0.2, -1.1],
            [0.9, -0.7],
            [1.2, 0.4],
            [0.5, 1.1],
            [-0.6, 0.9],
            [-0.1, 0.0],
            [0.55, 0.2],
        ]
    )
    query = np.array([[-0.3, -0.2], [0.2, 0.35], [0.75, -0.2], [-0.45, 0.55]])
    expected = np.array(
        [
            [
                0.1874290255921386,
                0.0833454011106457,
                0.0002366866860013,
                0.0,
                0.0,
                0.0464320470840236,
                0.6825568395271908,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.1673987581070807,
                0.0991207867537038,
                0.3502171651198253,
                0.3832632900193901,
            ],
            [
                0.0,
                0.0009216096884568,
                0.4570373759794244,
                0.1185942343731840,
                0.0,
                0.0,
                0.0559355794908082,
                0.3675112004681266,
            ],
            [
                0.0568307353360843,
                0.0,
                0.0,
                0.0,
                0.0229991789644525,
                0.6323181584385680,
                0.2824561831946050,
                0.0053957440662901,
            ],
        ]
    )

    mappings, _, weights, _, overflow, degenerate = _sibson_numpy(points, query)
    dense_weights = np.zeros_like(expected)
    for row in range(query.shape[0]):
        valid = mappings[row] >= 0
        np.add.at(dense_weights[row], mappings[row, valid], weights[row, valid])

    np.testing.assert_allclose(dense_weights, expected, atol=1.0e-13)
    assert not overflow.any()
    assert not degenerate.any()


def test__random_mesh__partition_of_unity_and_linear_precision():
    rng = np.random.default_rng(10)
    points = rng.uniform(-1.0, 1.0, size=(100, 2))
    query = rng.uniform(-0.8, 0.8, size=(500, 2))

    mappings, sizes, weights, cavity_sizes, overflow, degenerate = _sibson_numpy(
        points, query
    )

    reconstructed = np.zeros_like(query)
    for row in range(query.shape[0]):
        valid = mappings[row] >= 0
        reconstructed[row] = np.sum(
            points[mappings[row, valid]] * weights[row, valid, None], axis=0
        )

    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1.0e-12)
    inside = cavity_sizes > 0
    np.testing.assert_allclose(reconstructed[inside], query[inside], atol=1.0e-11)
    assert sizes.max() <= 12
    assert cavity_sizes.max() <= 12
    assert not overflow.any()
    assert not degenerate.any()


def test__cavity_cap__reports_overflow_instead_of_silent_approximation():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.1, 1.8], [1.8, 2.1], [1.0, 0.7]])
    query = np.array([[0.9, 0.9]])

    _, _, weights, cavity_sizes, overflow, _ = _sibson_numpy(
        points, query, max_cavity_triangles=1, max_neighbors=5
    )

    assert cavity_sizes[0] == 1
    assert overflow[0]
    assert np.isnan(weights[0]).all()
