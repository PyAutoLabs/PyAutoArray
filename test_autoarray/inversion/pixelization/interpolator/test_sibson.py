import hashlib

import numpy as np
import pytest

from autoarray.inversion.mesh.interpolator import sibson
from autoarray.inversion.mesh.interpolator.delaunay import (
    pix_indexes_delaunay_walk_from,
    scipy_delaunay_tri_only,
)
from autoarray.inversion.mesh.interpolator.sibson import (
    _bool_env,
    _positive_int_env,
    _sibson_unroll_candidates,
    delaunay_circumcircles_from,
    scipy_delaunay_nn,
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


def test__precomputed_circumcircles__match_the_default_call_bit_for_bit():
    """``sibson_mappings_weights_from_tables`` accepts the circumcircles of the
    frozen simplex table so a caller interpolating several query sets against
    one mesh computes them once (``jax_delaunay_nn``).  Passing them must be a
    pure hoist: every output identical, not merely close."""
    rng = np.random.default_rng(11)
    points = rng.uniform(-1.0, 1.0, size=(80, 2))
    query = rng.uniform(-0.8, 0.8, size=(120, 2))

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
    kwargs = dict(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=delaunay_mappings[:, 0],
        xp=np,
    )

    default = sibson_mappings_weights_from_tables(**kwargs)
    hoisted = sibson_mappings_weights_from_tables(
        circumcircles=delaunay_circumcircles_from(points, simplices, xp=np),
        **kwargs,
    )

    assert len(default) == len(hoisted) == 6
    for expected, actual in zip(default, hoisted):
        if np.issubdtype(expected.dtype, np.floating):
            assert np.array_equal(expected, actual, equal_nan=True)
        else:
            assert np.array_equal(expected, actual)


def test__scipy_delaunay_nn__fixed_seed_regression():
    """Guard the NumPy Sibson path against silent drift.

    The integer connectivity (mappings, sizes, cavity sizes) is exact, so it is
    hashed rather than inlined; the floating weights are pinned as a partition
    of unity plus one stored data row and one stored split row.  Values were
    computed on PyAutoArray ``main`` before the issue #532 JAX changes, which
    do not touch this path.
    """
    rng = np.random.default_rng(24)
    points = rng.uniform(-1.0, 1.0, size=(60, 2))
    query = rng.uniform(-0.7, 0.7, size=(40, 2))

    (
        _,
        _,
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
    ) = scipy_delaunay_nn(points, query, areas_factor=0.5)

    def integer_digest(*arrays):
        hasher = hashlib.sha256()
        for array in arrays:
            hasher.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
        return hasher.hexdigest()[:16]

    assert split_points.shape == (4 * points.shape[0], 2)
    assert integer_digest(mappings, sizes, cavity_sizes) == "b827766a6308f8a6"
    assert (
        integer_digest(splitted_mappings, splitted_sizes, split_cavity_sizes)
        == "09733405e7271ecd"
    )

    assert not overflow.any()
    assert not degenerate.any()
    assert not split_overflow.any()
    assert not split_degenerate.any()

    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(splitted_weights.sum(axis=1), 1.0, atol=1.0e-12)

    assert mappings[7][mappings[7] >= 0].tolist() == [9, 18, 20, 24, 34, 45]
    np.testing.assert_allclose(
        weights[7][:6],
        [
            0.1945365763239002,
            0.37915688518183144,
            0.1014855420340886,
            0.004891861361516876,
            0.18448730058462717,
            0.13544183451403582,
        ],
        atol=1.0e-14,
    )
    assert splitted_mappings[13][splitted_mappings[13] >= 0].tolist() == [3, 6, 13, 35]
    np.testing.assert_allclose(
        splitted_weights[13][:4],
        [
            0.5667287604913306,
            0.0769230792268042,
            0.25906145975374095,
            0.0972867005281243,
        ],
        atol=1.0e-14,
    )


def test__env_override_parsers__accept_valid_and_reject_invalid():
    """``PYAUTO_SIBSON_QUERY_CHUNK`` and ``PYAUTO_SIBSON_UNROLL_CANDIDATES``
    are read once at import, so the parsing itself is what the unit tests can
    reach; a bad value must fail loudly rather than fall back to the default."""
    assert _positive_int_env("CHUNK", None) is None
    assert _positive_int_env("CHUNK", "64") == 64
    assert _positive_int_env("CHUNK", "1024") == 1024
    for bad in ("0", "-1", "notanint", "2.5", ""):
        with pytest.raises(ValueError):
            _positive_int_env("CHUNK", bad)

    assert _bool_env("UNROLL", None) is None
    assert _bool_env("UNROLL", "1") is True
    assert _bool_env("UNROLL", "0") is False
    for bad in ("", "true", "yes", "2", "-1"):
        with pytest.raises(ValueError):
            _bool_env("UNROLL", bad)


def test__unroll_gate__module_override_wins_and_needs_no_backend(monkeypatch):
    """With the override set, the gate answers without importing JAX -- the
    unit suite must not pull in a backend. Both settings are bit-identical
    code paths; only their emitted program differs (issue #532)."""
    monkeypatch.setattr(sibson, "SIBSON_UNROLL_CANDIDATES", True)
    assert _sibson_unroll_candidates() is True

    monkeypatch.setattr(sibson, "SIBSON_UNROLL_CANDIDATES", False)
    assert _sibson_unroll_candidates() is False
