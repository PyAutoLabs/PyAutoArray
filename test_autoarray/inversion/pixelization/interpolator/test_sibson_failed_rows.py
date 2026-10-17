"""
Failed Sibson rows keep `size >= 1` so their NaN weights propagate (PyAutoArray#629).

A degenerate query whose Watson contributions are all non-finite selects no entry. Before #629 such a row came
out at `size == 0`, so every consumer (which reads only the first `size` entries) skipped its NaN weights and
the evidence was finite and wrong. The reference, the numba kernel and the JAX path now all keep the row at
size 1, mapped to the nearest seed vertex, with a NaN weight.

The degenerate case is built by triangulating a well-scaled point set and then evaluating the Sibson tables
at coordinates scaled by 1e150, where the Watson circle arithmetic overflows.
"""

import importlib.util

import numpy as np
import pytest

from autoarray.inversion.mesh.interpolator import sibson
from autoarray.inversion.mesh.interpolator.delaunay import (
    pix_indexes_delaunay_walk_from,
    scipy_delaunay_tri_only,
)

SCALE = 1.0e150


def _overflowing_tables():
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.0, 1.0, size=(30, 2))
    query = rng.uniform(-0.5, 0.5, size=(20, 2))
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
    scaled_points = points * SCALE
    with np.errstate(all="ignore"):
        circumcircles = sibson.delaunay_circumcircles_from(
            scaled_points, simplices, xp=np
        )
    return dict(
        query_points=query * SCALE,
        points=scaled_points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=delaunay_mappings[:, 0],
        circumcircles=circumcircles,
        max_cavity_triangles=32,
        max_neighbors=32,
    )


def _assert_failed_rows_kept_at_size_one(outputs, points_count):
    mappings, sizes, weights, _, overflow, degenerate = (np.asarray(o) for o in outputs)
    failed = overflow | degenerate

    assert failed.all()
    assert (sizes == 1).all()
    assert ((mappings[:, 0] >= 0) & (mappings[:, 0] < points_count)).all()
    assert (mappings[:, 1:] == -1).all()
    assert np.isnan(weights[:, 0]).all()


def test__reference__degenerate_row_with_no_selected_entry__keeps_size_one_and_nan():
    with np.errstate(all="ignore"):
        outputs = sibson._sibson_mappings_weights_numpy_reference(
            **_overflowing_tables()
        )

    _assert_failed_rows_kept_at_size_one(outputs, points_count=30)


def test__numba__degenerate_row_with_no_selected_entry__matches_reference():
    pytest.importorskip("numba")

    kwargs = _overflowing_tables()
    with np.errstate(all="ignore"):
        reference = sibson._sibson_mappings_weights_numpy_reference(**kwargs)
        numba_outputs = sibson._sibson_mappings_weights_numba(**kwargs)

    _assert_failed_rows_kept_at_size_one(numba_outputs, points_count=30)
    for expected, actual in zip(reference, numba_outputs):
        assert expected.dtype == actual.dtype
        assert np.array_equal(expected, actual, equal_nan=True)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
def test__jax__degenerate_row_with_no_selected_entry__matches_reference():
    import jax.numpy as jnp

    kwargs = _overflowing_tables()
    with np.errstate(all="ignore"):
        reference = sibson._sibson_mappings_weights_numpy_reference(**kwargs)

    jax_kwargs = {
        key: (jnp.asarray(value) if isinstance(value, np.ndarray) else value)
        for key, value in kwargs.items()
        if key != "circumcircles"
    }
    jax_outputs = sibson.sibson_mappings_weights_from_tables(xp=jnp, **jax_kwargs)

    _assert_failed_rows_kept_at_size_one(jax_outputs, points_count=30)
    for expected, actual in zip(reference[:3], jax_outputs[:3]):
        assert np.array_equal(expected, np.asarray(actual), equal_nan=True)
