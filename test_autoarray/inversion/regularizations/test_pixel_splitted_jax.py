"""
JAX leg of the split-pixel regularization matrix builder: the compacted scatter (a narrow main
pass plus a wide-row supplement, see ``SPLIT_REG_COMPACT_WIDTH``) must reproduce the NumPy
reference exactly, poison the matrix with NaN when the wide-row budget overflows, and stay
jit / vmap / grad friendly.

Skipped when JAX is absent (it is an optional dependency).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from autoarray.inversion.regularization import regularization_util  # noqa: E402
from autoarray.inversion.regularization.regularization_util import (  # noqa: E402
    SPLIT_REG_COMPACT_WIDTH,
)


def tables_from_sizes(sizes, width, seed=1):
    """
    A synthetic ``(4P, K)`` split stencil table, padded ``DelaunayNN`` style: every column beyond a
    row's size is mapping ``-1`` / weight ``0.0``. The number of pixels ``P`` is ``len(sizes) / 4``,
    every mapping is a pixel index and no row repeats a pixel, so a row can be at most ``P`` wide.
    """
    rng = np.random.default_rng(seed)

    sizes = np.asarray(sizes, dtype=np.int32)

    assert sizes.shape[0] % 4 == 0

    total_pixels = sizes.shape[0] // 4

    assert int(sizes.max()) <= total_pixels
    assert int(sizes.max()) <= width

    mappings = -np.ones((sizes.shape[0], width), dtype=np.int32)
    weights = np.zeros((sizes.shape[0], width), dtype=np.float64)

    for row, size in enumerate(sizes):
        pixels = rng.choice(total_pixels, size=int(size), replace=False)
        mappings[row, : int(size)] = pixels
        values = rng.uniform(0.05, 1.0, size=int(size))
        weights[row, : int(size)] = values / values.sum()

    regularization_weights = rng.uniform(0.5, 3.0, size=total_pixels)

    return regularization_weights, mappings, sizes, weights


NARROW_SIZES = [1, 3, 5, 8, 11, 4, 2, 9, 12, 6, 3, 7] * 4


def wide_sizes():
    """
    ``DelaunayNN``-shaped sizes: 144 rows (36 pixels) of which five exceed the compact width, one
    of them the full 33-column table width.
    """
    sizes = [1, 3, 5, 8, 11, 4, 2, 9, 12, 6, 3, 7] * 12

    for row, size in [(5, 21), (40, 13), (77, 33), (100, 14), (131, 20)]:
        sizes[row] = size

    return sizes


WIDE_SIZES = wide_sizes()


def matrix_np_from(regularization_weights, mappings, sizes, weights):
    return regularization_util.pixel_splitted_regularization_matrix_np_from(
        regularization_weights=np.copy(regularization_weights),
        splitted_mappings=np.copy(mappings),
        splitted_sizes=np.copy(sizes),
        splitted_weights=np.copy(weights),
    )


def matrix_jax_from(regularization_weights, mappings, sizes, weights, **kwargs):
    return regularization_util.pixel_splitted_regularization_matrix_from(
        regularization_weights=jnp.asarray(regularization_weights),
        splitted_mappings=jnp.asarray(mappings),
        splitted_sizes=jnp.asarray(sizes),
        splitted_weights=jnp.asarray(weights),
        xp=jnp,
        **kwargs,
    )


def assert_matrices_equal(matrix_jax, matrix_np):
    np.testing.assert_allclose(
        np.asarray(matrix_jax, dtype=np.float64), matrix_np, rtol=1.0e-12, atol=1.0e-14
    )


def test__padding_convention__valid_mask_agrees_with_sizes():
    """
    The compaction selects wide rows by ``splitted_sizes`` but masks entries by ``mapping != -1``:
    the two must describe the same occupied columns.
    """
    _, mappings, sizes, _ = tables_from_sizes(sizes=WIDE_SIZES, width=33)

    valid = mappings != -1
    assert np.array_equal(valid.sum(axis=1).astype(np.int32), sizes)
    assert np.array_equal(valid, np.arange(33)[None, :] < sizes[:, None])


def test__all_rows_narrower_than_compact_width__matches_numpy():
    reg_weights, mappings, sizes, weights = tables_from_sizes(
        sizes=NARROW_SIZES, width=33
    )

    assert int(sizes.max()) <= SPLIT_REG_COMPACT_WIDTH

    assert_matrices_equal(
        matrix_jax_from(reg_weights, mappings, sizes, weights),
        matrix_np_from(reg_weights, mappings, sizes, weights),
    )


def test__wide_rows_inside_budget__matches_numpy():
    """
    Rows wider than the compact width exercise the head x tail, tail x head and tail x tail
    supplement blocks. One row is exactly the full table width.
    """
    reg_weights, mappings, sizes, weights = tables_from_sizes(
        sizes=WIDE_SIZES, width=33, seed=3
    )

    assert int((sizes > SPLIT_REG_COMPACT_WIDTH).sum()) == 5
    assert int(sizes.max()) == 33

    assert_matrices_equal(
        matrix_jax_from(reg_weights, mappings, sizes, weights),
        matrix_np_from(reg_weights, mappings, sizes, weights),
    )


def test__wide_rows_at_the_budget__matches_numpy():
    reg_weights, mappings, sizes, weights = tables_from_sizes(
        sizes=WIDE_SIZES, width=33, seed=5
    )

    wide = int((sizes > SPLIT_REG_COMPACT_WIDTH).sum())

    assert_matrices_equal(
        matrix_jax_from(reg_weights, mappings, sizes, weights, wide_row_budget=wide),
        matrix_np_from(reg_weights, mappings, sizes, weights),
    )


def test__more_wide_rows_than_the_budget__matrix_is_nan():
    reg_weights, mappings, sizes, weights = tables_from_sizes(
        sizes=WIDE_SIZES, width=33, seed=5
    )

    wide = int((sizes > SPLIT_REG_COMPACT_WIDTH).sum())

    matrix = matrix_jax_from(
        reg_weights, mappings, sizes, weights, wide_row_budget=wide - 1
    )

    assert np.all(np.isnan(np.asarray(matrix)))


def test__table_narrower_than_compact_width__matches_numpy_with_no_supplement():
    """
    The ``Delaunay`` mesh's ``K = 4`` tables are already narrower than the compact width: the
    compaction is a no-op, the supplement is not built and the overflow guard is not applied.
    """
    reg_weights, mappings, sizes, weights = tables_from_sizes(
        sizes=[1, 2, 3, 4] * 6, width=4, seed=7
    )

    assert_matrices_equal(
        matrix_jax_from(reg_weights, mappings, sizes, weights),
        matrix_np_from(reg_weights, mappings, sizes, weights),
    )

    jaxpr = jax.make_jaxpr(
        lambda w, m, s, ws: regularization_util.pixel_splitted_regularization_matrix_from(
            regularization_weights=w,
            splitted_mappings=m,
            splitted_sizes=s,
            splitted_weights=ws,
            xp=jnp,
        )
    )(
        jnp.asarray(reg_weights),
        jnp.asarray(mappings),
        jnp.asarray(sizes),
        jnp.asarray(weights),
    )

    assert "top_k" not in str(jaxpr)


def test__jit_and_vmap__match_the_unbatched_result():
    tables = [
        tables_from_sizes(sizes=WIDE_SIZES, width=33, seed=seed) for seed in range(3)
    ]

    def matrix_from(reg_weights, mappings, sizes, weights):
        return regularization_util.pixel_splitted_regularization_matrix_from(
            regularization_weights=reg_weights,
            splitted_mappings=mappings,
            splitted_sizes=sizes,
            splitted_weights=weights,
            xp=jnp,
        )

    unbatched = [
        np.asarray(jax.jit(matrix_from)(*[jnp.asarray(a) for a in t])) for t in tables
    ]

    for matrix, table in zip(unbatched, tables):
        assert_matrices_equal(matrix, matrix_np_from(*table))

    batched = jax.jit(jax.vmap(matrix_from))(
        jnp.asarray(np.stack([t[0] for t in tables])),
        jnp.asarray(np.stack([t[1] for t in tables])),
        jnp.asarray(np.stack([t[2] for t in tables])),
        jnp.asarray(np.stack([t[3] for t in tables])),
    )

    for index in range(3):
        assert_matrices_equal(batched[index], unbatched[index])


def test__gradient_of_the_matrix_sum__is_finite_and_matches_finite_differences():
    reg_weights, mappings, sizes, weights = tables_from_sizes(
        sizes=WIDE_SIZES, width=33, seed=11
    )

    def matrix_sum(splitted_weights):
        return jnp.sum(
            regularization_util.pixel_splitted_regularization_matrix_from(
                regularization_weights=jnp.asarray(reg_weights),
                splitted_mappings=jnp.asarray(mappings),
                splitted_sizes=jnp.asarray(sizes),
                splitted_weights=splitted_weights,
                xp=jnp,
            )
        )

    gradient = np.asarray(jax.grad(matrix_sum)(jnp.asarray(weights)))

    assert np.all(np.isfinite(gradient))
    assert np.any(gradient != 0.0)

    # A row and column inside the wide-row supplement's tail block.
    row = int(np.argmax(sizes))
    column = 15

    step = 1.0e-6

    weights_up = np.copy(weights)
    weights_up[row, column] += step
    weights_down = np.copy(weights)
    weights_down[row, column] -= step

    finite_difference = (
        float(matrix_sum(jnp.asarray(weights_up)))
        - float(matrix_sum(jnp.asarray(weights_down)))
    ) / (2.0 * step)

    assert gradient[row, column] == pytest.approx(finite_difference, rel=1.0e-6)
