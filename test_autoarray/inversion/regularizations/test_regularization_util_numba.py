"""
Numba leg of the split-regularization builders: the ``@numba_util.jit()`` kernels behind
``reg_split_np_from`` and ``pixel_splitted_regularization_matrix_np_from`` must reproduce the
retained pure-Python ``_reference`` bodies bit-for-bit (``np.array_equal``, not ``allclose``),
leave their inputs untouched, and raise rather than write out of bounds.
"""

import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray.inversion.regularization import regularization_util


def tables_from_sizes(sizes, width, seed=1):
    """
    A synthetic ``(4P, K)`` split stencil table, padded ``DelaunayNN`` style: every column beyond a
    row's size is mapping ``-1`` / weight ``0.0``. Mirrors the generator of the same name in
    ``test_pixel_splitted_jax.py`` (replicated rather than imported, because that module is skipped
    when JAX is absent and this leg has no JAX dependency).
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
    sizes = [1, 3, 5, 8, 11, 4, 2, 9, 12, 6, 3, 7] * 12

    for row, size in [(5, 21), (40, 13), (77, 33), (100, 14), (131, 20)]:
        sizes[row] = size

    return sizes


WIDE_SIZES = wide_sizes()


def knn_tables(pixels=50, k_neighbors=3, seed=7):
    """
    A K-NN-shaped stencil table: ``4P`` rows of ``k_neighbors`` barycentric mappings plus the one
    reserved pad column (mapping ``-1``, weight ``0.0``) the interpolators hstack for the centre
    pixel insertion, with ``sizes`` reporting ``k_neighbors``.
    """
    rng = np.random.default_rng(seed)

    rows = 4 * pixels
    width = k_neighbors + 1

    mappings = -np.ones((rows, width), dtype=np.int32)
    weights = np.zeros((rows, width), dtype=np.float64)
    sizes = np.full(rows, k_neighbors, dtype=np.int32)

    for row in range(rows):
        mappings[row, :k_neighbors] = rng.choice(
            pixels, size=k_neighbors, replace=False
        )
        values = rng.uniform(0.05, 1.0, size=k_neighbors)
        weights[row, :k_neighbors] = values / values.sum()

    regularization_weights = rng.uniform(0.5, 3.0, size=pixels)

    return regularization_weights, mappings, sizes, weights


def reference_matrix_from(regularization_weights, mappings, sizes, weights):
    return regularization_util._pixel_splitted_regularization_matrix_reference(
        regularization_weights=np.copy(regularization_weights),
        splitted_mappings=np.copy(mappings),
        splitted_sizes=np.copy(sizes),
        splitted_weights=np.copy(weights),
    )


def kernel_matrix_from(regularization_weights, mappings, sizes, weights):
    return regularization_util.pixel_splitted_regularization_matrix_np_from(
        regularization_weights=np.copy(regularization_weights),
        splitted_mappings=np.copy(mappings),
        splitted_sizes=np.copy(sizes),
        splitted_weights=np.copy(weights),
    )


@pytest.mark.parametrize("sizes", [NARROW_SIZES, WIDE_SIZES])
def test__matrix__bit_identical_to_reference__delaunay_nn_shaped_tables(sizes):
    reg_weights, mappings, table_sizes, weights = tables_from_sizes(
        sizes=sizes, width=33
    )

    assert np.array_equal(
        kernel_matrix_from(reg_weights, mappings, table_sizes, weights),
        reference_matrix_from(reg_weights, mappings, table_sizes, weights),
    )


def test__matrix__bit_identical_to_reference__knn_tables_through_reg_split():
    reg_weights, mappings, sizes, weights = knn_tables()

    kernel_tables = regularization_util.reg_split_np_from(
        splitted_mappings=mappings,
        splitted_sizes=sizes,
        splitted_weights=weights,
    )

    reference_tables = regularization_util._reg_split_reference(
        splitted_mappings=np.copy(mappings),
        splitted_sizes=np.copy(sizes),
        splitted_weights=np.copy(weights),
    )

    assert np.array_equal(
        kernel_matrix_from(reg_weights, *kernel_tables),
        reference_matrix_from(reg_weights, *reference_tables),
    )


def mixed_self_tables(pixels=20, width=5, seed=11):
    """
    A stencil table where some rows already contain their own pixel (``i // 4``) and some do not,
    so the kernel's self detection and its pad-column insertion are both exercised. Every row has a
    size of at least one, so the reference's ``size == 0`` index leak is not in play.
    """
    rng = np.random.default_rng(seed)

    rows = 4 * pixels

    mappings = -np.ones((rows, width), dtype=np.int32)
    weights = np.zeros((rows, width), dtype=np.float64)
    sizes = np.zeros(rows, dtype=np.int32)

    for row in range(rows):
        size = int(rng.integers(1, width - 1))
        others = [p for p in range(pixels) if p != row // 4]
        pixel_list = list(rng.choice(others, size=size, replace=False))

        if row % 3 == 0:
            pixel_list[int(rng.integers(0, size))] = row // 4

        mappings[row, :size] = pixel_list
        values = rng.uniform(0.05, 1.0, size=size)
        weights[row, :size] = values / values.sum()
        sizes[row] = size

    return mappings, sizes, weights


def test__reg_split__bit_identical_to_reference__mixed_self_rows():
    mappings, sizes, weights = mixed_self_tables()

    kernel_mappings, kernel_sizes, kernel_weights = (
        regularization_util.reg_split_np_from(
            splitted_mappings=mappings,
            splitted_sizes=sizes,
            splitted_weights=weights,
        )
    )

    ref_mappings, ref_sizes, ref_weights = regularization_util._reg_split_reference(
        splitted_mappings=np.copy(mappings),
        splitted_sizes=np.copy(sizes),
        splitted_weights=np.copy(weights),
    )

    # Both self-containing and self-absent rows are present in the fixture.
    assert np.any(ref_sizes == sizes)
    assert np.any(ref_sizes == sizes + 1)

    assert np.array_equal(kernel_mappings, ref_mappings)
    assert np.array_equal(kernel_sizes, ref_sizes)
    assert np.array_equal(kernel_weights, ref_weights)


def test__reg_split__does_not_mutate_its_inputs():
    mappings, sizes, weights = mixed_self_tables()

    mappings_before = np.copy(mappings)
    sizes_before = np.copy(sizes)
    weights_before = np.copy(weights)

    regularization_util.reg_split_np_from(
        splitted_mappings=mappings,
        splitted_sizes=sizes,
        splitted_weights=weights,
    )

    assert np.array_equal(mappings, mappings_before)
    assert np.array_equal(sizes, sizes_before)
    assert np.array_equal(weights, weights_before)


def test__reg_split__empty_row__inserts_at_column_zero():
    """
    A row with ``size == 0`` inserts its pixel at column 0 and reports a size of 1.

    The reference deliberately diverges here: its ``for j in range(splitted_sizes[i])`` loop does
    not run, so ``j`` leaks in from the previous row and the flag-zero branch writes at a stale
    ``j + 1``. This test pins the kernel to the intended semantics, not to the reference.
    """
    mappings = np.array([[3, 1, -1, -1], [-1, -1, -1, -1]] * 2, dtype=np.int32)
    sizes = np.array([2, 0, 2, 0], dtype=np.int32)
    weights = np.array(
        [[0.4, 0.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float64
    )

    out_mappings, out_sizes, out_weights = regularization_util.reg_split_np_from(
        splitted_mappings=mappings,
        splitted_sizes=sizes,
        splitted_weights=weights,
    )

    assert np.array_equal(out_sizes, np.array([3, 1, 3, 1]))

    # Rows 1 and 3 are empty; both belong to pixel 0, inserted at column 0.
    assert np.array_equal(out_mappings[1], np.array([0, -1, -1, -1]))
    assert np.array_equal(out_mappings[3], np.array([0, -1, -1, -1]))
    assert np.array_equal(out_weights[1], np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(out_weights[3], np.array([1.0, 0.0, 0.0, 0.0]))

    # The reference's leaked `j` writes at column 2 of the empty rows instead.
    ref_mappings, _, _ = regularization_util._reg_split_reference(
        splitted_mappings=np.copy(mappings),
        splitted_sizes=np.copy(sizes),
        splitted_weights=np.copy(weights),
    )

    assert ref_mappings[1][2] == 0
    assert not np.array_equal(ref_mappings[1], out_mappings[1])


def test__reg_split__full_row_without_self_pixel__raises():
    mappings = np.array([[1, 2, 3, 4]] * 4, dtype=np.int32)
    sizes = np.array([4, 4, 4, 4], dtype=np.int32)
    weights = np.full((4, 4), 0.25)

    with pytest.raises(exc.InversionException):
        regularization_util.reg_split_np_from(
            splitted_mappings=mappings,
            splitted_sizes=sizes,
            splitted_weights=weights,
        )


def test__constant_split__matrix_from_mapper__matches_reference_chain(
    delaunay_mapper_9_3x3,
):
    mappings, sizes, weights = (
        delaunay_mapper_9_3x3.interpolator._mappings_sizes_weights_split
    )

    regularization = aa.reg.ConstantSplit(coefficient=2.0)

    matrix = regularization.regularization_matrix_from(linear_obj=delaunay_mapper_9_3x3)

    reference_tables = regularization_util._reg_split_reference(
        splitted_mappings=np.copy(mappings),
        splitted_sizes=np.copy(sizes),
        splitted_weights=np.copy(weights),
    )

    pixels = int(len(mappings) / 4)

    matrix_reference = (
        regularization_util._pixel_splitted_regularization_matrix_reference(
            regularization_weights=np.full(pixels, 2.0),
            splitted_mappings=reference_tables[0],
            splitted_sizes=reference_tables[1],
            splitted_weights=reference_tables[2],
        )
    )

    assert np.array_equal(matrix, matrix_reference)
