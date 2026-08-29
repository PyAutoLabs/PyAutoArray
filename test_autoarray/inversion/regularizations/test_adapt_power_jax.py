"""
JAX leg of the ``*Power`` gate: the new single-scatter matrix builder and the split path must agree
with their NumPy counterparts under ``jax.numpy``.

Skipped when JAX is absent (it is an optional dependency).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from autoarray.inversion.regularization.adapt import (  # noqa: E402
    weighted_regularization_matrix_single_scatter_from,
)
from autoarray.inversion.regularization.constant import (  # noqa: E402
    constant_regularization_matrix_from,
)
from autoarray.inversion.regularization import regularization_util  # noqa: E402


NEIGHBORS = np.array(
    [
        [1, 3, -1, -1],
        [0, 2, 4, -1],
        [1, 5, -1, -1],
        [0, 4, 6, -1],
        [1, 3, 5, 7],
        [2, 4, 8, -1],
        [3, 7, -1, -1],
        [4, 6, 8, -1],
        [5, 7, -1, -1],
    ]
)


def test__single_scatter_builder__numpy_and_jax_agree():
    weights = np.array([0.1, 5.0, 0.3, 2.0, 9.0, 0.05, 1.0, 4.0, 0.7])

    matrix_np = weighted_regularization_matrix_single_scatter_from(
        regularization_weights=weights, neighbors=NEIGHBORS
    )
    matrix_jax = weighted_regularization_matrix_single_scatter_from(
        regularization_weights=jnp.asarray(weights),
        neighbors=jnp.asarray(NEIGHBORS),
        xp=jnp,
    )

    assert np.asarray(matrix_jax) == pytest.approx(matrix_np, abs=1.0e-12)


def test__single_scatter_builder__jax_path_equals_constant_for_uniform_weights():
    coefficient = 3.0

    matrix_jax = weighted_regularization_matrix_single_scatter_from(
        regularization_weights=jnp.full(9, coefficient),
        neighbors=jnp.asarray(NEIGHBORS),
        xp=jnp,
    )
    matrix_constant = constant_regularization_matrix_from(
        coefficient=coefficient,
        neighbors=NEIGHBORS.copy(),
        neighbors_sizes=(NEIGHBORS >= 0).sum(axis=1),
    )

    assert np.asarray(matrix_jax) == pytest.approx(matrix_constant, abs=1.0e-12)


def test__single_scatter_builder__is_jittable_and_differentiable():
    def log_det(weights):
        matrix = weighted_regularization_matrix_single_scatter_from(
            regularization_weights=weights,
            neighbors=jnp.asarray(NEIGHBORS),
            xp=jnp,
        )
        return jnp.linalg.slogdet(matrix)[1]

    weights = jnp.array([0.5, 1.5, 0.8, 2.0, 3.0, 0.4, 1.0, 1.2, 0.9])

    value = jax.jit(log_det)(weights)
    gradient = jax.grad(log_det)(weights)

    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(gradient)))


def test__split_builder__numpy_and_jax_agree(delaunay_mapper_9_3x3):
    mappings, sizes, weights = (
        delaunay_mapper_9_3x3.interpolator._mappings_sizes_weights_split
    )

    regularization_weights = np.full(9, 2.0)

    def matrix_from(xp, mappings, sizes, weights, regularization_weights):
        (
            splitted_mappings,
            splitted_sizes,
            splitted_weights,
        ) = regularization_util.reg_split_from(
            splitted_mappings=mappings,
            splitted_sizes=sizes,
            splitted_weights=weights,
            xp=xp,
        )

        return regularization_util.pixel_splitted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
            xp=xp,
        )

    matrix_np = matrix_from(np, mappings, sizes, weights, regularization_weights)
    matrix_jax = matrix_from(
        jnp,
        jnp.asarray(mappings),
        jnp.asarray(sizes),
        jnp.asarray(weights),
        jnp.asarray(regularization_weights),
    )

    assert np.asarray(matrix_jax) == pytest.approx(matrix_np, abs=1.0e-10)
