import autoarray as aa
import numpy as np

import pytest


def test__zeroth_regularization_matrix_from__coefficient_1_pixels_3__identity_diagonal_matrix():
    regularization_matrix = aa.util.regularization.zeroth_regularization_matrix_from(
        coefficient=1.0, pixels=3
    )

    assert (
        regularization_matrix
        == np.array(([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    ).all()
    assert abs(np.linalg.det(regularization_matrix)) > 1e-8


def test__zeroth_regularization_matrix_from__coefficient_2_pixels_2__scaled_identity_matrix():
    regularization_matrix = aa.util.regularization.zeroth_regularization_matrix_from(
        coefficient=2.0, pixels=2
    )

    assert (regularization_matrix == np.array(([[4.0, 0.0], [0.0, 4.0]]))).all()
    assert abs(np.linalg.det(regularization_matrix)) > 1e-8


def test__constant_regularization_matrix_from__3_pixel_chain__matches_b_matrix_computation():
    neighbors = np.array([[1, 2, -1], [0, -1, -1], [0, -1, -1]])

    neighbors_sizes = np.array([2, 1, 1])

    b_matrix = np.array(
        [[-1, 1, 0], [-1, 0, 1], [0, 0, 0]]  # Pair 1  # Pair 2
    )  # Pair 1 flip

    test_regularization_matrix = np.matmul(b_matrix.T, b_matrix) + 1e-8 * np.identity(3)

    regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=1.0, neighbors=neighbors, neighbors_sizes=neighbors_sizes
    )

    assert (regularization_matrix == test_regularization_matrix).all()
    assert abs(np.linalg.det(regularization_matrix)) > 1e-8


def test__constant_regularization_matrix_from__4_pixel_ring_coefficient_1__matches_b_matrix_computation():
    b_matrix = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]])

    test_regularization_matrix = np.matmul(b_matrix.T, b_matrix) + 1e-8 * np.identity(4)

    neighbors = np.array(
        [[1, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1], [0, 2, -1, -1]]
    )

    neighbors_sizes = np.array([2, 2, 2, 2])

    regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=1.0, neighbors=neighbors, neighbors_sizes=neighbors_sizes
    )

    assert (regularization_matrix == test_regularization_matrix).all()
    assert abs(np.linalg.det(regularization_matrix)) > 1e-8


def test__constant_regularization_matrix_from__4_pixel_ring_coefficient_2__scaled_regularization_matrix():
    neighbors = np.array(
        [[1, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1], [0, 2, -1, -1]]
    )

    neighbors_sizes = np.array([2, 2, 2, 2])

    b_matrix = 2.0 * np.array(
        [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]]
    )

    test_regularization_matrix = np.matmul(b_matrix.T, b_matrix) + 1e-8 * np.identity(4)

    regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=2.0, neighbors=neighbors, neighbors_sizes=neighbors_sizes
    )

    assert (regularization_matrix == test_regularization_matrix).all()
    assert abs(np.linalg.det(regularization_matrix)) > 1e-8


def test__constant_regularization_matrix_from__9_pixel_grid__matches_b_matrix_computation():
    neighbors = np.array(
        [
            [1, 3, -1, -1],
            [4, 2, 0, -1],
            [1, 5, -1, -1],
            [4, 6, 0, -1],
            [7, 1, 5, 3],
            [4, 2, 8, -1],
            [7, 3, -1, -1],
            [4, 8, 6, -1],
            [7, 5, -1, -1],
        ]
    )

    neighbors_sizes = np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])

    b_matrix_0 = np.array(
        [
            [-1, 1, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 1, 0],
        ]
    )

    b_matrix_1 = np.array(
        [
            [0, 0, 0, 0, 0, -1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    test_regularization_matrix_0 = np.matmul(b_matrix_0.T, b_matrix_0)
    test_regularization_matrix_1 = np.matmul(b_matrix_1.T, b_matrix_1)

    test_regularization_matrix = (
        test_regularization_matrix_0
        + test_regularization_matrix_1
        + 1e-8 * np.identity(9)
    )

    regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=1.0, neighbors=neighbors, neighbors_sizes=neighbors_sizes
    )

    assert (regularization_matrix == test_regularization_matrix).all()
    assert abs(np.linalg.det(regularization_matrix)) > 1e-8


def test__constant_zeroth_regularization_matrix_from():
    neighbors = np.array([[1, 2, -1], [0, -1, -1], [0, -1, -1]])

    neighbors_sizes = np.array([2, 1, 1])

    regularization_matrix = (
        aa.util.regularization.constant_zeroth_regularization_matrix_from(
            coefficient=2.0,
            coefficient_zeroth=0.5,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )
    )

    assert regularization_matrix == pytest.approx(
        np.array([[8.25, -4.0, -4.0], [-4.0, 4.25, 0.0], [-4.0, 0.0, 4.25]]), 1.0e-4
    )
    assert abs(np.linalg.det(regularization_matrix)) > 1e-8


def test__adapt_regularization_weights_from__uniform_signals_equal_inner_outer__all_weights_one():
    pixel_signals = np.array([1.0, 1.0, 1.0])

    weight_list = aa.util.regularization.adapt_regularization_weights_from(
        inner_coefficient=1.0, outer_coefficient=1.0, pixel_signals=pixel_signals
    )

    assert (weight_list == np.array([1.0, 1.0, 1.0])).all()


def test__adapt_regularization_weights_from__non_uniform_signals_equal_inner_outer__all_weights_one():
    pixel_signals = np.array([0.25, 0.5, 0.75])

    weight_list = aa.util.regularization.adapt_regularization_weights_from(
        inner_coefficient=1.0, outer_coefficient=1.0, pixel_signals=pixel_signals
    )

    assert (weight_list == np.array([1.0, 1.0, 1.0])).all()


def test__adapt_regularization_weights_from__non_uniform_signals_outer_zero__weights_scale_with_signal():
    pixel_signals = np.array([0.25, 0.5, 0.75])

    weight_list = aa.util.regularization.adapt_regularization_weights_from(
        inner_coefficient=1.0, outer_coefficient=0.0, pixel_signals=pixel_signals
    )

    assert (weight_list == np.array([0.25**2.0, 0.5**2.0, 0.75**2.0])).all()


def test__adapt_regularization_weights_from__non_uniform_signals_inner_zero__weights_scale_inversely():
    pixel_signals = np.array([0.25, 0.5, 0.75])

    weight_list = aa.util.regularization.adapt_regularization_weights_from(
        inner_coefficient=0.0, outer_coefficient=1.0, pixel_signals=pixel_signals
    )

    assert (weight_list == np.array([0.75**2.0, 0.5**2.0, 0.25**2.0])).all()


def test__brightness_zeroth_regularization_weights_from__uniform_max_signals__all_zero_weights():
    pixel_signals = np.array([1.0, 1.0, 1.0])

    weight_list = aa.util.regularization.brightness_zeroth_regularization_weights_from(
        coefficient=1.0, pixel_signals=pixel_signals
    )

    assert (weight_list == np.array([0.0, 0.0, 0.0])).all()


def test__brightness_zeroth_regularization_weights_from__non_uniform_signals_coefficient_1__complement_weights():
    pixel_signals = np.array([0.25, 0.5, 0.75])

    weight_list = aa.util.regularization.brightness_zeroth_regularization_weights_from(
        coefficient=1.0, pixel_signals=pixel_signals
    )

    assert (weight_list == np.array([0.75, 0.5, 0.25])).all()


def test__brightness_zeroth_regularization_weights_from__non_uniform_signals_coefficient_2__scaled_complement_weights():
    pixel_signals = np.array([0.25, 0.5, 0.75])

    weight_list = aa.util.regularization.brightness_zeroth_regularization_weights_from(
        coefficient=2.0, pixel_signals=pixel_signals
    )

    assert (weight_list == np.array([1.5, 1.0, 0.5])).all()


def test__weighted_regularization_matrix_from__4_pixel_cycle_uniform_weights__matches_b_matrix():
    neighbors = np.array([[2], [3], [0], [1]])

    b_matrix = np.array([[-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]])

    test_regularization_matrix = np.matmul(b_matrix.T, b_matrix)

    regularization_weights = np.ones((4,))

    regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
        regularization_weights=regularization_weights,
        neighbors=neighbors,
    )

    assert regularization_matrix == pytest.approx(test_regularization_matrix, 1.0e-4)


def test__weighted_regularization_matrix_from__3_pixel_chain_uniform_weights__matches_b_matrix():
    # Here, we define the neighbors first here and make the B matrices based on them.

    # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
    # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
    # the rows of each B matrix wherever we like ;0.

    neighbors = np.array([[1, 2], [0, -1], [0, -1]])

    b_matrix_1 = np.array(
        [[-1, 1, 0], [-1, 0, 1], [1, -1, 0]]  # Pair 1  # Pair 2
    )  # Pair 1 flip

    test_regularization_matrix_1 = np.matmul(b_matrix_1.T, b_matrix_1)

    b_matrix_2 = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]])  # Pair 2 flip

    test_regularization_matrix_2 = np.matmul(b_matrix_2.T, b_matrix_2)

    test_regularization_matrix = (
        test_regularization_matrix_1 + test_regularization_matrix_2
    )

    regularization_weights = np.ones((3))

    regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
        regularization_weights=regularization_weights,
        neighbors=neighbors,
    )

    assert regularization_matrix == pytest.approx(test_regularization_matrix, 1.0e-4)


def test__weighted_regularization_matrix_from__4_pixel_ring_uniform_weights__matches_combined_b_matrix():
    b_matrix_1 = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]])

    test_regularization_matrix_1 = np.matmul(b_matrix_1.T, b_matrix_1)

    b_matrix_2 = np.array([[-1, 0, 0, 1], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])

    test_regularization_matrix_2 = np.matmul(b_matrix_2.T, b_matrix_2)

    test_regularization_matrix = (
        test_regularization_matrix_1 + test_regularization_matrix_2
    )

    neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

    regularization_weights = np.ones((4,))

    regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
        regularization_weights=regularization_weights,
        neighbors=neighbors,
    )

    assert regularization_matrix == pytest.approx(test_regularization_matrix, 1.0e-4)


def test__weighted_regularization_matrix_from__6_pixel_graph_uniform_weights__matches_b_matrix():
    # Again, lets exploit the freedom we have when setting up our B matrices to make matching it to pairs a
    # lot less Stressful.

    neighbors = np.array(
        [
            [2, 3, 4, -1],
            [2, 5, -1, -1],
            [0, 1, 3, 5],
            [0, 2, -1, -1],
            [5, 0, -1, -1],
            [4, 1, 2, -1],
        ]
    )

    neighbors_sizes = np.array([3, 2, 4, 2, 2, 3])

    b_matrix_1 = np.array(
        [
            [-1, 0, 1, 0, 0, 0],  # Pair 1
            [0, -1, 1, 0, 0, 0],  # Pair 2
            [-1, 0, 0, 1, 0, 0],  # Pair 3
            [0, 0, 0, 0, -1, 1],  # Pair 4
            [0, -1, 0, 0, 0, 1],  # Pair 5
            [-1, 0, 0, 0, 1, 0],
        ]
    )  # Pair 6

    test_regularization_matrix_1 = np.matmul(b_matrix_1.T, b_matrix_1)

    b_matrix_2 = np.array(
        [
            [0, 0, -1, 1, 0, 0],  # Pair 7
            [0, 0, -1, 0, 0, 1],  # Pair 8
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    test_regularization_matrix_2 = np.matmul(b_matrix_2.T, b_matrix_2)

    b_matrix_3 = np.array(
        [
            [1, 0, -1, 0, 0, 0],  # Pair 1 flip
            [0, 1, -1, 0, 0, 0],  # Pair 2 flip
            [1, 0, 0, -1, 0, 0],  # Pair 3 flip
            [0, 0, 0, 0, 1, -1],  # Pair 4 flip
            [0, 1, 0, 0, 0, -1],  # Pair 5 flip
            [1, 0, 0, 0, -1, 0],
        ]
    )  # Pair 6 flip

    test_regularization_matrix_3 = np.matmul(b_matrix_3.T, b_matrix_3)

    b_matrix_4 = np.array(
        [
            [0, 0, 1, -1, 0, 0],  # Pair 7 flip
            [0, 0, 1, 0, 0, -1],  # Pair 8 flip
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    test_regularization_matrix_4 = np.matmul(b_matrix_4.T, b_matrix_4)

    test_regularization_matrix = (
        test_regularization_matrix_1
        + test_regularization_matrix_2
        + test_regularization_matrix_3
        + +test_regularization_matrix_4
    )

    regularization_weights = np.ones((6))

    regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
        regularization_weights=regularization_weights,
        neighbors=neighbors,
    )

    assert regularization_matrix == pytest.approx(test_regularization_matrix, 1.0e-4)


def test__weighted_regularization_matrix_from__4_pixel_non_uniform_weights__matches_weighted_b_matrix():
    # Simple case, where we have just one regularization direction, regularizing pixel 0 -> 1 and 1 -> 2.

    # This means our B matrix is:

    # [-1, 1, 0]
    # [0, -1, 1]
    # [0, 0, -1]

    # Regularization Matrix, H = B * B.T.I can

    regularization_weights = np.array([2.0, 4.0, 1.0, 8.0])

    b_matrix_1 = np.array([[-2, 2, 0, 0], [-2, 0, 2, 0], [0, -4, 4, 0], [0, -4, 0, 4]])

    b_matrix_2 = np.array([[4, -4, 0, 0], [1, 0, -1, 0], [0, 1, -1, 0], [0, 8, 0, -8]])

    test_regularization_matrix_1 = np.matmul(b_matrix_1.T, b_matrix_1)
    test_regularization_matrix_2 = np.matmul(b_matrix_2.T, b_matrix_2)

    test_regularization_matrix = (
        test_regularization_matrix_1 + test_regularization_matrix_2
    )

    neighbors = np.array(
        [[1, 2, -1, -1], [0, 2, 3, -1], [0, 1, -1, -1], [1, -1, -1, -1]]
    )

    regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
        regularization_weights=regularization_weights,
        neighbors=neighbors,
    )

    assert regularization_matrix == pytest.approx(test_regularization_matrix, 1.0e-4)


def test__weighted_regularization_matrix_from__6_pixel_non_uniform_weights__matches_weighted_b_matrix():
    neighbors = np.array(
        [
            [1, 4, -1, -1],
            [2, 4, 0, -1],
            [3, 4, 5, 1],
            [5, 2, -1, -1],
            [5, 0, 1, 2],
            [2, 3, 4, -1],
        ]
    )

    regularization_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # I'm inputting the regularization weight_list directly thiss time, as it'd be a pain to multiply with a
    # loop.

    b_matrix_1 = np.array(
        [
            [-1, 1, 0, 0, 0, 0],  # Pair 1
            [-1, 0, 0, 0, 1, 0],  # Pair 2
            [0, -2, 2, 0, 0, 0],  # Pair 3
            [0, -2, 0, 0, 2, 0],  # Pair 4
            [0, 0, -3, 3, 0, 0],  # Pair 5
            [0, 0, -3, 0, 3, 0],
        ]
    )  # Pair 6

    b_matrix_2 = np.array(
        [
            [0, 0, -3, 0, 0, 3],  # Pair 7
            [0, 0, 0, -4, 0, 4],  # Pair 8
            [0, 0, 0, 0, -5, 5],  # Pair 9
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    # Now do the same pairs but with the regularization direction and weight_list swapped.

    b_matrix_3 = np.array(
        [
            [2, -2, 0, 0, 0, 0],  # Pair 1
            [5, 0, 0, 0, -5, 0],  # Pair 2
            [0, 3, -3, 0, 0, 0],  # Pair 3
            [0, 5, 0, 0, -5, 0],  # Pair 4
            [0, 0, 4, -4, 0, 0],  # Pair 5
            [0, 0, 5, 0, -5, 0],
        ]
    )  # Pair 6

    b_matrix_4 = np.array(
        [
            [0, 0, 6, 0, 0, -6],  # Pair 7
            [0, 0, 0, 6, 0, -6],  # Pair 8
            [0, 0, 0, 0, 6, -6],  # Pair 9
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    test_regularization_matrix_1 = np.matmul(b_matrix_1.T, b_matrix_1)
    test_regularization_matrix_2 = np.matmul(b_matrix_2.T, b_matrix_2)
    test_regularization_matrix_3 = np.matmul(b_matrix_3.T, b_matrix_3)
    test_regularization_matrix_4 = np.matmul(b_matrix_4.T, b_matrix_4)

    test_regularization_matrix = (
        test_regularization_matrix_1
        + test_regularization_matrix_2
        + test_regularization_matrix_3
        + test_regularization_matrix_4
    )

    regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
        regularization_weights=regularization_weights,
        neighbors=neighbors,
    )

    assert regularization_matrix == pytest.approx(test_regularization_matrix, 1.0e-4)


def test__brightness_zeroth_regularization_matrix_from__uniform_weights__identity_diagonal_matrix():
    regularization_weights = np.ones(shape=(3,))

    regularization_matrix = (
        aa.util.regularization.brightness_zeroth_regularization_matrix_from(
            regularization_weights=regularization_weights,
        )
    )

    assert regularization_matrix == pytest.approx(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 1.0e-4
    )


def test__brightness_zeroth_regularization_matrix_from__non_uniform_weights__squared_weights_on_diagonal():
    regularization_weights = np.array([1.0, 2.0, 3.0])

    regularization_matrix = (
        aa.util.regularization.brightness_zeroth_regularization_matrix_from(
            regularization_weights=regularization_weights,
        )
    )

    assert regularization_matrix == pytest.approx(
        np.array([[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]), 1.0e-4
    )


@pytest.fixture
def splitted_data():

    splitted_mappings = np.array(
        [
            [0, -1, -1, -1],
            [1, 3, -1, -1],
            [1, 4, 2, -1],
            [2, 3, -1, -1],
            [1, 2, 3, 4],
            [0, 3, 4, -1],
            [4, -1, -1, -1],
            [3, -1, -1, -1],
            [0, 3, -1, -1],
            [2, 3, -1, -1],
            [0, -1, -1, -1],
            [3, -1, -1, -1],
            [4, 2, -1, -1],
            [1, 4, -1, -1],
            [2, 4, -1, -1],
            [3, 1, 2, -1],
            [2, 1, 4, -1],
            [2, -1, -1, -1],
            [3, 4, -1, -1],
            [1, 4, -1, -1],
        ]
    )

    splitted_sizes = np.sum(splitted_mappings != -1, axis=1)

    splitted_weights = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.1, 0.3, 0.6, 0.0],
            [0.15, 0.85, 0.0, 0.0],
            [0.2, 0.25, 0.1, 0.45],
            [0.3, 0.6, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
            [0.36, 0.64, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.77, 0.23, 0.0, 0.0],
            [0.12, 0.4, 0.48, 0.0],
            [0.6, 0.15, 0.25, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.66, 0.34, 0.0, 0.0],
            [0.57, 0.43, 0.0, 0.0],
        ]
    )

    return splitted_mappings, splitted_sizes, splitted_weights


def test__reg_split_from__splitted_mapping_data__produces_correct_mappings_sizes_and_weights(
    splitted_data,
):

    splitted_mappings, splitted_sizes, splitted_weights = splitted_data

    splitted_mappings, splitted_sizes, splitted_weights = (
        aa.util.regularization.reg_split_from(
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )
    )

    expected_mappings = np.array(
        [
            [0, -1, -1, -1],
            [1, 3, 0, -1],
            [1, 4, 2, 0],
            [2, 3, 0, -1],
            [1, 2, 3, 4],
            [0, 3, 4, 1],
            [4, 1, -1, -1],
            [3, 1, -1, -1],
            [0, 3, 2, -1],
            [2, 3, -1, -1],
            [0, 2, -1, -1],
            [3, 2, -1, -1],
            [4, 2, 3, -1],
            [1, 4, 3, -1],
            [2, 4, 3, -1],
            [3, 1, 2, -1],
            [2, 1, 4, -1],
            [2, 4, -1, -1],
            [3, 4, -1, -1],
            [1, 4, -1, -1],
        ]
    )

    expected_sizes = np.array(
        [1, 3, 4, 3, 4, 4, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2]
    )

    expected_weights = np.array(
        [
            [0.00, -0.00, -0.00, -0.00],
            [-0.20, -0.80, 1.00, -0.00],
            [-0.10, -0.30, -0.60, 1.00],
            [-0.15, -0.85, 1.00, -0.00],
            [0.80, -0.25, -0.10, -0.45],
            [-0.30, -0.60, -0.10, 1.00],
            [-1.00, 1.00, -0.00, -0.00],
            [-1.00, 1.00, -0.00, -0.00],
            [-0.70, -0.30, 1.00, -0.00],
            [0.64, -0.64, -0.00, -0.00],
            [-1.00, 1.00, -0.00, -0.00],
            [-1.00, 1.00, -0.00, -0.00],
            [-0.95, -0.05, 1.00, -0.00],
            [-0.10, -0.90, 1.00, -0.00],
            [-0.77, -0.23, 1.00, -0.00],
            [0.88, -0.40, -0.48, -0.00],
            [-0.60, -0.15, 0.75, -0.00],
            [-1.00, 1.00, -0.00, -0.00],
            [-0.66, 0.66, -0.00, -0.00],
            [-0.57, 0.57, -0.00, -0.00],
        ]
    )

    assert splitted_mappings == pytest.approx(expected_mappings, abs=1.0e-4)
    assert splitted_sizes == pytest.approx(expected_sizes, abs=1.0e-4)
    assert splitted_weights == pytest.approx(expected_weights, abs=1.0e-4)


def test__pixel_splitted_regularization_matrix_from__uniform_weights__correct_regularization_matrix(
    splitted_data,
):

    splitted_mappings, splitted_sizes, splitted_weights = splitted_data

    pixels = int(len(splitted_mappings) / 4)

    regularization_matrix = (
        aa.util.regularization.pixel_splitted_regularization_matrix_from(
            regularization_weights=np.full(fill_value=1.0, shape=(pixels,)),
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )
    )

    expected_reg_matrix = np.array(
        [
            [2.58000001, 0.0, 0.0, 0.39, 0.03],
            [0.0, 0.60740001, 0.392, 0.228, 0.4926],
            [0.0, 0.392, 2.76040001, 0.4405, 0.6671],
            [0.39, 0.228, 0.4405, 4.68210001, 0.3294],
            [0.03, 0.4926, 0.6671, 0.3294, 3.43090001],
        ]
    )

    assert pytest.approx(regularization_matrix, 1e-4) == np.array(expected_reg_matrix)


def test__pixel_splitted_regularization_matrix_from__non_uniform_weights__correct_regularization_matrix(
    splitted_data,
):

    splitted_mappings, splitted_sizes, splitted_weights = splitted_data

    regularization_weights = np.array([2.0, 4.0, 2.0, 2.0, 2.0])

    regularization_matrix = (
        aa.util.regularization.pixel_splitted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )
    )

    expected_reg_matrix = np.array(
        [
            [11.40000001, 0.0, 0.0, 3.72, 0.48],
            [0.0, 2.90960001, 2.168, 1.152, 3.0504],
            [0.0, 2.168, 11.79160001, 2.062, 4.0184],
            [3.72, 1.152, 2.062, 35.16840001, 2.5776],
            [0.48, 3.0504, 4.0184, 2.5776, 28.27360001],
        ]
    )

    assert pytest.approx(regularization_matrix, 1e-4) == np.array(expected_reg_matrix)
