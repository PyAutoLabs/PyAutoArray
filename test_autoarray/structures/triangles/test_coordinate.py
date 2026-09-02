import numpy as np

import pytest

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR

from autoarray.structures.triangles.coordinate_array_np import (
    CoordinateArrayTrianglesNp,
)


def test__two(two_triangles):

    assert np.all(two_triangles.centres == np.array([[0, 0], [0.5, 0]]))
    assert two_triangles.triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, HEIGHT_FACTOR / 2],
                    [0.5, -HEIGHT_FACTOR / 2],
                    [-0.5, -HEIGHT_FACTOR / 2],
                ],
                [
                    [0.5, -HEIGHT_FACTOR / 2],
                    [0.0, HEIGHT_FACTOR / 2],
                    [1.0, HEIGHT_FACTOR / 2],
                ],
            ]
        ),
        1.0e-4,
    )


def test__trivial_triangles(one_triangle):
    assert one_triangle.flip_array == np.array([1])
    assert np.all(one_triangle.centres == np.array([[0, 0]]))
    assert one_triangle.triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, HEIGHT_FACTOR / 2],
                    [0.5, -HEIGHT_FACTOR / 2],
                    [-0.5, -HEIGHT_FACTOR / 2],
                ],
            ]
        ),
        1.0e-4,
    )


def test__above():
    triangles = CoordinateArrayTrianglesNp(
        coordinates=np.array([[0, 1]]),
        side_length=1.0,
    )
    assert triangles.up_sample().triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, 0.43301270189221935],
                    [-0.25, 0.8660254037844386],
                    [0.25, 0.8660254037844386],
                ],
                [
                    [0.25, 0.8660254037844388],
                    [0.0, 1.299038105676658],
                    [0.5, 1.299038105676658],
                ],
                [
                    [-0.25, 0.8660254037844388],
                    [-0.5, 1.299038105676658],
                    [0.0, 1.299038105676658],
                ],
                [
                    [0.0, 1.299038105676658],
                    [0.25, 0.8660254037844388],
                    [-0.25, 0.8660254037844388],
                ],
            ]
        ),
        1.0e-4,
    )


@pytest.fixture
def upside_down():
    return CoordinateArrayTrianglesNp(
        coordinates=np.array([[1, 0]]),
        side_length=1.0,
    )


def test_upside_down(upside_down):
    assert np.all(upside_down.centres == np.array([[0.5, 0]]))
    assert upside_down.triangles == pytest.approx(
        np.array(
            [
                [
                    [0.5, -HEIGHT_FACTOR / 2],
                    [0.0, HEIGHT_FACTOR / 2],
                    [1.0, HEIGHT_FACTOR / 2],
                ],
            ]
        ),
        1.0e-4,
    )


def test_up_sample(one_triangle):
    up_sampled = one_triangle.up_sample()
    assert up_sampled.side_length == 0.5
    assert up_sampled.triangles == pytest.approx(
        np.array(
            [
                [[0.0, -0.4330127018922193], [-0.25, 0.0], [0.25, 0.0]],
                [[0.25, 0.0], [0.5, -0.4330127018922193], [0.0, -0.4330127018922193]],
                [[-0.25, 0.0], [0.0, -0.4330127018922193], [-0.5, -0.4330127018922193]],
                [[0.0, 0.4330127018922193], [0.25, 0.0], [-0.25, 0.0]],
            ]
        ),
        1.0e-4,
    )


def test_up_sample_upside_down(upside_down):
    up_sampled = upside_down.up_sample()
    assert up_sampled.side_length == 0.5
    assert up_sampled.triangles == pytest.approx(
        np.array(
            [
                [[0.5, -0.4330127018922193], [0.25, 0.0], [0.75, 0.0]],
                [[0.75, 0.0], [0.5, 0.4330127018922193], [1.0, 0.4330127018922193]],
                [[0.25, 0.0], [0.0, 0.4330127018922193], [0.5, 0.4330127018922193]],
                [[0.5, 0.4330127018922193], [0.75, 0.0], [0.25, 0.0]],
            ]
        ),
        1.0e-4,
    )


def _test_up_sample_twice(one_triangle, plot):
    plot(one_triangle)
    one = one_triangle.up_sample()
    two = one.up_sample()
    three = two.up_sample()
    plot(three, color="blue")
    plot(two, color="green")
    plot(one, color="red")


def test_neighborhood(one_triangle):
    assert one_triangle.neighborhood().triangles == pytest.approx(
        np.array(
            [
                [
                    [-0.5, -0.4330127018922193],
                    [-1.0, 0.4330127018922193],
                    [0.0, 0.4330127018922193],
                ],
                [
                    [0.0, -1.299038105676658],
                    [-0.5, -0.4330127018922193],
                    [0.5, -0.4330127018922193],
                ],
                [
                    [0.0, 0.4330127018922193],
                    [0.5, -0.4330127018922193],
                    [-0.5, -0.4330127018922193],
                ],
                [
                    [0.5, -0.4330127018922193],
                    [0.0, 0.4330127018922193],
                    [1.0, 0.4330127018922193],
                ],
            ]
        ),
        1.0e-4,
    )


def test_upside_down_neighborhood(upside_down):
    assert upside_down.neighborhood().triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, 0.4330127018922193],
                    [0.5, -0.4330127018922193],
                    [-0.5, -0.4330127018922193],
                ],
                [
                    [0.5, -0.4330127018922193],
                    [0.0, 0.4330127018922193],
                    [1.0, 0.4330127018922193],
                ],
                [
                    [0.5, 1.299038105676658],
                    [1.0, 0.4330127018922193],
                    [0.0, 0.4330127018922193],
                ],
                [
                    [1.0, 0.4330127018922193],
                    [1.5, -0.4330127018922193],
                    [0.5, -0.4330127018922193],
                ],
            ]
        ),
        1.0e-4,
    )


def _test_complicated(plot, one_triangle):
    triangles = one_triangle.neighborhood().neighborhood()
    up_sampled = triangles.up_sample()


def test_vertices(one_triangle):
    assert one_triangle.vertices == pytest.approx(
        np.array(
            [
                [-0.5, -0.4330127018922193],
                [0.0, 0.4330127018922193],
                [0.5, -0.4330127018922193],
            ]
        ),
        1.0e-4,
    )


def test_up_sampled_vertices(one_triangle):
    assert one_triangle.up_sample().vertices[0:6, :] == pytest.approx(
        np.array(
            [
                [-0.5, -0.4330127018922193],
                [-0.25, 0.0],
                [0.0, -0.4330127018922193],
                [0.0, 0.4330127018922193],
                [0.25, 0.0],
                [0.5, -0.4330127018922193],
            ]
        ),
        1.0e-4,
    )


def test_with_vertices(one_triangle):
    triangle = one_triangle.with_vertices(np.array([[0, 0], [1, 0], [0.5, 1]]))
    assert triangle.triangles == pytest.approx(
        np.array([[[1.0, 0.0], [0.5, 1.0], [0.0, 0.0]]]), 1.0e-4
    )


def _test_multiple_with_vertices(one_triangle, plot):
    up_sampled = one_triangle.up_sample()
    plot(up_sampled.with_vertices(2 * up_sampled.vertices).triangles.tolist())


def test_for_indexes(two_triangles):
    assert two_triangles.for_indexes(np.array([0])).triangles == pytest.approx(
        np.array(
            [
                [
                    [0.0, 0.4330127018922193],
                    [0.5, -0.4330127018922193],
                    [-0.5, -0.4330127018922193],
                ]
            ]
        ),
        1.0e-4,
    )


def test_means(one_triangle):
    assert one_triangle.means == pytest.approx(
        np.array([[0.0, -0.14433756729740643]]), 1.0e-4
    )


def test_triangles_touch():
    triangles = CoordinateArrayTrianglesNp(
        np.array([[0, 0], [2, 0]]),
    )

    assert max(triangles.triangles[0][:, 0]) == min(triangles.triangles[1][:, 0])

    triangles = CoordinateArrayTrianglesNp(
        np.array([[0, 0], [0, 1]]),
    )
    assert max(triangles.triangles[0][:, 1]) == min(triangles.triangles[1][:, 1])


def test_from_grid_regression():
    triangles = CoordinateArrayTrianglesNp.for_limits_and_scale(
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
        scale=1.5,
    )

    x = triangles.vertices[:, 0]
    assert min(x) <= -2.0
    assert max(x) >= 2.0

    y = triangles.vertices[:, 1]
    assert min(y) <= -2.0
    assert max(y) >= 2.0


def test_means_up_sampled(one_triangle):
    assert len(one_triangle.means) == 1

    up_sampled = one_triangle.up_sample()
    neighborhood = up_sampled.neighborhood()
    assert np.count_nonzero(~np.isnan(neighborhood.means).any(axis=1)) == 10


ONE_TRIANGLE_AREA = HEIGHT_FACTOR * 0.5


def test_area(one_triangle):
    assert one_triangle.area == ONE_TRIANGLE_AREA
    assert one_triangle.up_sample().area == ONE_TRIANGLE_AREA

    neighborhood = one_triangle.neighborhood()
    assert neighborhood.area == 4 * ONE_TRIANGLE_AREA
    assert neighborhood.up_sample().area == 4 * ONE_TRIANGLE_AREA
    assert neighborhood.neighborhood().area == 10 * ONE_TRIANGLE_AREA


def test_for_limits_and_scale__element_0_spans_the_y_limits():
    """
    Regression: `for_limits_and_scale` named its first pair of limits ``x_min``/``x_max``
    and tiled them along element ``0`` of every vertex, while element ``0`` is the ``y``
    coordinate everywhere else (``ArrayTrianglesNp``, every PyAuto ``(y, x)`` grid, and
    the ``element 0 <-> element 0`` convention of ``Shape.contains`` / ``Shape.mask``).
    A rectangular extent was therefore tiled transposed, so the point solver searched a
    box the data does not occupy. A square extent hides it, which is why it shipped.
    """
    triangles = CoordinateArrayTrianglesNp.for_limits_and_scale(
        y_min=-2.0,
        y_max=2.0,
        x_min=-1.0,
        x_max=1.0,
        scale=0.5,
    )

    vertices = triangles.vertices

    assert min(vertices[:, 0]) <= -2.0
    assert max(vertices[:, 0]) >= 2.0

    assert min(vertices[:, 1]) <= -1.0
    assert max(vertices[:, 1]) >= 1.0

    # The transposed tiling would have run element 1 out to +/- 2.0.
    assert max(vertices[:, 1]) < 2.0


def test_for_grid__rectangular_grid_is_tiled_the_same_way_round():
    """
    ``AbstractTriangles.for_grid`` passes the limits *positionally* as
    ``(y_min, y_max, x_min, x_max)``, so the signature order matters as much as the
    tiling does.
    """
    from autoarray.structures.grids.uniform_2d import Grid2D

    grid = Grid2D.uniform(shape_native=(24, 80), pixel_scales=0.05)

    triangles = CoordinateArrayTrianglesNp.for_grid(grid=grid)

    vertices = triangles.vertices

    assert max(vertices[:, 0]) < max(vertices[:, 1])


def test_array_triangles_with_vertices_returns_new_triangles():
    """
    Regression: ``ArrayTrianglesNp.with_vertices`` carried a stray ``bbbb`` statement (a
    ``NameError`` on every call) from 2025-11, uncaught because the solver only ever
    reached the ``CoordinateArrayTrianglesNp`` override of the same name.
    """
    from autoarray.structures.triangles.array_np import ArrayTrianglesNp

    triangles = ArrayTrianglesNp(
        indices=np.array([[0, 1, 2]]),
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )

    new_vertices = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])

    assert (triangles.with_vertices(new_vertices).vertices == new_vertices).all()
