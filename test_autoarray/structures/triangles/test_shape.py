import numpy as np
import pytest

from autoarray.structures.triangles.shape import (
    Circle,
    Point,
    Polygon,
    Square,
    Triangle,
)


def triangle_array_with_centroid(coordinate, half_width=1.0e-3):
    """
    A single, non-degenerate triangle whose centroid is `coordinate`.

    Used to compare a `contains` test on a coordinate against the `mask` test
    on a triangle sitting at that coordinate.
    """
    coordinate_0, coordinate_1 = coordinate

    return np.array(
        [
            [
                [coordinate_0, coordinate_1 + 2.0 * half_width],
                [coordinate_0 - np.sqrt(3.0) * half_width, coordinate_1 - half_width],
                [coordinate_0 + np.sqrt(3.0) * half_width, coordinate_1 - half_width],
            ]
        ]
    )


"""
An asymmetric triangle, written as the (y, x) coordinates every PyAuto grid
uses: it spans 3 in y and 1 in x, so swapping a coordinate's two elements
changes the answer. Every convention test below relies on that asymmetry.
"""
ASYMMETRIC_TRIANGLE_YX = ((0.0, 0.0), (3.0, 0.0), (0.0, 1.0))
POINT_INSIDE_YX = (2.0, 0.2)


def test_contains_uses_the_same_axis_order_as_the_triangle_array():
    """
    `contains` compares element 0 of a point with `Shape.x` and element 1 with
    `Shape.y` — the same pairing `mask` uses when it reads `triangles[..., 0]`
    and `triangles[..., 1]`. Since every grid PyAuto produces is ordered
    `(y, x)`, and `PointSolver.solve` builds `Point(*source_plane_coordinate)`
    from such a tuple, the attribute named `x` holds the y coordinate.
    """
    triangles = np.array(ASYMMETRIC_TRIANGLE_YX)[None]

    point = POINT_INSIDE_YX
    swapped = point[::-1]

    assert Point(*point).mask(triangles) == np.array([True])
    assert Point(*swapped).mask(triangles) == np.array([False])

    assert Circle(*point, radius=1.0e-8).contains(np.array([point])) == np.array([True])
    assert Circle(*point, radius=1.0e-8).contains(np.array([swapped])) == np.array(
        [False]
    )

    triangle = Triangle(*ASYMMETRIC_TRIANGLE_YX)

    assert triangle.contains(np.array([point, swapped])).tolist() == [True, False]


def test_triangle_contains_mask_is_not_reflected():
    """
    Regression: `triangle_contains_mask` unpacked its own vertices as `y, x`
    (element 1 as "x") while testing a triangle centroid built as `x, y`
    (element 0 as "x"), so it tested the reflected triangle. It reported the
    genuinely-interior point (2.0, 0.2) as outside and its reflection
    (0.2, 2.0) as inside.
    """
    triangle = Triangle(*ASYMMETRIC_TRIANGLE_YX)

    assert triangle.triangle_contains_mask(
        triangle_array_with_centroid(POINT_INSIDE_YX)
    ) == np.array([True])
    assert triangle.triangle_contains_mask(
        triangle_array_with_centroid(POINT_INSIDE_YX[::-1])
    ) == np.array([False])


CIRCLE = Circle(x=1.0, y=2.0, radius=0.5)
TRIANGLE = Triangle(*ASYMMETRIC_TRIANGLE_YX)
POLYGON = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])
SQUARE = Square(top=0.0, bottom=1.0, left=0.0, right=2.0)


@pytest.mark.parametrize(
    "shape, inside, outside",
    [
        (
            CIRCLE,
            [(1.0, 2.0), (1.4, 2.0), (1.0, 2.5), (1.3, 2.3)],
            [(1.6, 2.0), (1.0, 2.6), (2.0, 3.0)],
        ),
        (
            TRIANGLE,
            [(0.1, 0.1), (2.0, 0.2), (0.0, 0.0), (3.0, 0.0)],
            [(2.0, 0.6), (-0.1, 0.5), (4.0, 0.0), (0.0, 1.5)],
        ),
        (
            POLYGON,
            [(1.0, 0.5), (0.0, 0.0), (2.0, 1.0)],
            [(3.0, 0.5), (1.0, 1.5), (-0.5, 0.5)],
        ),
        (
            SQUARE,
            [(1.0, 0.5), (0.0, 0.0), (2.0, 1.0)],
            [(2.5, 0.5), (1.0, 1.5), (-0.5, 0.5)],
        ),
    ],
)
def test_contains(shape, inside, outside):
    assert shape.contains(np.array(inside)).all()
    assert not shape.contains(np.array(outside)).any()


def test_square_contains_is_robust_to_coordinate_ordering():
    """
    `mask` and `area` assume `top < bottom` numerically (coordinates from the
    top-left corner of the image). `contains` and `boundary` sort the pair, so
    they behave identically for a square built from arcsec `(y, x)`
    coordinates, where the top of the image is the larger first coordinate.
    """
    flipped = Square(top=1.0, bottom=0.0, left=2.0, right=0.0)

    points = np.array([(1.0, 0.5), (0.0, 0.0), (2.5, 0.5), (1.0, 1.5)])

    assert flipped.contains(points).tolist() == SQUARE.contains(points).tolist()
    assert flipped.boundary().tolist() == SQUARE.boundary().tolist()


@pytest.mark.parametrize("shape", [CIRCLE, TRIANGLE, POLYGON, SQUARE])
def test_boundary_is_closed_and_its_edges_lie_inside(shape):
    boundary = shape.boundary()

    assert boundary.ndim == 2
    assert boundary.shape[1] == 2
    assert boundary[0] == pytest.approx(boundary[-1])

    midpoints = 0.5 * (boundary[:-1] + boundary[1:])

    assert shape.contains(midpoints).all()


def test_circle_boundary_sample_count():
    assert CIRCLE.boundary(n=8).shape == (9, 2)
    assert CIRCLE.boundary().shape == (101, 2)


@pytest.mark.parametrize(
    "point",
    [
        (1.0, 2.0),
        (1.4, 2.0),
        (1.6, 2.0),
        (1.0, 2.6),
        (0.0, 0.0),
    ],
)
def test_circle_contains_agrees_with_mask(point):
    """
    A triangle small enough to sit at a single coordinate is kept by
    `Circle.mask` exactly when that coordinate is inside the circle.
    """
    assert (
        CIRCLE.contains(np.array([point]))[0]
        == CIRCLE.mask(triangle_array_with_centroid(point))[0]
    )


def test_point_contains_raises():
    point = Point(1.0, 2.0)

    with pytest.raises(NotImplementedError):
        point.contains(np.array([[1.0, 2.0]]))


def test_point_boundary_is_a_single_row():
    assert Point(1.0, 2.0).boundary().tolist() == [[1.0, 2.0]]
