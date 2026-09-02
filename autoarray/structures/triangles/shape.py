from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class Shape(ABC):
    """
    A shape in the source plane for which we identify corresponding image plane
    pixels using up-sampling.

    Coordinate convention
    ---------------------
    Coordinates are indexed here exactly as they are in the triangle arrays:
    element ``0`` of a coordinate pair is the attribute these classes call
    ``x`` and element ``1`` is the attribute they call ``y``. For a triangle
    array of shape ``(n, 3, 2)`` this is ``triangles[..., 0]`` and
    ``triangles[..., 1]``, and the ``(N, 2)`` array of points passed to
    `contains` is ordered the same way, as is the polygon returned by
    `boundary`.

    The legacy attribute names do **not** correspond to the physical axes.
    Every grid PyAuto produces is ordered ``(y, x)``, and ``PointSolver.solve``
    builds its shape as ``Point(*source_plane_coordinate)`` from such a
    ``(y, x)`` tuple, so the attribute named `x` in fact holds the ``y``
    (first) coordinate and the attribute named `y` holds the ``x`` (second)
    coordinate. The names are kept for backwards compatibility; what matters
    is that points, vertices and triangle arrays all use element ``0`` for the
    first axis, so containment is computed consistently and a `boundary` can be
    plotted directly against an arcsec ``(y, x)`` grid.
    """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        The area of the shape.
        """

    @abstractmethod
    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles contain the shape.

        Parameters
        ----------
        triangles
            The vertices of the triangles.

        Returns
        -------
        A boolean array indicating which triangles contain the shape.
        """

    @abstractmethod
    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Determine which points lie inside the shape.

        Parameters
        ----------
        points
            An array of coordinates of shape ``(N, 2)``, ordered the same way
            as the triangle vertices (see the class docstring).

        Returns
        -------
        A boolean array of shape ``(N,)`` indicating which points lie inside
        the shape.
        """

    @abstractmethod
    def boundary(self, n: int = 100) -> np.ndarray:
        """
        The boundary of the shape, as a closed polygon.

        Parameters
        ----------
        n
            The number of samples used for curved boundaries (e.g. a circle).
            Ignored by shapes whose boundary is already polygonal.

        Returns
        -------
        An array of shape ``(M, 2)`` whose first row is repeated as its last
        row, ordered the same way as the triangle vertices.
        """


def _barycentric_contains(
    a_0,
    a_1,
    b_0,
    b_1,
    c_0,
    c_1,
    coordinate_0,
    coordinate_1,
) -> np.ndarray:
    """
    Determine which coordinates lie inside the triangle with vertices
    ``(a_0, a_1)``, ``(b_0, b_1)`` and ``(c_0, c_1)``, using barycentric
    coordinates.

    Vertices and coordinates are indexed with the same axis order (element 0
    against element 0) — see the `Shape` docstring. Either the vertices or the
    coordinates may be arrays, so this single implementation serves both
    "which triangles contain this shape" and "which points does this shape
    contain".

    Parameters
    ----------
    a_0, a_1, b_0, b_1, c_0, c_1
        The components of the three vertices of the triangle.
    coordinate_0, coordinate_1
        The components of the coordinates being tested.

    Returns
    -------
    A boolean array indicating which coordinates lie inside the triangle.
    """
    denominator = (b_1 - c_1) * (a_0 - c_0) + (c_0 - b_0) * (a_1 - c_1)

    alpha = (
        (b_1 - c_1) * (coordinate_0 - c_0) + (c_0 - b_0) * (coordinate_1 - c_1)
    ) / denominator
    beta = (
        (c_1 - a_1) * (coordinate_0 - c_0) + (a_0 - c_0) * (coordinate_1 - c_1)
    ) / denominator
    gamma = 1 - alpha - beta

    return (
        (0 <= alpha)
        & (alpha <= 1)
        & (0 <= beta)
        & (beta <= 1)
        & (0 <= gamma)
        & (gamma <= 1)
    )


class Point(Shape):
    def __init__(self, x: float, y: float):
        """
        A point in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        x
        y
            The coordinates of the point.
        """
        self.x = x
        self.y = y

    @property
    def area(self) -> float:
        """
        The area of the point.
        """
        return 0.0

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles contain the point.

        Parameters
        ----------
        triangles
            The vertices of the triangles

        Returns
        -------
        A boolean array indicating which triangles contain the point.
        """
        return _barycentric_contains(
            triangles[:, 0, 0],
            triangles[:, 0, 1],
            triangles[:, 1, 0],
            triangles[:, 1, 1],
            triangles[:, 2, 0],
            triangles[:, 2, 1],
            self.x,
            self.y,
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        A point has no area, so no coordinate lies inside it.

        Raises
        ------
        NotImplementedError
            Always. Use `Circle` (or another finite `Shape`) for a source-plane
            region with an interior, or `PointSolver` to find the image-plane
            positions to which a point source traces.
        """
        raise NotImplementedError(
            "A Point has zero area so it contains no coordinates. Use a Circle "
            "(or another finite Shape) for a source-plane region with an "
            "interior, or PointSolver to find the image plane positions of a "
            "point source."
        )

    def boundary(self, n: int = 100) -> np.ndarray:
        """
        The boundary of a point is the point itself, a single row.

        Parameters
        ----------
        n
            Unused; a point is not sampled.

        Returns
        -------
        An array of shape ``(1, 2)`` containing the coordinates of the point.
        """
        return np.array([[self.x, self.y]])

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (self.x, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            x=children[0],
            y=children[1],
        )


def centroid(triangles: np.ndarray):
    y1, x1 = triangles[:, 0, 1], triangles[:, 0, 0]
    y2, x2 = triangles[:, 1, 1], triangles[:, 1, 0]
    y3, x3 = triangles[:, 2, 1], triangles[:, 2, 0]

    return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3


class Circle(Point):
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
    ):
        """
        A circle in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        x
        y
            The coordinates of the center of the circle.
        radius
            The radius of the circle.
        """
        super().__init__(x, y)
        self.radius = radius

    @property
    def area(self) -> float:
        """
        The area of the circle.
        """
        return np.pi * self.radius**2

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles intersect the circle.

        This is approximated by checking if the centroid of the triangle is within
        the circle or if the triangle contains the centroid of the circle.

        Parameters
        ----------
        triangles
            The vertices of the triangles.

        Returns
        -------
        A boolean array indicating which triangles intersect the circle.
        """
        centroid_x, centroid_y = centroid(triangles)

        a = centroid_x - self.x
        b = centroid_y - self.y

        distance_squared = a * a + b * b

        radius_2 = self.radius * self.radius

        return (distance_squared <= radius_2) | super().mask(triangles)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Determine which points lie inside the circle.

        Parameters
        ----------
        points
            An array of coordinates of shape ``(N, 2)``, ordered the same way
            as the triangle vertices (see the `Shape` docstring).

        Returns
        -------
        A boolean array of shape ``(N,)``; points exactly on the circle count
        as inside.
        """
        points = np.asarray(points)

        delta_0 = points[:, 0] - self.x
        delta_1 = points[:, 1] - self.y

        return delta_0 * delta_0 + delta_1 * delta_1 <= self.radius * self.radius

    def boundary(self, n: int = 100) -> np.ndarray:
        """
        The circle sampled as a closed polygon of ``n`` points.

        Parameters
        ----------
        n
            The number of samples around the circle.

        Returns
        -------
        An array of shape ``(n + 1, 2)`` whose last row repeats its first.
        """
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

        boundary = np.stack(
            (
                self.x + self.radius * np.cos(angles),
                self.y + self.radius * np.sin(angles),
            ),
            axis=-1,
        )

        return np.append(boundary, boundary[:1], axis=0)

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (self.x, self.y, self.radius), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            x=children[0],
            y=children[1],
            radius=children[2],
        )


class Triangle(Point):
    def __init__(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
    ):
        """
        A triangle in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        a, b, c
            The vertices of the triangle.
        """
        xs, ys = zip(a, b, c)
        super().__init__(
            x=np.mean(xs),
            y=np.mean(ys),
        )
        self.a = a
        self.b = b
        self.c = c

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (
            self.a,
            self.b,
            self.c,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            *children,
        )

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        return self.triangle_contains_mask(triangles) | super().mask(triangles)

    def triangle_contains_mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles have their centroid inside this triangle.

        The vertices of this triangle and the centroids being tested are
        indexed with the same axis order (element 0 against element 0). This
        previously unpacked the vertices as ``y, x`` while testing a centroid
        built as ``x, y``, which tested the reflected triangle and so gave the
        wrong answer for any triangle that is not symmetric about
        ``element 0 == element 1``.

        Parameters
        ----------
        triangles
            The vertices of the triangles.

        Returns
        -------
        A boolean array indicating which triangle centroids lie inside this
        triangle.
        """
        centroid_0, centroid_1 = centroid(triangles)

        return _barycentric_contains(
            self.a[0],
            self.a[1],
            self.b[0],
            self.b[1],
            self.c[0],
            self.c[1],
            centroid_0,
            centroid_1,
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Determine which points lie inside the triangle.

        Parameters
        ----------
        points
            An array of coordinates of shape ``(N, 2)``, ordered the same way
            as the vertices (see the `Shape` docstring).

        Returns
        -------
        A boolean array of shape ``(N,)``; points on an edge count as inside.
        """
        points = np.asarray(points)

        return _barycentric_contains(
            self.a[0],
            self.a[1],
            self.b[0],
            self.b[1],
            self.c[0],
            self.c[1],
            points[:, 0],
            points[:, 1],
        )

    def boundary(self, n: int = 100) -> np.ndarray:
        """
        The three vertices of the triangle, closed by repeating the first.

        Parameters
        ----------
        n
            Unused; a triangle's boundary is exactly polygonal.

        Returns
        -------
        An array of shape ``(4, 2)``.
        """
        return np.array([self.a, self.b, self.c, self.a])

    @property
    def area(self) -> float:
        """
        The area of the triangle.
        """
        return 0.5 * abs(
            self.a[0] * (self.b[1] - self.c[1])
            + self.b[0] * (self.c[1] - self.a[1])
            + self.c[0] * (self.a[1] - self.b[1])
        )


class Polygon(Point):
    def __init__(
        self,
        vertices: List[Tuple[float, float]],
    ):
        """
        A polygon in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        vertices
            The vertices of the polygon.
        """
        self.vertices = vertices

        if len(vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")

        x = np.mean([vertex[0] for vertex in vertices])
        y = np.mean([vertex[1] for vertex in vertices])
        super().__init__(x, y)

        first = vertices[0]

        self.triangles = [
            Triangle(
                first,
                second,
                third,
            )
            for second, third in zip(vertices[1:], vertices[2:])
        ]

    @property
    def area(self) -> float:
        """
        The area of the polygon.
        """
        return sum(triangle.area for triangle in self.triangles)

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (self.vertices,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            vertices=children[0],
        )

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles intersect the cell.

        Parameters
        ----------
        triangles
            The vertices of the triangles

        Returns
        -------
        A boolean array indicating which triangles intersect the cell.
        """
        return np.any(
            [triangle.mask(triangles) for triangle in self.triangles],
            axis=0,
        ) | super().mask(triangles)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Determine which points lie inside the polygon.

        The polygon is decomposed into a fan of triangles about its first
        vertex, so a point is inside when it is inside any of them. This is
        exact for **convex** polygons only; for a concave polygon the fan
        covers its convex hull.

        Parameters
        ----------
        points
            An array of coordinates of shape ``(N, 2)``, ordered the same way
            as the vertices (see the `Shape` docstring).

        Returns
        -------
        A boolean array of shape ``(N,)``; points on an edge count as inside.
        """
        points = np.asarray(points)

        return np.any(
            [triangle.contains(points) for triangle in self.triangles],
            axis=0,
        )

    def boundary(self, n: int = 100) -> np.ndarray:
        """
        The vertices of the polygon, closed by repeating the first.

        Parameters
        ----------
        n
            Unused; a polygon's boundary is exactly polygonal.

        Returns
        -------
        An array of shape ``(len(vertices) + 1, 2)``.
        """
        return np.array(list(self.vertices) + [self.vertices[0]])


class Square(Point):
    def __init__(self, top, bottom, left, right):
        """
        A square in the source plane for which we want to identify pixels in the
        image plane that trace to it.

        Parameters
        ----------
        top
        bottom
        left
        right
            The coordinates of the top, bottom, left, and right edges of the square.
            Coordinates are from the top-left corner of the image.
        """
        x = (left + right) / 2
        y = (top + bottom) / 2
        super().__init__(x, y)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    @property
    def area(self) -> float:
        """
        The area of the square.
        """
        return (self.right - self.left) * (self.bottom - self.top)

    def mask(self, triangles: np.ndarray) -> np.ndarray:
        """
        Determine which triangles intersect the square.

        This is approximated by checking if the centroid of the triangle is within
        the square or if the triangle contains the centroid of the square.

        Parameters
        ----------
        triangles
            The vertices of the triangles.

        Returns
        -------
        A boolean array indicating which triangles intersect the square.
        """
        centroid_x, centroid_y = centroid(triangles)

        return (
            (self.left <= centroid_x)
            & (centroid_x <= self.right)
            & (self.bottom >= centroid_y)
            & (centroid_y >= self.top)
        ) | super().mask(triangles)

    @property
    def _bounds(self):
        """
        The bounds of the square as ``(low_0, high_0, low_1, high_1)``.

        `mask` and `area` assume ``left < right`` and ``top < bottom``
        numerically (coordinates measured from the top left corner of the
        image, so `top` is the smaller number). `contains` and `boundary` do
        not: they sort the pair, so they behave the same way when a square is
        built from arcsec ``(y, x)`` coordinates, where the top of the image is
        the *larger* first coordinate.
        """
        return (
            min(self.left, self.right),
            max(self.left, self.right),
            min(self.top, self.bottom),
            max(self.top, self.bottom),
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Determine which points lie inside the square.

        Parameters
        ----------
        points
            An array of coordinates of shape ``(N, 2)``, ordered the same way
            as the triangle vertices (see the `Shape` docstring), so element 0
            is bounded by `left` / `right` and element 1 by `top` / `bottom` —
            the same pairing `mask` uses.

        Returns
        -------
        A boolean array of shape ``(N,)``; points on an edge count as inside.
        """
        points = np.asarray(points)

        low_0, high_0, low_1, high_1 = self._bounds

        return (
            (low_0 <= points[:, 0])
            & (points[:, 0] <= high_0)
            & (low_1 <= points[:, 1])
            & (points[:, 1] <= high_1)
        )

    def boundary(self, n: int = 100) -> np.ndarray:
        """
        The four corners of the square, closed by repeating the first.

        Parameters
        ----------
        n
            Unused; a square's boundary is exactly polygonal.

        Returns
        -------
        An array of shape ``(5, 2)``.
        """
        low_0, high_0, low_1, high_1 = self._bounds

        return np.array(
            [
                [low_0, low_1],
                [low_0, high_1],
                [high_0, high_1],
                [high_0, low_1],
                [low_0, low_1],
            ]
        )
