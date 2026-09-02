import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.structures.triangles.array_np import ArrayTrianglesNp
from autoarray.structures.triangles.shape import Shape
from autonerves import cached_property


class CoordinateArrayTrianglesNp(AbstractTriangles):

    def __init__(
        self,
        coordinates: np.ndarray,
        side_length: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        flipped: bool = False,
    ):
        """
        Represents a set of triangles by integer coordinates.

        Parameters
        ----------
        coordinates
            Integer x y coordinates for each triangle.
        side_length
            The side length of the triangles.
        flipped
            Whether the triangles are flipped upside down.
        y_offset
            An y_offset to apply to the y coordinates so that up-sampled triangles align.
        """
        self.coordinates = coordinates
        self.side_length = side_length
        self.flipped = flipped

        self.scaling_factors = np.array(
            [0.5 * side_length, HEIGHT_FACTOR * side_length]
        )
        self.x_offset = x_offset
        self.y_offset = y_offset

    @property
    def vertices(self) -> np.ndarray:
        """
        The unique vertices of the triangles.
        """
        return self._vertices_and_indices[0]

    @property
    def indices(self) -> np.ndarray:
        """
        The indices of the vertices of the triangles.
        """
        return self._vertices_and_indices[1]

    @property
    def centres(self) -> np.ndarray:
        """
        The centres of the triangles.
        """
        return self.scaling_factors * self.coordinates + np.array(
            [self.x_offset, self.y_offset]
        )

    @cached_property
    def flip_mask(self) -> np.ndarray:
        """
        A mask for the triangles that are flipped.

        Every other triangle is flipped so that they tessellate.
        """
        mask = (self.coordinates[:, 0] + self.coordinates[:, 1]) % 2 != 0
        if self.flipped:
            mask = ~mask
        return mask

    @cached_property
    def flip_array(self) -> np.ndarray:
        """
        An array of 1s and -1s to flip the triangles.
        """
        array = np.ones(
            self.coordinates.shape[0],
            dtype=np.int32,
        )
        array[self.flip_mask] = -1

        return array[:, np.newaxis]

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float = 1.0,
        **_,
    ):
        """
        Tile the rectangle ``[y_min, y_max] x [x_min, x_max]`` with equilateral triangles.

        Element ``0`` of every vertex spans the ``y`` limits and element ``1`` the ``x``
        limits, matching `ArrayTrianglesNp.for_limits_and_scale`, the ``(y, x)`` order of
        every PyAuto grid, and the ``element 0 <-> element 0`` convention `Shape.contains`
        and `Shape.mask` are documented with.

        The two axes were previously the other way round while the signature named its
        first pair ``x_min``/``x_max``, so both the keyword callers (`AbstractSolver`) and
        the positional caller (`AbstractTriangles.for_grid`) tiled the *transposed*
        rectangle. On a square grid that is invisible; on a rectangular one the solver
        searched a box the data does not occupy and silently missed multiple images which
        lay inside the grid (a 24x80 grid of 0.05" pixels found one of an Isothermal's two
        images instead of both).

        Parameters
        ----------
        y_min, y_max, x_min, x_max
            The limits of the rectangle to tile.
        scale
            The side length of the triangles.
        """
        y_shift = int(2 * y_min / scale)
        x_shift = int(x_min / (HEIGHT_FACTOR * scale))

        coordinates = []

        for y in range(y_shift, int(2 * y_max / scale) + 1):
            for x in range(x_shift - 1, int(x_max / (HEIGHT_FACTOR * scale)) + 2):
                coordinates.append([y, x])

        return CoordinateArrayTrianglesNp(
            coordinates=np.array(coordinates, dtype=np.int32),
            side_length=scale,
        )

    @cached_property
    def triangles(self) -> np.ndarray:
        """
        The vertices of the triangles as an Nx3x2 array.
        """
        centres = self.centres
        return np.stack(
            (
                centres
                + self.flip_array
                * np.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + self.flip_array
                * np.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + self.flip_array
                * np.array(
                    [-0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
            ),
            axis=1,
        )

    def up_sample(self) -> "CoordinateArrayTrianglesNp":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.
        """
        new_coordinates = np.zeros(
            (4 * self.coordinates.shape[0], 2),
            dtype=np.int32,
        )
        n_normal = 4 * np.sum(~self.flip_mask)

        new_coordinates[:n_normal] = np.vstack(
            (
                2 * self.coordinates[~self.flip_mask],
                2 * self.coordinates[~self.flip_mask] + np.array([1, 0]),
                2 * self.coordinates[~self.flip_mask] + np.array([-1, 0]),
                2 * self.coordinates[~self.flip_mask] + np.array([0, 1]),
            )
        )
        new_coordinates[n_normal:] = np.vstack(
            (
                2 * self.coordinates[self.flip_mask],
                2 * self.coordinates[self.flip_mask] + np.array([1, 1]),
                2 * self.coordinates[self.flip_mask] + np.array([-1, 1]),
                2 * self.coordinates[self.flip_mask] + np.array([0, 1]),
            )
        )

        return CoordinateArrayTrianglesNp(
            coordinates=new_coordinates,
            side_length=self.side_length / 2,
            y_offset=self.y_offset + -0.25 * HEIGHT_FACTOR * self.side_length,
            x_offset=self.x_offset,
            flipped=True,
        )

    def neighborhood(self) -> "CoordinateArrayTrianglesNp":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Ensures that the new triangles are unique.
        """
        new_coordinates = np.zeros(
            (4 * self.coordinates.shape[0], 2),
            dtype=np.int32,
        )
        n_normal = 4 * np.sum(~self.flip_mask)

        new_coordinates[:n_normal] = np.vstack(
            (
                self.coordinates[~self.flip_mask],
                self.coordinates[~self.flip_mask] + np.array([1, 0]),
                self.coordinates[~self.flip_mask] + np.array([-1, 0]),
                self.coordinates[~self.flip_mask] + np.array([0, -1]),
            )
        )
        new_coordinates[n_normal:] = np.vstack(
            (
                self.coordinates[self.flip_mask],
                self.coordinates[self.flip_mask] + np.array([1, 0]),
                self.coordinates[self.flip_mask] + np.array([-1, 0]),
                self.coordinates[self.flip_mask] + np.array([0, 1]),
            )
        )
        return CoordinateArrayTrianglesNp(
            coordinates=np.unique(new_coordinates, axis=0),
            side_length=self.side_length,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
            flipped=self.flipped,
        )

    @cached_property
    def _vertices_and_indices(self):
        flat_triangles = self.triangles.reshape(-1, 2)
        vertices, inverse_indices = np.unique(
            flat_triangles,
            axis=0,
            return_inverse=True,
        )
        indices = inverse_indices.reshape(-1, 3)
        return vertices, indices

    def with_vertices(self, vertices: np.ndarray) -> ArrayTrianglesNp:
        """
        Create a new set of triangles with the vertices replaced.

        Parameters
        ----------
        vertices
            The new vertices to use.

        Returns
        -------
        The new set of triangles with the new vertices.
        """
        return ArrayTrianglesNp(
            indices=self.indices,
            vertices=vertices,
        )

    def for_indexes(self, indexes: np.ndarray) -> "CoordinateArrayTrianglesNp":
        """
        Create a new CoordinateArrayTrianglesNp containing triangles corresponding to the given indexes

        Parameters
        ----------
        indexes
            The indexes of the triangles to include in the new CoordinateArrayTrianglesNp.

        Returns
        -------
        The new CoordinateArrayTrianglesNp instance.
        """
        return CoordinateArrayTrianglesNp(
            coordinates=self.coordinates[indexes],
            side_length=self.side_length,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
            flipped=self.flipped,
        )

    def containing_indices(self, shape: Shape) -> np.ndarray:
        """
        Find the triangles that insect with a given shape.

        Parameters
        ----------
        shape
            The shape

        Returns
        -------
        The indices of triangles that intersect the shape.
        """
        return self.with_vertices(self.vertices).containing_indices(shape)

    @property
    def means(self):
        return np.mean(self.triangles, axis=1)
