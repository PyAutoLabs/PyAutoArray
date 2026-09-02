from abc import ABC

import numpy as np

from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.structures.triangles.shape import Shape
from autoarray.structures.triangles.abstract import HEIGHT_FACTOR


class ArrayTrianglesNp(AbstractTriangles, ABC):

    def __init__(
        self,
        indices,
        vertices,
        **kwargs,
    ):
        """
        Represents a set of triangles in efficient NumPy arrays.

        Parameters
        ----------
        indices
            The indices of the vertices of the triangles. This is a 2D array where each row is a triangle
            with the three indices of the vertices.
        vertices
            The vertices of the triangles.
        """

        self._indices = indices
        self._vertices = vertices

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
        **kwargs,
    ) -> "AbstractTriangles":
        height = scale * HEIGHT_FACTOR

        vertices = []
        indices = []
        vertex_dict = {}

        def add_vertex(v):
            if v not in vertex_dict:
                vertex_dict[v] = len(vertices)
                vertices.append(v)
            return vertex_dict[v]

        rows = []
        for row_y in np.arange(y_min, y_max + height, height):
            row = []
            offset = (len(rows) % 2) * scale / 2
            for col_x in np.arange(x_min - offset, x_max + scale, scale):
                row.append((row_y, col_x))
            rows.append(row)

        for i in range(len(rows) - 1):
            row = rows[i]
            next_row = rows[i + 1]
            for j in range(len(row)):
                if i % 2 == 0 and j < len(next_row) - 1:
                    t1 = [
                        add_vertex(row[j]),
                        add_vertex(next_row[j]),
                        add_vertex(next_row[j + 1]),
                    ]
                    if j < len(row) - 1:
                        t2 = [
                            add_vertex(row[j]),
                            add_vertex(row[j + 1]),
                            add_vertex(next_row[j + 1]),
                        ]
                        indices.append(t2)
                elif i % 2 == 1 and j < len(next_row) - 1:
                    t1 = [
                        add_vertex(row[j]),
                        add_vertex(next_row[j]),
                        add_vertex(row[j + 1]),
                    ]
                    indices.append(t1)
                    if j < len(next_row) - 1:
                        t2 = [
                            add_vertex(next_row[j]),
                            add_vertex(next_row[j + 1]),
                            add_vertex(row[j + 1]),
                        ]
                        indices.append(t2)
                else:
                    continue
                indices.append(t1)

        vertices = np.array(vertices)
        indices = np.array(indices)

        return ArrayTrianglesNp(
            indices=indices,
            vertices=vertices,
            **kwargs,
        )

    @property
    def triangles(self):
        return self.vertices[self.indices]

    @property
    def means(self):
        return np.mean(self.triangles, axis=1)

    def containing_indices(self, shape: Shape) -> np.ndarray:
        """
        Find the triangles that insect with a given shape.

        Parameters
        ----------
        shape
            The shape

        Returns
        -------
        The triangles that intersect the shape.
        """
        inside = shape.mask(self.triangles)

        return np.where(inside)[0]

    def for_indexes(self, indexes: np.ndarray) -> "ArrayTrianglesNp":
        """
        Create a new ArrayTrianglesNp containing indices and vertices corresponding to the given indexes
        but without duplicate vertices.

        Parameters
        ----------
        indexes
            The indexes of the triangles to include in the new ArrayTrianglesNp.

        Returns
        -------
        The new ArrayTrianglesNp instance.
        """
        selected_indices = self.indices[indexes]

        flat_indices = selected_indices.flatten()
        unique_vertices, inverse_indices = np.unique(
            self.vertices[flat_indices], axis=0, return_inverse=True
        )

        new_indices = inverse_indices.reshape(selected_indices.shape)

        return ArrayTrianglesNp(indices=new_indices, vertices=unique_vertices)

    def up_sample(self) -> "ArrayTrianglesNp":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.

        This means each triangle becomes four smaller triangles.
        """
        unique_vertices, inverse_indices = np.unique(
            self._up_sample_triangle().reshape(-1, 2), axis=0, return_inverse=True
        )
        new_indices = inverse_indices.reshape(-1, 3)

        return ArrayTrianglesNp(
            indices=new_indices,
            vertices=unique_vertices,
        )

    def neighborhood(self) -> "ArrayTrianglesNp":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Includes the current triangles and the triangles that share an edge with the current triangles.
        """
        unique_vertices, inverse_indices = np.unique(
            self._neighborhood_triangles().reshape(-1, 2),
            axis=0,
            return_inverse=True,
        )
        new_indices = inverse_indices.reshape(-1, 3)

        new_indices_sorted = np.sort(new_indices, axis=1)

        unique_triangles_indices, unique_index_positions = np.unique(
            new_indices_sorted, axis=0, return_index=True
        )

        return ArrayTrianglesNp(
            indices=unique_triangles_indices,
            vertices=unique_vertices,
        )

    def with_vertices(self, vertices: np.ndarray) -> "ArrayTrianglesNp":
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

    def _up_sample_triangle(self):
        triangles = self.triangles

        m01 = (triangles[:, 0] + triangles[:, 1]) / 2
        m12 = (triangles[:, 1] + triangles[:, 2]) / 2
        m20 = (triangles[:, 2] + triangles[:, 0]) / 2

        return np.concatenate(
            [
                np.stack([triangles[:, 1], m12, m01], axis=1),
                np.stack([triangles[:, 2], m20, m12], axis=1),
                np.stack([m01, m12, m20], axis=1),
                np.stack([triangles[:, 0], m01, m20], axis=1),
            ],
            axis=0,
        )

    def _neighborhood_triangles(self):
        triangles = self.triangles

        new_v0 = triangles[:, 1] + triangles[:, 2] - triangles[:, 0]
        new_v1 = triangles[:, 0] + triangles[:, 2] - triangles[:, 1]
        new_v2 = triangles[:, 0] + triangles[:, 1] - triangles[:, 2]

        return np.concatenate(
            [
                np.stack([new_v0, triangles[:, 1], triangles[:, 2]], axis=1),
                np.stack([triangles[:, 0], new_v1, triangles[:, 2]], axis=1),
                np.stack([triangles[:, 0], triangles[:, 1], new_v2], axis=1),
                triangles,
            ],
            axis=0,
        )

    def __iter__(self):
        return iter(self.triangles)
