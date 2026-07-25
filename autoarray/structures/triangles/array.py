import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR

from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.structures.triangles.shape import Shape

MAX_CONTAINING_SIZE = 15


class ArrayTriangles(AbstractTriangles):
    def __init__(
        self,
        indices,
        vertices,
        max_containing_size=MAX_CONTAINING_SIZE,
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
        self.max_containing_size = max_containing_size

    def __len__(self):
        return len(self.triangles)

    def __iter__(self):
        return iter(self.triangles)

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.indices)} triangles"

    def __repr__(self):
        return str(self)

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float,
        max_containing_size=MAX_CONTAINING_SIZE,
    ) -> "AbstractTriangles":

        import jax.numpy as jnp

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

        return cls(
            indices=jnp.array(indices),
            vertices=jnp.array(vertices),
            max_containing_size=max_containing_size,
        )

    @property
    def triangles(self) -> np.ndarray:
        """
        The triangles as a 3x2 array of vertices.
        """

        import jax.numpy as jnp

        invalid_mask = jnp.any(self.indices == -1, axis=1)
        nan_array = jnp.full(
            (self.indices.shape[0], 3, 2),
            jnp.nan,
            dtype=jnp.float32,
        )
        safe_indices = jnp.where(self.indices == -1, 0, self.indices)
        triangle_vertices = self.vertices[safe_indices]
        return jnp.where(invalid_mask[:, None, None], nan_array, triangle_vertices)

    @property
    def means(self) -> np.ndarray:
        """
        The mean of each triangle.
        """
        import jax.numpy as jnp

        return jnp.mean(self.triangles, axis=1)

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
        import jax.numpy as jnp

        inside = shape.mask(self.triangles)

        return jnp.where(
            inside,
            size=self.max_containing_size,
            fill_value=-1,
        )[0]

    def for_indexes(self, indexes: np.ndarray) -> "ArrayTriangles":
        """
        Create a new ArrayTriangles containing indices and vertices corresponding to the given indexes
        but without duplicate vertices.

        Parameters
        ----------
        indexes
            The indexes of the triangles to include in the new ArrayTriangles.

        Returns
        -------
        The new ArrayTriangles instance.
        """
        import jax.numpy as jnp

        selected_indices = select_and_handle_invalid(
            data=self.indices,
            indices=indexes,
            invalid_value=-1,
            invalid_replacement=jnp.array([-1, -1, -1], dtype=jnp.int32),
        )

        flat_indices = selected_indices.flatten()

        selected_vertices = select_and_handle_invalid(
            data=self.vertices,
            indices=flat_indices,
            invalid_value=-1,
            invalid_replacement=jnp.array([jnp.nan, jnp.nan], dtype=jnp.float32),
        )

        unique_vertices, inv_indices = jnp.unique(
            selected_vertices,
            axis=0,
            return_inverse=True,
            equal_nan=True,
            size=selected_indices.shape[0] * 3,
            fill_value=jnp.nan,
        )

        nan_mask = jnp.isnan(unique_vertices).any(axis=1)
        inv_indices = jnp.where(nan_mask[inv_indices], -1, inv_indices)

        new_indices = inv_indices.reshape(selected_indices.shape)

        new_indices_sorted = jnp.sort(new_indices, axis=1)

        unique_triangles_indices = jnp.unique(
            new_indices_sorted,
            axis=0,
            size=new_indices_sorted.shape[0],
            fill_value=-1,
        )

        return ArrayTriangles(
            indices=unique_triangles_indices,
            vertices=unique_vertices,
            max_containing_size=self.max_containing_size,
        )

    def _up_sample_triangle(self):
        import jax.numpy as jnp

        triangles = self.triangles

        m01 = (triangles[:, 0] + triangles[:, 1]) / 2
        m12 = (triangles[:, 1] + triangles[:, 2]) / 2
        m20 = (triangles[:, 2] + triangles[:, 0]) / 2

        return jnp.concatenate(
            [
                jnp.stack([triangles[:, 1], m12, m01], axis=1),
                jnp.stack([triangles[:, 2], m20, m12], axis=1),
                jnp.stack([m01, m12, m20], axis=1),
                jnp.stack([triangles[:, 0], m01, m20], axis=1),
            ],
            axis=0,
        )

    def up_sample(self) -> "ArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.

        This means each triangle becomes four smaller triangles.
        """
        new_indices, unique_vertices = remove_duplicates(self._up_sample_triangle())

        return ArrayTriangles(
            indices=new_indices,
            vertices=unique_vertices,
            max_containing_size=self.max_containing_size,
        )

    def _neighborhood_triangles(self):
        import jax.numpy as jnp

        triangles = self.triangles

        new_v0 = triangles[:, 1] + triangles[:, 2] - triangles[:, 0]
        new_v1 = triangles[:, 0] + triangles[:, 2] - triangles[:, 1]
        new_v2 = triangles[:, 0] + triangles[:, 1] - triangles[:, 2]

        return jnp.concatenate(
            [
                jnp.stack([new_v0, triangles[:, 1], triangles[:, 2]], axis=1),
                jnp.stack([triangles[:, 0], new_v1, triangles[:, 2]], axis=1),
                jnp.stack([triangles[:, 0], triangles[:, 1], new_v2], axis=1),
                triangles,
            ],
            axis=0,
        )

    def neighborhood(self) -> "ArrayTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Includes the current triangles and the triangles that share an edge with the current triangles.
        """
        new_indices, unique_vertices = remove_duplicates(self._neighborhood_triangles())

        return ArrayTriangles(
            indices=new_indices,
            vertices=unique_vertices,
            max_containing_size=self.max_containing_size,
        )

    def with_vertices(self, vertices: np.ndarray) -> "ArrayTriangles":
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
        return ArrayTriangles(
            indices=self.indices,
            vertices=vertices,
            max_containing_size=self.max_containing_size,
        )

    def tree_flatten(self):
        """
        Flatten this model as a PyTree.
        """
        return (
            self.indices,
            self.vertices,
        ), (self.max_containing_size,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a PyTree into a model.
        """
        return cls(
            indices=children[0],
            vertices=children[1],
            max_containing_size=aux_data[0],
        )


def select_and_handle_invalid(
    data: np.ndarray,
    indices: np.ndarray,
    invalid_value,
    invalid_replacement,
):
    """
    Select data based on indices, handling invalid indices by replacing them with a specified value.

    Parameters
    ----------
    data
        The array from which to select data.
    indices
        The indices used to select data from the array.
    invalid_value
        The value representing invalid indices.
    invalid_replacement
        The value to use for invalid entries in the result.

    Returns
    -------
    An array with selected data, where invalid indices are replaced with `invalid_replacement`.
    """
    import jax.numpy as jnp

    invalid_mask = indices == invalid_value
    safe_indices = jnp.where(invalid_mask, 0, indices)
    selected_data = data[safe_indices]
    selected_data = jnp.where(
        invalid_mask[..., None],
        invalid_replacement,
        selected_data,
    )

    return selected_data


def remove_duplicates(new_triangles):
    import jax.numpy as jnp

    unique_vertices, inverse_indices = jnp.unique(
        new_triangles.reshape(-1, 2),
        axis=0,
        return_inverse=True,
        size=2 * new_triangles.shape[0],
        fill_value=jnp.nan,
        equal_nan=True,
    )

    inverse_indices_flat = inverse_indices.reshape(-1)
    selected_vertices = unique_vertices[inverse_indices_flat]
    mask = jnp.any(jnp.isnan(selected_vertices), axis=1)
    inverse_indices_flat = jnp.where(mask, -1, inverse_indices_flat)
    inverse_indices = inverse_indices_flat.reshape(inverse_indices.shape)

    new_indices = inverse_indices.reshape(-1, 3)

    new_indices_sorted = jnp.sort(new_indices, axis=1)

    unique_triangles_indices = jnp.unique(
        new_indices_sorted,
        axis=0,
        size=new_indices_sorted.shape[0],
        fill_value=jnp.array(
            [-1, -1, -1],
            dtype=jnp.int32,
        ),
    )

    return unique_triangles_indices, unique_vertices
