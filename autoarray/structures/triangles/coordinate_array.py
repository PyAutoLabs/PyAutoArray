from abc import ABC
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.abstract import AbstractTriangles
from autoarray.structures.triangles.array import ArrayTriangles


def _lattice_coordinates(
    y_min: float, y_max: float, x_min: float, x_max: float, scale: float
) -> np.ndarray:
    """
    The integer ``(y, x)`` lattice coordinates `CoordinateArrayTriangles.for_limits_and_scale`
    tiles the rectangle with, as an ``(N, 2)`` int array.
    """
    y_shift = int(2 * y_min / scale)
    x_shift = int(x_min / (HEIGHT_FACTOR * scale))

    coordinates = []

    for y in range(y_shift, int(2 * y_max / scale) + 1):
        for x in range(x_shift - 1, int(x_max / (HEIGHT_FACTOR * scale)) + 2):
            coordinates.append([y, x])

    return np.array(coordinates)


@lru_cache(maxsize=32)
def static_vertex_table(
    y_min: float, y_max: float, x_min: float, x_max: float, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The geometrically unique vertices of the initial triangle lattice
    `CoordinateArrayTriangles.for_limits_and_scale` builds, plus the index map from each
    triangle's three vertices into them.

    The lattice is fixed by its limits and scale, so it is known before any tracing: on the JAX
    `PointSolver` path its vertices are a compile-time constant, and only the geometrically
    distinct ones need deflecting. Of the ``3N`` vertex slots most are shared by up to six
    triangles (for the ``+-9.9"`` / ``0.2"`` lattice: 69 849 slots, 11 859 distinct points). They
    cannot be deduplicated on the floats -- the same point computed from two neighbouring
    triangle centres can differ by one ulp (28 665 exact-float distinct rows) -- so they are keyed
    on the integer lattice position instead. Vertex ``k`` of the triangle at integer coordinates
    ``(cy, cx)`` with flip ``f = +-1`` sits at ``(0.5 * s * (cy + f * dy), 0.5 * h * s * (2 * cx +
    f * dx))`` with ``(dy, dx)`` in ``((0, 1), (1, -1), (-1, -1))``, ``s`` the side length and ``h``
    the height factor; ``(cy + f * dy, 2 * cx + f * dx)`` is therefore an exact integer key.

    Each unique vertex takes the float value of its first occurrence, computed with the same
    arithmetic as `CoordinateArrayTriangles.triangles`, so ``vertices[indices]`` equals
    ``triangles`` to within an ulp per element and is bit-identical for every first occurrence.

    Built in NumPy (never staged into a JAX trace) and cached per geometry; the arrays are
    read-only because the cache hands the same objects to every caller.

    Parameters
    ----------
    y_min, y_max, x_min, x_max
        The limits of the rectangle the lattice tiles.
    scale
        The side length of the triangles.

    Returns
    -------
    ``(vertices, indices)``: the ``(V, 2)`` float64 unique vertices and the ``(N, 3)`` int index map
    such that ``vertices[indices]`` is the ``(N, 3, 2)`` triangle array.
    """
    coordinates = _lattice_coordinates(y_min, y_max, x_min, x_max, scale)

    flip = np.where((coordinates[:, 0] + coordinates[:, 1]) % 2 != 0, -1, 1)[:, None]

    # Same operation order as `CoordinateArrayTriangles.centres` / `.triangles` (zero offsets).
    scaling_factors = np.array([0.5 * scale, HEIGHT_FACTOR * scale])
    centres = scaling_factors * coordinates + np.array([0.0, 0.0])
    triangles = np.stack(
        (
            centres + flip * np.array([0.0, 0.5 * scale * HEIGHT_FACTOR]),
            centres + flip * np.array([0.5 * scale, -0.5 * scale * HEIGHT_FACTOR]),
            centres + flip * np.array([-0.5 * scale, -0.5 * scale * HEIGHT_FACTOR]),
        ),
        axis=1,
    )

    offsets = np.array([[0, 1], [1, -1], [-1, -1]])
    keys = np.stack(
        (
            coordinates[:, None, 0] + flip * offsets[None, :, 0],
            2 * coordinates[:, None, 1] + flip * offsets[None, :, 1],
        ),
        axis=-1,
    ).reshape(-1, 2)

    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    vertices = np.ascontiguousarray(triangles.reshape(-1, 2)[first], dtype=np.float64)
    indices = np.ascontiguousarray(inverse.reshape(-1, 3))

    vertices.setflags(write=False)
    indices.setflags(write=False)

    return vertices, indices


class CoordinateArrayTriangles(AbstractTriangles, ABC):

    def __init__(
        self,
        coordinates: np.ndarray,
        side_length: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        flipped: bool = False,
        vertex_table: Optional[Tuple[np.ndarray, np.ndarray]] = None,
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
        vertex_table
            An optional precomputed ``(vertices, indices)`` pair (see `static_vertex_table`) that
            `vertices` and `indices` return instead of the flat per-triangle table. It must describe
            exactly these ``coordinates``; only `for_limits_and_scale(..., static_vertices=True)`
            sets it. Derived lattices (`for_indexes`, `up_sample`, `neighborhood`) do not inherit
            it, and it is not part of the pytree (`tree_flatten`), so an unflattened copy falls
            back to the flat table -- which is still correct, only unshared.
        """
        import jax.numpy as jnp

        self.coordinates = coordinates
        self.vertex_table = vertex_table
        self.side_length = side_length
        self.flipped = flipped

        self.scaling_factors = jnp.array(
            [0.5 * side_length, HEIGHT_FACTOR * side_length]
        )
        self.x_offset = x_offset
        self.y_offset = y_offset

    @classmethod
    def for_limits_and_scale(
        cls,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        scale: float = 1.0,
        static_vertices: bool = False,
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
        static_vertices
            If ``True``, attach the cached `static_vertex_table` for this geometry, so `vertices`
            is the ``(V, 2)`` table of geometrically unique lattice vertices (11 859 rows rather
            than 69 849 for the ``+-9.9"`` / ``0.2"`` lattice) and `indices` maps each triangle
            into it. Consumers that deflect `vertices` then evaluate each lattice point once. The
            limits and scale must be concrete Python / NumPy numbers (they are the cache key).
        """
        import jax.numpy as jnp

        vertex_table = None
        if static_vertices:
            vertex_table = static_vertex_table(
                float(y_min), float(y_max), float(x_min), float(x_max), float(scale)
            )

        return cls(
            coordinates=jnp.array(
                _lattice_coordinates(y_min, y_max, x_min, x_max, scale)
            ),
            side_length=scale,
            vertex_table=vertex_table,
        )

    def tree_flatten(self):
        return (
            self.coordinates,
            self.side_length,
            self.x_offset,
            self.y_offset,
        ), (self.flipped,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Create a prior from a flattened PyTree

        Parameters
        ----------
        aux_data
            Auxiliary information that remains unchanged including
            the keys of the dict
        children
            Child objects subject to change

        Returns
        -------
        An instance of this class
        """
        return cls(*children, flipped=aux_data[0])

    def __len__(self):
        import jax.numpy as jnp

        return jnp.count_nonzero(~jnp.isnan(self.coordinates).any(axis=1))

    def __iter__(self):
        return iter(self.triangles)

    @property
    def centres(self) -> np.ndarray:
        """
        The centres of the triangles.
        """
        import jax.numpy as jnp

        centres = self.scaling_factors * self.coordinates + jnp.array(
            [self.x_offset, self.y_offset]
        )
        return centres

    @property
    def vertex_coordinates(self) -> np.ndarray:
        """
        The vertices of the triangles as an Nx3x2 array.
        """
        import jax.numpy as jnp

        coordinates = self.coordinates
        return jnp.concatenate(
            [
                coordinates + self.flip_array * np.array([0, 1], dtype=np.int32),
                coordinates + self.flip_array * np.array([1, -1], dtype=np.int32),
                coordinates + self.flip_array * np.array([-1, -1], dtype=np.int32),
            ],
            dtype=np.int32,
        )

    @property
    def triangles(self) -> np.ndarray:
        """
        The vertices of the triangles as an Nx3x2 array.
        """
        import jax.numpy as jnp

        centres = self.centres
        return jnp.stack(
            (
                centres
                + self.flip_array
                * jnp.array(
                    [0.0, 0.5 * self.side_length * HEIGHT_FACTOR],
                ),
                centres
                + self.flip_array
                * jnp.array(
                    [0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
                centres
                + self.flip_array
                * jnp.array(
                    [-0.5 * self.side_length, -0.5 * self.side_length * HEIGHT_FACTOR]
                ),
            ),
            axis=1,
        )

    @property
    def flip_mask(self) -> np.ndarray:
        """
        A mask for the triangles that are flipped.

        Every other triangle is flipped so that they tessellate.
        """
        mask = (self.coordinates[:, 0] + self.coordinates[:, 1]) % 2 != 0
        if self.flipped:
            mask = ~mask
        return mask

    @property
    def flip_array(self) -> np.ndarray:
        """
        An array of 1s and -1s to flip the triangles.
        """
        import jax.numpy as jnp

        array = jnp.where(self.flip_mask, -1, 1)
        return array[:, None]

    def up_sample(self) -> "CoordinateArrayTriangles":
        """
        Up-sample the triangles by adding a new vertex at the midpoint of each edge.
        """
        import jax.numpy as jnp

        coordinates = self.coordinates
        flip_mask = self.flip_mask

        coordinates = 2 * coordinates

        n = coordinates.shape[0]

        shift0 = jnp.zeros((n, 2))
        shift3 = jnp.tile(jnp.array([0, 1]), (n, 1))
        shift1 = jnp.stack([jnp.ones(n), jnp.where(flip_mask, 1, 0)], axis=1)
        shift2 = jnp.stack([-jnp.ones(n), jnp.where(flip_mask, 1, 0)], axis=1)
        shifts = jnp.stack([shift0, shift1, shift2, shift3], axis=1)

        coordinates_expanded = coordinates[:, None, :]
        new_coordinates = coordinates_expanded + shifts
        new_coordinates = new_coordinates.reshape(-1, 2)

        return CoordinateArrayTriangles(
            coordinates=new_coordinates,
            side_length=self.side_length / 2,
            flipped=True,
            y_offset=self.y_offset + -0.25 * HEIGHT_FACTOR * self.side_length,
            x_offset=self.x_offset,
        )

    def neighborhood(self) -> "CoordinateArrayTriangles":
        """
        Create a new set of triangles that are the neighborhood of the current triangles.

        Ensures that the new triangles are unique and adjusts the mask accordingly.
        """
        import jax.numpy as jnp

        coordinates = self.coordinates
        flip_mask = self.flip_mask

        shift0 = jnp.zeros((coordinates.shape[0], 2))
        shift1 = jnp.tile(jnp.array([1, 0]), (coordinates.shape[0], 1))
        shift2 = jnp.tile(jnp.array([-1, 0]), (coordinates.shape[0], 1))
        shift3 = jnp.where(
            flip_mask[:, None],
            jnp.tile(jnp.array([0, 1]), (coordinates.shape[0], 1)),
            jnp.tile(jnp.array([0, -1]), (coordinates.shape[0], 1)),
        )

        shifts = jnp.stack([shift0, shift1, shift2, shift3], axis=1)

        coordinates_expanded = coordinates[:, None, :]
        new_coordinates = coordinates_expanded + shifts
        new_coordinates = new_coordinates.reshape(-1, 2)

        expected_size = 4 * coordinates.shape[0]
        unique_coords, indices = jnp.unique(
            new_coordinates,
            axis=0,
            size=expected_size,
            fill_value=jnp.nan,
            return_index=True,
        )

        return CoordinateArrayTriangles(
            coordinates=unique_coords,
            side_length=self.side_length,
            flipped=self.flipped,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
        )

    @property
    def _vertices_and_indices(self):
        """
        The flat ``(3N, 2)`` vertex table and the ``(N, 3)`` index map into it.

        On this JAX path the table is deliberately *not* deduplicated: vertex ``3 * i + k`` is
        vertex ``k`` of triangle ``i`` and ``indices`` is simply ``arange(3N).reshape(N, 3)``.
        Under ``jit``, ``jnp.unique`` needs a static ``size``, so a deduplicated table was padded
        back to 3N rows anyway (no deflection evaluations saved) while costing a lexicographic
        sort of 3N fp64 rows, twice per solver refinement step (PyAutoArray#568,
        autolens_profiling#297). NaN padding rows (from `for_indexes`) trace to NaN triangles,
        which every `Shape.mask` rejects, so containment is unchanged. The NumPy sibling
        `CoordinateArrayTrianglesNp` still deduplicates, because its shapes are dynamic.

        When a `vertex_table` is attached (the static initial lattice, see `static_vertex_table`)
        it is returned instead: a compile-time-constant ``(V, 2)`` table of the geometrically
        unique vertices and its ``(N, 3)`` index map.
        """
        import jax.numpy as jnp

        if self.vertex_table is not None:
            vertices, indices = self.vertex_table
            return jnp.asarray(vertices), jnp.asarray(indices)

        flat = self.triangles.reshape(-1, 2)
        indices = jnp.arange(flat.shape[0]).reshape(-1, 3)
        return flat, indices

    def with_vertices(self, vertices: np.ndarray) -> ArrayTriangles:
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
        )

    def for_indexes(self, indexes: np.ndarray) -> "CoordinateArrayTriangles":
        """
        Create a new CoordinateArrayTriangles containing triangles corresponding to the given indexes

        Parameters
        ----------
        indexes
            The indexes of the triangles to include in the new CoordinateArrayTriangles.

        Returns
        -------
        The new CoordinateArrayTriangles instance.
        """
        import jax.numpy as jnp

        mask = indexes == -1
        safe_indexes = jnp.where(mask, 0, indexes)
        coordinates = jnp.take(self.coordinates, safe_indexes, axis=0)
        coordinates = jnp.where(mask[:, None], jnp.nan, coordinates)

        return CoordinateArrayTriangles(
            coordinates=coordinates,
            side_length=self.side_length,
            y_offset=self.y_offset,
            x_offset=self.x_offset,
            flipped=self.flipped,
        )

    @property
    def vertices(self) -> np.ndarray:
        """
        The vertices of the triangles as a flat ``(3N, 2)`` table, row ``3 * i + k`` being vertex
        ``k`` of triangle ``i``.

        Not deduplicated on this JAX path (see `_vertices_and_indices`): a static-shape
        ``jnp.unique`` returned 3N rows regardless and only added a sort. Rows of NaN padding
        triangles are NaN. `CoordinateArrayTrianglesNp.vertices` is deduplicated.
        """
        return self._vertices_and_indices[0]

    @property
    def indices(self) -> np.ndarray:
        """
        The indices of the vertices of the triangles, an ``(N, 3)`` map into `vertices`.

        On this JAX path it is ``arange(3N).reshape(N, 3)`` (no deduplication, see
        `_vertices_and_indices`); padding triangles keep valid indices and are carried as NaN
        vertices instead of ``-1`` entries.
        """
        return self._vertices_and_indices[1]

    @property
    def means(self):
        import jax.numpy as jnp

        return jnp.mean(self.triangles, axis=1)

    @property
    def area(self):
        return (3**0.5 / 4 * self.side_length**2) * len(self)
