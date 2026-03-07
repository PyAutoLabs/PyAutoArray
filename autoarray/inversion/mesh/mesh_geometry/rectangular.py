import numpy as np
from typing import List, Tuple

from autoarray.geometry.geometry_2d import Geometry2D
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.mesh.mesh_geometry.abstract import AbstractMeshGeometry


def rectangular_neighbors_from(
    shape_native: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the 4 (or less) adjacent neighbors of every pixel on a rectangular pixelization as an ndarray of shape
    [total_pixels, 4], called `neighbors`. This uses the uniformity of the rectangular grid's geometry to speed
    up the computation.

    Entries with values of `-1` signify edge pixels which do not have neighbors. This function therefore also returns
    an ndarray with the number of neighbors of every pixel, `neighbors_sizes`, which is iterated over when using
    the `neighbors` ndarray.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For example, on a 3x3 grid:

    - Pixel 0 is at the top-left and has two neighbors, the pixel to its right  (with index 1) and the pixel below
      it (with index 3). Therefore, the neighbors[0,:] = [1, 3, -1, -1] and neighbors_sizes[0] = 2.

    - Pixel 1 is at the top-middle and has three neighbors, to its left (index 0, right (index 2) and below it
      (index 4). Therefore, neighbors[1,:] = [0, 2, 4, -1] and neighbors_sizes[1] = 3.

    - For pixel 4, the central pixel, neighbors[4,:] = [1, 3, 5, 7] and neighbors_sizes[4] = 4.

    Parameters
    ----------
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The ndarrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixels = int(shape_native[0] * shape_native[1])

    neighbors = -1 * np.ones(shape=(pixels, 4))
    neighbors_sizes = np.zeros(pixels)

    neighbors, neighbors_sizes = rectangular_corner_neighbors(
        neighbors=neighbors, neighbors_sizes=neighbors_sizes, shape_native=shape_native
    )
    neighbors, neighbors_sizes = rectangular_top_edge_neighbors(
        neighbors=neighbors, neighbors_sizes=neighbors_sizes, shape_native=shape_native
    )
    neighbors, neighbors_sizes = rectangular_left_edge_neighbors(
        neighbors=neighbors, neighbors_sizes=neighbors_sizes, shape_native=shape_native
    )
    neighbors, neighbors_sizes = rectangular_right_edge_neighbors(
        neighbors=neighbors, neighbors_sizes=neighbors_sizes, shape_native=shape_native
    )
    neighbors, neighbors_sizes = rectangular_bottom_edge_neighbors(
        neighbors=neighbors, neighbors_sizes=neighbors_sizes, shape_native=shape_native
    )
    neighbors, neighbors_sizes = rectangular_central_neighbors(
        neighbors=neighbors, neighbors_sizes=neighbors_sizes, shape_native=shape_native
    )

    return neighbors, neighbors_sizes


def rectangular_corner_neighbors(
    neighbors: np.ndarray, neighbors_sizes: np.ndarray, shape_native: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `neighbors` and `neighbors_sizes` arrays described in the function
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the corners.

    Parameters
    ----------
    neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """

    pixels = int(shape_native[0] * shape_native[1])

    neighbors[0, 0:2] = np.array([1, shape_native[1]])
    neighbors_sizes[0] = 2

    neighbors[shape_native[1] - 1, 0:2] = np.array(
        [shape_native[1] - 2, shape_native[1] + shape_native[1] - 1]
    )
    neighbors_sizes[shape_native[1] - 1] = 2

    neighbors[pixels - shape_native[1], 0:2] = np.array(
        [pixels - shape_native[1] * 2, pixels - shape_native[1] + 1]
    )
    neighbors_sizes[pixels - shape_native[1]] = 2

    neighbors[pixels - 1, 0:2] = np.array([pixels - shape_native[1] - 1, pixels - 2])
    neighbors_sizes[pixels - 1] = 2

    return neighbors, neighbors_sizes


def rectangular_top_edge_neighbors(
    neighbors: np.ndarray, neighbors_sizes: np.ndarray, shape_native: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `neighbors` and `neighbors_sizes` arrays described in the function
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the top edge.

    Parameters
    ----------
    neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    """
    Vectorized version of the top edge neighbor update using NumPy arithmetic.
    """
    # Pixels along the top edge, excluding corners
    top_edge_pixels = np.arange(1, shape_native[1] - 1)

    neighbors[top_edge_pixels, 0] = top_edge_pixels - 1
    neighbors[top_edge_pixels, 1] = top_edge_pixels + 1
    neighbors[top_edge_pixels, 2] = top_edge_pixels + shape_native[1]
    neighbors_sizes[top_edge_pixels] = 3

    return neighbors, neighbors_sizes


def rectangular_left_edge_neighbors(
    neighbors: np.ndarray, neighbors_sizes: np.ndarray, shape_native: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `neighbors` and `neighbors_sizes` arrays described in the function
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the left edge.

    Parameters
    ----------
    neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    # Row indices (excluding top and bottom corners)
    rows = np.arange(1, shape_native[0] - 1)

    # Convert to flat pixel indices for the left edge (first column)
    pixel_indices = rows * shape_native[1]

    neighbors[pixel_indices, 0] = pixel_indices - shape_native[1]
    neighbors[pixel_indices, 1] = pixel_indices + 1
    neighbors[pixel_indices, 2] = pixel_indices + shape_native[1]
    neighbors_sizes[pixel_indices] = 3

    return neighbors, neighbors_sizes


def rectangular_right_edge_neighbors(
    neighbors: np.ndarray, neighbors_sizes: np.ndarray, shape_native: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `neighbors` and `neighbors_sizes` arrays described in the function
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the right edge.

    Parameters
    ----------
    neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    # Rows excluding the top and bottom corners
    rows = np.arange(1, shape_native[0] - 1)

    # Flat indices for the right edge pixels
    pixel_indices = rows * shape_native[1] + shape_native[1] - 1

    neighbors[pixel_indices, 0] = pixel_indices - shape_native[1]
    neighbors[pixel_indices, 1] = pixel_indices - 1
    neighbors[pixel_indices, 2] = pixel_indices + shape_native[1]
    neighbors_sizes[pixel_indices] = 3

    return neighbors, neighbors_sizes


def rectangular_bottom_edge_neighbors(
    neighbors: np.ndarray, neighbors_sizes: np.ndarray, shape_native: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `neighbors` and `neighbors_sizes` arrays described in the function
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are on the bottom edge.

    Parameters
    ----------
    neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    n_rows, n_cols = shape_native
    pixels = n_rows * n_cols

    # Horizontal pixel positions along bottom row, excluding corners
    cols = np.arange(1, n_cols - 1)
    pixel_indices = pixels - cols - 1  # Reverse order from right to left

    neighbors[pixel_indices, 0] = pixel_indices - n_cols
    neighbors[pixel_indices, 1] = pixel_indices - 1
    neighbors[pixel_indices, 2] = pixel_indices + 1
    neighbors_sizes[pixel_indices] = 3

    return neighbors, neighbors_sizes


def rectangular_central_neighbors(
    neighbors: np.ndarray, neighbors_sizes: np.ndarray, shape_native: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the `neighbors` and `neighbors_sizes` arrays described in the function
    `rectangular_neighbors_from()` for pixels on the rectangular pixelization that are in the centre and thus have 4
    adjacent neighbors.

    Parameters
    ----------
    neighbors
        An array of dimensions [total_pixels, 4] which provides the index of all neighbors of every pixel in
        the rectangular pixelization (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of dimensions [total_pixels] which gives the number of neighbors of every pixel in the rectangular
        pixelization.
    shape_native
        The shape of the rectangular 2D pixelization which pixels are defined on.

    Returns
    -------
    The arrays containing the 1D index of every pixel's neighbors and the number of neighbors that each pixel has.
    """
    n_rows, n_cols = shape_native

    # Grid coordinates excluding edges
    xs = np.arange(1, n_rows - 1)
    ys = np.arange(1, n_cols - 1)

    # 2D grid of central pixel indices
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    pixel_indices = grid_x * n_cols + grid_y
    pixel_indices = pixel_indices.ravel()

    # Compute neighbor indices
    neighbors[pixel_indices, 0] = pixel_indices - n_cols  # Up
    neighbors[pixel_indices, 1] = pixel_indices - 1  # Left
    neighbors[pixel_indices, 2] = pixel_indices + 1  # Right
    neighbors[pixel_indices, 3] = pixel_indices + n_cols  # Down

    neighbors_sizes[pixel_indices] = 4

    return neighbors, neighbors_sizes


def rectangular_edges_from(shape_native, pixel_scales, xp=np):
    """
    Returns all pixel edges for a rectangular grid as a JAX array of shape (N, 4, 2, 2),
    where N = Ny * Nx. Edge order per pixel matches the user's convention:

      0: (x1, y0) -> (x1, y1)
      1: (x1, y1) -> (x0, y1)
      2: (x0, y1) -> (x0, y0)
      3: (x0, y0) -> (x1, y0)

    Notes
    -----
    - x is flipped so that the leftmost column has the largest +x (e.g. centres start at x=+1.0).
    - y increases upward (top row has the most negative y when dy>0).
    """
    Ny, Nx = shape_native
    dy, dx = pixel_scales

    # Grid edge coordinates. Flip x so leftmost column has largest +x, matching your convention.
    x_edges = ((xp.arange(Nx + 1) - Nx / 2) * dx)[::-1]
    y_edges = (xp.arange(Ny + 1) - Ny / 2) * dy

    edges_list = []

    # Pixel order: row-major (y outer, x inner). If you want column-major, swap the loop nesting.
    for j in range(Ny):
        for i in range(Nx):
            y0, y1 = y_edges[i], y_edges[i + 1]
            xa, xb = (
                x_edges[j],
                x_edges[j + 1],
            )  # xa is the "right" boundary in your convention

            # Edge order to match your pytest: [(xa,y0)->(xa,y1), (xa,y1)->(xb,y1), (xb,y1)->(xb,y0), (xb,y0)->(xa,y0)]
            e0 = xp.array([[xa, y0], [xa, y1]])  # "top" in your test (vertical at x=xa)
            e1 = xp.array(
                [[xa, y1], [xb, y1]]
            )  # "right" in your test (horizontal at y=y1)
            e2 = xp.array(
                [[xb, y1], [xb, y0]]
            )  # "bottom" in your test (vertical at x=xb)
            e3 = xp.array(
                [[xb, y0], [xa, y0]]
            )  # "left" in your test (horizontal at y=y0)

            edges_list.append(xp.stack([e0, e1, e2, e3], axis=0))

    return xp.stack(edges_list, axis=0)


def rectangular_edge_pixel_list_from(
    shape_native: Tuple[int, int],
) -> List[int]:
    """
    Returns a list of the 1D indices of all pixels on the edge of a rectangular pixelization,
    based on its 2D shape.

    Parameters
    ----------
    shape_native
        The (rows, cols) shape of the rectangular 2D pixel grid.

    Returns
    -------
    A list of the 1D indices of all edge pixels.
    """
    rows, cols = shape_native

    # Top row
    top = np.arange(0, cols)

    # Bottom row
    bottom = np.arange((rows - 1) * cols, rows * cols)

    # Left column (excluding corners)
    left = np.arange(1, rows - 1) * cols

    # Right column (excluding corners)
    right = (np.arange(1, rows - 1) + 1) * cols - 1

    edge_pixel_indices = np.concatenate([top, left, right, bottom])

    return np.sort(edge_pixel_indices).tolist()


def adaptive_rectangular_areas_from(
    source_grid_shape, data_grid, mesh_weight_map=None, xp=np
):

    from autoarray.inversion.mesh.interpolator.rectangular import (
        create_transforms,
    )

    edges_y = xp.linspace(1, 0, source_grid_shape[0] + 1)
    edges_x = xp.linspace(0, 1, source_grid_shape[1] + 1)

    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    pixel_edges = inv_full(xp.stack([edges_y, edges_x]).T)
    pixel_lengths = xp.diff(pixel_edges, axis=0).squeeze()  # shape (N_source, 2)

    dy = pixel_lengths[:, 0]
    dx = pixel_lengths[:, 1]

    return xp.abs(xp.outer(dy, dx).flatten())


class MeshGeometryRectangular(AbstractMeshGeometry):

    @property
    def shape(self):
        """
        The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        """
        return self.mesh.shape

    @property
    def shape_native(self):
        """
        The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        """
        return self.shape

    @property
    def geometry(self):

        xmin = np.min(self.mesh_grid[:, 1])
        xmax = np.max(self.mesh_grid[:, 1])
        ymin = np.min(self.mesh_grid[:, 0])
        ymax = np.max(self.mesh_grid[:, 0])

        pixel_scales = (ymax - ymin) / (self.shape[0] - 1), (xmax - xmin) / (
            self.shape[1] - 1
        )

        origin = ((ymax + ymin) / 2.0, (xmax + xmin) / 2.0)

        return Geometry2D(
            shape_native=self.shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @property
    def pixel_scales(self) -> Tuple[float, float]:
        return self.geometry.pixel_scales

    @property
    def origin(self) -> Tuple[float, float]:
        return self.geometry.origin

    @property
    def areas_transformed(self):
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `rectangular_neighbors_from`.
        """
        return adaptive_rectangular_areas_from(
            source_grid_shape=self.shape_native,
            data_grid=self.data_grid.over_sampled,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )

    @property
    def areas_for_magnification(self):
        """
        The area of every pixel in the rectangular pixelization.

        Returns
        -------
        ndarray
            The area of every pixel in the rectangular pixelization.
        """
        return self.areas_transformed

    @property
    def edges_transformed(self):
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `rectangular_neighbors_from`.
        """

        from autoarray.inversion.mesh.interpolator.rectangular import (
            adaptive_rectangular_transformed_grid_from,
        )

        # edges defined in 0 -> 1 space, there is one more edge than pixel centers on each side
        edges_y = self._xp.linspace(1, 0, self.shape_native[0] + 1)
        edges_x = self._xp.linspace(0, 1, self.shape_native[1] + 1)

        edges_reshaped = self._xp.stack([edges_y, edges_x]).T

        return adaptive_rectangular_transformed_grid_from(
            data_grid=self.data_grid.array,
            grid=edges_reshaped,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )

    @property
    def neighbors(self) -> Neighbors:
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `rectangular_neighbors_from`.
        """
        neighbors, sizes = rectangular_neighbors_from(shape_native=self.shape)

        return Neighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))
