import numpy as np


class Neighbors(np.ndarray):
    def __new__(cls, arr: np.ndarray, sizes: np.ndarray):
        """
        Class packaging ndarrays describing the neighbors of every pixel in a mesh (e.g. `RectangularAdaptDensity`).

        The array `arr` contains the pixel indexes of the neighbors of every pixel. Its has shape [total_pixels,
        max_neighbors_in_single_pixel].

        The array `sizes` contains the number of neighbors of every pixel in the pixelixzation.

        For example, for a 3x3 `RectangularAdaptDensity` grid:

        - `total_pixels=9` and `max_neighbors_in_single_pixel=4` (because the central pixel has 4 neighbors whereas
          edge / corner pixels have 3 and 2).

        - The shape of `arr` is therefore [9, 4], with entries where there is no neighbor (e.g. arr[0, 3]) containing
          values of -1.

        - Pixel 0 is at the top-left of the rectangular mesh and has two neighbors, the pixel to its right
          (with index 1) and the pixel below it (with index 3). Therefore, `arr[0,:] = [1, 3, -1, -1]` and `sizes[0] = 2`.

        - Pixel 1 is at the top-middle and has three neighbors, to its left (index 0, right (index 2) and below it
          (index 4). Therefore, neighbors[1,:] = [0, 2, 4, 1] and neighbors_sizes[1] = 3.

        - For pixel 4, the central pixel, neighbors[4,:] = [1, 3, 5, 7] and neighbors_sizes[4] = 4.

        The same arrays can be generalized for other pixelizations, for example a `Delaunay` grid.

        Parameters
        ----------
        arr
            An array which maps every pixelization pixel to the indexes of its neighbors.
        sizes
            An array containing the number of neighbors of every pixelization pixel.
        """
        obj = arr.view(cls)
        obj.sizes = sizes

        return obj
