"""
Plotting-agnostic objects describing how a region of the source-plane maps to the image-plane.

A `Mapping` pairs one source-plane region (a group of mesh pixels, e.g. a clump of the
reconstruction) with the list of image-plane connected regions (the multiple images) that the
region maps to. Both planes are described as polygons in scaled (e.g. arc-second) coordinates,
so that they can be overlaid on any plot without depending on how the underlying data is binned,
zoomed or cropped.

Everything in this module is pure numpy and is never jitted, because it is diagnostic /
visualization code which runs once per figure.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


COLORS_DEFAULT = ["r", "g", "b", "m", "c", "y"]


@dataclass(frozen=True, eq=False)
class ImageRegion:
    """
    A single connected region of image-plane pixels which a source-plane region maps to.

    For a strong lens this is one of the multiple images of a source clump.

    Parameters
    ----------
    slim_indexes
        The slim indexes (indexes into the masked data, without over-sampling) of every image
        pixel in the region.
    mask
        A `Mask2D` whose `False` entries are the pixels inside the region, using the same 2D
        shape, pixel scales and origin as the data the region was computed from.
    contours
        The boundary loops of the region, each an ``(N, 2)`` array of ``(y, x)`` scaled
        coordinates whose first and last entries are identical (a closed loop). A region with a
        hole, or which pinches at a corner, has more than one loop.
    centre
        The ``(y, x)`` scaled coordinate of the geometric centre (unweighted mean) of the region.
    """

    slim_indexes: np.ndarray
    mask: "Mask2D"
    contours: List[np.ndarray]
    centre: Tuple[float, float]

    @property
    def pixel_coordinates(self) -> np.ndarray:
        """
        The ``(y, x)`` native pixel coordinates of every pixel inside the region, shape ``(N, 2)``.
        """
        return np.argwhere(~np.asarray(self.mask))

    @property
    def scaled_coordinates(self) -> np.ndarray:
        """
        The ``(y, x)`` scaled coordinates of the centre of every pixel inside the region, shape
        ``(N, 2)``.
        """
        geometry = self.mask.geometry

        pixels = self.pixel_coordinates

        y = geometry.scaled_maxima[0] - (pixels[:, 0] + 0.5) * geometry.pixel_scales[0]
        x = geometry.scaled_minima[1] + (pixels[:, 1] + 0.5) * geometry.pixel_scales[1]

        return np.stack([y, x], axis=-1)

    def area(self) -> float:
        """
        The area of the region in scaled units squared (e.g. arcsec^2), computed as its number of
        pixels multiplied by the area of a single pixel.

        Returns
        -------
        The scaled area of the region.
        """
        pixel_scales = self.mask.geometry.pixel_scales

        return float(len(self.slim_indexes) * pixel_scales[0] * pixel_scales[1])

    def values_from(self, array) -> np.ndarray:
        """
        The values of `array` at every pixel inside the region.

        Parameters
        ----------
        array
            An `Array2D` (or 2D numpy array in native format) defined on the same mask geometry as
            the region.

        Returns
        -------
        The values of `array` inside the region, ordered as `pixel_coordinates`.
        """
        native = (
            np.asarray(array.native.array)
            if hasattr(array, "native")
            else np.asarray(array)
        )

        if native.ndim != 2:
            raise ValueError(
                "The array input to an ImageRegion must be an Array2D or a 2D native numpy array, "
                f"but an array of shape {native.shape} was input."
            )

        pixels = self.pixel_coordinates

        return native[pixels[:, 0], pixels[:, 1]]

    def brightest_coordinate_from(self, array) -> Tuple[float, float]:
        """
        The ``(y, x)`` scaled coordinate of the brightest pixel of `array` inside the region.

        Parameters
        ----------
        array
            An `Array2D` (or 2D native numpy array) defined on the same mask geometry as the region.

        Returns
        -------
        The ``(y, x)`` scaled coordinate of the region's brightest pixel.
        """
        values = self.values_from(array=array)

        index = int(np.argmax(values))

        coordinate = self.scaled_coordinates[index]

        return float(coordinate[0]), float(coordinate[1])

    def centroid_from(self, array) -> Tuple[float, float]:
        """
        The flux-weighted ``(y, x)`` scaled centroid of `array` inside the region.

        Parameters
        ----------
        array
            An `Array2D` (or 2D native numpy array) defined on the same mask geometry as the region.

        Returns
        -------
        The ``(y, x)`` scaled coordinate of the region's flux-weighted centroid.
        """
        values = self.values_from(array=array)

        total = float(np.sum(values))

        if total == 0.0:
            return self.centre

        coordinates = self.scaled_coordinates

        y = float(np.sum(coordinates[:, 0] * values) / total)
        x = float(np.sum(coordinates[:, 1] * values) / total)

        return y, x

    def flux_from(self, array) -> float:
        """
        The summed value of `array` over every pixel inside the region.

        Parameters
        ----------
        array
            An `Array2D` (or 2D native numpy array) defined on the same mask geometry as the region.

        Returns
        -------
        The total flux of the region.
        """
        return float(np.sum(self.values_from(array=array)))


@dataclass(frozen=True, eq=False)
class Mapping:
    """
    One source-plane region and the image-plane regions (multiple images) it maps to.

    Parameters
    ----------
    pix_indexes
        The indexes of the mesh pixels in the source-plane region.
    source_contours
        The boundaries of the source-plane region, each an ``(N, 2)`` array of ``(y, x)`` scaled
        coordinates forming a closed loop. One loop per mesh pixel (rectangular cells / Voronoi
        cells), so the region is drawn as the union of its pixels.
    source_centre
        The ``(y, x)`` scaled coordinate of the mean of the region's mesh pixel centres.
    image_regions
        The connected image-plane regions the source region maps to.
    peak_value
        The maximum reconstructed value over the region's mesh pixels, or `None` when the mapping
        was computed from a bare `Mapper` (which has no reconstruction).
    """

    pix_indexes: np.ndarray
    source_contours: List[np.ndarray]
    source_centre: Tuple[float, float]
    image_regions: List[ImageRegion]
    peak_value: Optional[float] = None

    @property
    def image_contours(self) -> List[np.ndarray]:
        """
        Every image-plane boundary loop of the mapping, flattened across its image regions.
        """
        return [contour for region in self.image_regions for contour in region.contours]


def pix_index_groups_from(pix_indexes: Sequence) -> List[np.ndarray]:
    """
    Normalise a user-input `pix_indexes` into a list of mesh-pixel index groups.

    A flat sequence of indexes (e.g. ``[0, 1, 2]``) is a single group; a nested sequence
    (e.g. ``[[0, 1], [5]]``) is one group per entry.

    Parameters
    ----------
    pix_indexes
        The mesh pixel indexes, either flat or nested.

    Returns
    -------
    A list of 1D integer arrays, one per group.
    """
    groups = list(pix_indexes)

    if len(groups) == 0:
        return []

    if np.ndim(groups[0]) == 0:
        return [np.asarray(groups, dtype=int)]

    return [np.asarray(group, dtype=int) for group in groups]


def connected_components_from(indexes, neighbors) -> List[np.ndarray]:
    """
    Split `indexes` into connected components of the mesh neighbour graph.

    Two mesh pixels are in the same component if a path of neighbouring pixels between them exists
    which passes only through pixels in `indexes`.

    Parameters
    ----------
    indexes
        The mesh pixel indexes to group.
    neighbors
        A `Neighbors` object mapping every mesh pixel to the indexes of its neighbours, whose
        `sizes` attribute gives the number of valid entries of each row.

    Returns
    -------
    A list of 1D integer arrays, one per connected component, each sorted ascending. Components
    are ordered by the first appearance of one of their members in `indexes`.
    """
    indexes = np.asarray(indexes, dtype=int).ravel()

    neighbors_arr = np.asarray(neighbors)
    sizes = np.asarray(neighbors.sizes, dtype=int)

    in_group = set(int(index) for index in indexes)
    visited = set()

    components = []

    for index in indexes:
        index = int(index)

        if index in visited:
            continue

        visited.add(index)

        stack = [index]
        component = []

        while stack:
            pixel = stack.pop()
            component.append(pixel)

            row = neighbors_arr[pixel][: sizes[pixel]]

            for neighbor in row:
                neighbor = int(neighbor)

                if neighbor < 0 or neighbor in visited or neighbor not in in_group:
                    continue

                visited.add(neighbor)
                stack.append(neighbor)

        components.append(np.sort(np.asarray(component, dtype=int)))

    return components


def contours_from_bool_native(bool_native: np.ndarray, geometry) -> List[np.ndarray]:
    """
    The boundary loops of the `True` pixels of a 2D boolean array, in scaled coordinates.

    Every edge of a `True` pixel which is not shared with another `True` pixel is a boundary edge.
    The boundary edges are directed so that the region is always on the same side of them, and are
    then chained end-to-end into closed loops. A region with a hole emits the hole as an extra
    loop, and a region which pinches to a point at a corner emits one loop per lobe.

    Parameters
    ----------
    bool_native
        A 2D boolean array in native format, whose `True` entries are the region.
    geometry
        A `Geometry2D` (e.g. `mask.geometry`) giving the `pixel_scales`, `scaled_maxima` and
        `scaled_minima` used to convert pixel corners to scaled coordinates.

    Returns
    -------
    A list of ``(N, 2)`` arrays of ``(y, x)`` scaled coordinates, each a closed loop whose first
    and last entries are identical.
    """
    bool_native = np.asarray(bool_native, dtype=bool)

    padded = np.zeros(
        shape=(bool_native.shape[0] + 2, bool_native.shape[1] + 2), dtype=bool
    )
    padded[1:-1, 1:-1] = bool_native

    inside = padded[1:-1, 1:-1]

    # Directed boundary edges in pixel-corner space, each stored as start -> end. The directions
    # are chosen so that the region is consistently on one side, which makes the chaining below
    # trace each loop without ambiguity except at pinch vertices (where either choice is valid).

    edges = {}

    def _add(start, end):
        edges.setdefault(start, []).append(end)

    rows, cols = np.nonzero(inside)

    for row, col in zip(rows.tolist(), cols.tolist()):
        if not padded[row, col + 1]:
            _add((row, col), (row, col + 1))
        if not padded[row + 1, col + 2]:
            _add((row, col + 1), (row + 1, col + 1))
        if not padded[row + 2, col + 1]:
            _add((row + 1, col + 1), (row + 1, col))
        if not padded[row + 1, col]:
            _add((row + 1, col), (row, col))

    loops = []

    starts = list(edges.keys())

    for start in starts:
        while edges.get(start):
            loop = [start]
            point = start

            while True:
                ends = edges.get(point)

                if not ends:
                    raise ValueError(
                        "The boundary edges of a region could not be chained into a closed loop, "
                        "which should be impossible for a boolean pixel region."
                    )

                point = ends.pop()
                loop.append(point)

                if point == start:
                    break

            loops.append(np.asarray(loop, dtype=int))

    y_max = geometry.scaled_maxima[0]
    x_min = geometry.scaled_minima[1]
    pixel_scales = geometry.pixel_scales

    contours = []

    for loop in loops:
        y = y_max - loop[:, 0] * pixel_scales[0]
        x = x_min + loop[:, 1] * pixel_scales[1]

        contours.append(np.stack([y, x], axis=-1))

    return contours


def image_regions_from_slim_mask(
    mask, slim_bool, min_pixels: int = 1
) -> List[ImageRegion]:
    """
    The connected image-plane regions of a boolean flag over the masked data pixels.

    The slim boolean is lifted to native 2D, split into connected components with 8-connectivity
    (`scipy.ndimage.label`), and each component larger than `min_pixels` is returned as an
    `ImageRegion` with its boundary contours in scaled coordinates.

    Parameters
    ----------
    mask
        The `Mask2D` of the data, whose `False` entries are the (slim ordered) data pixels.
    slim_bool
        A 1D boolean array of length `mask.pixels_in_mask`, `True` for pixels in the region.
    min_pixels
        Connected components with fewer than this many pixels are discarded.

    Returns
    -------
    The connected regions, ordered largest first.
    """
    from scipy import ndimage

    from autoarray.mask.mask_2d import Mask2D

    mask_arr = np.asarray(mask)

    slim_bool = np.asarray(slim_bool, dtype=bool)

    region_native = np.zeros(shape=mask_arr.shape, dtype=bool)
    region_native[~mask_arr] = slim_bool

    # The slim index of every unmasked pixel, in the native raster order the slim ordering uses.
    slim_index_native = np.full(shape=mask_arr.shape, fill_value=-1, dtype=int)
    slim_index_native[~mask_arr] = np.arange(int(np.sum(~mask_arr)))

    labels, total_labels = ndimage.label(
        region_native, structure=np.ones(shape=(3, 3), dtype=int)
    )

    regions = []

    for label in range(1, total_labels + 1):
        label_native = labels == label

        slim_indexes = np.sort(slim_index_native[label_native])

        if slim_indexes.shape[0] < min_pixels:
            continue

        region_mask = Mask2D(
            mask=~label_native,
            pixel_scales=mask.pixel_scales,
            origin=mask.origin,
        )

        contours = contours_from_bool_native(
            bool_native=label_native, geometry=mask.geometry
        )

        pixels = np.argwhere(label_native)

        geometry = mask.geometry

        centre = (
            float(
                np.mean(
                    geometry.scaled_maxima[0]
                    - (pixels[:, 0] + 0.5) * geometry.pixel_scales[0]
                )
            ),
            float(
                np.mean(
                    geometry.scaled_minima[1]
                    + (pixels[:, 1] + 0.5) * geometry.pixel_scales[1]
                )
            ),
        )

        regions.append(
            ImageRegion(
                slim_indexes=slim_indexes,
                mask=region_mask,
                contours=contours,
                centre=centre,
            )
        )

    regions.sort(key=lambda region: -region.slim_indexes.shape[0])

    return regions


def image_regions_from(
    mapper,
    pix_indexes,
    weight_threshold: float = 0.0,
    min_pixels: int = 1,
) -> List[ImageRegion]:
    """
    The connected image-plane regions which a group of mesh pixels maps to.

    The mapper's `mapping_matrix` (shape ``[data_pixels, mesh_pixels]``, already folded over
    over-sampled sub-pixels) is summed over `pix_indexes`, giving the total weight with which every
    data pixel maps to the group. Data pixels whose weight exceeds `weight_threshold` form the
    image-plane region, which is then split into connected components.

    Parameters
    ----------
    mapper
        The `Mapper` whose mappings between the data and the mesh are used.
    pix_indexes
        The indexes of the mesh pixels forming the source-plane region.
    weight_threshold
        A data pixel is in the image-plane region if its summed mapping weight exceeds this value.
        Raising it keeps only the data pixels which map dominantly to the source region.
    min_pixels
        Connected image-plane regions with fewer than this many pixels are discarded.

    Returns
    -------
    The connected image-plane regions, ordered largest first.
    """
    pix_indexes = np.asarray(pix_indexes, dtype=int).ravel()

    mapping_matrix = np.asarray(mapper.mapping_matrix)

    weights = mapping_matrix[:, pix_indexes].sum(axis=1)

    return image_regions_from_slim_mask(
        mask=mapper.mask,
        slim_bool=weights > weight_threshold,
        min_pixels=min_pixels,
    )


def _rectangular_edges_from(mapper) -> Tuple[np.ndarray, np.ndarray]:
    """
    The ``(y, x)`` cell edge coordinates of a rectangular mesh, one more edge than pixels per axis.

    Uniform rectangular meshes derive their edges from the mesh geometry's pixel scales and origin
    (matching the `imshow` extent the reconstruction is drawn with), whereas adaptive rectangular
    meshes use the mesh geometry's `edges_transformed` (matching the `pcolormesh` grid).

    Parameters
    ----------
    mapper
        A `Mapper` whose interpolator is rectangular.

    Returns
    -------
    The ``(y_edges, x_edges)`` arrays, of length ``shape[0] + 1`` and ``shape[1] + 1``.
    """
    from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
        InterpolatorRectangularUniform,
    )

    mesh_geometry = mapper.mesh_geometry

    shape = mesh_geometry.shape

    if isinstance(mapper.interpolator, InterpolatorRectangularUniform):
        geometry = mesh_geometry.geometry

        y_edges = (
            geometry.scaled_maxima[0]
            - np.arange(shape[0] + 1) * geometry.pixel_scales[0]
        )
        x_edges = (
            geometry.scaled_minima[1]
            + np.arange(shape[1] + 1) * geometry.pixel_scales[1]
        )

        return y_edges, x_edges

    y_edges, x_edges = np.asarray(mesh_geometry.edges_transformed).T

    return y_edges, x_edges


def _voronoi_contours_from(mapper, pix_indexes: np.ndarray) -> List[np.ndarray]:
    """
    The Voronoi cell polygons of a group of Delaunay (or KNN) mesh vertices.

    A Delaunay mesh pixel is a vertex of the triangulation, whose natural area is its Voronoi cell.
    Cells which are unbounded (the mesh's convex hull) have no polygon and are omitted; bounded
    cells are clipped to the bounding box of the mesh grid so a large boundary cell cannot stretch
    the figure.

    Parameters
    ----------
    mapper
        A `Mapper` whose interpolator is Delaunay or K nearest neighbour.
    pix_indexes
        The indexes of the mesh pixels to return cells for.

    Returns
    -------
    A list of ``(N, 2)`` arrays of ``(y, x)`` scaled coordinates, each a closed loop.
    """
    voronoi = mapper.mesh_geometry.voronoi

    mesh_grid = np.asarray(
        mapper.source_plane_mesh_grid.array
        if hasattr(mapper.source_plane_mesh_grid, "array")
        else mapper.source_plane_mesh_grid
    )

    y_min, y_max = float(np.min(mesh_grid[:, 0])), float(np.max(mesh_grid[:, 0]))
    x_min, x_max = float(np.min(mesh_grid[:, 1])), float(np.max(mesh_grid[:, 1]))

    contours = []

    for pix_index in pix_indexes:
        region = voronoi.regions[voronoi.point_region[int(pix_index)]]

        if len(region) == 0 or -1 in region:
            continue

        # `mesh_grid_xy` feeds scipy the mesh grid in its native (y, x) column order, so the
        # Voronoi vertices come back in (y, x) too.
        vertices = np.asarray(voronoi.vertices)[np.asarray(region, dtype=int)]

        vertices = np.stack(
            [
                np.clip(vertices[:, 0], y_min, y_max),
                np.clip(vertices[:, 1], x_min, x_max),
            ],
            axis=-1,
        )

        contours.append(np.vstack([vertices, vertices[0]]))

    return contours


def source_contours_from(mapper, pix_indexes) -> List[np.ndarray]:
    """
    The source-plane cell boundaries of a group of mesh pixels, in scaled coordinates.

    Rectangular meshes return the four corners of each cell; Delaunay and K nearest neighbour
    meshes return each vertex's Voronoi cell.

    Parameters
    ----------
    mapper
        The `Mapper` whose mesh the pixels belong to.
    pix_indexes
        The indexes of the mesh pixels forming the source-plane region.

    Returns
    -------
    A list of ``(N, 2)`` arrays of ``(y, x)`` scaled coordinates, one closed loop per mesh pixel.
    """
    from autoarray.inversion.mesh.interpolator.rectangular import (
        InterpolatorRectangular,
    )
    from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
        InterpolatorRectangularUniform,
    )
    from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
    from autoarray.inversion.mesh.interpolator.knn import InterpolatorKNearestNeighbor

    pix_indexes = np.asarray(pix_indexes, dtype=int).ravel()

    if isinstance(
        mapper.interpolator, (InterpolatorRectangular, InterpolatorRectangularUniform)
    ):
        y_edges, x_edges = _rectangular_edges_from(mapper=mapper)

        total_x = mapper.mesh_geometry.shape[1]

        contours = []

        for pix_index in pix_indexes:
            row = int(pix_index) // total_x
            col = int(pix_index) % total_x

            y0, y1 = float(y_edges[row]), float(y_edges[row + 1])
            x0, x1 = float(x_edges[col]), float(x_edges[col + 1])

            contours.append(
                np.asarray([[y0, x0], [y0, x1], [y1, x1], [y1, x0], [y0, x0]])
            )

        return contours

    if isinstance(
        mapper.interpolator, (InterpolatorDelaunay, InterpolatorKNearestNeighbor)
    ):
        return _voronoi_contours_from(mapper=mapper, pix_indexes=pix_indexes)

    raise NotImplementedError(
        f"Source-plane contours are not implemented for the interpolator "
        f"{type(mapper.interpolator).__name__}."
    )


def mappings_from(
    mapper,
    pix_indexes,
    weight_threshold: float = 0.0,
    min_pixels: int = 1,
    peak_values: Optional[List[float]] = None,
) -> List["Mapping"]:
    """
    The `Mapping` objects of one or more groups of mesh pixels.

    Parameters
    ----------
    mapper
        The `Mapper` whose mappings between the data and the mesh are used.
    pix_indexes
        The mesh pixel indexes, either a flat sequence (one group) or a nested sequence (one group
        per entry).
    weight_threshold
        A data pixel is in an image-plane region if its summed mapping weight exceeds this value.
    min_pixels
        Connected image-plane regions with fewer than this many pixels are discarded.
    peak_values
        The peak reconstructed value of each group, stored on the returned `Mapping` objects. `None`
        leaves every `peak_value` as `None` (the bare-`Mapper` case, which has no reconstruction).

    Returns
    -------
    One `Mapping` per group of mesh pixels.
    """
    groups = pix_index_groups_from(pix_indexes=pix_indexes)

    mesh_grid = np.asarray(
        mapper.source_plane_mesh_grid.array
        if hasattr(mapper.source_plane_mesh_grid, "array")
        else mapper.source_plane_mesh_grid
    )

    mappings = []

    for i, group in enumerate(groups):
        centre = np.mean(mesh_grid[group], axis=0)

        mappings.append(
            Mapping(
                pix_indexes=group,
                source_contours=source_contours_from(mapper=mapper, pix_indexes=group),
                source_centre=(float(centre[0]), float(centre[1])),
                image_regions=image_regions_from(
                    mapper=mapper,
                    pix_indexes=group,
                    weight_threshold=weight_threshold,
                    min_pixels=min_pixels,
                ),
                peak_value=None if peak_values is None else float(peak_values[i]),
            )
        )

    return mappings
