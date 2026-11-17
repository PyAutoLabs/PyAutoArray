"""
Standalone functions for plotting inversion / pixelization reconstructions.

Replaces the inversion-specific paths in ``MatPlot2D.plot_mapper``.
"""

from typing import List, Optional, Tuple

import numpy as np

from autoarray.plot.utils import (
    subplots,
    apply_extent,
    apply_labels,
    conf_figsize,
    save_figure,
    norm_from,
    _conf_imshow_origin,
    _overlay_yx_for_origin,
    numpy_grid,
    plot_regions,
)


def plot_inversion_reconstruction(
    pixel_values: np.ndarray,
    mapper,
    ax=None,
    # --- cosmetics --------------------------------------------------------------
    title: str = "Reconstruction",
    xlabel: str = 'x (")',
    ylabel: str = 'y (")',
    colormap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_log10: bool = False,
    cb_unit: Optional[str] = None,
    zoom_to_brightest: bool = True,
    zoom_extent_scale: float = 1.0,
    arcsec_after_decimal: Optional[bool] = None,
    interpolated: bool = False,
    interpolated_shape_native: Optional[Tuple[int, int]] = None,
    # --- overlays ---------------------------------------------------------------
    lines: Optional[List[np.ndarray]] = None,
    line_colors: Optional[List] = None,
    grid: Optional[np.ndarray] = None,
    regions: Optional[List] = None,
    region_colors: Optional[List] = None,
    region_alpha: float = 0.25,
    region_labels: Optional[List[str]] = None,
    # --- figure control (used only when ax is None) -----------------------------
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    output_filename: str = "reconstruction",
    output_format: str = None,
) -> None:
    """
    Plot an inversion reconstruction using the appropriate mapper type.

    Chooses between rectangular (``imshow``/``pcolormesh``) and Delaunay
    (``tripcolor``) rendering based on the mapper's interpolator type.

    Parameters
    ----------
    pixel_values
        1D array of reconstructed flux values, one per source pixel.
    mapper
        Autoarray mapper object exposing ``interpolator``, ``mesh_geometry``,
        ``source_plane_mesh_grid``, etc.
    ax
        Existing ``Axes``.  ``None`` creates a new figure.
    title, xlabel, ylabel
        Text labels.
    colormap
        Matplotlib colormap name.
    vmin, vmax
        Explicit colour scale limits.
    use_log10
        Apply ``LogNorm``.
    cb_unit
        Unit appended to the middle colour-bar label. ``None`` reads it from config; ``""`` shows no unit.
    zoom_to_brightest
        Pass through to ``mapper.extent_from``.
    arcsec_after_decimal
        Per-call override of the ``ticks.symbol_over_decimal`` config flag.
        When ``True`` the arcsecond tick labels place the ``″`` symbol over the
        decimal point (``3.″8``) instead of suffixing it (``3.8"``); when ``False``
        the suffix form is forced.  ``None`` (the default) reads the config.
    interpolated
        For Delaunay / kNN / natural-neighbour meshes, draw the reconstruction
        the likelihood actually uses between the mesh vertices (the mapper's
        own interpolant evaluated on a uniform grid, see
        :func:`interpolated_reconstruction_from`) instead of flat-shaded
        triangles. Ignored for rectangular meshes, whose pixels already are the
        reconstruction.
    interpolated_shape_native
        The ``(y, x)`` shape of the uniform grid the interpolated view is
        evaluated on. ``None`` reads ``visualize/general.yaml``
        ``inversion: interpolated_pixels`` (default 400 x 400).
    lines
        Line overlays (e.g. critical curves).
    grid
        Scatter overlay (e.g. data-plane grid).
    regions
        List of source-plane regions to overlay as filled polygons, each region being a list of
        ``(N, 2)`` arrays of ``(y, x)`` coordinates (e.g. the ``source_contours`` of a ``Mapping``).
        Every region is drawn in its own colour, matching the colour its image-plane counterpart is
        drawn with by :func:`~autoarray.plot.array.plot_array`.
    region_colors
        The colour of each region, cycled if there are more regions than colours.  ``None`` uses
        ``["r", "g", "b", "m", "c", "y"]``.
    region_alpha
        The alpha of each region's fill; the outline is always drawn opaque.
    region_labels
        A text label drawn at the centre of each region, e.g. ``["1", "2"]``.
    figsize, output_path, output_filename, output_format
        Figure output controls.
    """
    from autoarray.inversion.mesh.interpolator.rectangular import (
        InterpolatorRectangular,
    )
    from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
        InterpolatorRectangularUniform,
    )
    from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
    from autoarray.inversion.mesh.interpolator.knn import InterpolatorKNearestNeighbor

    if colormap is None:
        from autoarray.plot.utils import _default_colormap

        colormap = _default_colormap()

    owns_figure = ax is None
    if owns_figure:
        figsize = figsize or conf_figsize("figures")
        fig, ax = subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- colour normalisation --------------------------------------------------
    norm = norm_from(array=pixel_values, use_log10=use_log10, vmin=vmin, vmax=vmax)

    extent = mapper.extent_from(
        values=pixel_values,
        zoom_to_brightest=zoom_to_brightest,
        zoom_extent_scale=zoom_extent_scale,
    )

    is_subplot = not owns_figure

    # Extent of an ``imshow`` raster drawn on *ax* (uniform rectangular meshes
    # only); ``None`` for ``pcolormesh`` / ``tripcolor`` panels, which are drawn
    # in data coordinates and need no origin handling.
    raster_extent = None

    if isinstance(
        mapper.interpolator, (InterpolatorRectangular, InterpolatorRectangularUniform)
    ):
        raster_extent = _plot_rectangular(
            ax,
            pixel_values,
            mapper,
            norm,
            colormap,
            extent,
            is_subplot=is_subplot,
            cb_unit=cb_unit,
        )
    elif interpolated and isinstance(
        mapper.interpolator, (InterpolatorDelaunay, InterpolatorKNearestNeighbor)
    ):
        _plot_interpolated(
            ax,
            pixel_values,
            mapper,
            norm,
            colormap,
            extent,
            shape_native=interpolated_shape_native,
            is_subplot=is_subplot,
            cb_unit=cb_unit,
        )
    elif isinstance(
        mapper.interpolator, (InterpolatorDelaunay, InterpolatorKNearestNeighbor)
    ):
        _plot_delaunay(
            ax,
            pixel_values,
            mapper,
            norm,
            colormap,
            extent,
            is_subplot=is_subplot,
            cb_unit=cb_unit,
        )

    # --- overlays --------------------------------------------------------------
    # Over an ``imshow`` raster drawn with ``origin="lower"`` the (y, x)
    # overlays are reflected for display so they stay registered to the
    # pixels (PyAutoArray#565); every other case is left unchanged.
    origin_imshow = _conf_imshow_origin() if raster_extent is not None else None
    if origin_imshow == "lower":

        def _for_origin(yx):
            return _overlay_yx_for_origin(yx, raster_extent, origin_imshow)

        if lines is not None:
            lines = [
                (
                    _for_origin(np.asarray(line).reshape(-1, 2))
                    if line is not None and len(line) > 0
                    else line
                )
                for line in lines
            ]
        if regions is not None:
            regions = [
                (
                    None
                    if region is None
                    else (
                        _for_origin(region)
                        if isinstance(region, np.ndarray) and region.ndim == 2
                        else [_for_origin(polygon) for polygon in region]
                    )
                )
                for region in regions
            ]
        grid = _for_origin(numpy_grid(grid))

    if lines is not None:
        for i, line in enumerate(lines):
            if line is not None and len(line) > 0:
                line = np.asarray(line).reshape(-1, 2)
                color = (
                    line_colors[i]
                    if (line_colors is not None and i < len(line_colors))
                    else None
                )
                kw = {"linewidth": 1}
                if color is not None:
                    kw["color"] = color
                ax.plot(line[:, 1], line[:, 0], **kw)

    plot_regions(
        ax,
        regions=regions,
        region_colors=region_colors,
        region_alpha=region_alpha,
        region_labels=region_labels,
    )

    if grid is not None:
        ax.scatter(grid[:, 1], grid[:, 0], s=1, c="w", alpha=0.5)

    apply_extent(ax, extent, symbol_over_decimal=arcsec_after_decimal)

    apply_labels(
        ax,
        title=title,
        xlabel="" if is_subplot else xlabel,
        ylabel="" if is_subplot else ylabel,
        is_subplot=is_subplot,
    )

    if owns_figure:
        save_figure(
            fig,
            path=output_path or "",
            filename=output_filename,
            format=output_format,
        )


def _plot_rectangular(
    ax, pixel_values, mapper, norm, colormap, extent, is_subplot=False, cb_unit=None
):
    """Render a rectangular pixelization reconstruction onto *ax*.

    Uses ``imshow`` for uniform rectangular grids
    (``InterpolatorRectangularUniform``) and ``pcolormesh`` for non-uniform
    rectangular grids.  Both paths add a colorbar.

    Parameters
    ----------
    ax
        Matplotlib ``Axes`` to draw onto.
    pixel_values
        1-D array of reconstructed flux values, one per source pixel.
        ``None`` renders a zero-filled image.
    mapper
        Mapper object exposing ``interpolator``, ``mesh_geometry``, and
        (for uniform grids) ``pixel_scales`` / ``origin``.
    norm
        ``matplotlib.colors.Normalize`` (or ``LogNorm``) instance, or
        ``None`` for automatic scaling.
    colormap
        Matplotlib colormap name.
    extent
        ``[xmin, xmax, ymin, ymax]`` spatial extent; passed to ``imshow``.
    is_subplot
        When ``True`` uses ``labelsize_subplot`` from config for the colorbar
        tick labels (matches the behaviour of :func:`~autoarray.plot.array.plot_array`).

    Returns
    -------
    tuple or None
        The ``imshow`` extent when the uniform path draws a raster (so callers
        can register overlays to its origin), else ``None``.
    """
    from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
        InterpolatorRectangularUniform,
    )
    import numpy as np

    shape_native = mapper.mesh_geometry.shape

    if pixel_values is None:
        pixel_values = np.zeros(shape_native[0] * shape_native[1])

    xmin, xmax, ymin, ymax = extent
    x_range = abs(xmax - xmin)
    y_range = abs(ymax - ymin)
    box_aspect = (x_range / y_range) if y_range > 0 else 1.0
    ax.set_aspect(box_aspect, adjustable="box")

    from autoarray.plot.utils import _apply_colorbar

    if isinstance(mapper.interpolator, InterpolatorRectangularUniform):
        from autoarray.structures.arrays.uniform_2d import Array2D
        from autoarray.structures.arrays import array_2d_util

        solution_array_2d = array_2d_util.array_2d_native_from(
            array_2d_slim=pixel_values,
            mask_2d=np.full(fill_value=False, shape=shape_native),
        )
        pix_array = Array2D.no_mask(
            values=solution_array_2d,
            pixel_scales=mapper.mesh_geometry.pixel_scales,
            origin=mapper.mesh_geometry.origin,
        )
        im = ax.imshow(
            pix_array.native.array,
            cmap=colormap,
            norm=norm,
            extent=pix_array.geometry.extent,
            aspect="auto",
            origin=_conf_imshow_origin(),
        )
        _apply_colorbar(im, ax, cb_unit=cb_unit, is_subplot=is_subplot)
        return pix_array.geometry.extent
    else:
        y_edges, x_edges = mapper.mesh_geometry.edges_transformed.T
        Y, X = np.meshgrid(y_edges, x_edges, indexing="ij")
        im = ax.pcolormesh(
            X,
            Y,
            pixel_values.reshape(shape_native),
            shading="flat",
            norm=norm,
            cmap=colormap,
        )
        _apply_colorbar(im, ax, cb_unit=cb_unit, is_subplot=is_subplot)
        return None


def _plot_delaunay(
    ax, pixel_values, mapper, norm, colormap, extent, is_subplot=False, cb_unit=None
):
    """Render a Delaunay or KNN pixelization reconstruction onto *ax*.

    This is the **default** source-plane view, and it deliberately draws the
    mesh rather than a smooth image: ``ax.tripcolor`` with its default *flat*
    shading paints every triangle in one colour (the mean of its three vertex
    values). Small triangles are where the mesh spends its resolution and large
    ones where it does not, so the figure shows at a glance how finely each
    part of the source is being reconstructed -- information a smooth image
    hides.

    It is **not** the reconstruction the likelihood uses between vertices. The
    likelihood maps every data point to the source through the mapper's
    interpolant (barycentric within each triangle for ``Delaunay``, the kNN or
    natural-neighbour weights for those meshes), which varies smoothly across a
    triangle rather than being constant on it. To see that reconstruction, pass
    ``interpolated=True`` to :func:`plot_inversion_reconstruction` (see
    :func:`interpolated_reconstruction_from`); ``inversion_#.png`` always shows
    it in its last panel.

    The triangulation is taken from the mapper's own Delaunay tables
    (``mapper.interpolator.delaunay.simplices``, with the ``-1`` padded rows
    dropped) so the figure shows the mesh the inversion actually used rather
    than a triangulation matplotlib re-derives. Mappers with no such tables
    (kNN, mocks) fall back to matplotlib's own triangulation.

    A colorbar is attached after rendering.

    Parameters
    ----------
    ax
        Matplotlib ``Axes`` to draw onto.
    pixel_values
        1-D array of reconstructed flux values (one per source-plane pixel),
        or an autoarray object exposing a ``.array`` attribute.
    mapper
        Mapper object exposing ``source_plane_mesh_grid`` — an ``(N, 2)``
        array of ``(y, x)`` mesh-point coordinates.
    norm
        ``matplotlib.colors.Normalize`` (or ``LogNorm``) instance, or
        ``None`` for automatic scaling.
    colormap
        Matplotlib colormap name.
    extent
        ``[xmin, xmax, ymin, ymax]`` spatial extent; used to set the axes
        aspect ratio to match rectangular pixelization plots.
    is_subplot
        When ``True`` uses ``labelsize_subplot`` from config for the colorbar
        tick labels (matches the behaviour of :func:`~autoarray.plot.array.plot_array`).
    """
    xmin, xmax, ymin, ymax = extent
    x_range = abs(xmax - xmin)
    y_range = abs(ymax - ymin)
    box_aspect = (x_range / y_range) if y_range > 0 else 1.0
    ax.set_aspect(box_aspect, adjustable="box")

    mesh_grid = mapper.source_plane_mesh_grid

    if hasattr(mesh_grid, "array"):
        mesh_grid = mesh_grid.array

    mesh_grid = np.asarray(mesh_grid)

    x = np.asarray(mesh_grid[:, 1], dtype=float)
    y = np.asarray(mesh_grid[:, 0], dtype=float)

    if hasattr(pixel_values, "array"):
        vals = pixel_values.array
    else:
        vals = pixel_values

    # The mesh grid is (y,x) and tripcolor is given x=column 1, y=column 0; the
    # simplices index the same points in the same order, so they pass through
    # unchanged.
    triangles = None

    try:
        simplices = np.asarray(mapper.interpolator.delaunay.simplices)
        if simplices.ndim == 2 and simplices.shape[1] == 3:
            triangles = simplices[simplices[:, 0] >= 0]
    except AttributeError:
        triangles = None

    if triangles is not None and triangles.shape[0] > 0:
        tc = ax.tripcolor(x, y, triangles, vals, cmap=colormap, norm=norm)
    else:
        tc = ax.tripcolor(x, y, vals, cmap=colormap, norm=norm)
    from autoarray.plot.utils import _apply_colorbar

    _apply_colorbar(tc, ax, cb_unit=cb_unit, is_subplot=is_subplot)


def mapper_has_interpolated_view(mapper) -> bool:
    """
    Whether *mapper* has an interpolated view distinct from its pixel plot.

    ``True`` for irregular meshes (Delaunay, kNN, natural-neighbour), whose
    likelihood interpolates between mesh vertices; ``False`` for rectangular
    meshes (whose pixels already are the reconstruction) and for anything
    without an interpolator.
    """
    from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay

    return isinstance(getattr(mapper, "interpolator", None), InterpolatorDelaunay)


def _conf_interpolated_shape_native() -> Tuple[int, int]:
    try:
        from autonerves import conf

        pixels = int(
            conf.instance["visualize"]["general"]["inversion"]["interpolated_pixels"]
        )
    except Exception:
        pixels = 400
    return (pixels, pixels)


def interpolated_reconstruction_from(
    mapper,
    pixel_values,
    extent: Tuple[float, float, float, float],
    shape_native: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Evaluate an irregular-mesh reconstruction on a uniform grid using the
    mapper's own interpolant -- the reconstruction the likelihood uses.

    A fresh interpolator of the mapper's interpolator type is built with the
    same ``mesh`` and ``mesh_grid`` but with ``data_grid`` set to a uniform grid
    of pixel centres covering *extent* (no over-sampling). Its mappings and
    weights are exactly those the mapper would compute for a data point at each
    grid position, so the returned image equals that interpolator's
    ``mapping_matrix @ pixel_values``: barycentric interpolation for
    ``Delaunay``, the kNN weights for ``KNearestNeighbor`` /
    ``KNNBarycentric``, the Sibson weights for ``DelaunayNN``.

    Grid points outside the convex hull of the mesh vertices are returned as
    ``NaN`` (drawn transparent), so the interpolators' nearest-vertex fill
    beyond the mesh is not shown as source light. The cost is one
    interpolation of ``shape_native[0] * shape_native[1]`` points (a few
    neighbours each) and a triangulation of the mesh; no dense mapping matrix
    is formed.

    Parameters
    ----------
    mapper
        A mapper whose ``interpolator`` is an irregular-mesh interpolator.
    pixel_values
        The reconstruction, one value per mesh vertex.
    extent
        ``(x_min, x_max, y_min, y_max)`` covered by the grid (pixel edges).
    shape_native
        ``(y, x)`` shape of the grid; ``None`` reads the visualize config
        (default 400 x 400).

    Returns
    -------
    np.ndarray
        ``shape_native`` image, row 0 at ``y_max``.
    """
    from scipy.spatial import Delaunay

    from autoarray.structures.grids.uniform_2d import Grid2D

    if shape_native is None:
        shape_native = _conf_interpolated_shape_native()

    shape_native = (int(shape_native[0]), int(shape_native[1]))

    x_min, x_max, y_min, y_max = (float(e) for e in extent)
    pixel_scales = (
        (y_max - y_min) / shape_native[0],
        (x_max - x_min) / shape_native[1],
    )
    grid = Grid2D.uniform(
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=(0.5 * (y_max + y_min), 0.5 * (x_max + x_min)),
        over_sample_size=1,
        # The grid must cover `extent` exactly; small-dataset mode would shrink
        # its shape without rescaling its pixels.
        respect_small_datasets=False,
    )

    interpolator = mapper.interpolator
    plot_interpolator = type(interpolator)(
        mesh=interpolator.mesh,
        mesh_grid=interpolator.mesh_grid,
        data_grid=grid,
        adapt_data=interpolator.adapt_data,
    )

    mappings, sizes, weights = plot_interpolator._mappings_sizes_weights

    mappings = np.asarray(mappings)
    sizes = np.asarray(sizes)
    weights = np.asarray(weights, dtype=float)

    if hasattr(pixel_values, "array"):
        pixel_values = pixel_values.array
    values = np.asarray(pixel_values, dtype=float)

    used = np.arange(mappings.shape[1])[None, :] < sizes[:, None]
    safe_mappings = np.where(used & (mappings >= 0), mappings, 0)
    image = np.sum(np.where(used, weights * values[safe_mappings], 0.0), axis=1)

    mesh_grid = np.asarray(
        getattr(interpolator.mesh_grid, "array", interpolator.mesh_grid), dtype=float
    )
    grid_points = np.asarray(grid.array, dtype=float)
    try:
        outside = Delaunay(mesh_grid).find_simplex(grid_points) < 0
        image = np.where(outside, np.nan, image)
    except Exception:
        # A degenerate (e.g. collinear) mesh has no hull to mask with.
        pass

    return image.reshape(shape_native)


def _plot_interpolated(
    ax,
    pixel_values,
    mapper,
    norm,
    colormap,
    extent,
    shape_native=None,
    is_subplot=False,
    cb_unit=None,
):
    """Render an irregular-mesh reconstruction as the interpolant the likelihood uses.

    The image from :func:`interpolated_reconstruction_from` is drawn with
    ``imshow`` over *extent*, in data coordinates (row 0 at ``y_max``), so
    overlays drawn in ``(y, x)`` arcsec need no reflection and the panel lines
    up with the flat-triangle view of the same mapper. The colour norm is the
    one built from the vertex values, so both views share a colour scale; under
    a log norm values below its floor are clipped up to it, as
    :func:`~autoarray.plot.utils.norm_from` does for the vertex values.
    """
    from matplotlib.colors import LogNorm

    from autoarray.plot.utils import _apply_colorbar

    xmin, xmax, ymin, ymax = extent
    x_range = abs(xmax - xmin)
    y_range = abs(ymax - ymin)
    box_aspect = (x_range / y_range) if y_range > 0 else 1.0
    ax.set_aspect(box_aspect, adjustable="box")

    image = interpolated_reconstruction_from(
        mapper=mapper,
        pixel_values=pixel_values,
        extent=extent,
        shape_native=shape_native,
    )

    if isinstance(norm, LogNorm) and norm.vmin is not None:
        image = np.where(np.isnan(image), image, np.maximum(image, norm.vmin))

    im = ax.imshow(
        image,
        cmap=colormap,
        norm=norm,
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    _apply_colorbar(im, ax, cb_unit=cb_unit, is_subplot=is_subplot)
