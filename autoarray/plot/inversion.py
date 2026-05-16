"""
Standalone functions for plotting inversion / pixelization reconstructions.

Replaces the inversion-specific paths in ``MatPlot2D.plot_mapper``.
"""

from typing import List, Optional, Tuple

import numpy as np
from matplotlib.colors import LogNorm, Normalize

from autoarray.plot.utils import subplots, apply_extent, apply_labels, conf_figsize, save_figure, _conf_imshow_origin


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
    zoom_to_brightest: bool = True,
    # --- overlays ---------------------------------------------------------------
    lines: Optional[List[np.ndarray]] = None,
    line_colors: Optional[List] = None,
    grid: Optional[np.ndarray] = None,
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
    zoom_to_brightest
        Pass through to ``mapper.extent_from``.
    lines
        Line overlays (e.g. critical curves).
    grid
        Scatter overlay (e.g. data-plane grid).
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
    if use_log10:
        vmin_log = vmin if (vmin is not None and np.isfinite(vmin)) else 1e-4
        if vmax is not None and np.isfinite(vmax):
            vmax_log = vmax
        elif pixel_values is not None:
            with np.errstate(all="ignore"):
                vmax_log = float(np.nanmax(np.asarray(pixel_values)))
            if not np.isfinite(vmax_log) or vmax_log <= vmin_log:
                vmax_log = vmin_log * 10.0
        else:
            vmax_log = vmin_log * 10.0
        norm = LogNorm(vmin=vmin_log, vmax=vmax_log)
    elif vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    extent = mapper.extent_from(
        values=pixel_values, zoom_to_brightest=zoom_to_brightest
    )

    is_subplot = not owns_figure

    if isinstance(
        mapper.interpolator, (InterpolatorRectangular, InterpolatorRectangularUniform)
    ):
        _plot_rectangular(ax, pixel_values, mapper, norm, colormap, extent, is_subplot=is_subplot)
    elif isinstance(
        mapper.interpolator, (InterpolatorDelaunay, InterpolatorKNearestNeighbor)
    ):
        _plot_delaunay(ax, pixel_values, mapper, norm, colormap, extent, is_subplot=is_subplot)

    # --- overlays --------------------------------------------------------------
    if lines is not None:
        for i, line in enumerate(lines):
            if line is not None and len(line) > 0:
                line = np.asarray(line).reshape(-1, 2)
                color = line_colors[i] if (line_colors is not None and i < len(line_colors)) else None
                kw = {"linewidth": 1}
                if color is not None:
                    kw["color"] = color
                ax.plot(line[:, 1], line[:, 0], **kw)

    if grid is not None:
        ax.scatter(grid[:, 1], grid[:, 0], s=1, c="w", alpha=0.5)

    apply_extent(ax, extent)

    apply_labels(ax, title=title, xlabel="" if is_subplot else xlabel, ylabel="" if is_subplot else ylabel, is_subplot=is_subplot)

    if owns_figure:
        save_figure(
            fig,
            path=output_path or "",
            filename=output_filename,
            format=output_format,
        )


def _plot_rectangular(ax, pixel_values, mapper, norm, colormap, extent, is_subplot=False):
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
        _apply_colorbar(im, ax, is_subplot=is_subplot)
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
        _apply_colorbar(im, ax, is_subplot=is_subplot)


def _plot_delaunay(ax, pixel_values, mapper, norm, colormap, extent, is_subplot=False):
    """Render a Delaunay or KNN pixelization reconstruction onto *ax*.

    Uses ``ax.tripcolor`` with Gouraud shading so that the reconstructed
    flux is interpolated smoothly across the triangulated source-plane mesh.
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

    tc = ax.tripcolor(x, y, vals, cmap=colormap, norm=norm)
    from autoarray.plot.utils import _apply_colorbar
    _apply_colorbar(tc, ax, is_subplot=is_subplot)
