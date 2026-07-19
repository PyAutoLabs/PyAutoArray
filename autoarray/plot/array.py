"""
Standalone function for plotting a 2D array (image) directly with matplotlib.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from autoarray.plot.utils import (
    subplots,
    apply_extent,
    apply_labels,
    conf_figsize,
    save_figure,
    zoom_array,
    auto_mask_edge,
    numpy_grid,
    numpy_lines,
    numpy_positions,
    _apply_contours,
    _conf_imshow_origin,
)

_zoom_array_2d = zoom_array
_mask_edge_coords = auto_mask_edge


def plot_array(
    array,
    ax=None,
    # --- spatial metadata -------------------------------------------------------
    extent: Optional[Tuple[float, float, float, float]] = None,
    # --- overlays ---------------------------------------------------------------
    mask: Optional[np.ndarray] = None,
    border=None,
    origin=None,
    grid=None,
    mesh_grid=None,
    positions=None,
    lines=None,
    vector_yx: Optional[np.ndarray] = None,
    array_overlay=None,
    patches: Optional[List] = None,
    fill_region: Optional[List] = None,
    contours: Optional[int] = None,
    # --- cosmetics --------------------------------------------------------------
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    colormap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_log10: bool = False,
    cb_unit: Optional[str] = None,
    line_colors: Optional[List] = None,
    origin_imshow: Optional[str] = None,
    # --- figure control (used only when ax is None) -----------------------------
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    output_filename: str = "array",
    output_format: str = None,
) -> None:
    """
    Plot a 2D array (image) using ``plt.imshow``.

    Parameters
    ----------
    array
        2D numpy array of pixel values.
    ax
        Existing matplotlib ``Axes`` to draw onto.  If ``None`` a new figure
        is created and saved / shown according to *output_path*.
    extent
        ``[xmin, xmax, ymin, ymax]`` spatial extent in data coordinates.
    mask
        Array of shape ``(N, 2)`` with ``(y, x)`` coordinates of masked
        pixels to overlay as black dots (auto-derived from array.mask by caller).
    border
        Array of shape ``(N, 2)`` with ``(y, x)`` border pixel coordinates.
    origin
        ``(y, x)`` origin coordinate(s) to scatter as a marker.
    grid
        Array of shape ``(N, 2)`` with ``(y, x)`` coordinates to scatter.
    mesh_grid
        Array of shape ``(N, 2)`` mesh grid coordinates to scatter.
    positions
        List of ``(N, 2)`` arrays; each is scattered as a distinct group.
    lines
        List of ``(N, 2)`` arrays with ``(y, x)`` columns to plot as lines.
    vector_yx
        Array of shape ``(N, 4)`` — ``(y, x, vy, vx)`` — plotted as quiver.
    array_overlay
        A second 2D array rendered on top of *array* with partial alpha.
    patches
        List of matplotlib ``Patch`` objects to draw over the image.
    fill_region
        List of two arrays ``[y1_arr, y2_arr]`` passed to ``ax.fill_between``.
    title
        Figure title string.
    xlabel, ylabel
        Axis label strings.
    colormap
        Matplotlib colormap name.
    vmin, vmax
        Explicit color scale limits.
    use_log10
        When ``True`` a ``LogNorm`` is applied.
    origin_imshow
        Passed directly to ``imshow`` (``"upper"`` or ``"lower"``).
    figsize
        Figure size in inches.
    output_path
        Directory to save the figure.  When empty / ``None`` ``plt.show()``
        is called instead.
    output_filename
        Base file name (without extension).
    output_format
        File format, e.g. ``"png"``.
    """
    if origin_imshow is None:
        origin_imshow = _conf_imshow_origin()

    # --- autoarray extraction --------------------------------------------------
    array = zoom_array(array)
    try:
        if extent is None:
            extent = array.geometry.extent
        if mask is None:
            mask = auto_mask_edge(array)
        array = array.native.array
    except AttributeError:
        array = np.asarray(array)

    if array is None or array.size == 0:
        return

    is_rgb = array.ndim == 3 and array.shape[2] in (3, 4)

    if colormap is None:
        from autoarray.plot.utils import _default_colormap
        colormap = _default_colormap()

    # convert overlay params (safe for None and already-numpy inputs)
    border = numpy_grid(border)
    origin = numpy_grid(origin)
    grid = numpy_grid(grid)
    mesh_grid = numpy_grid(mesh_grid)
    positions = numpy_positions(positions)
    lines = numpy_lines(lines)
    if array_overlay is not None:
        try:
            array_overlay = array_overlay.native.array
        except AttributeError:
            array_overlay = np.asarray(array_overlay)

    owns_figure = ax is None
    if owns_figure:
        figsize = figsize or conf_figsize("figures")
        fig, ax = subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- colour normalisation --------------------------------------------------
    if use_log10:
        try:
            from autonerves import conf as _conf

            log10_min = _conf.instance["visualize"]["general"]["general"][
                "log10_min_value"
            ]
        except Exception:
            log10_min = 1.0e-4
        clipped = np.clip(array, log10_min, None)
        vmin_log = vmin if (vmin is not None and np.isfinite(vmin)) else log10_min
        if vmax is not None and np.isfinite(vmax):
            vmax_log = vmax
        else:
            with np.errstate(all="ignore"):
                vmax_log = np.nanmax(clipped)
        if not np.isfinite(vmax_log) or vmax_log <= vmin_log:
            vmax_log = vmin_log * 10.0
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin_log, vmax=vmax_log)
    elif vmin is not None or vmax is not None:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Compute the axes-box aspect ratio from the data extent so that the
    # physical cell is correctly shaped and tight_layout has no whitespace
    # to absorb.  This reproduces the old "square" subplot behaviour where
    # ratio = x_range / y_range was passed to plt.subplot(aspect=ratio).
    if extent is not None:
        x_range = abs(extent[1] - extent[0])
        y_range = abs(extent[3] - extent[2])
        _box_aspect = (x_range / y_range) if y_range > 0 else 1.0
    else:
        h, w = array.shape[:2]
        _box_aspect = (w / h) if h > 0 else 1.0

    if is_rgb:
        im = ax.imshow(
            array,
            extent=extent,
            aspect="auto",
            origin=origin_imshow,
        )
    else:
        im = ax.imshow(
            array,
            cmap=colormap,
            norm=norm,
            extent=extent,
            aspect="auto",  # image fills the axes box; box shape set below
            origin=origin_imshow,
        )

    # Shape the axes box to match the data so there is no surrounding
    # whitespace when the panel is embedded in a subplot grid.
    ax.set_aspect(_box_aspect, adjustable="box")

    if not is_rgb:
        from autoarray.plot.utils import _apply_colorbar
        _apply_colorbar(im, ax, cb_unit=cb_unit, is_subplot=not owns_figure)

    # --- overlays --------------------------------------------------------------
    if array_overlay is not None:
        ax.imshow(
            array_overlay,
            cmap="Greys",
            alpha=0.5,
            extent=extent,
            aspect="auto",
            origin=origin_imshow,
        )

    if mask is not None:
        ax.scatter(mask[:, 1], mask[:, 0], s=1, c="k")

    if border is not None:
        ax.scatter(border[:, 1], border[:, 0], s=1, c="b")

    if origin is not None:
        origin_arr = np.asarray(origin)
        if origin_arr.ndim == 1:
            origin_arr = origin_arr[np.newaxis, :]
        ax.scatter(
            origin_arr[:, 1], origin_arr[:, 0], s=20, c="r", marker="x", zorder=6
        )

    if grid is not None:
        ax.scatter(grid[:, 1], grid[:, 0], s=1, c="k")

    if mesh_grid is not None:
        ax.scatter(mesh_grid[:, 1], mesh_grid[:, 0], s=1, c="w", alpha=0.5)

    if positions is not None:
        colors = ["k", "g", "b", "m", "c", "y"]
        for i, pos in enumerate(positions):
            pos = np.asarray(pos).reshape(-1, 2)
            ax.scatter(pos[:, 1], pos[:, 0], s=20, c=colors[i % len(colors)], zorder=5)


    if lines is not None:
        for i, line in enumerate(lines):
            if line is not None and len(line) > 0:
                line = np.asarray(line).reshape(-1, 2)
                color = line_colors[i] if (line_colors is not None and i < len(line_colors)) else None
                kw = {"linewidth": 1}
                if color is not None:
                    kw["color"] = color
                ax.plot(line[:, 1], line[:, 0], **kw)

    if vector_yx is not None:
        ax.quiver(
            vector_yx[:, 1],
            vector_yx[:, 0],
            vector_yx[:, 3],
            vector_yx[:, 2],
        )

    if patches is not None:
        for patch in patches:
            import copy

            ax.add_patch(copy.copy(patch))

    if fill_region is not None:
        y1, y2 = fill_region[0], fill_region[1]
        x_fill = np.arange(len(y1))
        ax.fill_between(x_fill, y1, y2, alpha=0.3)

    # Contours: auto-enabled for log10 plots; explicit int enables linear contours.
    if use_log10 or (contours is not None and contours > 0):
        _apply_contours(
            ax, array, extent,
            use_log10=use_log10,
            n=contours if (contours is not None and contours > 0) else None,
        )

    # --- labels / ticks --------------------------------------------------------
    apply_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel, is_subplot=not owns_figure)

    if extent is not None:
        apply_extent(ax, extent)

    # --- output ----------------------------------------------------------------
    if owns_figure:
        save_figure(
            fig,
            path=output_path or "",
            filename=output_filename,
            format=output_format,
        )
