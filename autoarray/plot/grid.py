"""
Standalone function for plotting a 2D grid of (y, x) coordinates.

This replaces the ``MatPlot2D.plot_grid`` / ``MatWrap`` system.
"""

from typing import Iterable, List, Optional, Tuple

import numpy as np

from autoarray.plot.utils import (
    subplots,
    get_cmap,
    apply_extent,
    apply_labels,
    conf_figsize,
    save_figure,
    numpy_lines,
)


def plot_grid(
    grid,
    ax=None,
    # --- errors -----------------------------------------------------------------
    y_errors: Optional[np.ndarray] = None,
    x_errors: Optional[np.ndarray] = None,
    # --- overlays ---------------------------------------------------------------
    lines=None,
    color_array: Optional[np.ndarray] = None,
    indexes: Optional[List] = None,
    # --- cosmetics --------------------------------------------------------------
    title: str = "",
    xlabel: str = 'x (")',
    ylabel: str = 'y (")',
    colormap: Optional[str] = None,
    buffer: float = 0.1,
    extent: Optional[Tuple[float, float, float, float]] = None,
    force_symmetric_extent: bool = True,
    # --- figure control (used only when ax is None) -----------------------------
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    output_filename: str = "grid",
    output_format: str = None,
) -> None:
    """
    Plot a 2D grid of ``(y, x)`` coordinates as a scatter plot.

    This is the direct-matplotlib replacement for ``MatPlot2D.plot_grid``.

    Parameters
    ----------
    grid
        Array of shape ``(N, 2)``; column 0 is *y*, column 1 is *x*.
    ax
        Existing ``Axes`` to draw onto.  ``None`` creates a new figure.
    y_errors, x_errors
        Per-point error values; when provided ``plt.errorbar`` is used.
    lines
        Iterable of ``(N, 2)`` arrays (y, x columns) drawn as lines.
    color_array
        1D array of scalar values for colouring each point; triggers a
        colorbar.
    title
        Figure title.
    xlabel, ylabel
        Axis labels.
    colormap
        Matplotlib colormap name.
    buffer
        Fractional padding for the auto-computed extent.  The grid's
        ``extent_with_buffer_from`` method is called when *extent* is
        ``None`` and the grid object exposes that method.
    extent
        Manual axis limits ``[xmin, xmax, ymin, ymax]``.  Auto-computed
        when ``None``.
    force_symmetric_extent
        When ``True`` (and *extent* is auto-computed) the limits are made
        symmetric about the origin so the plot is centred.
    figsize
        Figure size in inches ``(width, height)``.
    output_path
        Directory for saving.  Empty string / ``None`` triggers
        ``plt.show()``.
    output_filename
        Base file name without extension.
    output_format
        File format, e.g. ``"png"``.
    """
    # --- autoarray extraction --------------------------------------------------
    # Compute extent before converting to numpy so grid methods are available.
    if extent is None:
        try:
            extent = grid.extent_with_buffer_from(buffer=buffer)
        except AttributeError:
            pass  # computed from numpy values below

    if hasattr(grid, "array"):
        grid = np.array(grid.array)
    else:
        grid = np.asarray(grid)

    lines = numpy_lines(lines)

    if colormap is None:
        from autoarray.plot.utils import _default_colormap
        colormap = _default_colormap()

    owns_figure = ax is None
    if owns_figure:
        figsize = figsize or conf_figsize("figures")
        fig, ax = subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- scatter / errorbar ----------------------------------------------------
    if color_array is not None:
        cmap = get_cmap(colormap)
        colors = cmap((color_array - color_array.min()) / (np.ptp(color_array) or 1))

        if y_errors is None and x_errors is None:
            sc = ax.scatter(grid[:, 1], grid[:, 0], s=1, c=color_array, cmap=colormap)
        else:
            sc = ax.scatter(grid[:, 1], grid[:, 0], s=1, c=color_array, cmap=colormap)
            ax.errorbar(
                grid[:, 1],
                grid[:, 0],
                yerr=y_errors,
                xerr=x_errors,
                fmt="none",
                ecolor=colors,
            )

        from autoarray.plot.utils import _apply_colorbar
        _apply_colorbar(sc, ax)
    else:
        if y_errors is None and x_errors is None:
            ax.scatter(grid[:, 1], grid[:, 0], s=1, c="k")
        else:
            ax.errorbar(
                grid[:, 1],
                grid[:, 0],
                yerr=y_errors,
                xerr=x_errors,
                fmt="o",
                markersize=2,
                color="k",
            )

    # --- line overlays ---------------------------------------------------------
    if lines is not None:
        for line in lines:
            if line is not None and len(line) > 0:
                ax.plot(line[:, 1], line[:, 0], linewidth=1)

    # --- labels ----------------------------------------------------------------
    apply_labels(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    # --- extent ----------------------------------------------------------------
    if extent is None:
        y_vals = grid[:, 0]
        x_vals = grid[:, 1]
        extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]

    if indexes is not None:
        colors = ["r", "g", "b", "m", "c", "y"]
        for i, idx_list in enumerate(indexes):
            ax.scatter(
                grid[idx_list, 1],
                grid[idx_list, 0],
                s=10,
                c=colors[i % len(colors)],
                zorder=5,
            )

    if force_symmetric_extent and extent is not None:
        x_abs = max(abs(extent[0]), abs(extent[1]))
        y_abs = max(abs(extent[2]), abs(extent[3]))
        extent = [-x_abs, x_abs, -y_abs, y_abs]

    apply_extent(ax, extent)

    # --- output ----------------------------------------------------------------
    if owns_figure:
        save_figure(
            fig,
            path=output_path or "",
            filename=output_filename,
            format=output_format,
        )
