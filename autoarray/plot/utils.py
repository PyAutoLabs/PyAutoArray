"""
Shared utilities for the direct-matplotlib plot functions.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


_FAST_PLOTS = os.environ.get("PYAUTO_FAST_PLOTS") == "1"


def subplots(*args, **kwargs):
    """Lazy wrapper around ``plt.subplots`` that defers the matplotlib import."""
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


def get_cmap(name):
    """Lazy wrapper around ``plt.get_cmap`` that defers the matplotlib import."""
    import matplotlib.pyplot as plt

    return plt.get_cmap(name)


def tight_layout():
    """Call ``plt.tight_layout()`` unless fast-plot mode is active.

    When ``PYAUTO_FAST_PLOTS=1`` the expensive layout-optimisation pass
    is skipped.  All figure creation, data computation, and rendering
    still execute — only the final spacing adjustment is bypassed.
    """
    if _FAST_PLOTS:
        return

    import matplotlib.pyplot as plt

    plt.tight_layout()


# ---------------------------------------------------------------------------
# autoarray → numpy conversion helpers (used by high-level plot functions)
# ---------------------------------------------------------------------------


def auto_mask_edge(array) -> Optional[np.ndarray]:
    """Return the edge-pixel ``(y, x)`` coordinates of an autoarray mask.

    Used to overlay the mask boundary on ``plot_array`` images.  If *array*
    has no ``mask`` attribute, or the mask is fully unmasked, ``None`` is
    returned so no overlay is drawn.

    Parameters
    ----------
    array
        An autoarray ``Array2D`` (or any object with a ``.mask`` attribute
        that exposes ``.derive_grid.edge.array``).

    Returns
    -------
    numpy.ndarray or None
        Shape ``(N, 2)`` float array of ``(y, x)`` edge coordinates, or
        ``None`` when the array is unmasked or has no mask.
    """
    try:
        if not array.mask.is_all_false:
            return np.array(array.mask.derive_grid.edge.array)
    except AttributeError:
        pass
    return None


def zoom_array(array):
    """Crop *array* around its mask when ``zoom_around_mask`` is enabled in config.

    Reads ``visualize/general/general/zoom_around_mask`` from the autoconf
    configuration.  When the flag is ``True`` and *array* carries a non-trivial
    mask the array is cropped via ``Zoom2D`` so that downstream ``imshow``
    calls fill the axes without empty black borders.

    Parameters
    ----------
    array
        An autoarray ``Array2D`` (or any object).  Plain numpy arrays are
        returned unchanged.

    Returns
    -------
    array
        The (potentially cropped) array.  If the config flag is ``False``, or
        *array* has no mask / the mask is all-``False``, the input is returned
        unmodified.
    """
    try:
        from autoconf import conf

        zoom_around_mask = conf.instance["visualize"]["general"]["general"][
            "zoom_around_mask"
        ]
    except Exception:
        zoom_around_mask = False

    if zoom_around_mask and hasattr(array, "mask") and hasattr(array.mask, "is_all_false") and not array.mask.is_all_false:
        from autoarray.mask.derive.zoom_2d import Zoom2D

        return Zoom2D(mask=array.mask).array_2d_from(array=array, buffer=1)
    return array


def numpy_grid(grid) -> Optional[np.ndarray]:
    """Convert a grid-like object to a plain ``(N, 2)`` numpy array, or ``None``.

    Accepts autoarray ``Grid2D`` / ``Grid2DIrregular`` objects (via their
    ``.array`` attribute) as well as bare numpy arrays.  ``None`` inputs are
    passed through so callers can use this as a safe no-op.

    Parameters
    ----------
    grid
        An autoarray grid, a ``(N, 2)`` numpy array, or ``None``.

    Returns
    -------
    numpy.ndarray or None
        Plain ``(N, 2)`` float array with ``(y, x)`` columns, or ``None``.
    """
    if grid is None:
        return None
    try:
        return np.array(grid.array if hasattr(grid, "array") else grid)
    except Exception:
        return None


def numpy_lines(lines) -> Optional[List[np.ndarray]]:
    """Convert a collection of lines to a list of ``(N, 2)`` numpy arrays.

    Accepts autoarray ``Grid2DIrregular`` objects or any iterable of
    ``(N, 2)`` array-like sequences.  Each element is converted to a plain
    numpy array; elements that cannot be converted are silently skipped.

    Parameters
    ----------
    lines
        An autoarray grid collection, a list of ``(N, 2)`` arrays, or ``None``.

    Returns
    -------
    list of numpy.ndarray or None
        List of ``(N, 2)`` float arrays (``y`` column 0, ``x`` column 1), or
        ``None`` when *lines* is ``None`` or no valid lines are found.
    """
    if lines is None:
        return None
    result = []
    try:
        for line in lines:
            try:
                arr = np.array(line.array if hasattr(line, "array") else line)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    result.append(arr)
            except Exception:
                pass
    except TypeError:
        pass
    return result or None


def numpy_positions(positions) -> Optional[List[np.ndarray]]:
    """Convert a positions object to a list of ``(N, 2)`` numpy arrays.

    Positions can be a single ``Grid2DIrregular`` (treated as one group),
    a plain ``(N, 2)`` array (treated as one group), or a list of such
    objects (each becomes one group, scatter-plotted in a distinct colour).

    Parameters
    ----------
    positions
        An autoarray ``Grid2DIrregular``, a ``(N, 2)`` numpy array, a list
        of the above, or ``None``.

    Returns
    -------
    list of numpy.ndarray or None
        Each element is a ``(N, 2)`` array of ``(y, x)`` coordinates
        representing one group of positions, or ``None`` when *positions*
        is ``None`` or cannot be converted.
    """
    if positions is None:
        return None
    try:
        arr = np.array(positions.array if hasattr(positions, "array") else positions)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [arr]
    except Exception:
        pass
    if isinstance(positions, list):
        result = []
        for p in positions:
            try:
                result.append(np.array(p.array if hasattr(p, "array") else p))
            except Exception:
                pass
        return result or None
    return None


def symmetric_vmin_vmax(array):
    """Return ``(-abs_max, abs_max)`` colour limits for a symmetric residual colormap.

    Computes the maximum absolute value of *array* and returns symmetric limits
    so that zero maps to the centre of the colormap.  Typically applied to
    residual maps and normalised residual maps.

    Parameters
    ----------
    array
        An autoarray ``Array2D`` (uses ``.native.array``) or a plain numpy
        array.

    Returns
    -------
    tuple of (float, float) or (None, None)
        ``(vmin, vmax)`` where ``vmin == -vmax == -abs_max``.  Returns
        ``(None, None)`` if the computation fails (e.g. all-NaN input).
    """
    try:
        arr = array.native.array if hasattr(array, "native") else np.asarray(array)
        abs_max = float(np.nanmax(np.abs(arr)))
        return -abs_max, abs_max
    except Exception:
        return None, None


def symmetric_cmap_from(array, symmetric_value=None):
    """Return a matplotlib ``Normalize`` centred on zero for a symmetric colormap.

    Parameters
    ----------
    array
        The data array (autoarray or numpy).  Used to compute ``abs_max`` when
        *symmetric_value* is not provided.
    symmetric_value
        If given, fix the half-range to this value (``vmin=-symmetric_value``,
        ``vmax=+symmetric_value``).

    Returns
    -------
    matplotlib.colors.Normalize or None
    """
    import matplotlib.colors as colors

    if symmetric_value is not None:
        abs_max = float(symmetric_value)
    else:
        vmin, vmax = symmetric_vmin_vmax(array)
        if vmin is None:
            return None
        abs_max = max(abs(vmin), abs(vmax))

    return colors.Normalize(vmin=-abs_max, vmax=abs_max)


def set_with_color_values(ax, cmap, color_values, norm=None):
    """Attach a colorbar to *ax* driven by *color_values* rather than a plotted artist.

    Useful for Delaunay mapper visualisation where ``ax.tripcolor`` already draws
    the mesh but we need a separate colorbar tied to specific solution values.

    Parameters
    ----------
    ax
        The matplotlib axes to attach the colorbar to.
    cmap
        A matplotlib colormap name or object.
    color_values
        The 1-D array of values that define the colorbar range.
    norm
        A ``matplotlib.colors.Normalize`` instance.  If ``None`` a default
        ``Normalize(vmin, vmax)`` is created from *color_values*.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if norm is None:
        arr = np.asarray(color_values)
        norm = mcolors.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(color_values)
    return _apply_colorbar(mappable, ax)


def _output_mode_save(fig, filename):
    import matplotlib.pyplot as plt

    """If ``PYAUTO_OUTPUT_MODE=1``, save *fig* to a numbered file in
    ``./output_mode/<script_name>/`` and return ``True``.  Otherwise return
    ``False`` so the caller can proceed with normal saving.

    The counter is stored as a function attribute to avoid a global variable.
    """
    if os.environ.get("PYAUTO_OUTPUT_MODE") != "1":
        return False

    import sys

    script_name = Path(sys.argv[0]).stem
    output_path = str(Path(os.getcwd()) / "output_mode" / script_name)
    os.makedirs(output_path, exist_ok=True)

    count = getattr(_output_mode_save, "_count", -1) + 1
    _output_mode_save._count = count

    try:
        fig.savefig(
            str(Path(output_path) / f"{count}_{filename}.png"),
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.1,
        )
    except Exception as exc:
        logger.warning(f"output_mode: could not save {count}_{filename}.png: {exc}")

    plt.close(fig)
    return True


def subplot_save(fig, output_path, output_filename, output_format=None):
    """Save a subplot figure to disk, or display it, then close it.

    All ``subplot_*`` functions call this as their final step.  When
    *output_format* is ``"show"`` (the config default) or *output_path*
    is empty, ``plt.show()`` is called; otherwise the figure is written to
    ``<output_path>/<output_filename>.<output_format>``.
    ``plt.close(fig)`` is always called to release memory.

    Parameters
    ----------
    fig
        The matplotlib ``Figure`` to save or show.
    output_path
        Directory to write the file.  Creates the directory if needed.
        ``None`` or an empty string causes ``plt.show()`` to be called.
    output_filename
        Base file name without extension.
    output_format
        File format string, e.g. ``"png"`` or ``"pdf"``.  ``"show"``
        displays the figure interactively.  ``None`` reads from config.
    """
    import matplotlib.pyplot as plt

    if output_format is None:
        output_format = _conf_output_format()

    if _output_mode_save(fig, output_filename):
        return

    if _FAST_PLOTS:
        plt.close(fig)
        return

    if output_format == "show" or not output_path:
        plt.show()
    else:
        os.makedirs(output_path, exist_ok=True)
        try:
            fig.savefig(
                str(Path(output_path) / f"{output_filename}.{output_format}"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
        except Exception as exc:
            logger.warning(
                f"subplot_save: could not save {output_filename}.{output_format}: {exc}"
            )
    plt.close(fig)


def conf_mat_plot_fontsize(section: str, default: int) -> int:
    """Read a font size from the ``mat_plot`` section of ``visualize/general.yaml``.

    Parameters
    ----------
    section
        Sub-key inside ``mat_plot``, e.g. ``"title"``, ``"xlabel"``,
        ``"ylabel"``, ``"xticks"``, or ``"yticks"``.
    default
        Value returned when the config key is absent or unreadable.

    Returns
    -------
    int
        The configured font size.
    """
    try:
        from autoconf import conf

        return int(
            conf.instance["visualize"]["general"]["mat_plot"][section]["fontsize"]
        )
    except Exception:
        return default


def _parse_figsize(raw) -> Tuple[int, int]:
    """Convert *raw* (a tuple/list or a string like ``"(7, 7)"``) to a 2-tuple."""
    if isinstance(raw, (tuple, list)):
        return tuple(raw)
    import ast

    return tuple(ast.literal_eval(str(raw)))


def conf_figsize(context: str = "figures") -> Tuple[int, int]:
    """
    Read figsize from ``visualize/general.yaml`` for the given context.

    For single-panel figures the value is taken from
    ``mat_plot/figure/figsize``; the *context* argument is kept for
    backward compatibility with subplot callers that pass ``"subplots"``.

    Parameters
    ----------
    context
        ``"figures"`` (single-panel) or ``"subplots"`` (multi-panel).
    """
    try:
        from autoconf import conf

        if context == "figures":
            raw = conf.instance["visualize"]["general"]["mat_plot"]["figure"]["figsize"]
            return _parse_figsize(raw)
        return tuple(conf.instance["visualize"]["general"][context]["figsize"])
    except Exception:
        return (7, 7) if context == "figures" else (19, 16)


def conf_subplot_figsize(rows: int, cols: int) -> Tuple[int, int]:
    """Compute figsize for a subplot grid from config.

    Reads ``mat_plot/figure/subplot_shape_to_figsize_factor`` from
    ``visualize/general.yaml`` (default ``(6, 6)``) and returns
    ``(cols * fx, rows * fy)``.
    """
    try:
        from autoconf import conf

        raw = conf.instance["visualize"]["general"]["mat_plot"]["figure"][
            "subplot_shape_to_figsize_factor"
        ]
        fx, fy = _parse_figsize(raw)
    except Exception:
        fx, fy = 6, 6
    return (cols * fx, rows * fy)


def apply_labels(
    ax,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    is_subplot: bool = False,
) -> None:
    """Apply title, axis labels, and tick font sizes to *ax* from config.

    Reads font sizes from the ``mat_plot`` section of
    ``visualize/general.yaml``.  When *is_subplot* is ``True``, reads
    ``*_subplot`` keys (defaulting to the single-figure values / 10 for ticks).
    """
    if is_subplot:
        title_fs = conf_mat_plot_fontsize("title_subplot", default=20)
        xlabel_fs = conf_mat_plot_fontsize("xlabel_subplot", default=conf_mat_plot_fontsize("xlabel", default=14))
        ylabel_fs = conf_mat_plot_fontsize("ylabel_subplot", default=conf_mat_plot_fontsize("ylabel", default=14))
        xticks_fs = conf_mat_plot_fontsize("xticks_subplot", default=18)
        yticks_fs = conf_mat_plot_fontsize("yticks_subplot", default=18)
    else:
        title_fs = conf_mat_plot_fontsize("title", default=24)
        xlabel_fs = conf_mat_plot_fontsize("xlabel", default=14)
        ylabel_fs = conf_mat_plot_fontsize("ylabel", default=14)
        xticks_fs = conf_mat_plot_fontsize("xticks", default=12)
        yticks_fs = conf_mat_plot_fontsize("yticks", default=12)

    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel(xlabel, fontsize=xlabel_fs)
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.tick_params(axis="x", labelsize=xticks_fs)
    ax.tick_params(axis="y", labelsize=yticks_fs, labelrotation=90)


def save_figure(
    fig,
    path: str,
    filename: str,
    format: str = None,
    dpi: Optional[int] = None,
) -> None:
    """
    Save *fig* to ``<path>/<filename>.<format>`` then close it.

    If *format* is ``"show"`` (the config default) or *path* is empty/``None``,
    ``plt.show()`` is called instead of saving.
    After either action ``plt.close(fig)`` is always called to free memory.

    For FITS output use ``fits_array`` (in ``autogalaxy.plot``) instead.

    Parameters
    ----------
    fig
        The matplotlib figure to save.
    path
        Directory where the file is written.  Created if it does not exist.
    filename
        File name without extension.
    format
        File format(s) passed to ``fig.savefig``.  Either a single string
        (e.g. ``"png"``) or a list/tuple of strings (e.g. ``["png", "pdf"]``)
        to save in multiple formats in one call.  ``"show"`` displays the
        figure interactively.  ``None`` reads the default from config.
    dpi
        Resolution in dots per inch.
    """
    import matplotlib.pyplot as plt

    if format is None:
        format = _conf_output_format()

    if dpi is None:
        from autoconf import conf
        dpi = int(conf.instance["visualize"]["general"]["general"]["dpi"])

    if _output_mode_save(fig, filename):
        return

    if _FAST_PLOTS:
        plt.close(fig)
        return

    formats = format if isinstance(format, (list, tuple)) else [format]

    if all(f == "show" for f in formats) or not path:
        plt.show()
    else:
        os.makedirs(path, exist_ok=True)
        for fmt in formats:
            if fmt == "show":
                continue
            try:
                fig.savefig(
                    str(Path(path) / f"{filename}.{fmt}"),
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
            except Exception as exc:
                logger.warning(
                    f"save_figure: could not save {filename}.{fmt}: {exc}"
                )

    plt.close(fig)


def plot_visibilities_1d(vis, ax, title: str = "") -> None:
    """Plot the real and imaginary components of a visibilities array as 1D line plots.

    Draws two overlapping lines — one for the real part and one for the
    imaginary part — with a legend.  Used by interferometer subplot functions
    to visualise raw or residual visibilities.

    Parameters
    ----------
    vis
        A ``Visibilities`` autoarray object (accessed via ``.slim``) or any
        array-like that can be cast to a complex numpy array.
    ax
        Matplotlib ``Axes`` to draw onto.
    title
        Axes title string.
    """
    try:
        y = np.array(vis.slim if hasattr(vis, "slim") else vis)
    except Exception:
        y = np.asarray(vis)
    ax.plot(y.real, label="Real", alpha=0.7)
    ax.plot(y.imag, label="Imaginary", alpha=0.7)
    ax.set_title(title)
    ax.legend(fontsize=8)


def _conf_colorbar(key: str, default):
    try:
        from autoconf import conf
        return conf.instance["visualize"]["general"]["colorbar"][key]
    except Exception:
        return default


def _colorbar_tick_values(norm) -> Optional[List[float]]:
    """Return [min, mid, max] tick positions from *norm*, with mid in log-space for LogNorm."""
    if norm is None or norm.vmin is None or norm.vmax is None:
        return None
    import matplotlib.colors as mcolors
    lo, hi = float(norm.vmin), float(norm.vmax)
    if isinstance(norm, mcolors.LogNorm):
        mid = 10 ** ((np.log10(lo) + np.log10(hi)) / 2.0)
    else:
        mid = (lo + hi) / 2.0
    return [lo, mid, hi]


_SUPERSCRIPT_DIGITS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _to_scientific(v: float) -> Optional[str]:
    """Convert *v* to Unicode scientific notation (e.g. ``4.3×10⁴``).

    Returns ``None`` when ``f"{v:.2g}"`` does not produce an exponent (unusual
    edge case for certain values near the g-format threshold).
    """
    s = f"{v:.2g}"
    if "e" not in s:
        return None
    mantissa, exp = s.split("e")
    sign = "-" if exp.startswith("-") else ""
    exp_num = exp.lstrip("+-").lstrip("0") or "0"
    superscript = f"{sign}{exp_num}".translate(_SUPERSCRIPT_DIGITS)
    return f"{mantissa}×10{superscript}"


def _fmt_tick(v: float) -> str:
    """Format a single tick value compactly.

    Values with 5 or more digits (abs(v) >= 10000) or very small values
    (abs(v) < 0.001) are rendered as compact scientific notation using
    Unicode superscripts, e.g. ``4.3×10⁴`` or ``1.2×10⁻⁵``.  This avoids
    LaTeX expansion that would overflow the colorbar width.  Values in
    between are rendered with ``:.2f``.
    """
    abs_v = abs(v)
    if abs_v != 0 and (abs_v >= 10000 or abs_v < 0.001):
        sci = _to_scientific(v)
        return sci if sci is not None else f"{v:.2g}"
    return f"{v:.2f}"


def _colorbar_tick_labels(tick_values: List[float], cb_unit: Optional[str] = None) -> List[str]:
    """Format tick values, appending *cb_unit* to the middle label.

    All three labels use a consistent notation style: if any tick is rendered
    in scientific notation (``×10ⁿ``), every non-zero tick is forced through
    the same format.  This prevents the central tick from showing e.g.
    ``-5000.00`` when the outer ticks show ``-2×10⁴`` / ``1.5×10⁴`` because
    the midpoint happens to fall below the per-value threshold.

    If *cb_unit* is ``None`` the unit is read from config; pass ``""`` for
    unitless panels.
    """
    if cb_unit is None:
        try:
            from autoconf import conf
            cb_unit = conf.instance["visualize"]["general"]["units"]["cb_unit"]
        except Exception:
            cb_unit = ""
    labels = [_fmt_tick(v) for v in tick_values]
    mid = len(labels) // 2

    # Enforce consistent notation: if any label uses ×10, convert all others.
    if any("×10" in lbl for lbl in labels):
        for i, (lbl, v) in enumerate(zip(labels, tick_values)):
            if "×10" not in lbl:
                if v == 0:
                    labels[i] = "0"
                else:
                    sci = _to_scientific(v)
                    if sci is not None:
                        labels[i] = sci

    labels[mid] = f"{labels[mid]}{cb_unit}"
    return labels


def _apply_colorbar(
    mappable,
    ax,
    cb_unit: Optional[str] = None,
    is_subplot: bool = False,
) -> None:
    """Create a colorbar with 3 ticks (min/mid/max), unit on middle label, config styling.

    Parameters
    ----------
    cb_unit
        Override the unit string on the middle tick.  Pass ``""`` for unitless panels.
        ``None`` reads the unit from config.
    is_subplot
        When ``True`` uses ``labelsize_subplot`` from config (default 16) instead of
        the single-figure ``labelsize`` (default 16).
    """
    import matplotlib.pyplot as plt

    tick_values = _colorbar_tick_values(getattr(mappable, "norm", None))

    cb = plt.colorbar(
        mappable,
        ax=ax,
        fraction=float(_conf_colorbar("fraction", 0.047)),
        pad=float(_conf_colorbar("pad", 0.01)),
        ticks=tick_values,
    )
    labelsize_key = "labelsize_subplot" if is_subplot else "labelsize"
    labelsize = float(_conf_colorbar(labelsize_key, 16))
    labelrotation = float(_conf_colorbar("labelrotation", 90))
    if tick_values is not None:
        cb.ax.set_yticklabels(
            _colorbar_tick_labels(tick_values, cb_unit=cb_unit),
            va="center",
            fontsize=labelsize,
        )
    # tick_params stores the setting for ticks created during draw;
    # axis='y' is explicit since colorbars are vertical.
    cb.ax.tick_params(axis="y", labelsize=labelsize, labelrotation=labelrotation)
    # Also drive it through the yaxis object directly so it survives
    # any internal colorbar redraw that recreates tick Text objects.
    cb.ax.yaxis.set_tick_params(labelsize=labelsize, labelrotation=labelrotation)


def _apply_contours(
    ax,
    array: np.ndarray,
    extent,
    use_log10: bool = False,
    n: Optional[int] = None,
) -> None:
    """Draw contour lines over a 2D image panel.

    For log10 plots contours are drawn automatically with log-spaced levels.
    For linear plots contours are only drawn when *n* is given explicitly.

    Level count and label visibility are read from the ``contour`` section of
    ``visualize/general.yaml`` (keys ``total_contours`` and
    ``include_values``).  The *n* argument overrides ``total_contours`` when
    provided.

    Parameters
    ----------
    ax
        The axes to draw on.
    array
        2D numpy array of the plotted data (after any clipping/normalisation).
    extent
        ``[xmin, xmax, ymin, ymax]`` passed to ``ax.contour``.
    use_log10
        When ``True`` levels are log-spaced between the positive minimum and
        maximum of *array*.
    n
        Explicit number of contour levels (overrides config).  When ``None``
        the config value is used.
    """
    try:
        from autoconf import conf
        _c = conf.instance["visualize"]["general"]["contour"]
        total = int(n if n is not None else _c.get("total_contours", 10))
        include_values = bool(_c.get("include_values", True))
    except Exception:
        total = int(n) if n is not None else 10
        include_values = True

    try:
        if use_log10:
            try:
                from autoconf import conf
                log10_min = float(conf.instance["visualize"]["general"]["general"]["log10_min_value"])
            except Exception:
                log10_min = 1.0e-4

            positive = array[array > 0]
            if positive.size == 0:
                return
            min_value = float(np.nanmin(positive))
            if min_value < log10_min:
                min_value = log10_min
            levels = np.logspace(
                np.log10(min_value),
                np.log10(float(np.nanmax(array))),
                total,
            )
        else:
            levels = np.linspace(float(np.nanmin(array)), float(np.nanmax(array)), total)

        # Build explicit coordinate grids so the contours align with imshow.
        # With origin="upper" row 0 maps to ymax (Y decreases across rows);
        # with origin="lower" row 0 maps to ymin (Y increases across rows).
        ny, nx = array.shape[:2]
        if extent is not None:
            origin = _conf_imshow_origin()
            xs = np.linspace(extent[0], extent[1], nx)
            if origin == "upper":
                ys = np.linspace(extent[3], extent[2], ny)  # ymax → ymin
            else:
                ys = np.linspace(extent[2], extent[3], ny)  # ymin → ymax
            X, Y = np.meshgrid(xs, ys)
            cs = ax.contour(X, Y, array, levels=levels, colors="k", alpha=0.5)
        else:
            cs = ax.contour(array, levels=levels, colors="k", alpha=0.5)

        if include_values:
            try:
                cs.clabel(levels=levels, inline=True, fontsize=10, fmt="%.2g")
            except (ValueError, IndexError):
                pass
    except Exception:
        pass


def hide_unused_axes(axes) -> None:
    """Turn off any axes in the flattened *axes* array that have no plotted data."""
    for ax in axes:
        if not ax.has_data():
            ax.axis("off")


def _default_colormap() -> str:
    """Return the colormap name from config, registering the custom one if needed."""
    try:
        from autoconf import conf
        name = conf.instance["visualize"]["general"]["colormap"]
    except Exception:
        name = "autoarray"
    if name == "autoarray":
        from autoarray.plot.segmentdata import register
        register()
    return name


def _conf_imshow_origin() -> str:
    """Return the imshow origin from config (``"upper"`` or ``"lower"``)."""
    try:
        from autoconf import conf
        return conf.instance["visualize"]["general"]["general"]["imshow_origin"]
    except Exception:
        return "upper"


def _conf_output_format() -> str:
    """Return the default output_format from config (``"show"``, ``"png"``, etc.)."""
    try:
        from autoconf import conf
        return conf.instance["visualize"]["general"]["general"]["output_format"]
    except Exception:
        return "show"


def _conf_ticks(key: str, default: float) -> float:
    try:
        from autoconf import conf
        return float(conf.instance["visualize"]["general"]["ticks"][key])
    except Exception:
        return default


def _inward_ticks(lo: float, hi: float, factor: float, n: int) -> np.ndarray:
    """Return *n* tick positions pulled inward from the extent edges by *factor*."""
    centre = (lo + hi) / 2.0
    return np.linspace(
        centre + (lo - centre) * factor,
        centre + (hi - centre) * factor,
        n,
    )


def _round_ticks(values: np.ndarray, sig: int = 2) -> np.ndarray:
    """Round *values* to *sig* significant figures.

    After rounding, values smaller than 1e-10 of the overall tick scale are
    clamped to zero so that floating-point noise (e.g. 1e-16 centre ticks on
    symmetric extents) does not appear as scientific notation in labels.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        nonzero = np.where(values != 0, np.abs(values), 1.0)
        mags = np.where(values != 0, 10 ** (sig - 1 - np.floor(np.log10(nonzero))), 1.0)
    rounded = np.round(values * mags) / mags
    scale = float(np.max(np.abs(rounded))) if len(rounded) > 0 else 1.0
    if scale > 0:
        rounded[np.abs(rounded) < scale * 1e-10] = 0.0
    return rounded


def _arcsec_labels(ticks) -> List[str]:
    """Format tick values as arcsecond coordinate strings.

    Values that all end in ``.0`` are stripped of the decimal point before the
    ``"`` suffix is appended, so ``[-1.0, 0.0, 1.0]`` → ``['-1"', '0"', '1"']``.
    """
    labels = [f'{v:g}' for v in ticks]
    if all(label.endswith(".0") for label in labels):
        labels = [label[:-2] for label in labels]
    return [f'{label}"' for label in labels]


def apply_extent(
    ax,
    extent: Tuple[float, float, float, float],
) -> None:
    """
    Apply axis limits and inward-pulled, rounded, arcsecond-labelled ticks to *ax*.

    Tick count and inward factor are read from ``visualize/general.yaml``
    (``ticks.number_of_ticks_2d`` and ``ticks.extent_factor_2d``), defaulting
    to 3 ticks and factor 0.75.
    """
    factor = _conf_ticks("extent_factor_2d", 0.75)
    n = int(_conf_ticks("number_of_ticks_2d", 3))

    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    xticks = _round_ticks(_inward_ticks(xmin, xmax, factor, n))
    yticks = _round_ticks(_inward_ticks(ymin, ymax, factor, n))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(_arcsec_labels(xticks))
    ax.set_yticklabels(_arcsec_labels(yticks))
