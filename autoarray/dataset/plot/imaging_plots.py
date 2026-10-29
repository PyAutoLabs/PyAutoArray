from typing import Optional


from autoarray.plot.utils import subplots, subplot_save, conf_subplot_figsize, tight_layout


def subplot_imaging_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "dataset",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    grid=None,
    positions=None,
    lines=None,
    title_prefix: str = None,
):
    """
    3×3 subplot of core ``Imaging`` dataset components.

    Panels (row-major):
    0. Data
    1. Data (log10)
    2. Noise-Map
    3. PSF (if present)
    4. PSF (log10, if present)
    5. Signal-To-Noise Map
    6. Over Sample Size (Light Profiles, if present)
    7. Over Sample Size (Pixelization, if present)

    Parameters
    ----------
    dataset
        An ``Imaging`` dataset instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format, e.g. ``"png"``.
    colormap
        Matplotlib colormap name.  ``None`` uses the package default.
    use_log10
        Apply log10 normalisation to non-log panels.
    grid, positions, lines
        Optional overlays forwarded to every panel.
    """
    if isinstance(output_format, (list, tuple)):
        output_format = output_format[0]

    from autoarray.plot.array import plot_array

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    fig, axes = subplots(3, 3, figsize=conf_subplot_figsize(3, 3))
    axes = axes.flatten()

    plot_array(
        dataset.data,
        ax=axes[0],
        title=_pf("Data"),
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        dataset.data,
        ax=axes[1],
        title=_pf("Data (log10)"),
        colormap=colormap,
        use_log10=True,
        grid=grid,
        positions=positions,
        lines=lines,
    )
    plot_array(
        dataset.noise_map,
        ax=axes[2],
        title=_pf("Noise-Map"),
        colormap=colormap,
        use_log10=use_log10,
        grid=grid,
        positions=positions,
        lines=lines,
    )

    if dataset.psf is not None:
        plot_array(
            dataset.psf.kernel,
            ax=axes[3],
            title=_pf("Point Spread Function"),
            colormap=colormap,
            use_log10=use_log10,
            cb_unit="",
        )
        plot_array(
            dataset.psf.kernel,
            ax=axes[4],
            title=_pf("PSF (log10)"),
            colormap=colormap,
            use_log10=True,
            cb_unit="",
        )

    plot_array(
        dataset.signal_to_noise_map,
        ax=axes[5],
        title=_pf("Signal-To-Noise Map"),
        colormap=colormap,
        use_log10=use_log10,
        cb_unit="",
        grid=grid,
        positions=positions,
        lines=lines,
    )

    over_sample_size_lp = getattr(getattr(dataset, "grids", None), "over_sample_size_lp", None)
    if over_sample_size_lp is not None:
        plot_array(
            over_sample_size_lp,
            ax=axes[6],
            title="Over Sample Size (Light Profiles)",
            colormap=colormap,
            use_log10=use_log10,
            cb_unit="",
        )

    over_sample_size_pix = getattr(getattr(dataset, "grids", None), "over_sample_size_pixelization", None)
    if over_sample_size_pix is not None:
        plot_array(
            over_sample_size_pix,
            ax=axes[7],
            title="Over Sample Size (Pixelization)",
            colormap=colormap,
            use_log10=use_log10,
            cb_unit="",
        )

    from autoarray.plot.utils import hide_unused_axes
    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)



def _intro_panel(array, radius=None, excluded_pixels=None):
    """Prepare a display-only crop with excluded values missing, not zeroed.

    Preserve the native pixel/coordinate registration; fitting arrays are never
    mutated. Coordinates, ticks and colour mapping remain the renderer's job.
    """
    import numpy as np
    from autoarray.plot.utils import zoom_array

    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError("radius must be a finite positive number")
    if radius is None and excluded_pixels is None:
        array = zoom_array(array)
    values = np.array(array.native.array, dtype=float, copy=True)
    values[np.asarray(array.mask.array, dtype=bool)] = np.nan
    if excluded_pixels is not None:
        excluded = np.asarray(excluded_pixels, dtype=bool)
        if excluded.shape != values.shape:
            raise ValueError("excluded_pixels must have the dataset image shape")
        values[excluded] = np.nan
    xmin, xmax, ymin, ymax = array.geometry.extent
    if radius is not None:
        ny, nx = values.shape
        dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
        x = xmin + (np.arange(nx) + 0.5) * dx
        y = ymax - (np.arange(ny) + 0.5) * dy
        rows, cols = np.flatnonzero(abs(y) <= radius), np.flatnonzero(abs(x) <= radius)
        if not len(rows) or not len(cols):
            raise ValueError("radius does not include any pixel centres")
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        values = values[r0:r1, c0:c1]
        xmin, xmax, ymin, ymax = xmin + c0 * dx, xmin + c1 * dx, ymax - r1 * dy, ymax - r0 * dy
    return values, (xmin, xmax, ymin, ymax)


def subplot_imaging_dataset_intro(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "dataset_intro",
    output_format: str = None,
    colormap=None,
    stretch: Optional[float] = 0.01,
    radius: Optional[float] = None,
    excluded_pixels=None,
):
    """Four-panel introduction: Image, Noise map, Signal-to-noise map and PSF.

    Uses the ordinary plot renderer and configured colormap, fonts, titles,
    arcsecond ticks, colourbars and saving. Unlike the diagnostic 3x3 subplot,
    it omits duplicate log panels and over-sampling maps. Numerical sampling
    and the dataset itself are unchanged.

    ``stretch`` is the image's asinh linear-width fraction: smaller values
    reveal faint structure next to bright light; None selects linear display.
    The S/N and PSF panels use asinh fractions 0.1 and 0.003, respectively;
    noise remains linear. All colourbars retain original measurement values.
    ``colormap=None`` inherits configuration on every panel.

    ``radius`` optionally crops the first three panels about (0, 0), in
    arcseconds, for display only. None follows the usual mask-zoom convention.
    ``excluded_pixels`` is a native-shaped boolean array (True = exclude) for
    deliberately down-weighted neighbours. These pixels, dataset-masked values
    and non-finite measurements use the colormap's standard bad-value colour;
    they do not determine the display scale. No clipping or data mutation.
    """
    from autoarray.plot.array import plot_array
    from autoarray.plot.utils import apply_labels

    panels = [
        (dataset.data, "Image", stretch, None),
        (dataset.noise_map, "Noise map", None, None),
        (dataset.signal_to_noise_map, "Signal-to-noise map", 0.1, ""),
    ]
    prepared = [(_intro_panel(array, radius, excluded_pixels), title, scale, unit)
                for array, title, scale, unit in panels]
    fig, axes = subplots(2, 2, figsize=conf_subplot_figsize(2, 2))
    for ax, ((values, extent), title, scale, unit) in zip(axes.flat, prepared):
        plot_array(values, ax=ax, extent=extent, title=title,
                   colormap=colormap, stretch=scale, cb_unit=unit)
    if dataset.psf is not None:
        plot_array(dataset.psf.kernel, ax=axes[1, 1], title="PSF",
                   colormap=colormap, stretch=0.003, cb_unit="")
    else:
        apply_labels(axes[1, 1], title="PSF", is_subplot=True)
        axes[1, 1].text(0.5, 0.5, "No PSF supplied", ha="center", va="center",
                        transform=axes[1, 1].transAxes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
    return fig


def subplot_imaging_dataset_list(
    dataset_list,
    output_path=None,
    output_filename: str = "dataset_combined",
    output_format=None,
    title_prefix: str = None,
):
    """
    n×3 subplot showing core components for each dataset in a list.

    Each row shows: Data | Noise Map | Signal-To-Noise Map

    Parameters
    ----------
    dataset_list
        List of ``Imaging`` dataset instances.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format string or list, e.g. ``"png"`` or ``["png"]``.
    title_prefix
        Optional string prepended (with an automatic space) to every panel title.
    """
    if isinstance(output_format, (list, tuple)):
        output_format = output_format[0]

    from autoarray.plot.array import plot_array

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    n = len(dataset_list)
    fig, axes = subplots(n, 3, figsize=conf_subplot_figsize(n, 3))
    if n == 1:
        axes = [axes]
    for i, dataset in enumerate(dataset_list):
        plot_array(dataset.data, ax=axes[i][0], title=_pf("Data"))
        plot_array(dataset.noise_map, ax=axes[i][1], title=_pf("Noise Map"))
        plot_array(dataset.signal_to_noise_map, ax=axes[i][2], title=_pf("Signal-To-Noise Map"))
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def fits_imaging(
    dataset,
    file_path=None,
    data_path=None,
    psf_path=None,
    noise_map_path=None,
    overwrite=False,
):
    """Write an ``Imaging`` dataset to FITS.

    Supports two modes:

    * **Separate files** — pass ``data_path``, ``psf_path``, ``noise_map_path``
      to write each component to its own single-HDU FITS file.
    * **Single multi-HDU file** — pass ``file_path`` to write all components
      into one FITS file with named extensions (``mask``, ``data``, ``psf``,
      ``noise_map``).

    Parameters
    ----------
    dataset
        The ``Imaging`` dataset to write.
    file_path : str or Path, optional
        Path for a single multi-HDU FITS file.
    data_path, psf_path, noise_map_path : str or Path, optional
        Paths for individual component files.
    overwrite : bool
        If ``True`` existing files are replaced.
    """
    from autonerves.fitsable import output_to_fits, hdu_list_for_output_from, write_hdu_list

    header_dict = dataset.data.mask.header_dict if hasattr(dataset.data.mask, "header_dict") else None

    if file_path is not None:
        values_list = [dataset.data.mask.astype("float")]
        ext_name_list = ["mask"]

        values_list.append(dataset.data.native.array.astype("float"))
        ext_name_list.append("data")

        if dataset.psf is not None:
            values_list.append(dataset.psf.kernel.native.array.astype("float"))
            ext_name_list.append("psf")

        if dataset.noise_map is not None:
            values_list.append(dataset.noise_map.native.array.astype("float"))
            ext_name_list.append("noise_map")

        hdu_list = hdu_list_for_output_from(
            values_list=values_list,
            ext_name_list=ext_name_list,
            header_dict=header_dict,
        )
        write_hdu_list(hdu_list, file_path=file_path, overwrite=overwrite)
    else:
        if data_path is not None:
            output_to_fits(
                values=dataset.data.native.array.astype("float"),
                file_path=data_path, overwrite=overwrite, header_dict=header_dict,
            )
        if dataset.psf is not None and psf_path is not None:
            output_to_fits(
                values=dataset.psf.kernel.native.array.astype("float"),
                file_path=psf_path, overwrite=overwrite,
            )
        if dataset.noise_map is not None and noise_map_path is not None:
            output_to_fits(
                values=dataset.noise_map.native.array.astype("float"),
                file_path=noise_map_path, overwrite=overwrite, header_dict=header_dict,
            )
