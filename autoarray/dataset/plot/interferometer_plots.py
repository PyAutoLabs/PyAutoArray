import numpy as np
from typing import Optional


from autoarray.plot.array import plot_array
from autoarray.plot.grid import plot_grid
from autoarray.plot.yx import plot_yx
from autoarray.plot.utils import subplots, subplot_save, hide_unused_axes, conf_subplot_figsize, tight_layout
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


def _subplot_natural_dataset(
    dataset,
    output_path,
    output_filename,
    output_format,
    colormap,
    use_log10,
    title_prefix=None,
):
    """
    1x2 subplot of the natural-weighted dirty image and dirty beam of an array-free
    ``Interferometer`` (built by ``from_stream`` / ``from_sparse_terms``), which carries no
    visibilities, uv-wavelengths or transformer, so only these real-space terms can be drawn.
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    fig, axes = subplots(1, 2, figsize=conf_subplot_figsize(1, 2))

    plot_array(
        dataset.dirty_image_natural,
        ax=axes[0],
        title=_pf("Dirty Image (Natural)"),
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        dataset.dirty_beam,
        ax=axes[1],
        title=_pf("Dirty Beam (Natural)"),
        colormap=colormap,
        use_log10=use_log10,
    )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def subplot_interferometer_dataset(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "dataset",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    title_prefix: str = None,
):
    """
    2x3 subplot of interferometer dataset components.

    Panels: Visibilities | UV-Wavelengths | Amplitudes vs UV-distances |
            Phases vs UV-distances | Dirty Image | Dirty S/N Map

    An array-free dataset (``dataset.is_array_free``) has no visibilities, so a 1x2 subplot
    of its ``dirty_image_natural`` and ``dirty_beam`` is written to the same filename instead.

    Parameters
    ----------
    dataset
        An ``Interferometer`` dataset instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation to image panels.
    """
    if dataset.is_array_free:
        return _subplot_natural_dataset(
            dataset,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
            colormap=colormap,
            use_log10=use_log10,
            title_prefix=title_prefix,
        )

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    fig, axes = subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes = axes.flatten()

    plot_grid(dataset.data.in_grid, ax=axes[0], title=_pf("Visibilities"), xlabel="", ylabel="")
    plot_grid(
        Grid2DIrregular.from_yx_1d(
            y=dataset.uv_wavelengths[:, 1] / 10**3.0,
            x=dataset.uv_wavelengths[:, 0] / 10**3.0,
        ),
        ax=axes[1],
        title=_pf("UV-Wavelengths"),
        xlabel="",
        ylabel="",
    )
    plot_yx(
        dataset.amplitudes,
        dataset.uv_distances / 10**3.0,
        ax=axes[2],
        title=_pf("Amplitudes vs UV-distances"),
        xtick_suffix='"',
        ytick_suffix="Jy",
        plot_axis_type="scatter",
    )
    plot_yx(
        dataset.phases,
        dataset.uv_distances / 10**3.0,
        ax=axes[3],
        title=_pf("Phases vs UV-distances"),
        xtick_suffix='"',
        ytick_suffix="deg",
        plot_axis_type="scatter",
    )
    plot_array(
        dataset.dirty_image,
        ax=axes[4],
        title=_pf("Dirty Image"),
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        dataset.dirty_signal_to_noise_map,
        ax=axes[5],
        title=_pf("Dirty Signal-To-Noise Map"),
        colormap=colormap,
        use_log10=use_log10,
    )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def subplot_interferometer_dirty_images(
    dataset,
    output_path: Optional[str] = None,
    output_filename: str = "dirty_images",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
):
    """
    1x3 subplot of dirty image, dirty noise map, and dirty S/N map.

    An array-free dataset (``dataset.is_array_free``) has no visibilities, so a 1x2 subplot
    of its ``dirty_image_natural`` and ``dirty_beam`` is written to the same filename instead.

    Parameters
    ----------
    dataset
        An ``Interferometer`` dataset instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    """
    if dataset.is_array_free:
        return _subplot_natural_dataset(
            dataset,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
            colormap=colormap,
            use_log10=use_log10,
        )

    fig, axes = subplots(1, 3, figsize=conf_subplot_figsize(1, 3))

    plot_array(
        dataset.dirty_image,
        ax=axes[0],
        title="Dirty Image",
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        dataset.dirty_noise_map,
        ax=axes[1],
        title="Dirty Noise Map",
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        dataset.dirty_signal_to_noise_map,
        ax=axes[2],
        title="Dirty Signal-To-Noise Map",
        colormap=colormap,
        use_log10=use_log10,
    )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def fits_interferometer(
    dataset,
    file_path=None,
    data_path=None,
    noise_map_path=None,
    uv_wavelengths_path=None,
    overwrite=False,
):
    """Write an ``Interferometer`` dataset to FITS.

    Supports two modes:

    * **Separate files** -- pass ``data_path``, ``noise_map_path``,
      ``uv_wavelengths_path`` to write each component to its own FITS file.
    * **Single multi-HDU file** -- pass ``file_path`` to write all components
      into one FITS file with named extensions (``data``, ``noise_map``,
      ``uv_wavelengths``). An array-free dataset (``from_stream`` /
      ``from_sparse_terms``) has none of these, so its natural-weighted
      ``dirty_image_natural`` and ``dirty_beam`` are written instead.

    Parameters
    ----------
    dataset
        The ``Interferometer`` dataset to write.
    file_path : str or Path, optional
        Path for a single multi-HDU FITS file.
    data_path, noise_map_path, uv_wavelengths_path : str or Path, optional
        Paths for individual component files.
    overwrite : bool
        If ``True`` existing files are replaced.
    """
    from autonerves.fitsable import output_to_fits, hdu_list_for_output_from, write_hdu_list

    if file_path is not None:
        values_list = []
        ext_name_list = []

        if dataset.data is not None:
            values_list.append(np.asarray(dataset.data.in_array))
            ext_name_list.append("data")

        if dataset.noise_map is not None:
            values_list.append(np.asarray(dataset.noise_map.in_array))
            ext_name_list.append("noise_map")

        if dataset.uv_wavelengths is not None:
            values_list.append(np.asarray(dataset.uv_wavelengths))
            ext_name_list.append("uv_wavelengths")

        if dataset.is_array_free:
            values_list.append(np.asarray(dataset.dirty_image_natural.native))
            ext_name_list.append("dirty_image_natural")
            values_list.append(np.asarray(dataset.dirty_beam.native))
            ext_name_list.append("dirty_beam")

        hdu_list = hdu_list_for_output_from(
            values_list=values_list,
            ext_name_list=ext_name_list,
        )
        write_hdu_list(hdu_list, file_path=file_path, overwrite=overwrite)
    else:
        if dataset.data is not None and data_path is not None:
            output_to_fits(
                values=np.asarray(dataset.data.in_array),
                file_path=data_path, overwrite=overwrite,
            )
        if dataset.noise_map is not None and noise_map_path is not None:
            output_to_fits(
                values=np.asarray(dataset.noise_map.in_array),
                file_path=noise_map_path, overwrite=overwrite,
            )
        if dataset.uv_wavelengths is not None and uv_wavelengths_path is not None:
            output_to_fits(
                values=dataset.uv_wavelengths,
                file_path=uv_wavelengths_path, overwrite=overwrite,
            )
