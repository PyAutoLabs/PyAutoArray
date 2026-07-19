import numpy as np
from typing import Optional


from autoarray.plot.array import plot_array
from autoarray.plot.grid import plot_grid
from autoarray.plot.yx import plot_yx
from autoarray.plot.utils import subplots, subplot_save, hide_unused_axes, conf_subplot_figsize, tight_layout
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


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
      ``uv_wavelengths``).

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

        values_list.append(np.asarray(dataset.data.in_array))
        ext_name_list.append("data")

        if dataset.noise_map is not None:
            values_list.append(np.asarray(dataset.noise_map.in_array))
            ext_name_list.append("noise_map")

        if dataset.uv_wavelengths is not None:
            values_list.append(np.asarray(dataset.uv_wavelengths))
            ext_name_list.append("uv_wavelengths")

        hdu_list = hdu_list_for_output_from(
            values_list=values_list,
            ext_name_list=ext_name_list,
        )
        write_hdu_list(hdu_list, file_path=file_path, overwrite=overwrite)
    else:
        if data_path is not None:
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
