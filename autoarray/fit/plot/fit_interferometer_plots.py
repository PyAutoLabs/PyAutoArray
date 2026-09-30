import numpy as np
from typing import Optional


from autoarray.plot.array import plot_array
from autoarray.plot.yx import plot_yx
from autoarray.plot.utils import subplots, subplot_save, symmetric_vmin_vmax, hide_unused_axes, conf_subplot_figsize, tight_layout


def _subplot_fit_natural(
    fit,
    model_image,
    output_path,
    output_filename,
    output_format,
    colormap,
    use_log10,
    residuals_symmetric_cmap,
):
    """
    Subplot of the natural-weighted dirty images of a ``FitInterferometer`` whose dataset is
    array-free (built by ``from_stream`` / ``from_sparse_terms``), which has no visibilities,
    transformer or visibility-space residuals.

    Panels: Dirty Image (Natural) | Dirty Model Image (Natural) | Dirty Residual Map (Natural),
    the last two only when the real-space ``model_image`` is given. The dirty model image is
    ``W~ m / sum(w)`` (see ``dirty_model_image_natural_from``) and the dirty residual map is the
    dirty image minus it.
    """
    from autoarray.fit.fit_interferometer import dirty_model_image_natural_from

    dataset = fit.dataset
    dirty_image = dataset.dirty_image_natural

    if model_image is None:
        fig, axes = subplots(1, 1, figsize=conf_subplot_figsize(1, 1))
        axes = [axes]
    else:
        fig, axes = subplots(1, 3, figsize=conf_subplot_figsize(1, 3))

    plot_array(
        dirty_image,
        ax=axes[0],
        title="Dirty Image (Natural)",
        colormap=colormap,
        use_log10=use_log10,
    )

    if model_image is not None:
        dirty_model_image = dirty_model_image_natural_from(
            dataset=dataset, image=model_image
        )
        dirty_residual_map = dirty_image - dirty_model_image

        if residuals_symmetric_cmap:
            vmin_r, vmax_r = symmetric_vmin_vmax(dirty_residual_map)
        else:
            vmin_r = vmax_r = None

        plot_array(
            dirty_model_image,
            ax=axes[1],
            title="Dirty Model Image (Natural)",
            colormap=colormap,
            use_log10=use_log10,
        )
        plot_array(
            dirty_residual_map,
            ax=axes[2],
            title="Dirty Residual Map (Natural)",
            colormap=colormap,
            use_log10=False,
            vmin=vmin_r,
            vmax=vmax_r,
        )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def subplot_fit_interferometer(
    fit,
    output_path: Optional[str] = None,
    output_filename: str = "fit",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    residuals_symmetric_cmap: bool = True,
    model_image=None,
):
    """
    2×3 subplot of ``FitInterferometer`` residuals in UV-plane.

    Panels (real then imaginary): Residual Map | Norm Residual Map | Chi-Squared Map

    Parameters
    ----------
    fit
        A ``FitInterferometer`` instance.
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
    residuals_symmetric_cmap
        Not used here (UV-plane residuals are scatter plots); kept for API
        consistency.
    model_image
        The real-space model image, used only when the fit's dataset is array-free
        (``fit.dataset.is_array_free``): the visibility-space panels cannot be drawn, so
        the natural-weighted dirty image, dirty model image and dirty residual map are
        plotted to the same filename instead (the last two only if this is given).
    """
    if fit.dataset.is_array_free:
        return _subplot_fit_natural(
            fit,
            model_image=model_image,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
            colormap=colormap,
            use_log10=use_log10,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

    fig, axes = subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes = axes.flatten()

    uv = fit.dataset.uv_distances / 10**3.0

    plot_yx(
        np.real(fit.residual_map),
        uv,
        ax=axes[0],
        title="Residual vs UV-Distance (real)",
        xlabel="k$\\lambda$",
        plot_axis_type="scatter",
    )
    plot_yx(
        np.real(fit.normalized_residual_map),
        uv,
        ax=axes[1],
        title="Norm Residual vs UV-Distance (real)",
        ylabel="$\\sigma$",
        xlabel="k$\\lambda$",
        plot_axis_type="scatter",
    )
    plot_yx(
        np.real(fit.chi_squared_map),
        uv,
        ax=axes[2],
        title="Chi-Squared vs UV-Distance (real)",
        ylabel="$\\chi^2$",
        xlabel="k$\\lambda$",
        plot_axis_type="scatter",
    )
    plot_yx(
        np.imag(fit.residual_map),
        uv,
        ax=axes[3],
        title="Residual vs UV-Distance (imag)",
        xlabel="k$\\lambda$",
        plot_axis_type="scatter",
    )
    plot_yx(
        np.imag(fit.normalized_residual_map),
        uv,
        ax=axes[4],
        title="Norm Residual vs UV-Distance (imag)",
        ylabel="$\\sigma$",
        xlabel="k$\\lambda$",
        plot_axis_type="scatter",
    )
    plot_yx(
        np.imag(fit.chi_squared_map),
        uv,
        ax=axes[5],
        title="Chi-Squared vs UV-Distance (imag)",
        ylabel="$\\chi^2$",
        xlabel="k$\\lambda$",
        plot_axis_type="scatter",
    )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)


def subplot_fit_interferometer_dirty_images(
    fit,
    output_path: Optional[str] = None,
    output_filename: str = "fit_dirty_images",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    residuals_symmetric_cmap: bool = True,
    model_image=None,
):
    """
    2×3 subplot of ``FitInterferometer`` dirty-image components.

    Panels: Dirty Image | Dirty S/N Map | Dirty Model Image |
            Dirty Residual Map | Dirty Norm Residual Map | Dirty Chi-Squared Map

    Parameters
    ----------
    fit
        A ``FitInterferometer`` instance.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation to non-residual panels.
    residuals_symmetric_cmap
        Centre residual colour scale symmetrically around zero.
    model_image
        The real-space model image, used only when the fit's dataset is array-free
        (``fit.dataset.is_array_free``): the natural-weighted dirty image, dirty model
        image and dirty residual map are plotted to the same filename instead (the last
        two only if this is given).
    """
    if fit.dataset.is_array_free:
        return _subplot_fit_natural(
            fit,
            model_image=model_image,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
            colormap=colormap,
            use_log10=use_log10,
            residuals_symmetric_cmap=residuals_symmetric_cmap,
        )

    fig, axes = subplots(2, 3, figsize=conf_subplot_figsize(2, 3))
    axes = axes.flatten()

    plot_array(
        fit.dirty_image,
        ax=axes[0],
        title="Dirty Image",
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        fit.dirty_signal_to_noise_map,
        ax=axes[1],
        title="Dirty Signal-To-Noise Map",
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        fit.dirty_model_image,
        ax=axes[2],
        title="Dirty Model Image",
        colormap=colormap,
        use_log10=use_log10,
    )

    if residuals_symmetric_cmap:
        vmin_r, vmax_r = symmetric_vmin_vmax(fit.dirty_residual_map)
        vmin_n, vmax_n = symmetric_vmin_vmax(fit.dirty_normalized_residual_map)
    else:
        vmin_r = vmax_r = vmin_n = vmax_n = None

    plot_array(
        fit.dirty_residual_map,
        ax=axes[3],
        title="Dirty Residual Map",
        colormap=colormap,
        use_log10=False,
        vmin=vmin_r,
        vmax=vmax_r,
    )
    plot_array(
        fit.dirty_normalized_residual_map,
        ax=axes[4],
        title="Dirty Normalized Residual Map",
        colormap=colormap,
        use_log10=False,
        vmin=vmin_n,
        vmax=vmax_n,
        cb_unit=r"$\sigma$",
    )
    plot_array(
        fit.dirty_chi_squared_map,
        ax=axes[5],
        title="Dirty Chi-Squared Map",
        colormap=colormap,
        use_log10=use_log10,
        cb_unit=r"$\chi^2$",
    )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
