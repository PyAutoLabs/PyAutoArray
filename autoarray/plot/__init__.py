def _set_backend():
    try:
        import matplotlib
        from autonerves import conf

        backend = conf.get_matplotlib_backend()
        if backend != "default":
            matplotlib.use(backend)
        try:
            hpc_mode = conf.instance["general"]["hpc"]["hpc_mode"]
        except KeyError:
            hpc_mode = False
        if hpc_mode:
            matplotlib.use("Agg")
    except Exception:
        pass


_set_backend()

from autoarray.plot.output import Output

from autoarray.plot.array import plot_array
from autoarray.plot.grid import plot_grid
from autoarray.plot.yx import plot_yx
from autoarray.plot.inversion import plot_inversion_reconstruction
from autoarray.plot.utils import (
    apply_extent,
    apply_labels,
    conf_figsize,
    conf_subplot_figsize,
    conf_mat_plot_fontsize,
    save_figure,
    subplot_save,
    tight_layout,
    auto_mask_edge,
    zoom_array,
    numpy_grid,
    numpy_lines,
    numpy_positions,
    symmetric_vmin_vmax,
    symmetric_cmap_from,
    set_with_color_values,
    plot_visibilities_1d,
)

from autoarray.structures.plot.structure_plots import (
    plot_array_2d,
    plot_grid_2d,
    plot_yx_1d,
)

from autoarray.fit.plot.fit_imaging_plots import subplot_fit_imaging
from autoarray.fit.plot.fit_interferometer_plots import (
    subplot_fit_interferometer,
    subplot_fit_interferometer_dirty_images,
)

from autoarray.inversion.plot.mapper_plots import (
    plot_mapper,
    subplot_image_and_mapper,
)
from autoarray.inversion.plot.inversion_plots import (
    subplot_of_mapper,
    subplot_mappings,
)
