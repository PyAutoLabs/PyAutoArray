import csv
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

from autonerves import conf

from autoarray.inversion.mappers.abstract import Mapper
from autoarray.plot.array import plot_array
from autoarray.plot.utils import subplots, numpy_grid, numpy_lines, numpy_positions, subplot_save, hide_unused_axes, conf_subplot_figsize, tight_layout
from autoarray.inversion.plot.mapper_plots import plot_mapper
from autoarray.structures.arrays.uniform_2d import Array2D

logger = logging.getLogger(__name__)


def subplot_of_mapper(
    inversion,
    mapper_index: int = 0,
    output_path: Optional[str] = None,
    output_filename: str = "inversion",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    mesh_grid=None,
    lines=None,
    grid=None,
    positions=None,
    title_prefix: str = None,
):
    """
    3×4 subplot showing all pixelization diagnostics for one mapper.

    Parameters
    ----------
    inversion
        An ``AbstractInversion`` instance.
    mapper_index
        Which mapper in the inversion to visualise.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename prefix (``_{mapper_index}`` is appended).
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    mesh_grid, lines, grid, positions
        Optional overlays.
    """
    mapper = inversion.cls_list_from(cls=Mapper)[mapper_index]

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    fig, axes = subplots(3, 4, figsize=conf_subplot_figsize(3, 4))
    axes = axes.flatten()

    # panel 0: data subtracted
    try:
        array = inversion.data_subtracted_dict[mapper]
        from autoarray.structures.visibilities import Visibilities

        if isinstance(array, Visibilities):
            array = inversion.transformer.image_from(visibilities=array)
        plot_array(
            array,
            ax=axes[0],
            title=_pf("Data Subtracted"),
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    # panels 1-3: reconstructed operated data (plain, log10, + mesh grid overlay)
    def _recon_array():
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities

        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        return array

    try:
        plot_array(
            _recon_array(),
            ax=axes[1],
            title=_pf("Reconstructed Image"),
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
        plot_array(
            _recon_array(),
            ax=axes[2],
            title=_pf("Reconstructed Image (log10)"),
            colormap=colormap,
            use_log10=True,
            grid=grid,
            positions=positions,
            lines=lines,
        )
        plot_array(
            _recon_array(),
            ax=axes[3],
            title=_pf("Mesh Pixel Grid Overlaid"),
            colormap=colormap,
            use_log10=use_log10,
            grid=numpy_grid(mapper.image_plane_mesh_grid),
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    # panels 4-5: source reconstruction zoomed / unzoomed
    pixel_values = inversion.reconstruction_dict[mapper]
    try:
        recon_vmax = float(np.max(np.asarray(_recon_array())))
    except Exception:
        recon_vmax = None
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[4],
        title=_pf("Source Plane (Zoom)"),
        colormap=colormap,
        use_log10=use_log10,
        vmax=recon_vmax,
        zoom_to_brightest=True,
        mesh_grid=mesh_grid,
        lines=lines,
    )
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[5],
        title=_pf("Source Plane (No Zoom)"),
        colormap=colormap,
        use_log10=use_log10,
        vmax=recon_vmax,
        zoom_to_brightest=False,
        mesh_grid=mesh_grid,
        lines=lines,
    )

    # panel 6: noise map
    try:
        nm = inversion.reconstruction_noise_map_dict[mapper]
        plot_mapper(
            mapper,
            solution_vector=nm,
            ax=axes[6],
            title=_pf("Noise-Map (No Zoom)"),
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=False,
            mesh_grid=mesh_grid,
            lines=lines,
        )
    except (KeyError, TypeError, np.linalg.LinAlgError):
        pass

    # panel 7: regularization weights
    try:
        rw = inversion.regularization_weights_mapper_dict[mapper]
        plot_mapper(
            mapper,
            solution_vector=rw,
            ax=axes[7],
            title=_pf("Regularization (No Zoom)"),
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=False,
            mesh_grid=mesh_grid,
            lines=lines,
        )
    except (IndexError, ValueError, KeyError, TypeError):
        pass

    # panel 8: sub pixels per image pixels
    try:
        sub_size = Array2D(
            values=mapper.over_sampler.sub_size, mask=inversion.dataset.mask
        )
        plot_array(
            sub_size,
            ax=axes[8],
            title=_pf("Sub Pixels Per Image Pixels"),
            colormap=colormap,
            use_log10=use_log10,
        )
    except Exception:
        pass

    # panel 9: mesh pixels per image pixels
    try:
        plot_array(
            mapper.mesh_pixels_per_image_pixels,
            ax=axes[9],
            title=_pf("Mesh Pixels Per Image Pixels"),
            colormap=colormap,
            use_log10=use_log10,
        )
    except Exception:
        pass

    # panel 10: image pixels per mesh pixel
    try:
        pw = mapper.data_weight_total_for_pix_from()
        plot_mapper(
            mapper,
            solution_vector=pw,
            ax=axes[10],
            title=_pf("Image Pixels Per Source Pixel"),
            colormap=colormap,
            use_log10=use_log10,
            zoom_to_brightest=True,
            mesh_grid=mesh_grid,
            lines=lines,
        )
    except (TypeError, Exception):
        pass

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(fig, output_path, f"{output_filename}_{mapper_index}", output_format)


def subplot_mappings(
    inversion,
    pixelization_index: int = 0,
    output_path: Optional[str] = None,
    output_filename: str = "mappings",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    mesh_grid=None,
    lines=None,
    grid=None,
    positions=None,
):
    """
    2×2 subplot showing data, model image, reconstruction and unzoomed reconstruction.

    Parameters
    ----------
    inversion
        An ``AbstractInversion`` instance.
    pixelization_index
        Which mapper in the inversion to visualise.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename prefix (``_{pixelization_index}`` is appended).
    output_format
        File format.
    colormap
        Matplotlib colormap name.
    use_log10
        Apply log10 normalisation.
    mesh_grid, lines, grid, positions
        Optional overlays.
    """
    mapper = inversion.cls_list_from(cls=Mapper)[pixelization_index]

    try:
        total_pixels = conf.instance["visualize"]["general"]["inversion"][
            "total_mappings_pixels"
        ]
    except Exception:
        total_pixels = 10

    pix_indexes = inversion.max_pixel_list_from(
        total_pixels=total_pixels,
        filter_neighbors=True,
        mapper_index=pixelization_index,
    )
    mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

    fig, axes = subplots(2, 2, figsize=conf_subplot_figsize(2, 2))
    axes = axes.flatten()

    # panel 0: data subtracted
    try:
        array = inversion.data_subtracted_dict[mapper]
        from autoarray.structures.visibilities import Visibilities

        if isinstance(array, Visibilities):
            array = inversion.transformer.image_from(visibilities=array)
        plot_array(
            array,
            ax=axes[0],
            title="Data Subtracted",
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    # panel 1: reconstructed operated data
    try:
        array = inversion.mapped_reconstructed_operated_data_dict[mapper]
        from autoarray.structures.visibilities import Visibilities

        if isinstance(array, Visibilities):
            array = inversion.mapped_reconstructed_data_dict[mapper]
        plot_array(
            array,
            ax=axes[1],
            title="Reconstructed Image",
            colormap=colormap,
            use_log10=use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )
    except (AttributeError, KeyError):
        pass

    pixel_values = inversion.reconstruction_dict[mapper]
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[2],
        title="Source Plane (Zoom)",
        colormap=colormap,
        use_log10=use_log10,
        zoom_to_brightest=True,
        mesh_grid=mesh_grid,
        lines=lines,
    )
    plot_mapper(
        mapper,
        solution_vector=pixel_values,
        ax=axes[3],
        title="Source Plane (No Zoom)",
        colormap=colormap,
        use_log10=use_log10,
        zoom_to_brightest=False,
        mesh_grid=mesh_grid,
        lines=lines,
    )

    hide_unused_axes(axes)
    tight_layout()
    subplot_save(
        fig, output_path, f"{output_filename}_{pixelization_index}", output_format
    )


def save_reconstruction_csv(
    inversion,
    output_path: Union[str, Path],
) -> None:
    """Write a CSV of each mapper's reconstruction and noise map to *output_path*.

    One file is written per mapper: ``source_plane_reconstruction_{i}.csv``,
    with columns ``y``, ``x``, ``reconstruction``, ``noise_map``.

    The reconstruction noise map inverts the curvature regularization matrix, which is
    singular for rank-deficient inversions (e.g. the reduced-iteration searches used by
    test profiles). The column schema is fixed, because consumers index the CSV by
    column name, so in that case the ``noise_map`` column is written as ``nan`` (with a
    logged warning) rather than omitted, and the file is still written: the
    reconstruction is a science product in its own right and must not be lost, nor may
    a failure here abort the enclosing model-fit.

    Parameters
    ----------
    inversion
        An ``AbstractInversion`` instance.
    output_path
        Directory in which to write the CSV files.
    """
    output_path = Path(output_path)
    mapper_list = inversion.cls_list_from(cls=Mapper)

    for i, mapper in enumerate(mapper_list):
        y = mapper.source_plane_mesh_grid[:, 0]
        x = mapper.source_plane_mesh_grid[:, 1]
        reconstruction = inversion.reconstruction_dict[mapper]

        try:
            noise_map = inversion.reconstruction_noise_map_dict[mapper]
        except np.linalg.LinAlgError:
            logger.warning(
                f"save_reconstruction_csv: could not compute the reconstruction noise map for "
                f"mapper {i} (singular curvature_reg_matrix); writing the noise_map column of "
                f"source_plane_reconstruction_{i}.csv as nan."
            )
            noise_map = None

        with open(output_path / f"source_plane_reconstruction_{i}.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["y", "x", "reconstruction", "noise_map"])
            for j in range(len(x)):
                noise_value = float("nan") if noise_map is None else float(noise_map[j])
                writer.writerow([float(y[j]), float(x[j]), float(reconstruction[j]), noise_value])
