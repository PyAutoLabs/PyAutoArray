import csv
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from autonerves import conf

from autoarray.inversion.mappers.abstract import Mapper
from autoarray.plot.array import plot_array
from autoarray.plot.utils import (
    subplots,
    numpy_grid,
    numpy_lines,
    numpy_positions,
    subplot_save,
    hide_unused_axes,
    conf_subplot_figsize,
    tight_layout,
)
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


def _conf_mappings(key: str, default, old_key: Optional[str] = None):
    """
    Read a `subplot_mappings` setting from the ``visualize/general.yaml`` ``inversion`` section.

    A config which predates the mappings rewrite has none of the new keys, so `old_key` names the
    key it did have (``total_mappings_pixels``) and is read as a fallback for one release. When
    neither key is present the documented `default` is used.

    Parameters
    ----------
    key
        The config key to read.
    default
        The value used when neither `key` nor `old_key` is in the config.
    old_key
        The superseded key read when `key` is absent.

    Returns
    -------
    The config value.
    """
    inversion = conf.instance["visualize"]["general"]["inversion"]

    try:
        return inversion[key]
    except KeyError:
        pass

    if old_key is not None:
        try:
            return inversion[old_key]
        except KeyError:
            pass

    return default


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
    threshold: Optional[float] = None,
    min_pixels: Optional[int] = None,
    total_clumps: Optional[int] = None,
    pix_indexes: Optional[List] = None,
    weight_threshold: float = 0.0,
    region_alpha: float = 0.25,
):
    """
    2x2 subplot showing how the brightest clumps of the source reconstruction map to the image-plane.

    The panels are the data with all other linear objects subtracted, the reconstructed image, the
    source reconstruction zoomed to its brightest region, and the source reconstruction unzoomed.
    Every source clump is drawn as a filled polygon on the two source-plane panels and its
    image-plane regions (the multiple images it maps to) are drawn in the same colour on the two
    image-plane panels, each labelled with the same number in both planes.

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
    threshold
        A source pixel joins a clump if its reconstructed value exceeds this fraction of the
        reconstruction's maximum.  ``None`` reads ``mappings_threshold`` from the visualize config.
    min_pixels
        Connected groups of source pixels smaller than this are not drawn.  ``None`` reads
        ``mappings_min_pixels`` from the visualize config.
    total_clumps
        The maximum number of clumps drawn, keeping the brightest.  ``None`` reads
        ``total_mappings`` from the visualize config.
    pix_indexes
        An explicit list of source pixel index groups which bypasses the clump finding, e.g.
        ``[[0, 1], [10]]``.
    weight_threshold
        A data pixel is in an image-plane region if its summed mapping weight to the clump exceeds
        this value.
    region_alpha
        The alpha of each drawn region's fill.
    """
    mapper = inversion.cls_list_from(cls=Mapper)[pixelization_index]

    if threshold is None:
        threshold = float(_conf_mappings(key="mappings_threshold", default=0.5))
    if min_pixels is None:
        min_pixels = int(_conf_mappings(key="mappings_min_pixels", default=3))
    if total_clumps is None:
        total_clumps = int(
            _conf_mappings(
                key="total_mappings", default=5, old_key="total_mappings_pixels"
            )
        )

    mappings = inversion.mappings_from(
        mapper_index=pixelization_index,
        weight_threshold=weight_threshold,
        threshold=threshold,
        min_pixels=min_pixels,
        total_clumps=total_clumps,
        pix_indexes=pix_indexes,
    )

    image_regions = [mapping.image_contours for mapping in mappings]
    source_regions = [mapping.source_contours for mapping in mappings]
    region_labels = [str(i + 1) for i in range(len(mappings))]

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
            regions=image_regions,
            region_alpha=region_alpha,
            region_labels=region_labels,
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
            regions=image_regions,
            region_alpha=region_alpha,
            region_labels=region_labels,
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
        regions=source_regions,
        region_alpha=region_alpha,
        region_labels=region_labels,
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
        regions=source_regions,
        region_alpha=region_alpha,
        region_labels=region_labels,
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

        with open(
            output_path / f"source_plane_reconstruction_{i}.csv", mode="w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["y", "x", "reconstruction", "noise_map"])
            for j in range(len(x)):
                noise_value = float("nan") if noise_map is None else float(noise_map[j])
                writer.writerow(
                    [float(y[j]), float(x[j]), float(reconstruction[j]), noise_value]
                )
