import logging
from typing import Optional


from autoarray.plot.array import plot_array
from autoarray.plot.inversion import plot_inversion_reconstruction
from autoarray.plot.utils import (
    subplots,
    numpy_grid,
    numpy_lines,
    subplot_save,
    conf_subplot_figsize,
    tight_layout,
)

logger = logging.getLogger(__name__)


def plot_mapper(
    mapper,
    solution_vector=None,
    output_path: Optional[str] = None,
    output_filename: str = "mapper",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    vmin=None,
    vmax=None,
    mesh_grid=None,
    lines=None,
    line_colors=None,
    regions=None,
    region_colors=None,
    region_alpha: float = 0.25,
    region_labels=None,
    title: str = "Pixelization Mesh (Source-Plane)",
    zoom_to_brightest: bool = True,
    zoom_extent_scale: float = 1.0,
    ax=None,
):
    """
    Plot a pixelization mapper reconstruction.

    Parameters
    ----------
    mapper
        A ``Mapper`` instance.
    solution_vector
        Per-pixel flux values.  ``None`` uses uniform colours.
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
    mesh_grid
        Mesh grid to overlay as scatter points.
    lines
        Lines to overlay.
    regions
        Source-plane regions to overlay as filled polygons, each a list of ``(N, 2)`` arrays of
        ``(y, x)`` coordinates (e.g. the ``source_contours`` of a ``Mapping``).
    region_colors
        The colour of each region, cycled if there are more regions than colours.
    region_alpha
        The alpha of each region's fill.
    region_labels
        A text label drawn at the centre of each region.
    title
        Figure title.
    zoom_to_brightest
        Zoom the source plane to the brightest region.
    ax
        Existing ``Axes`` to draw onto.
    """
    try:
        plot_inversion_reconstruction(
            pixel_values=solution_vector,
            mapper=mapper,
            ax=ax,
            title=title,
            colormap=colormap,
            use_log10=use_log10,
            vmin=vmin,
            vmax=vmax,
            zoom_to_brightest=zoom_to_brightest,
            zoom_extent_scale=zoom_extent_scale,
            lines=numpy_lines(lines),
            line_colors=line_colors,
            regions=regions,
            region_colors=region_colors,
            region_alpha=region_alpha,
            region_labels=region_labels,
            grid=numpy_grid(mesh_grid),
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
        )
    except Exception as exc:
        logger.info(f"Could not plot the source-plane via the Mapper: {exc}")


def subplot_image_and_mapper(
    mapper,
    image,
    output_path: Optional[str] = None,
    output_filename: str = "image_and_mapper",
    output_format: str = None,
    colormap=None,
    use_log10: bool = False,
    mesh_grid=None,
    lines=None,
    regions=None,
    region_colors=None,
    region_labels=None,
):
    """
    1×2 subplot: image-plane image (left) and pixelization mesh (right).

    Parameters
    ----------
    mapper
        A ``Mapper`` instance.
    image
        An ``Array2D`` instance to show in the image plane.
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
    mesh_grid
        Mesh grid to overlay on the reconstruction panel.
    lines
        Lines to overlay on both panels.
    regions
        A list of ``Mapping`` objects (e.g. from ``mapper.mappings_from(pix_indexes=...)``), drawn
        as filled polygons in matched colours: each mapping's image-plane regions on the left panel
        and its source-plane mesh pixels on the right panel.
    region_colors
        The colour of each region, cycled if there are more regions than colours.
    region_labels
        A text label drawn at the centre of each region, e.g. ``["1", "2"]``.
    """
    fig, axes = subplots(1, 2, figsize=conf_subplot_figsize(1, 2))

    image_regions = (
        None if regions is None else [mapping.image_contours for mapping in regions]
    )
    source_regions = (
        None if regions is None else [mapping.source_contours for mapping in regions]
    )

    plot_array(
        image,
        ax=axes[0],
        title="Image (Image-Plane)",
        colormap=colormap,
        use_log10=use_log10,
        lines=lines,
        regions=image_regions,
        region_colors=region_colors,
        region_labels=region_labels,
    )
    plot_mapper(
        mapper,
        colormap=colormap,
        use_log10=use_log10,
        mesh_grid=mesh_grid,
        lines=lines,
        regions=source_regions,
        region_colors=region_colors,
        region_labels=region_labels,
        ax=axes[1],
    )

    tight_layout()
    subplot_save(fig, output_path, output_filename, output_format)
