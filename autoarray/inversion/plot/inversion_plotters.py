import numpy as np

from autoconf import conf

from autoarray.inversion.mappers.abstract import Mapper
from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.plot.mapper_plotters import MapperPlotter


class InversionPlotter(AbstractPlotter):
    def __init__(
        self,
        inversion: AbstractInversion,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `Inversion` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Inversion` and plotted via the visuals object.

        Parameters
        ----------
        inversion
            The inversion the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        """
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.inversion = inversion
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def mapper_plotter_from(self, mapper_index: int) -> MapperPlotter:
        """
        Returns a `MapperPlotter` corresponding to the `Mapper` in the `Inversion`'s `linear_obj_list` given an input
        `mapper_index`.

        Parameters
        ----------
        mapper_index
            The index of the mapper in the inversion which is used to create the `MapperPlotter`.

        Returns
        -------
        MapperPlotter
            An object that plots mappers which is used for plotting attributes of the inversion.
        """
        return MapperPlotter(
            mapper=self.inversion.cls_list_from(cls=Mapper)[mapper_index],
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
        )

    def figures_2d(self, reconstructed_operated_data: bool = False):
        """
        Plots the individual attributes of the plotter's `Inversion` object in 2D.

        The API is such that every plottable attribute of the `Inversion` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        reconstructed_operated_data
            Whether to make a 2D plot (via `imshow`) of the reconstructed image data.
        """
        if reconstructed_operated_data:
            try:
                self.mat_plot_2d.plot_array(
                    array=self.inversion.mapped_reconstructed_operated_data,
                    visuals_2d=self.visuals_2d,
                    auto_labels=AutoLabels(
                        title="Reconstructed Image",
                        filename="reconstructed_operated_data",
                    ),
                )
            except AttributeError:
                self.mat_plot_2d.plot_array(
                    array=self.inversion.mapped_reconstructed_data,
                    visuals_2d=self.visuals_2d,
                    auto_labels=AutoLabels(
                        title="Reconstructed Image", filename="reconstructed_data"
                    ),
                )

    def figures_2d_of_pixelization(
        self,
        pixelization_index: int = 0,
        data_subtracted: bool = False,
        reconstructed_operated_data: bool = False,
        reconstruction: bool = False,
        reconstruction_noise_map: bool = False,
        signal_to_noise_map: bool = False,
        regularization_weights: bool = False,
        sub_pixels_per_image_pixels: bool = False,
        mesh_pixels_per_image_pixels: bool = False,
        image_pixels_per_mesh_pixel: bool = False,
        magnification_per_mesh_pixel: bool = False,
        zoom_to_brightest: bool = True,
    ):
        """
        Plots the individual attributes of a specific `Mapper` of the plotter's `Inversion` object in 2D.

        The API is such that every plottable attribute of the `Mapper` and `Inversion` object is an input parameter of
        type bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        pixelization_index
            The index of the `Mapper` in the `Inversion`'s `linear_obj_list` that is plotted.
        reconstructed_operated_data
            Whether to make a 2D plot (via `imshow`) of the mapper's reconstructed image data.
        reconstruction
            Whether to make a 2D plot (via `imshow` or `fill`) of the mapper's source-plane reconstruction.
        reconstruction_noise_map
            Whether to make a 2D plot (via `imshow` or `fill`) of the mapper's source-plane noise-map.
        signal_to_noise_map
            Whether to make a 2D plot (via `imshow` or `fill`) of the mapper's source-plane signal-to-noise-map.
        sub_pixels_per_image_pixels
            Whether to make a 2D plot (via `imshow`) of the number of sub pixels per image pixels in the 2D
            data's mask.
        mesh_pixels_per_image_pixels
            Whether to make a 2D plot (via `imshow`) of the number of image-mesh pixels per image pixels in the 2D
            data's mask (only valid for pixelizations which use an `image_mesh`, e.g. Hilbert, KMeans).
        image_pixels_per_mesh_pixel
            Whether to make a 2D plot (via `imshow`) of the number of image pixels per source plane pixel, therefore
            indicating how many image pixels map to each source pixel.
        magnification_per_mesh_pixel
            Whether to make a 2D plot (via `imshow`) of the magnification of each mesh pixel, which is the area
            ratio of the image pixel to source pixel.
        zoom_to_brightest
            For images not in the image-plane (e.g. the `plane_image`), whether to automatically zoom the plot to
            the brightest regions of the galaxies being plotted as opposed to the full extent of the grid.
        """

        if not self.inversion.has(cls=Mapper):
            return

        mapper_plotter = self.mapper_plotter_from(mapper_index=pixelization_index)

        if data_subtracted:
            # Attribute error is cause this raises an error for interferometer inversion, because the data is
            # visibilities not an image. Update this to be handled better in future.

            try:
                array = self.inversion.data_subtracted_dict[mapper_plotter.mapper]

                self.mat_plot_2d.plot_array(
                    array=array,
                    visuals_2d=self.visuals_2d,
                    grid_indexes=mapper_plotter.mapper.over_sampler.uniform_over_sampled,
                    auto_labels=AutoLabels(
                        title="Data Subtracted", filename="data_subtracted"
                    ),
                )
            except AttributeError:
                pass

        if reconstructed_operated_data:
            array = self.inversion.mapped_reconstructed_operated_data_dict[
                mapper_plotter.mapper
            ]

            from autoarray.structures.visibilities import Visibilities

            if isinstance(array, Visibilities):
                array = self.inversion.mapped_reconstructed_data_dict[
                    mapper_plotter.mapper
                ]

            self.mat_plot_2d.plot_array(
                array=array,
                visuals_2d=self.visuals_2d,
                grid_indexes=mapper_plotter.mapper.over_sampler.uniform_over_sampled,
                auto_labels=AutoLabels(
                    title="Reconstructed Image", filename="reconstructed_operated_data"
                ),
            )

        if reconstruction:
            vmax_custom = False

            if "vmax" in self.mat_plot_2d.cmap.kwargs:
                if self.mat_plot_2d.cmap.kwargs["vmax"] is None:
                    reconstruction_vmax_factor = conf.instance["visualize"]["general"][
                        "inversion"
                    ]["reconstruction_vmax_factor"]

                    self.mat_plot_2d.cmap.kwargs["vmax"] = (
                        reconstruction_vmax_factor
                        * np.max(self.inversion.reconstruction)
                    )
                    vmax_custom = True

            pixel_values = self.inversion.reconstruction_dict[mapper_plotter.mapper]

            mapper_plotter.plot_source_from(
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
                auto_labels=AutoLabels(
                    title="Source Reconstruction", filename="reconstruction"
                ),
            )

            if vmax_custom:
                self.mat_plot_2d.cmap.kwargs["vmax"] = None

        if reconstruction_noise_map:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=self.inversion.reconstruction_noise_map_dict[
                        mapper_plotter.mapper
                    ],
                    auto_labels=AutoLabels(
                        title="Noise Map", filename="reconstruction_noise_map"
                    ),
                )

            except TypeError:
                pass

        if signal_to_noise_map:
            try:
                signal_to_noise_values = (
                    self.inversion.reconstruction_dict[mapper_plotter.mapper]
                    / self.inversion.reconstruction_noise_map_dict[
                        mapper_plotter.mapper
                    ]
                )

                mapper_plotter.plot_source_from(
                    pixel_values=signal_to_noise_values,
                    auto_labels=AutoLabels(
                        title="Signal To Noise Map", filename="signal_to_noise_map"
                    ),
                )

            except TypeError:
                pass

        if regularization_weights:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=self.inversion.regularization_weights_mapper_dict[
                        mapper_plotter.mapper
                    ],
                    auto_labels=AutoLabels(
                        title="Regularization weight_list",
                        filename="regularization_weights",
                    ),
                )
            except (IndexError, ValueError):
                pass

        if sub_pixels_per_image_pixels:
            sub_size = Array2D(
                values=mapper_plotter.mapper.over_sampler.sub_size,
                mask=self.inversion.dataset.mask,
            )

            self.mat_plot_2d.plot_array(
                array=sub_size,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Sub Pixels Per Image Pixels",
                    filename="sub_pixels_per_image_pixels",
                ),
            )

        if mesh_pixels_per_image_pixels:
            try:
                mesh_pixels_per_image_pixels = (
                    mapper_plotter.mapper.mesh_pixels_per_image_pixels
                )

                self.mat_plot_2d.plot_array(
                    array=mesh_pixels_per_image_pixels,
                    visuals_2d=self.visuals_2d,
                    auto_labels=AutoLabels(
                        title="Mesh Pixels Per Image Pixels",
                        filename="mesh_pixels_per_image_pixels",
                    ),
                )
            except Exception:
                pass

        if image_pixels_per_mesh_pixel:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=mapper_plotter.mapper.data_weight_total_for_pix_from(),
                    auto_labels=AutoLabels(
                        title="Image Pixels Per Source Pixel",
                        filename="image_pixels_per_mesh_pixel",
                    ),
                )

            except TypeError:
                pass

    def subplot_of_mapper(
        self, mapper_index: int = 0, auto_filename: str = "subplot_inversion"
    ):
        """
        Plots the individual attributes of a specific `Mapper` of the plotter's `Inversion` object in 2D on a subplot.

        Parameters
        ----------
        mapper_index
            The index of the `Mapper` in the `Inversion`'s `linear_obj_list` that is plotted.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """

        self.open_subplot_figure(number_subplots=12)

        contour_original = self.mat_plot_2d.contour

        if self.mat_plot_2d.use_log10:
            self.mat_plot_2d.contour = False

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, data_subtracted=True
        )

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_operated_data=True
        )

        self.mat_plot_2d.use_log10 = True
        self.mat_plot_2d.contour = False

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_operated_data=True
        )

        self.mat_plot_2d.use_log10 = False

        mapper = self.inversion.cls_list_from(cls=Mapper)[mapper_index]

        self.visuals_2d += Visuals2D(mesh_grid=mapper.image_plane_mesh_grid)

        self.set_title(label="Mesh Pixel Grid Overlaid")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_operated_data=True
        )
        self.set_title(label=None)

        self.visuals_2d.mesh_grid = None

        # self.include_2d._mapper_image_plane_mesh_grid = False

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstruction=True
        )

        self.set_title(label="Source Reconstruction (Unzoomed)")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstruction=True,
            zoom_to_brightest=False,
        )
        self.set_title(label=None)

        self.set_title(label="Noise-Map (Unzoomed)")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstruction_noise_map=True,
            zoom_to_brightest=False,
        )

        self.set_title(label="Regularization Weights (Unzoomed)")
        try:
            self.figures_2d_of_pixelization(
                pixelization_index=mapper_index,
                regularization_weights=True,
                zoom_to_brightest=False,
            )
        except IndexError:
            pass
        self.set_title(label=None)

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, sub_pixels_per_image_pixels=True
        )

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, mesh_pixels_per_image_pixels=True
        )

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, image_pixels_per_mesh_pixel=True
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"{auto_filename}_{mapper_index}"
        )

        self.mat_plot_2d.contour = contour_original

        self.close_subplot_figure()

    def subplot_mappings(
        self, pixelization_index: int = 0, auto_filename: str = "subplot_mappings"
    ):
        self.open_subplot_figure(number_subplots=4)

        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, data_subtracted=True
        )

        total_pixels = conf.instance["visualize"]["general"]["inversion"][
            "total_mappings_pixels"
        ]

        mapper = self.inversion.cls_list_from(cls=Mapper)[pixelization_index]

        pix_indexes = self.inversion.max_pixel_list_from(
            total_pixels=total_pixels,
            filter_neighbors=True,
            mapper_index=pixelization_index,
        )

        indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

        self.visuals_2d.indexes = indexes

        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, reconstructed_operated_data=True
        )

        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, reconstruction=True
        )

        self.set_title(label="Source Reconstruction (Unzoomed)")
        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index,
            reconstruction=True,
            zoom_to_brightest=False,
        )
        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"{auto_filename}_{pixelization_index}"
        )

        self.close_subplot_figure()
