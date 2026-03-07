import itertools
import numpy as np
from typing import List, Optional, Tuple

from autoconf import conf
from autoconf import cached_property

from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.func_list import UniqueMappings
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.settings import Settings
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.inversion.mappers import mapper_util
from autoarray.inversion.mappers import mapper_numba_util


class Mapper(LinearObj):
    def __init__(
        self,
        interpolator,
        regularization: Optional[AbstractRegularization] = None,
        settings: Settings = None,
        image_plane_mesh_grid=None,
        xp=np,
    ):
        """
        To understand a `Mapper` one must be familiar `Mesh` objects and the `mesh` and `pixelization` packages, where
        the four grids are explained (`image_plane_data_grid`, `source_plane_data_grid`,
        `image_plane_mesh_grid`,`source_plane_mesh_grid`)

        If you are unfamliar withe above objects, read through the docstrings of the `pixelization`, `mesh` and
        `image_mesh` packages.

        A `Mapper` determines the mappings between the masked data grid's pixels (`image_plane_data_grid` and
        `source_plane_data_grid`) and the pxelization's pixels (`image_plane_mesh_grid` and `source_plane_mesh_grid`).

        The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
        change the indexing, such that `source_plane_data_grid[0]` corresponds to the transformed value
        of `image_plane_data_grid[0]` and so on).

        A mapper therefore only needs to determine the index mappings between the `grid_slim` and `mesh_grid`,
        noting that associations are made by pairing `source_plane_mesh_grid` with `source_plane_data_grid`.

        Mappings are represented in the 2D ndarray `pix_indexes_for_sub_slim_index`, whereby the index of
        a pixel on the `mesh_grid` maps to the index of a pixel on the `grid_slim` as follows:

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel (index 0) maps to the
          pixelization's 1st pixel (index 0).
        - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel (index 1) maps to the
          pixelization's 4th pixel (index 3).
        - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel (index 2) maps to the
          pixelization's 2nd pixel (index 1).

        The second dimension of this array (where all three examples above are 0) is used for cases where a
        single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, using a
        `Delaunay` pixelization, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles):

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel (index 0) maps to the
        pixelization's 1st pixel (index 0).
        - pix_indexes_for_sub_slim_index[0, 1] = 3: the data's 1st sub-pixel (index 0) also maps to the
        pixelization's 2nd pixel (index 3).
        - pix_indexes_for_sub_slim_index[0, 2] = 5: the data's 1st sub-pixel (index 0) also maps to the
        pixelization's 6th pixel (index 5).

        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked data pixel annd the pixels of a pixelization. This matrix is the basis of performing an `Inversion`,
        which reconstructs the data using the `source_plane_mesh_grid`.

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_plane_mesh_grid`.
        adapt_data
            An image which is used to determine the `image_plane_mesh_grid` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        mesh_weight_map
            The weight map used to weight the creation of the rectangular mesh grid, which is used for the
            `RectangularBrightness` mesh which adapts the size of its pixels to where the source is reconstructed.
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution,
            which for a mapper smooths neighboring pixels on the mesh.
        """

        super().__init__(regularization=regularization, xp=xp)

        self.interpolator = interpolator

        self.image_plane_mesh_grid = image_plane_mesh_grid

        self.settings = settings or Settings()

    @property
    def params(self) -> int:
        return self.source_plane_mesh_grid.shape[0]

    @property
    def pixels(self) -> int:
        return self.params

    @property
    def mask(self):
        return self.source_plane_data_grid.mask

    @property
    def mesh(self):
        return self.interpolator.mesh

    @property
    def mesh_geometry(self):
        return self.interpolator.mesh_geometry

    @property
    def over_sampler(self):
        return self.source_plane_data_grid.over_sampler

    @property
    def source_plane_data_grid(self):
        return self.interpolator.data_grid

    @property
    def source_plane_mesh_grid(self):
        return self.interpolator.mesh_grid

    @property
    def adapt_data(self):
        return self.interpolator.adapt_data

    @property
    def neighbors(self) -> Neighbors:
        return self.mesh_geometry.neighbors

    @property
    def pix_indexes_for_sub_slim_index(self) -> np.ndarray:
        """
        The mapping of every data pixel (given its `sub_slim_index`) to pixelization pixels (given their `pix_indexes`).

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the pixelization's
        third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the pixelization's
        fifth (index 4) pixel.
        """
        return self.interpolator.mappings

    @property
    def pix_sizes_for_sub_slim_index(self) -> np.ndarray:
        """
        The number of mappings of every data pixel to pixelization pixels.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_sizes_for_sub_slim_index[0] = 2`: The data's first (index 0) sub-pixel maps to 2 pixels in the
        pixelization.

        - `pix_sizes_for_sub_slim_index[2] = 4`: The data's third (index 2) sub-pixel maps to 4 pixels in the
        pixelization.
        """
        return self.interpolator.sizes

    @property
    def pix_weights_for_sub_slim_index(self) -> np.ndarray:
        """
        The interoplation weights of the mapping of every data pixel (given its `sub_slim_index`) to pixelization
        pixels (given their `pix_indexes`).

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_weights_for_sub_slim_index[0, 0] = 0.1`: The data's first (index 0) sub-pixel mapping to the
        pixelization's third (index 2) pixel has an interpolation weight of 0.1.

        - `pix_weights_for_sub_slim_index[2, 0] = 0.2`: The data's third (index 2) sub-pixel mapping to the
        pixelization's fifth (index 4) pixel has an interpolation weight of 0.2.
        """
        return self.interpolator.weights

    @property
    def slim_index_for_sub_slim_index(self) -> np.ndarray:
        """
        The mappings between every sub-pixel data point on the sub-gridded data and each data point for a grid which
        does not use sub gridding (e.g. `sub_size=1`).
        """
        return self.over_sampler.slim_for_sub_slim

    @property
    def sub_slim_indexes_for_pix_index(self) -> List[List]:
        """
        Returns the index mappings between each of the pixelization's pixels and the masked data's sub-pixels.

        Given that even pixelization pixel maps to multiple data sub-pixels, index mappings are returned as a list of
        lists where the first entries are the pixelization index and second entries store the data sub-pixel indexes.

        For example, if `sub_slim_indexes_for_pix_index[2][4] = 10`, the pixelization pixel with index 2
        (e.g. `mesh_grid[2,:]`) has a mapping to a data sub-pixel with index 10 (e.g. `grid_slim[10, :]).

        This is effectively a reversal of the array `pix_indexes_for_sub_slim_index`.
        """
        sub_slim_indexes_for_pix_index = [[] for _ in range(self.pixels)]

        pix_indexes_for_sub_slim_index = self.pix_indexes_for_sub_slim_index

        for slim_index, pix_indexes in enumerate(pix_indexes_for_sub_slim_index):
            for pix_index in pix_indexes:
                sub_slim_indexes_for_pix_index[int(pix_index)].append(slim_index)

        return sub_slim_indexes_for_pix_index

    @cached_property
    def unique_mappings(self) -> UniqueMappings:
        """
        Returns the unique mappings of every unmasked data pixel's (e.g. `grid_slim`) sub-pixels (e.g. `grid_sub_slim`)
        to their corresponding pixelization pixels (e.g. `mesh_grid`).

        To perform an `Inversion` efficiently the linear algebra can bypass the calculation of a `mapping_matrix` and
        instead use the w-tilde formalism, which requires these unique mappings for efficient computation. For
        convenience, these mappings and associated metadata are packaged into the class `UniqueMappings`.

        A full description of these mappings is given in the
        function `mapper_util.data_slim_to_pixelization_unique_from()`.
        """
        (
            data_to_pix_unique,
            data_weights,
            pix_lengths,
        ) = mapper_numba_util.data_slim_to_pixelization_unique_from(
            data_pixels=self.over_sampler.mask.pixels_in_mask,
            pix_indexes_for_sub_slim_index=np.array(
                self.pix_indexes_for_sub_slim_index
            ),
            pix_sizes_for_sub_slim_index=np.array(self.pix_sizes_for_sub_slim_index),
            pix_weights_for_sub_slim_index=np.array(
                self.pix_weights_for_sub_slim_index
            ),
            pix_pixels=self.params,
            sub_size=np.array(self.over_sampler.sub_size).astype("int"),
        )

        return UniqueMappings(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
        )

    @cached_property
    def mapping_matrix(self) -> np.ndarray:
        """
        The `mapping_matrix` of a linear object describes the mappings between the observed data's data-points / pixels
        and the linear object parameters. It is used to construct the simultaneous linear equations which reconstruct
        the data.

        The matrix has shape [total_data_points, data_linear_object_parameters], whereby all non-zero entries
        indicate that a data point maps to a linear object parameter.

        It is described in the following paper as matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf and in more
        detail in the function  `mapper_util.mapping_matrix_from()`.
        """

        return mapper_util.mapping_matrix_from(
            pix_indexes_for_sub_slim_index=self.pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=self.pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=self.pix_weights_for_sub_slim_index,
            pixels=self.pixels,
            total_mask_pixels=self.over_sampler.mask.pixels_in_mask,
            slim_index_for_sub_slim_index=self.slim_index_for_sub_slim_index,
            sub_fraction=self.over_sampler.sub_fraction.array,
            use_mixed_precision=self.settings.use_mixed_precision,
            xp=self._xp,
        )

    @cached_property
    def sparse_triplets_data(self):
        """
        Sparse triplet representation of the (unblurred) mapping operator on the *slim data grid*.

        This property returns the mapping between image-plane subpixels and source pixels in
        sparse COO triplet form:

            (rows, cols, vals)

        where each triplet encodes one non-zero entry of the mapping matrix:

            A[row, col] += val

        The returned indices correspond to:

        - `rows`: slim masked image pixel indices (one per subpixel contribution)
        - `cols`: source pixel indices in the pixelization
        - `vals`: interpolation weights, including oversampling normalization

        This representation is used for efficient computation of quantities such as the
        data vector:

            D = Aᵀ d

        without ever forming the dense mapping matrix explicitly.

        Notes
        -----
        - This version keeps `rows` in *slim masked pixel coordinates*, which is the natural
          indexing convention for data-vector calculations using `psf_operated_data`.
        - The triplets contain only non-zero contributions, making them significantly faster
          than dense matrix operations.

        Returns
        -------
        rows : ndarray of shape (nnz,)
            Slim masked image pixel index for each non-zero mapping entry.

        cols : ndarray of shape (nnz,)
            Source pixel index for each mapping entry.

        vals : ndarray of shape (nnz,)
            Mapping weight for each entry, including subpixel normalization.
        """

        rows, cols, vals = mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=self.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=self.pix_weights_for_sub_slim_index,
            slim_index_for_sub=self.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=self.mask.fft_index_for_masked_pixel,
            sub_fraction_slim=self.over_sampler.sub_fraction.array,
            xp=self._xp,
        )

        return rows, cols, vals

    @cached_property
    def sparse_triplets_curvature(self):
        """
        Sparse triplet representation of the mapping operator on the *rectangular FFT grid*.

        This property returns the same sparse mapping triplets as `sparse_triplets_data`,
        but with the row indices converted from slim masked pixel coordinates into the
        rectangular FFT indexing system used in the w-tilde curvature formalism.

        This is required because curvature matrix calculations involve applying the
        PSF precision operator:

            W = Hᵀ N⁻¹ H

        via FFT-based convolution on a rectangular grid. Therefore the mapping operator
        must be expressed in terms of rectangular pixel indices.

        Specifically:

        - `rows` are converted from slim masked pixel indices into FFT-grid indices via:

              rows_rect = fft_index_for_masked_pixel[rows_slim]

        The resulting triplets are used in curvature matrix assembly:

            F = Aᵀ W A

        Notes
        -----
        - Use `sparse_triplets_data` for data-vector calculations.
        - Use `sparse_triplets_curvature` for curvature matrix calculations with FFT-based
          PSF operators.

        Returns
        -------
        rows : ndarray of shape (nnz,)
            Rectangular FFT-grid pixel index for each mapping entry.

        cols : ndarray of shape (nnz,)
            Source pixel index for each mapping entry.

        vals : ndarray of shape (nnz,)
            Mapping weight for each entry.
        """

        rows, cols, vals = mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=self.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=self.pix_weights_for_sub_slim_index,
            slim_index_for_sub=self.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=self.mask.fft_index_for_masked_pixel,
            sub_fraction_slim=self.over_sampler.sub_fraction.array,
            xp=self._xp,
            return_rows_slim=False,
        )

        return rows, cols, vals

    def pixel_signals_from(self, signal_scale: float, xp=np) -> np.ndarray:
        """
        Returns the signal in each pixelization pixel, where this signal is an estimate of the expected signal
        each pixelization pixel contains given the data pixels it maps too.

        A full description of this is given in the function `mapper_util.adaptive_pixel_signals_from().

        Parameters
        ----------
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
            low signal regions.
        """

        return mapper_util.adaptive_pixel_signals_from(
            pixels=self.pixels,
            signal_scale=signal_scale,
            pixel_weights=self.pix_weights_for_sub_slim_index,
            pix_indexes_for_sub_slim_index=self.pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=self.pix_sizes_for_sub_slim_index,
            slim_index_for_sub_slim_index=self.over_sampler.slim_for_sub_slim,
            adapt_data=self.adapt_data.array,
            xp=xp,
        )

    def slim_indexes_for_pix_indexes(self, pix_indexes: List) -> List[List]:
        """
        Returns the index mappings between every masked data-point (not subgridded) on the data and the mapper
        pixels / parameters that it maps too.

        The `slim_index` refers to the masked data pixels (without subgridding) and `pix_indexes` the pixelization
        pixel indexes, for example:

        - `slim_indexes_for_pix_indexes[0] = [2, 3]`: The data's first (index 0) pixel maps to the
        pixelization's third (index 2) and fourth (index 3) pixels.

        Parameters
        ----------
        pix_indexes
            A list of all pixelization indexes for which the data-points that map to these pixelization pixels are
            computed.
        """
        image_for_source = self.sub_slim_indexes_for_pix_index

        if not any(isinstance(i, list) for i in pix_indexes):
            return list(
                itertools.chain.from_iterable(
                    [image_for_source[index] for index in pix_indexes]
                )
            )
        else:
            indexes = []
            for source_pixel_index_list in pix_indexes:
                indexes.append(
                    list(
                        itertools.chain.from_iterable(
                            [
                                image_for_source[index]
                                for index in source_pixel_index_list
                            ]
                        )
                    )
                )
            return indexes

    def data_weight_total_for_pix_from(self) -> np.ndarray:
        """
        Returns the total weight of every pixelization pixel, which is the sum of the weights of all data-points that
        map to that pixel.
        """

        return mapper_util.data_weight_total_for_pix_from(
            pix_indexes_for_sub_slim_index=np.array(
                self.pix_indexes_for_sub_slim_index
            ),
            pix_weights_for_sub_slim_index=np.array(
                self.pix_weights_for_sub_slim_index
            ),
            pixels=self.pixels,
        )

    def data_pixel_area_for_pix_from(self) -> np.ndarray:
        pass

    def mapped_to_source_from(self, array: Array2D) -> np.ndarray:
        """
        Map a masked 2d image in the image domain to the source domain and sum up all mappings on the source-pixels.

        For example, suppose we have an image and a mapper. We can map every image-pixel to its corresponding mapper's
        source pixel and sum the values based on these mappings.

        This will produce something similar to a `reconstruction`, by passing the linear algebra / inversion.

        Parameters
        ----------
        array_slim
            The masked 2D array of values in its slim representation (e.g. the image data) which are mapped to the
            source domain in order to compute their average values.
        """
        return mapper_util.mapped_to_source_via_mapping_matrix_from(
            mapping_matrix=np.array(self.mapping_matrix),
            array_slim=array.slim.array,
        )

    def extent_from(
        self,
        values: np.ndarray = None,
        zoom_to_brightest: bool = True,
        zoom_percent: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:

        from autoarray.geometry import geometry_util

        if zoom_to_brightest and values is not None:
            if zoom_percent is None:
                zoom_percent = conf.instance["visualize"]["general"]["zoom"][
                    "inversion_percent"
                ]

            fractional_value = np.max(values) * zoom_percent
            fractional_bool = values > fractional_value
            true_indices = np.argwhere(fractional_bool)
            true_grid = self.source_plane_mesh_grid[true_indices]

            try:
                return geometry_util.extent_symmetric_from(
                    extent=(
                        np.min(true_grid[:, 0, 1]),
                        np.max(true_grid[:, 0, 1]),
                        np.min(true_grid[:, 0, 0]),
                        np.max(true_grid[:, 0, 0]),
                    )
                )
            except ValueError:
                return geometry_util.extent_symmetric_from(
                    extent=self.source_plane_mesh_grid.geometry.extent
                )

        return geometry_util.extent_symmetric_from(
            extent=self.source_plane_mesh_grid.geometry.extent
        )

    @property
    def image_plane_data_grid(self):
        return self.mask.derive_grid.unmasked

    @property
    def mesh_pixels_per_image_pixels(self):

        from autoarray.structures.grids import grid_2d_util

        mesh_pixels_per_image_pixels = grid_2d_util.grid_pixels_in_mask_pixels_from(
            grid=np.array(self.image_plane_mesh_grid),
            shape_native=self.mask.shape_native,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

        return Array2D(
            values=mesh_pixels_per_image_pixels,
            mask=self.mask,
        )
