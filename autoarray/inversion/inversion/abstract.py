import copy

import numpy as np
from typing import Dict, List, Optional, Type, Union

from autoconf import cached_property

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.settings import Settings
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.visibilities import Visibilities

from autoarray.util import misc_util
from autoarray.inversion.inversion import inversion_util


class AbstractInversion:
    def __init__(
        self,
        dataset: Union[Imaging, Interferometer, DatasetInterface],
        linear_obj_list: List[LinearObj],
        settings: Settings = None,
        xp=np,
    ):
        """
        An `Inversion` reconstructs an input dataset using a list of linear objects (e.g. a list of analytic functions
        or a pixelized grid).

        The inversion constructs simultaneous linear equations (via vectors and matrices) which allow for the values
        of the linear object parameters that best reconstruct the dataset to be solved, via linear matrix algebra.

        The inversion may be regularized, whereby the parameters of the linear objects used to reconstruct the data
        are smoothed with one another such that their solved for values conform to certain properties (e.g. smoothness
        based regularization requires that parameters in the linear objects which neighbor one another have similar
        values).

        This object contains properties which compute all of the different matrices necessary to perform the inversion.

        The linear algebra required to perform an `Inversion` depends on the type of dataset being fitted (e.g.
        `Imaging`, `Interferometer) and the formalism chosen (e.g. a using a `mapping_matrix` or the
        sparse linear algebra formalism). The children of this class overwrite certain methods in order to be appropriate for
        certain datasets or use a specific formalism.

        Inversions use the formalism's outlined in the following Astronomy papers:

        https://arxiv.org/pdf/astro-ph/0302587.pdf
        https://arxiv.org/abs/1708.07377
        https://arxiv.org/abs/astro-ph/0601493

        Parameters
        ----------
        data
            The data of the dataset (e.g. the `image` of `Imaging` data) which may have been changed.
        noise_map
            The noise_map of the noise_mapset (e.g. the `noise_map` of `Imaging` noise_map) which may have been changed.
        linear_obj_list
            The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
            input dataset's data and whose values are solved for via the inversion.
        settings
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        """

        self.dataset = dataset

        self.linear_obj_list = linear_obj_list

        self.settings = settings or Settings()

        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    @property
    def data(self):
        return self.dataset.data

    @property
    def noise_map(self):
        return self.dataset.noise_map

    def has(self, cls: Type) -> bool:
        """
        Does this `Inversion` have an attribute which is of type `cls`?

        Parameters
        ----------
        dict_values
            A class dictionary of values which instances of the input `cls` are checked to see if it has an instance
            of that class.
        cls
            The type of class that is checked if the object has an instance of.
        """
        return misc_util.has(
            values=self.linear_obj_list + self.regularization_list, cls=cls
        )

    def total(self, cls: Type) -> int:
        """
        Returns the total number of attribute in the `Inversion` which are of type `cls`?

        Parameters
        ----------
        cls
            The type of class that is checked if the object has an instance of.
        """
        return misc_util.total(
            values=self.linear_obj_list + self.regularization_list, cls=cls
        )

    def param_range_list_from(self, cls: Type) -> List[List[int]]:
        """
        Each linear object in the `Inversion` has N parameters, and these parameters correspond to a certain range
        of indexing values in the matrices used to perform the inversion.

        This function returns the `param_range_list` of an input type of linear object, which gives the indexing range
        of each linear object of the input type.

        For example, if an `Inversion` has:

        - A `LinearFuncList` linear object with 3 `params`.
        - A `Mapper` with 100 `params`.
        - A `Mapper` with 200 `params`.

        The corresponding matrices of this inversion (e.g. the `curvature_matrix`) have `shape=(303, 303)` where:

        - The `LinearFuncList` values are in the entries `[0:3]`.
        - The first `Mapper` values are in the entries `[3:103]`.
        - The second `Mapper` values are in the entries `[103:303]

        For this example, `param_range_list_from(cls=Mapper)` therefore returns the
        list `[[3, 103], [103, 303]]`.

        Parameters
        ----------
        cls
            The type of class that the list of their parameter range index values are returned for.

        Returns
        -------
        A list of the index range of the parameters of each linear object in the inversion of the input cls type.
        """
        return inversion_util.param_range_list_from(
            cls=cls, linear_obj_list=self.linear_obj_list
        )

    def cls_list_from(self, cls: Type, cls_filtered: Optional[Type] = None) -> List:
        """
        Returns a list of objects in the `Inversion` which are an instance of the input `cls`.

        The optional `cls_filtered` input removes classes of an input instance type.

        For example:

        - If the input is `cls=aa.mesh.Mesh`, a list containing all pixelizations in the class are returned.

        - If `cls=aa.mesh.Mesh` and `cls_filtered=aa.mesh.RectangularAdaptDensity`, a list of all pixelizations
        excluding those which are `RectangularAdaptDensity` pixelizations will be returned.

        Parameters
        ----------
        cls
            The type of class that a list of instances of this class in the galaxy are returned for.
        cls_filtered
            A class type which is filtered and removed from the class list.
        """
        return misc_util.cls_list_from(
            values=self.linear_obj_list + self.regularization_list,
            cls=cls,
            cls_filtered=cls_filtered,
        )

    @property
    def total_params(self) -> int:
        """
        Returns the total number of parameters used by this `Inversion`, where:

        - Each function in a linear function object list is a parameter.
        - Each pixel of a `Mapper` object is a parameter.

        Returns
        -------
        The total number of parameters used by this inversion.
        """
        return sum(linear_obj.params for linear_obj in self.linear_obj_list)

    @property
    def regularization_list(self) -> List[AbstractRegularization]:
        return [linear_obj.regularization for linear_obj in self.linear_obj_list]

    @property
    def all_linear_obj_have_regularization(self) -> bool:
        return len(self.linear_obj_list) == len(
            list(filter(None, self.regularization_list))
        )

    @property
    def total_regularizations(self) -> int:
        return sum(
            regularization is not None for regularization in self.regularization_list
        )

    @property
    def no_regularization_index_list(self) -> List[int]:
        # TODO : Needs to be range based on pixels.

        no_regularization_index_list = []

        param_range_list = self.param_range_list_from(cls=LinearObj)

        for linear_obj, regularization, param_range in zip(
            self.linear_obj_list, self.regularization_list, param_range_list
        ):
            if regularization is None:
                no_regularization_index_list += range(param_range[0], param_range[1])

        return no_regularization_index_list

    @property
    def mapper_indices(self) -> np.ndarray:

        mapper_indices = []

        param_range_list = self.param_range_list_from(cls=Mapper)

        for param_range in param_range_list:

            mapper_indices += range(param_range[0], param_range[1])

        return np.array(mapper_indices)

    @property
    def mask(self) -> Array2D:
        return self.data.mask

    @property
    def mapping_matrix(self) -> np.ndarray:
        """
        The `mapping_matrix` of a linear object describes the mappings between the observed data's data-points / pixels
        and the linear object parameters. It is used to construct the simultaneous linear equations which reconstruct
        the data.

        The matrix has shape [total_data_points, data_linear_object_parameters], whereby all non-zero entries
        indicate that a data point maps to a linear object parameter.

        It is described in the following paper as matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf and in more
        detail in the function  `mapper_util.mapping_matrix_from()`.

        If there are multiple linear objects, the mapping matrices are stacked such that their simultaneous linear
        equations are solved simultaneously. This property returns the stacked mapping matrix.
        """
        return self._xp.hstack(
            [linear_obj.mapping_matrix for linear_obj in self.linear_obj_list]
        )

    @property
    def operated_mapping_matrix_list(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    def operated_mapping_matrix(self) -> np.ndarray:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        If there are multiple linear objects, the blurred mapping matrices are stacked such that their simultaneous
        linear equations are solved simultaneously.
        """
        return self._xp.hstack(self.operated_mapping_matrix_list)

    @property
    def data_vector(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def curvature_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    def regularization_matrix(self) -> Optional[np.ndarray]:
        """
        The regularization matrix H is used to impose smoothness on our inversion's reconstruction. This enters the
        linear algebra system we solve for using D and F above and is given by
        equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        A complete description of regularization is given in the `regularization.py` and `regularization_util.py`
        modules.

        For multiple mappers, the regularization matrix is computed as the block diagonal of each individual mapper.
        The scipy function `block_diag` has an overhead associated with it and if there is only one mapper and
        regularization it is bypassed.
        """
        if self._xp.__name__.startswith("jax"):
            from jax.scipy.linalg import block_diag

            return block_diag(
                *[
                    linear_obj.regularization_matrix
                    for linear_obj in self.linear_obj_list
                ]
            )
        from scipy.linalg import block_diag

        return block_diag(
            *[linear_obj.regularization_matrix for linear_obj in self.linear_obj_list]
        )

    @cached_property
    def regularization_matrix_reduced(self) -> Optional[np.ndarray]:
        """
        The regularization matrix H is used to impose smoothness on our inversion's reconstruction. This enters the
        linear algebra system we solve for using D and F above and is given by
        equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        A complete description of regularization is given in the `regularization.py` and `regularization_util.py`
        modules.

        For multiple mappers, the regularization matrix is computed as the block diagonal of each individual mapper.
        The scipy function `block_diag` has an overhead associated with it and if there is only one mapper and
        regularization it is bypassed.
        """
        if self.all_linear_obj_have_regularization:
            return self.regularization_matrix

        # ids of values which are on edge so zero-d and not solved for.
        ids_to_keep = self.mapper_indices

        # Zero rows and columns in the matrix we want to ignore
        return self.regularization_matrix[ids_to_keep][:, ids_to_keep]

    @property
    def curvature_reg_matrix(self) -> np.ndarray:
        """
        The linear system of equations solves for F + regularization_coefficient*H, which is computed below.

        For a single mapper, this function overwrites the cached `curvature_matrix`, because for large matrices this
        avoids overheads in memory allocation. The `curvature_matrix` is removed as a cached property as a result,
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        if not self.has(cls=AbstractRegularization):
            return self.curvature_matrix

        return self._xp.add(self.curvature_matrix, self.regularization_matrix)

    @property
    def curvature_reg_matrix_reduced(self) -> Optional[np.ndarray]:
        """
        The regularization matrix H is used to impose smoothness on our inversion's reconstruction. This enters the
        linear algebra system we solve for using D and F above and is given by
        equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        A complete description of regularization is given in the `regularization.py` and `regularization_util.py`
        modules.

        For multiple mappers, the regularization matrix is computed as the block diagonal of each individual mapper.
        The scipy function `block_diag` has an overhead associated with it and if there is only one mapper and
        regularization it is bypassed.
        """
        if self.all_linear_obj_have_regularization:
            return self.curvature_reg_matrix

        # ids of values which are on edge so zero-d and not solved for.
        ids_to_keep = self.mapper_indices

        # Zero rows and columns in the matrix we want to ignore
        return self.curvature_reg_matrix[ids_to_keep][:, ids_to_keep]

    @cached_property
    def zeroed_ids_to_keep(self):
        """
        Return the **positive global indices** of linear parameters that should be
        kept (solved for) in the inversion, accounting for **zeroed pixel indices**
        from one or more mappers.

        ---------------------------------------------------------------------------
        Parameter vector layout
        ---------------------------------------------------------------------------
        This method assumes the full linear parameter vector is ordered as:

            [ non-pixel linear objects ][ mapper_0 pixels ][ mapper_1 pixels ] ... [ mapper_M pixels ]

        where:

        - *Non-pixel linear objects* include quantities such as analytic light
          profiles, regularization amplitudes, etc.
        - Each mapper contributes a contiguous block of pixel-based linear parameters.
        - The concatenated pixel blocks occupy the **final** entries of the parameter
          vector, with total length:

              total_pixels = sum(mapper.mesh.pixels for mapper in mappers)

        ---------------------------------------------------------------------------
        Zeroed pixel convention
        ---------------------------------------------------------------------------
        For each mapper:

        - `mapper.mesh.zeroed_pixels` must be a 1D array of **positive, mesh-local**
          pixel indices in the range `[0, mapper.mesh.pixels - 1]`.
        - These indices identify pixels that should be **excluded** from the linear
          solve (e.g. edge pixels, masked regions, or padding pixels).
        - Indexing is defined purely within the mapper’s own pixelization (e.g.
          row-major flattening for rectangular meshes).

        This method converts all mesh-local zeroed pixel indices into **global
        parameter indices**, correctly offsetting for:
          - the presence of non-pixel linear objects at the start of the vector
          - the cumulative pixel counts of preceding mappers

        ---------------------------------------------------------------------------
        Backend and implementation details
        ---------------------------------------------------------------------------
        - The implementation is backend-agnostic and supports both NumPy and JAX via
          `self._xp`.
        - The returned indices are **positive global indices**, suitable for advanced
          indexing of:
              - `self.data_vector`
              - `self.curvature_reg_matrix`
        - When using JAX, this method avoids backend-incompatible operations and
          preserves JIT compatibility under the same constraints as the rest of the
          inversion pipeline.

        Returns
        -------
        array-like
            A 1D array of **positive global indices**, sorted in ascending order,
            corresponding to linear parameters that should be kept in the inversion.
        """

        mapper_list = self.cls_list_from(cls=Mapper)

        n_total = int(self.total_params)

        pixels_per_mapper = [int(m.mesh.pixels) for m in mapper_list]
        total_pixels = int(sum(pixels_per_mapper))

        # Global start index of concatenated pixel block
        pixel_start = n_total - total_pixels

        # Total number of zeroed pixels across all mappers (Python int => static)
        total_zeroed = int(sum(len(m.mesh.zeroed_pixels) for m in mapper_list))
        n_keep = int(n_total - total_zeroed)

        # Build global indices-to-zero across all mappers
        zeros_global_list = []
        offset = 0
        for m, n_pix in zip(mapper_list, pixels_per_mapper):
            zeros_local = self._xp.asarray(m.mesh.zeroed_pixels, dtype=self._xp.int32)
            zeros_global_list.append(pixel_start + offset + zeros_local)
            offset += n_pix

        zeros_global = (
            self._xp.concatenate(zeros_global_list)
            if len(zeros_global_list) > 0
            else self._xp.asarray([], dtype=self._xp.int32)
        )

        keep = self._xp.ones((n_total,), dtype=bool)

        if self._xp is np:
            keep[zeros_global] = False
            keep_ids = self._xp.nonzero(keep)[0]

        else:
            keep = keep.at[zeros_global].set(False)
            keep_ids = self._xp.nonzero(keep, size=n_keep)[0]

        return keep_ids

    @cached_property
    def reconstruction(self) -> np.ndarray:
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf (Positive-Negative solution)

        ============================================================================================

        Solve the Eq.(2) of https://arxiv.org/pdf/astro-ph/0302587.pdf (Non-negative solution)
        Find non-negative solution that minimizes |Z * S - x|^2.

        We use fnnls (https://github.com/jvendrow/fnnls) to optimize the quadratic value. Two commonly used
        variables in the code are defined as follows:
            ZTZ := np.dot(Z.T, Z)
            ZTx := np.dot(Z.T, x)
        """

        if self.settings.use_positive_only_solver:

            if self.settings.use_edge_zeroed_pixels and self.has(cls=Mapper):

                # Use advanced indexing to select rows/columns
                data_vector = self.data_vector[self.zeroed_ids_to_keep]
                curvature_reg_matrix = self.curvature_reg_matrix[
                    self.zeroed_ids_to_keep
                ][:, self.zeroed_ids_to_keep]

                # Perform reconstruction via fnnls
                reconstruction_partial = (
                    inversion_util.reconstruction_positive_only_from(
                        data_vector=data_vector,
                        curvature_reg_matrix=curvature_reg_matrix,
                        settings=self.settings,
                        xp=self._xp,
                    )
                )

                # Allocate full solution array
                reconstruction = self._xp.zeros(self.data_vector.shape[0])

                # Scatter the partial solution back to the full shape
                if self._xp.__name__.startswith("jax"):
                    reconstruction = reconstruction.at[self.zeroed_ids_to_keep].set(
                        reconstruction_partial
                    )
                else:
                    reconstruction[self.zeroed_ids_to_keep] = reconstruction_partial

                return reconstruction

            else:

                return inversion_util.reconstruction_positive_only_from(
                    data_vector=self.data_vector,
                    curvature_reg_matrix=self.curvature_reg_matrix,
                    settings=self.settings,
                    xp=self._xp,
                )

        return inversion_util.reconstruction_positive_negative_from(
            data_vector=self.data_vector,
            curvature_reg_matrix=self.curvature_reg_matrix,
            xp=self._xp,
        )

    @cached_property
    def reconstruction_reduced(self) -> np.ndarray:
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf

        S is the vector of reconstructed inversion values.
        """
        if self.all_linear_obj_have_regularization:
            return self.reconstruction

        # ids of values which are on edge so zero-d and not solved for.
        ids_to_keep = self.mapper_indices

        # Zero rows and columns in the matrix we want to ignore
        return self.reconstruction[ids_to_keep]

    @property
    def reconstruction_dict(self) -> Dict[LinearObj, np.ndarray]:
        return self.source_quantity_dict_from(source_quantity=self.reconstruction)

    def source_quantity_dict_from(
        self, source_quantity: np.ndarray
    ) -> Dict[LinearObj, np.ndarray]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays via stacking. This does not track
        which quantities belong to which linear objects, therefore the linear equation's solutions (which are returned
        as ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `source_quantity` (like a `reconstruction`) to a dictionary of ndarrays,
        where the keys are the instances of each mapper in the inversion.

        Parameters
        ----------
        source_quantity
            The quantity whose values are mapped to a dictionary of values for each individual mapper.

        Returns
        -------
        The dictionary of ndarrays of values for each individual mapper.
        """
        source_quantity_dict = {}

        index = 0

        for linear_obj in self.linear_obj_list:
            source_quantity_dict[linear_obj] = source_quantity[
                index : index + linear_obj.params
            ]

            index += linear_obj.params

        return source_quantity_dict

    @property
    def mapped_reconstructed_operated_data_dict(self) -> Dict[LinearObj, Array2D]:
        raise NotImplementedError

    @property
    def mapped_reconstructed_data(self) -> Union[Array2D, Visibilities]:
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image data.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """
        return sum(self.mapped_reconstructed_data_dict.values())

    @property
    def mapped_reconstructed_operated_data(self) -> Union[Array2D, Visibilities]:
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image data.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """
        return sum(self.mapped_reconstructed_operated_data_dict.values())

    @property
    def data_subtracted_dict(self) -> Dict[LinearObj, Array2D]:
        """
        Returns a dictionary of the data subtracted by the reconstructed images of combinations of all but one of the
        linear objects the inversion.

        This produces images of the data showing what each linear object is actually fitted to, after accounting for
        the signal in the other linear objects.

        Returns
        -------
        A dictionary of the data subtracted by the reconstructed images of combinations of all but one of the
        linear objects the inversion.
        """

        data_subtracted_dict = {}

        for linear_obj in self.linear_obj_list:
            data_subtracted_dict[linear_obj] = copy.copy(self.data)

            for linear_obj_other in self.linear_obj_list:
                if linear_obj != linear_obj_other:
                    data_subtracted_dict[
                        linear_obj
                    ] -= self.mapped_reconstructed_operated_data_dict[linear_obj_other]

        return data_subtracted_dict

    @property
    def regularization_term(self) -> float:
        """
        Returns the regularization term of an inversion. This term represents the sum of the difference in flux
        between every pair of neighboring pixels.

        This is computed as:

        s_T * H * s = solution_vector.T * regularization_matrix * solution_vector

        The term is referred to as *G_l* in Warren & Dye 2003, Nightingale & Dye 2015.

        The above works include the regularization_matrix coefficient (lambda) in this calculation. In PyAutoLens,
        this is already in the regularization matrix and thus implicitly included in the matrix multiplication.
        """
        if not self.has(cls=AbstractRegularization):
            return 0.0

        return self._xp.matmul(
            self.reconstruction_reduced.T,
            self._xp.matmul(
                self.regularization_matrix_reduced, self.reconstruction_reduced
            ),
        )

    @property
    def log_det_curvature_reg_matrix_term(self) -> float:
        """
        The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

        This uses the Cholesky decomposition which is already computed before solving the reconstruction.
        """
        if not self.has(cls=AbstractRegularization):
            return 0.0

        return 2.0 * self._xp.sum(
            self._xp.log(
                self._xp.diag(
                    self._xp.linalg.cholesky(self.curvature_reg_matrix_reduced)
                )
            )
        )

    @property
    def log_det_regularization_matrix_term(self) -> float:
        """
        The Bayesian evidence of an inversion which quantifies its overall goodness-of-fit uses the log determinant
        of regularization matrix, Log[Det[Lambda*H]].

        Unlike the determinant of the curvature reg matrix, which uses an existing preloading Cholesky decomposition
        used for the source reconstruction, this uses scipy sparse linear algebra to solve the determinant efficiently.

        Returns
        -------
        float
            The log determinant of the regularization matrix.
        """
        if not self.has(cls=AbstractRegularization):
            return 0.0

        return 2.0 * self._xp.sum(
            self._xp.log(
                self._xp.diag(
                    self._xp.linalg.cholesky(self.regularization_matrix_reduced)
                )
            )
        )

    @property
    def reconstruction_noise_map_with_covariance(self) -> np.ndarray:
        """
        Returns the noise-map of the reconstruction as a two dimension matrix which accounts for the covariance
        of the noise between pixels.

        The diagonal of this matrix is the noise-map of the reconstruction, which can be used for analysing the
        reconstruction with noise properties that are representative of the fit and therefore should be used
        for any scientific analysis (e.g. source reconstructions of strong lenses).

        This noise-map is defined as the RMS standard deviation of the noise in every pixel of the reconstruction.
        This definition is identical to the `noise_map` attributes of dataset objects.

        It is computed as the square root of the inverse of the curvature matrix with regularization, which is the
        same matrix used to solve for the reconstruction via the linear inversion.

        Returns
        -------
        The noise-map of the reconstruction as a two dimension matrix which accounts for the covariance of the noise
        between pixels.
        """
        return np.sqrt(np.linalg.inv(self.curvature_reg_matrix))

    @property
    def reconstruction_noise_map(self):
        """
        Returns the noise-map of the reconstruction as a one dimensional ndarray, which does not account for the
        covariance of the noise between pixels.

        This matrix is representative of the noise properties of the fit and should be used for any scientific
        analysis (e.g. source reconstructions of strong lenses).

        The noise-map of the reconstruction is the RMS standard deviation of the noise in every pixel of the
        reconstruction. This definition is identical to the `noise_map` attributes of dataset objects.

        It is computed as the square root of the diagonal of the `reconstruction_noise_map_with_covariance` matrix,
        which is the same matrix used to solve for the reconstruction via the linear inversion.

        Returns
        -------
        The noise-map of the reconstruction as a one dimensional ndarray, which does not account for the covariance
        of the noise between pixels.
        """
        return np.diagonal(self.reconstruction_noise_map_with_covariance)

    @property
    def reconstruction_noise_map_dict(self) -> Dict[LinearObj, np.ndarray]:
        return self.source_quantity_dict_from(
            source_quantity=self.reconstruction_noise_map
        )

    def regularization_weights_from(self, index: int) -> np.ndarray:
        linear_obj = self.linear_obj_list[index]
        regularization = self.regularization_list[index]

        if regularization is None:
            pixels = linear_obj.params

            return np.zeros((pixels,))

        return regularization.regularization_weights_from(
            linear_obj=linear_obj, xp=self._xp
        )

    @property
    def regularization_weights_mapper_dict(self) -> Dict[LinearObj, np.ndarray]:
        regularization_weights_dict = {}

        for index, mapper in enumerate(self.cls_list_from(cls=Mapper)):
            regularization_weights_dict[mapper] = self.regularization_weights_from(
                index=index,
            )

        return regularization_weights_dict

    @property
    def _data_vector_mapper(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def _curvature_matrix_mapper_diag(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    @property
    def linear_func_operated_mapping_matrix_dict(self) -> Dict:
        raise NotImplementedError

    @property
    def data_linear_func_matrix_dict(self):
        raise NotImplementedError

    @property
    def mapper_operated_mapping_matrix_dict(self) -> Dict:
        raise NotImplementedError

    def max_pixel_list_from(
        self,
        total_pixels: int = 1,
        filter_neighbors: bool = False,
        mapper_index: int = 0,
    ) -> List[List[int]]:
        """
        Returns a list of lists of the maximum cell or pixel values in the mapper.

        Neighbors can be filtered such that each maximum value in a pixel is higher than all surrounding pixels,
        thus forming a `peak` in the mapper values.

        For example, if a `reconstruction` is the mapper values and neighbor filtering is on, this would return the
        brightest pixels in the mapper reconstruction which are brighter than all pixels around them.

        In gravitational lensing, these peaks are the brightest regions of the source reconstruction and correspond
        to features like the centre of the source galaxy and knots of star formation in a galaxy.

        Parameters
        ----------
        total_pixels
            The total number of pixels to return in the list of peak pixels.
        filter_neighbors
            If True, the peak pixels are filtered such that they are the brightest pixel in the mapper and all
            of its neighbors.
        mapper_index
            The index of the mapper in the inversion to compute the max pixels for, where there may be multiple
            mappers in the inversion.

        Returns
        -------

        """
        mapper = self.cls_list_from(cls=Mapper)[mapper_index]
        reconstruction = self.reconstruction_dict[mapper]

        max_pixel_list = []

        pixel_list = []

        pixels_ascending_list = list(reversed(np.argsort(reconstruction)))

        for pixel in range(total_pixels):
            pixel_index = pixels_ascending_list[pixel]

            add_pixel = True

            if filter_neighbors:
                pixel_neighbors = mapper.neighbors[pixel_index]
                pixel_neighbors = pixel_neighbors[pixel_neighbors >= 0]

                max_value = reconstruction[pixel_index]
                max_value_neighbors = reconstruction[pixel_neighbors]

                if max_value < np.max(max_value_neighbors):
                    add_pixel = False

            if add_pixel:
                pixel_list.append(pixel_index)

        max_pixel_list.append(pixel_list)

        return max_pixel_list

    def max_pixel_centre(self, mapper_index: int = 0) -> Grid2DIrregular:
        """
        Returns the centre of the brightest pixel in the mapper values.

        Parameters
        ----------
        mapper_index
            The index of the mapper in the inversion to compute the max pixels for, where there may be multiple
            mappers in the inversion.

        Returns
        -------
        The centre of the brightest pixel in the mapper values.
        """
        mapper = self.cls_list_from(cls=Mapper)[mapper_index]
        reconstruction = self.reconstruction_dict[mapper]

        max_pixel = np.argmax(reconstruction)

        max_pixel_centre = Grid2DIrregular(
            values=[mapper.source_plane_mesh_grid.array[max_pixel]]
        )

        return max_pixel_centre
