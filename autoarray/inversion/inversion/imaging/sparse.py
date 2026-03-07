import numpy as np
from typing import Dict, List, Optional, Union

from autoconf import cached_property

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.imaging.abstract import AbstractInversionImaging
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.settings import Settings
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.inversion.inversion.imaging import inversion_imaging_util


class InversionImagingSparse(AbstractInversionImaging):
    def __init__(
        self,
        dataset: Union[Imaging, DatasetInterface],
        linear_obj_list: List[LinearObj],
        settings: Settings = None,
        xp=np,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.inversion.abstract.AbstractInversion`) for a full description.

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Imaging` objects, where the data is an image
        and the mappings may include a convolution operation described by the imaging data's PSF.

        This class uses the w-tilde formalism, which speeds up the construction of the simultaneous linear equations by
        bypassing the construction of a `mapping_matrix`.

        Parameters
        ----------
        dataset
            The dataset containing the image data, noise-map and psf which is fitted by the inversion.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        """

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
        )

    @cached_property
    def psf_weighted_data(self):
        return inversion_imaging_util.psf_weighted_data_from(
            weight_map_native=self.dataset.sparse_operator.weight_map.array,
            kernel_native=self.psf.kernel.native,
            native_index_for_slim_index=self.data.mask.derive_indexes.native_for_slim,
            xp=self._xp,
        )

    @property
    def _data_vector_mapper(self) -> np.ndarray:
        """
        Returns the `data_vector` of all mappers, a 1D vector whose values are solved for by the simultaneous
        linear equations constructed by this object. The object is described in full in the method `data_vector`.

        This method is used to compute part of the `data_vector` if there are also linear function list objects
        in the inversion, and is separated into a separate method to enable preloading of the mapper `data_vector`.
        """

        if not self.has(cls=Mapper):
            return None

        data_vector = self._xp.zeros(self.total_params)

        mapper_list = self.cls_list_from(cls=Mapper)
        mapper_param_range = self.param_range_list_from(cls=Mapper)

        for mapper_index, mapper in enumerate(mapper_list):

            rows, cols, vals = mapper.sparse_triplets_data

            data_vector_mapper = (
                inversion_imaging_util.data_vector_via_psf_weighted_data_from(
                    psf_weighted_data=self.psf_weighted_data,
                    rows=rows,
                    cols=cols,
                    vals=vals,
                    S=mapper.params,
                )
            )
            param_range = mapper_param_range[mapper_index]

            start = param_range[0]
            end = param_range[1]

            if self._xp is np:
                data_vector[start:end] = data_vector_mapper
            else:
                data_vector = data_vector.at[start:end].set(data_vector_mapper)

        return data_vector

    @cached_property
    def data_vector(self) -> np.ndarray:
        """
        Returns the `data_vector`, a 1D vector whose values are solved for by the simultaneous linear equations
        constructed by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects a `data_vector` is computed for each one, which are concatenated
        ensuring their values are solved for simultaneously.

        The calculation is described in more detail in `inversion_util.psf_weighted_data_from`.
        """
        if self.has(cls=AbstractLinearObjFuncList):
            return self._data_vector_func_list_and_mapper
        elif self.total(cls=Mapper) == 1:
            return self._data_vector_x1_mapper
        return self._data_vector_multi_mapper

    @property
    def _data_vector_x1_mapper(self) -> np.ndarray:
        """
        Returns the `data_vector`, a 1D vector whose values are solved for by the simultaneous linear equations
        constructed by this object. The object is described in full in the method `data_vector`.

        This method computes the `data_vector` whenthere is a single mapper object in the `Inversion`,
        which circumvents `np.concatenate` for speed up.
        """
        linear_obj = self.linear_obj_list[0]

        rows, cols, vals = linear_obj.sparse_triplets_data

        return inversion_imaging_util.data_vector_via_psf_weighted_data_from(
            psf_weighted_data=self.psf_weighted_data,
            rows=rows,
            cols=cols,
            vals=vals,
            S=linear_obj.params,
        )

    @property
    def _data_vector_multi_mapper(self) -> np.ndarray:
        """
        Returns the `data_vector`, a 1D vector whose values are solved for by the simultaneous linear equations
        constructed by this object. The object is described in full in the method `data_vector`.

        This method computes the `data_vector` when there are multiple mapper objects in the `Inversion`,
        which computes the `data_vector` of each object and concatenates them.
        """

        data_vector_list = []

        for mapper in self.cls_list_from(cls=Mapper):

            rows, cols, vals = mapper.sparse_triplets_data

            data_vector_mapper = (
                inversion_imaging_util.data_vector_via_psf_weighted_data_from(
                    psf_weighted_data=self.psf_weighted_data,
                    rows=rows,
                    cols=cols,
                    vals=vals,
                    S=mapper.params,
                )
            )

            data_vector_list.append(data_vector_mapper)

        return self._xp.concatenate(data_vector_list)

    @property
    def _data_vector_func_list_and_mapper(self) -> np.ndarray:
        """
        Returns the `data_vector`, a 1D vector whose values are solved for by the simultaneous linear equations
        constructed by this object. The object is described in full in the method `data_vector`.

        This method computes the `data_vector` when there are one or more mapper objects in the `Inversion`,
        which are combined with linear function list objects.

        The `data_vector` corresponding to all mapper objects is computed first, in a separate function. This
        separation of functions enables the `data_vector` to be preloaded in certain circumstances.
        """

        data_vector = self._xp.array(self._data_vector_mapper)

        linear_func_param_range = self.param_range_list_from(
            cls=AbstractLinearObjFuncList
        )

        for linear_func_index, linear_func in enumerate(
            self.cls_list_from(cls=AbstractLinearObjFuncList)
        ):
            operated_mapping_matrix = self.linear_func_operated_mapping_matrix_dict[
                linear_func
            ]

            diag = inversion_imaging_util.data_vector_via_blurred_mapping_matrix_from(
                blurred_mapping_matrix=operated_mapping_matrix,
                image=self.data.array,
                noise_map=self.noise_map.array,
            )

            param_range = linear_func_param_range[linear_func_index]

            start = param_range[0]
            end = param_range[1]

            if self._xp is np:
                data_vector[start:end] = diag
            else:
                data_vector = data_vector.at[start:end].set(diag)

        return data_vector

    @cached_property
    def curvature_matrix(self) -> np.ndarray:
        """
        Returns the `curvature_matrix`, a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes F using the sparse linear algebra formalism, which is faster as it precomputes the PSF convolution
        of different noise-map pixels (see `curvature_matrix_diag_via_sparse_operator_from`).

        If there are multiple linear objects the curvature_matrices are combined to ensure their values are solved
        for simultaneously. In the w-tilde formalism this requires us to consider the mappings between data and every
        linear object, meaning that the linear alegbra has both on and off diagonal terms.

        The `curvature_matrix` computed here is overwritten in memory when the regularization matrix is added to it,
        because for large matrices this avoids overhead. For this reason, `curvature_matrix` is not a cached property
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        if self.has(cls=AbstractLinearObjFuncList):
            curvature_matrix = self._curvature_matrix_func_list_and_mapper
        elif self.total(cls=Mapper) == 1:
            curvature_matrix = self._curvature_matrix_x1_mapper
        else:
            curvature_matrix = self._curvature_matrix_multi_mapper

        curvature_matrix = inversion_imaging_util.curvature_matrix_mirrored_from(
            curvature_matrix=curvature_matrix,
            xp=self._xp,
        )

        if len(self.no_regularization_index_list) > 0:
            curvature_matrix = (
                inversion_imaging_util.curvature_matrix_with_added_to_diag_from(
                    curvature_matrix=curvature_matrix,
                    value=self.settings.no_regularization_add_to_curvature_diag_value,
                    no_regularization_index_list=self.no_regularization_index_list,
                    xp=self._xp,
                )
            )

        return curvature_matrix

    @property
    def _curvature_matrix_mapper_diag(self) -> Optional[np.ndarray]:
        """
        Returns the diagonal regions of the `curvature_matrix`, a 2D matrix which uses the mappings between the data
        and the linear objects to construct the simultaneous linear equations. The object is described in full in
        the method `curvature_matrix`.

        This method computes the diagonal entries of all mapper objects in the `curvature_matrix`. It is separate from
        other calculations to enable preloading of this calculation.
        """

        if not self.has(cls=Mapper):
            return None

        curvature_matrix = self._xp.zeros((self.total_params, self.total_params))

        mapper_list = self.cls_list_from(cls=Mapper)
        mapper_param_range_list = self.param_range_list_from(cls=Mapper)

        for i in range(len(mapper_list)):
            mapper_i = mapper_list[i]
            mapper_param_range_i = mapper_param_range_list[i]

            rows, cols, vals = mapper_i.sparse_triplets_curvature

            diag = self.dataset.sparse_operator.curvature_matrix_diag_from(
                rows=rows,
                cols=cols,
                vals=vals,
                S=mapper_i.params,
            )

            start, end = mapper_param_range_i

            if self._xp is np:
                curvature_matrix[start:end, start:end] = diag
            else:
                curvature_matrix = curvature_matrix.at[start:end, start:end].set(diag)

            if self.total(cls=Mapper) == 1:
                return curvature_matrix

        return curvature_matrix

    def _curvature_matrix_off_diag_from(
        self, mapper_0: Mapper, mapper_1: Mapper
    ) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes the off-diagonal terms of F using the sparse linear algebra formalism.
        """

        rows0, cols0, vals0 = mapper_0.sparse_triplets_curvature
        rows1, cols1, vals1 = mapper_1.sparse_triplets_curvature

        S0 = mapper_0.params
        S1 = mapper_1.params

        return self.dataset.sparse_operator.curvature_matrix_off_diag_from(
            rows0=rows0,
            cols0=cols0,
            vals0=vals0,
            rows1=rows1,
            cols1=cols1,
            vals1=vals1,
            S0=S0,
            S1=S1,
        )

    @property
    def _curvature_matrix_x1_mapper(self) -> np.ndarray:
        """
        Returns the `curvature_matrix`, a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations. The object is described in full in the method `curvature_matrix`.

        This method computes the `curvature_matrix` when there is a single mapper object in the `Inversion`,
        which circumvents `block_diag` for speed up.
        """
        return self._curvature_matrix_mapper_diag

    @property
    def _curvature_matrix_multi_mapper(self) -> np.ndarray:
        """
        Returns the `curvature_matrix`, a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations. The object is described in full in the method `curvature_matrix`.

        This method computes the `curvature_matrix` when there are multiple mapper objects in the `Inversion`,
        by computing each one (and their off-diagonal matrices) and combining them via the `block_diag` method.
        """

        curvature_matrix = self._curvature_matrix_mapper_diag

        if self.total(cls=Mapper) == 1:
            return curvature_matrix

        mapper_list = self.cls_list_from(cls=Mapper)
        mapper_param_range_list = self.param_range_list_from(cls=Mapper)

        for i in range(len(mapper_list)):
            mapper_i = mapper_list[i]
            mapper_param_range_i = mapper_param_range_list[i]

            for j in range(i + 1, len(mapper_list)):
                mapper_j = mapper_list[j]
                mapper_param_range_j = mapper_param_range_list[j]

                off_diag = self._curvature_matrix_off_diag_from(
                    mapper_0=mapper_i, mapper_1=mapper_j
                )

                curvature_matrix[
                    mapper_param_range_i[0] : mapper_param_range_i[1],
                    mapper_param_range_j[0] : mapper_param_range_j[1],
                ] = off_diag

        return curvature_matrix

    @property
    def _curvature_matrix_func_list_and_mapper(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes the diagonal terms of F using the sparse linear algebra formalism.
        """

        curvature_matrix = self._curvature_matrix_multi_mapper

        mapper_list = self.cls_list_from(cls=Mapper)
        mapper_param_range_list = self.param_range_list_from(cls=Mapper)

        linear_func_list = self.cls_list_from(cls=AbstractLinearObjFuncList)
        linear_func_param_range_list = self.param_range_list_from(
            cls=AbstractLinearObjFuncList
        )

        for i in range(len(mapper_list)):
            mapper = mapper_list[i]
            mapper_param_range = mapper_param_range_list[i]

            for func_index, linear_func in enumerate(linear_func_list):
                linear_func_param_range = linear_func_param_range_list[func_index]

                curvature_weights = (
                    self.linear_func_operated_mapping_matrix_dict[linear_func]
                    / self.noise_map[:, None] ** 2
                )

                rows, cols, vals = mapper.sparse_triplets_curvature

                off_diag = self.dataset.sparse_operator.curvature_matrix_off_diag_func_list_from(
                    curvature_weights=curvature_weights,
                    fft_index_for_masked_pixel=self.mask.fft_index_for_masked_pixel,
                    rows=rows,
                    cols=cols,
                    vals=vals,
                    S=mapper.params,
                )

                if self._xp is np:

                    curvature_matrix[
                        mapper_param_range[0] : mapper_param_range[1],
                        linear_func_param_range[0] : linear_func_param_range[1],
                    ] = off_diag
                else:

                    curvature_matrix = curvature_matrix.at[
                        mapper_param_range[0] : mapper_param_range[1],
                        linear_func_param_range[0] : linear_func_param_range[1],
                    ].set(off_diag)

        for index_0, linear_func_0 in enumerate(linear_func_list):

            linear_func_param_range_0 = linear_func_param_range_list[index_0]

            weighted_vector_0 = (
                self.linear_func_operated_mapping_matrix_dict[linear_func_0]
                / self.noise_map[:, None]
            )

            for index_1, linear_func_1 in enumerate(linear_func_list):
                linear_func_param_range_1 = linear_func_param_range_list[index_1]

                weighted_vector_1 = (
                    self.linear_func_operated_mapping_matrix_dict[linear_func_1]
                    / self.noise_map[:, None]
                )

                diag = self._xp.dot(
                    weighted_vector_0.T,
                    weighted_vector_1,
                )

                if self._xp is np:

                    curvature_matrix[
                        linear_func_param_range_0[0] : linear_func_param_range_0[1],
                        linear_func_param_range_1[0] : linear_func_param_range_1[1],
                    ] = diag

                else:

                    curvature_matrix = curvature_matrix.at[
                        linear_func_param_range_0[0] : linear_func_param_range_0[1],
                        linear_func_param_range_1[0] : linear_func_param_range_1[1],
                    ].set(diag)

        return curvature_matrix

    def _mapped_reconstructed_data_dict_from(
        self,
        *,
        use_operated_for_linear_func: bool,
    ) -> Dict["LinearObj", "Array2D"]:
        """
        Shared implementation for mapping a reconstruction to image-plane arrays for each linear object.

        For Mapper objects this uses the sparse operator mapping, and optionally applies the PSF.
        For linear-func objects this uses either the operated or unoperated mapping matrix dict.
        """
        mapped_dict = {}

        reconstruction_dict = self.source_quantity_dict_from(
            source_quantity=self.reconstruction
        )

        for linear_obj in self.linear_obj_list:

            reconstruction = reconstruction_dict[linear_obj]

            if isinstance(linear_obj, Mapper):

                rows, cols, vals = linear_obj.sparse_triplets_curvature

                mapped = inversion_imaging_util.mapped_reconstructed_operated_data_via_sparse_operator_from(
                    reconstruction=reconstruction,
                    rows=rows,
                    cols=cols,
                    vals=vals,
                    fft_index_for_masked_pixel=self.mask.fft_index_for_masked_pixel,
                    data_shape=self.mask.shape_native,
                )

                mapped = Array2D(values=mapped, mask=self.mask)

                if use_operated_for_linear_func:

                    mapped = self.psf.convolved_image_from(
                        image=mapped,
                        blurring_image=None,
                        xp=self._xp,
                    ).array

                    mapped = Array2D(values=mapped, mask=self.mask)

            else:

                if use_operated_for_linear_func:
                    mapping_matrix = self.linear_func_operated_mapping_matrix_dict[
                        linear_obj
                    ]
                else:
                    mapping_matrix = self.linear_func_mapping_matrix_dict[linear_obj]

                mapped = self._xp.sum(reconstruction * mapping_matrix, axis=1)
                mapped = Array2D(values=mapped, mask=self.mask)

            mapped_dict[linear_obj] = mapped

        return mapped_dict

    @property
    def mapped_reconstructed_data_dict(self) -> Dict[LinearObj, Array2D]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays via stacking. This does not track
        which quantities belong to which linear objects, therefore the linear equation's solutions (which are returned
        as ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed data values, where the keys are the instances of each mapper in the inversion.

        The images are the unconvolved reconstructed data values, meaning they are the solved for reconstruction
        with PSF operations removed.

        The w-tilde formalism bypasses the calculation of the `mapping_matrix` and it therefore cannot be used to map
        the reconstruction's values to the image-plane. Instead, the unique data-to-pixelization mappings are used,
        including the 2D convolution operation after mapping is complete.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """
        return self._mapped_reconstructed_data_dict_from(
            use_operated_for_linear_func=False
        )

    @property
    def mapped_reconstructed_operated_data_dict(self) -> Dict[LinearObj, Array2D]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays via stacking. This does not track
        which quantities belong to which linear objects, therefore the linear equation's solutions (which are returned
        as ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed data values, where the keys are the instances of each mapper in the inversion.

        The images are the convolved reconstructed data values, meaning they are the solved for reconstruction with PSF
        operations included.

        The w-tilde formalism bypasses the calculation of the `mapping_matrix` and it therefore cannot be used to map
        the reconstruction's values to the image-plane. Instead, the unique data-to-pixelization mappings are used,
        including the 2D convolution operation after mapping is complete.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """
        return self._mapped_reconstructed_data_dict_from(
            use_operated_for_linear_func=True
        )
