import numpy as np
from typing import Dict, List, Optional, Union

from autoconf import cached_property

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.imaging.abstract import AbstractInversionImaging
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.settings import Settings
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.imaging import inversion_imaging_util


class InversionImagingMapping(AbstractInversionImaging):
    def __init__(
        self,
        dataset: Union[Imaging, DatasetInterface],
        linear_obj_list: List[LinearObj],
        settings: Settings = None,
        xp=np,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.inversion.abstract.AbstractInversion` for a full description.

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Imaging` objects, where the data is an image
        and the mappings may include a convolution operation described by the imaging data's PSF.

        This class uses the mapping formalism, which constructs the simultaneous linear equations using the
        `mapping_matrix` of every linear object.

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
    def data_vector(self) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects their `operated_mapping_matrix` properties will have already been
        concatenated ensuring their `data_vector` values are solved for simultaneously.

        The calculation is described in more detail in `inversion_util.data_vector_via_blurred_mapping_matrix_from`.
        """
        return inversion_imaging_util.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=self.operated_mapping_matrix,
            image=self.data.array,
            noise_map=self.noise_map.array,
        )

    @cached_property
    def curvature_matrix(self):
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        If there are multiple linear objects their `operated_mapping_matrix` properties will have already been
        concatenated ensuring their `curvature_matrix` values are solved for simultaneously. This includes all
        diagonal and off-diagonal terms describing the covariances between linear objects.

        The `curvature_matrix` computed here is overwritten in memory when the regularization matrix is added to it,
        because for large matrices this avoids overhead. For this reason, `curvature_matrix` is not a cached property
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """

        return inversion_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=self.operated_mapping_matrix,
            noise_map=self.noise_map,
            settings=self.settings,
            add_to_curvature_diag=True,
            no_regularization_index_list=self.no_regularization_index_list,
            xp=self._xp,
        )

    def _mapped_reconstructed_data_dict_from(
        self,
        mapping_matrix_list,
    ) -> Dict["LinearObj", "Array2D"]:
        """
        Shared implementation for mapping a reconstruction to image-plane arrays
        using a provided list of mapping matrices (operated or unoperated).
        """
        mapped_reconstructed_data_dict = {}

        reconstruction_dict = self.source_quantity_dict_from(
            source_quantity=self.reconstruction
        )

        for index, linear_obj in enumerate(self.linear_obj_list):
            reconstruction = reconstruction_dict[linear_obj]

            mapped_reconstructed_operated_data = (
                inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
                    mapping_matrix=mapping_matrix_list[index],
                    reconstruction=reconstruction,
                    xp=self._xp,
                )
            )

            mapped_reconstructed_operated_data = Array2D(
                values=mapped_reconstructed_operated_data,
                mask=self.mask,
            )

            mapped_reconstructed_data_dict[linear_obj] = (
                mapped_reconstructed_operated_data
            )

        return mapped_reconstructed_data_dict

    @property
    def mapped_reconstructed_data_dict(self) -> Dict["LinearObj", "Array2D"]:
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

        To perform this mapping the `mapping_matrix` is used, which straightforwardly describes how every value of
        the `reconstruction` maps to pixels in the data-frame after the 2D convolution operation has been performed.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """
        return self._mapped_reconstructed_data_dict_from(
            mapping_matrix_list=self.mapping_matrix_list
        )

    @property
    def mapped_reconstructed_operated_data_dict(self) -> Dict["LinearObj", "Array2D"]:
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

        To perform this mapping the `mapping_matrix` is used, which straightforwardly describes how every value of
        the `reconstruction` maps to pixels in the data-frame after the 2D convolution operation has been performed.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """
        return self._mapped_reconstructed_data_dict_from(
            mapping_matrix_list=self.operated_mapping_matrix_list
        )
