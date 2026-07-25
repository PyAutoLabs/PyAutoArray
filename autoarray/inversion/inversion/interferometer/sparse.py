import numpy as np
from typing import Dict, List, Union

from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.interferometer.abstract import (
    AbstractInversionInterferometer,
)
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.mappers import mapper_util
from autoarray.settings import Settings
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.structures.visibilities import Visibilities


class InversionInterferometerSparse(AbstractInversionInterferometer):
    def __init__(
        self,
        dataset: Union[Interferometer, DatasetInterface],
        linear_obj_list: List[LinearObj],
        settings: Settings = None,
        xp=np,
        preloads=None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.inversion.abstract.AbstractInversion` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Interferometer` objects, where the data is an
        an array of visibilities and the mappings include a non-uniform fast Fourier transform operation described by
        the interferometer dataset's transformer.

        This class uses the w-tilde formalism, which speeds up the construction of the simultaneous linear equations by
        bypassing the construction of a `mapping_matrix`.

        Parameters
        ----------
        noise_map
            The noise-map of the observed interferometer data which values are solved for.
        transformer
            The transformer which performs a non-uniform fast Fourier transform operations on the mapping matrix
            with the interferometer data's transformer.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        """
        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            preloads=preloads,
        )

        self.settings = settings

    @property
    def data_vector(self) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects the `data_vectors` are concatenated ensuring their values are solved
        for simultaneously.

        The calculation is described in more detail in `inversion_util.weighted_data_interferometer_from`.
        """
        return self._xp.dot(
            self.mapping_matrix.T, self.dataset.sparse_operator.dirty_image
        )

    @property
    def curvature_matrix(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        If there are multiple linear objects their `operated_mapping_matrix` properties will have already been
        concatenated ensuring their `curvature_matrix` values are solved for simultaneously. This includes all
        diagonal and off-diagonal terms describing the covariances between linear objects.

        If a `preloads.curvature_matrix` was injected (e.g. the datacube shared-state path, where `F` is
        identical for every channel) it is returned directly, so the dominant `F = LᵀW̃L` build is skipped.
        This is the only invariant inversion-setup quantity that must be preloaded explicitly: the mapper
        (and therefore `mapping_matrix` and `regularization_matrix`) is already reused by passing the same
        `linear_obj_list` to every channel's inversion.
        """
        if self._preloads is not None and self._preloads.curvature_matrix is not None:
            return self._preloads.curvature_matrix

        return self.curvature_matrix_diag

    @property
    def curvature_matrix_diag(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes the diagonal terms of F using the sparse linear algebra formalism.
        """
        mapper = self.cls_list_from(cls=Mapper)[0]

        # The interferometer W~ operator lives on the unmasked-extent rectangular
        # grid (shape_native_masked_pixels), not the full native grid used by
        # the imaging path. Build sparse triplets with extent-flat row indices
        # so they match the operator's (M = extent_y * extent_x, B) scatter buffer.
        rows, cols, vals = mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
            slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=self.mask.extent_index_for_masked_pixel,
            sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
            return_rows_slim=False,
            xp=self._xp,
        )

        return self.dataset.sparse_operator.curvature_matrix_diag_from(
            rows=rows,
            cols=cols,
            vals=vals,
            S=mapper.params,
        )

    @property
    def mapped_reconstructed_operated_data_dict(
        self,
    ) -> Dict[LinearObj, Visibilities]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays. This does not track which
        quantities belong to which linear objects, therefore the linear equation's solutions (which are returned as
        ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed images, where the keys are the instances of each mapper in the inversion.

        To perform this mapping the `mapping_matrix` is used, which straightforwardly describes how every value of
        the `reconstruction` maps to pixels in the data-frame after the 2D non-uniform fast Fourier transformer
        operation has been performed.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """
        mapped_reconstructed_operated_data_dict = {}

        image_dict = self.mapped_reconstructed_data_dict

        for linear_obj in self.linear_obj_list:
            visibilities = self.transformer.visibilities_from(
                image=image_dict[linear_obj], xp=self._xp
            )

            visibilities = Visibilities(visibilities=visibilities)

            mapped_reconstructed_operated_data_dict[linear_obj] = visibilities

        return mapped_reconstructed_operated_data_dict
