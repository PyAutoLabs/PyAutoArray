import numpy as np
from typing import Dict, List, Optional, Union

from autoarray import exc
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.settings import Settings
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.inversion.inversion import inversion_util


class AbstractInversionInterferometer(AbstractInversion):
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

        Parameters
        ----------
        dataset
            The interferometer dataset being reconstructed (e.g. an `Interferometer` dataset or a `DatasetInterface`
            whose attributes like `data`, `noise_map`, and `transformer` may have been modified).
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        settings
            Settings controlling how an inversion is fitted, for example which linear algebra formalism is used.
        xp
            The array module to use (`numpy` by default; pass `jax.numpy` for JAX support).
        """

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            preloads=preloads,
        )

    @property
    def transformer(self):
        return self.dataset.transformer

    @property
    def mask(self) -> Mask2D:
        return self.transformer.real_space_mask

    @property
    def operated_mapping_matrix_list(self) -> List[np.ndarray]:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values
        and the linear objects model, including a non-uniform fast Fourier transform operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        This property returns the a list of each linear object's transformed mapping matrix.

        A linear object may have a `operated_mapping_matrix_override` property, which bypasses the `mapping_matrix`
        computation and transformer operation and is directly placed in the `operated_mapping_matrix_list`. Because
        the override bypasses the transformer it must already be in the data's visibility space, with (complex)
        shape [total_visibilities, params] (e.g. computed via an analytic Fourier transform).
        """
        operated_mapping_matrix_list = []

        for linear_obj in self.linear_obj_list:
            operated_mapping_matrix_override = linear_obj.operated_mapping_matrix_override

            if operated_mapping_matrix_override is not None:
                expected_shape = (self.data.shape[0], linear_obj.params)

                if tuple(operated_mapping_matrix_override.shape) != expected_shape:
                    raise exc.InversionException(
                        f"The `operated_mapping_matrix_override` of a linear object input to an interferometer "
                        f"inversion has shape {tuple(operated_mapping_matrix_override.shape)} but shape "
                        f"{expected_shape} ([total_visibilities, params]) is required.\n\n"
                        f"For an interferometer dataset the override bypasses the transformer entirely and is "
                        f"placed directly in the `operated_mapping_matrix_list`, therefore it must be in the "
                        f"data's visibility space (unlike the real-space `mapping_matrix`, which the transformer "
                        f"maps to visibilities)."
                    )

                operated_mapping_matrix_list.append(operated_mapping_matrix_override)

            else:
                operated_mapping_matrix_list.append(
                    self.transformer.transform_mapping_matrix(
                        mapping_matrix=linear_obj.mapping_matrix, xp=self._xp
                    )
                )

        return operated_mapping_matrix_list

    @property
    def mapped_reconstructed_data_dict(
        self,
    ) -> Dict[LinearObj, Array2D]:
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

        For the linear equations which fit interferometer datasets, the reconstructed data is its visibilities. Thus,
        the reconstructed image is computed separately by performing a non-uniform fast Fourier transform which maps
        the `reconstruction`'s values to real space.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """
        mapped_reconstructed_data_dict = {}

        reconstruction_dict = self.source_quantity_dict_from(
            source_quantity=self.reconstruction
        )

        for linear_obj in self.linear_obj_list:
            reconstruction = reconstruction_dict[linear_obj]

            mapped_reconstructed_data = (
                inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
                    mapping_matrix=linear_obj.mapping_matrix,
                    reconstruction=reconstruction,
                    xp=self._xp,
                )
            )

            mapped_reconstructed_data = Array2D(
                values=mapped_reconstructed_data, mask=self.mask
            )

            mapped_reconstructed_data_dict[linear_obj] = mapped_reconstructed_data

        return mapped_reconstructed_data_dict

    @property
    def fast_chi_squared(self):
        """
        Returns the chi-squared of the interferometer inversion without needing to form the full residual visibilities.

        This is computed directly from the reconstruction and the matrices of the inversion:

        chi_squared = s^T F s - 2 s^T D + sum(d_r^2/sigma_r^2) + sum(d_i^2/sigma_i^2)

        where `s` is the reconstruction vector, `F` is the curvature matrix, `D` is the data vector,
        and `d_r`/`d_i` are the real/imaginary parts of the observed visibilities.

        This avoids computing the full mapped reconstructed visibilities and is faster than computing
        `chi_squared` via the residual visibilities when many source pixels are used.
        """
        xp = self._xp

        chi_squared_term_1 = xp.linalg.multi_dot(
            [
                self.reconstruction.T,  # (M,)
                self.curvature_matrix,  # (M, M)
                self.reconstruction,  # (M,)
            ]
        )

        chi_squared_term_2 = -2.0 * xp.linalg.multi_dot(
            [
                self.reconstruction.T,  # (M,)
                self.data_vector,  # (M,)
            ]
        )

        chi_squared_term_3 = xp.sum(
            self.dataset.data.array.real**2.0 / self.dataset.noise_map.array.real**2.0
        ) + xp.sum(
            self.dataset.data.array.imag**2.0 / self.dataset.noise_map.array.imag**2.0
        )

        return chi_squared_term_1 + chi_squared_term_2 + chi_squared_term_3
