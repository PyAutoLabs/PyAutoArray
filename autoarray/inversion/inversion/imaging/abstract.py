import numpy as np
from typing import Dict, List, Union, Type

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.settings import Settings

from autoarray.inversion.inversion.imaging import inversion_imaging_util
from autoarray import exc


class AbstractInversionImaging(AbstractInversion):
    def __init__(
        self,
        dataset: Union[Imaging, DatasetInterface],
        linear_obj_list: List[LinearObj],
        settings: Settings = None,
        xp=np,
    ):
        """
        An `Inversion` reconstructs an input dataset using a list of linear objects (e.g. a list of analytic functions
        or a pixelized grid).

        The inversion constructs simultaneous linear equations (via vectors and matrices) which allow for the values
        of the linear object parameters that best reconstruct the dataset to be solved, via linear matrix algebra.

        This object contains matrices and vectors which perform an inversion for fits to an `Imaging` dataset. This
        includes operations which use a PSF in order to incorporate blurring into the solved for linear object pixels.

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
        dataset
            The imaging dataset being reconstructed (e.g. an `Imaging` dataset or a `DatasetInterface` whose
            attributes like `data`, `noise_map`, and `psf` may have been modified before being passed in).
        linear_obj_list
            The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
            input dataset's data and whose values are solved for via the inversion.
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
        )

    @property
    def psf(self):
        return self.dataset.psf

    def _mapping_matrix_for_convolution_from(self, linear_obj) -> np.ndarray:
        """
        The mapping matrix a linear object contributes to PSF convolution.

        For a regular PSF this is the image-resolution `mapping_matrix`. For an
        oversampled PSF (`convolve_over_sample_size > 1`) convolution must happen
        before the sub-pixel fold, so a `Mapper` contributes its
        `mapping_matrix_over_sampled`; linear function objects are evaluated at
        image resolution and cannot be convolved at the fine resolution, so they
        raise (deferred formalism — see the design approved in #353, phase 2b).
        """
        if self.psf.convolve_over_sample_size == 1:
            return linear_obj.mapping_matrix

        if isinstance(linear_obj, Mapper):
            return linear_obj.mapping_matrix_over_sampled

        raise exc.InversionException(
            "Oversampled PSF convolution (convolve_over_sample_size > 1) currently "
            "supports Mapper linear objects only. Linear function objects (e.g. "
            "linear light profiles without an operated override) are evaluated at "
            "image resolution and their fine-resolution convolution is deferred "
            "future work."
        )

    @property
    def mapping_matrix_list(self) -> List[np.ndarray]:
        """
        The `mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear object's model, before an operation like a convolution is applied.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        This property returns a list containing each linear object's unoperated (unconvolved) `mapping_matrix`
        as defined on the corresponding `LinearObj` instance.
        """
        return [linear_obj.mapping_matrix for linear_obj in self.linear_obj_list]

    @property
    def operated_mapping_matrix_list(self) -> List[np.ndarray]:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        This property returns the a list of each linear object's blurred mapping matrix, which is computed by
        blurring each linear object's `mapping_matrix` property with the `psf` operator.

        A linear object may have a `operated_mapping_matrix_override` property, which bypasses  the `mapping_matrix`
        computation and convolution operator and is directly placed in the `operated_mapping_matrix_list`.
        """

        return [
            (
                self.psf.convolved_mapping_matrix_from(
                    mapping_matrix=self._mapping_matrix_for_convolution_from(
                        linear_obj
                    ),
                    mask=self.mask,
                    use_mixed_precision=self.settings.use_mixed_precision,
                    xp=self._xp,
                )
                if linear_obj.operated_mapping_matrix_override is None
                else self.linear_func_operated_mapping_matrix_dict[linear_obj]
            )
            for linear_obj in self.linear_obj_list
        ]

    def _updated_cls_key_dict_from(self, cls: Type, preload_dict: Dict) -> Dict:
        cls_dict = {}

        for linear_func, values in zip(
            self.cls_list_from(cls=cls),
            preload_dict.values(),
        ):
            cls_dict[linear_func] = values

        return cls_dict

    @property
    def linear_func_mapping_matrix_dict(self) -> Dict:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation. It is described fully in the method
        `operated_mapping_matrix`.

        This property returns a dictionary mapping every linear func object to its corresponded operated mapping
        matrix, which is used for constructing the matrices that perform the linear inversion in an efficent way
        for the psf precision operator calculation.

        Returns
        -------
        A dictionary mapping every linear function object to its operated mapping matrix.
        """

        linear_func_mapping_matrix_dict = {}

        for linear_func in self.cls_list_from(cls=AbstractLinearObjFuncList):

            linear_func_mapping_matrix_dict[linear_func] = linear_func.mapping_matrix

        return linear_func_mapping_matrix_dict

    @property
    def linear_func_operated_mapping_matrix_dict(self) -> Dict:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation. It is described fully in the method
        `operated_mapping_matrix`.

        This property returns a dictionary mapping every linear func object to its corresponded operated mapping
        matrix, which is used for constructing the matrices that perform the linear inversion in an efficent way
        for the psf precision operator calculation.

        Returns
        -------
        A dictionary mapping every linear function object to its operated mapping matrix.
        """

        linear_func_operated_mapping_matrix_dict = {}

        for linear_func in self.cls_list_from(cls=AbstractLinearObjFuncList):
            if linear_func.operated_mapping_matrix_override is not None:
                operated_mapping_matrix = linear_func.operated_mapping_matrix_override
            else:
                operated_mapping_matrix = self.psf.convolved_mapping_matrix_from(
                    mapping_matrix=self._mapping_matrix_for_convolution_from(
                        linear_func
                    ),
                    mask=self.mask,
                    xp=self._xp,
                )

            linear_func_operated_mapping_matrix_dict[linear_func] = (
                operated_mapping_matrix
            )

        return linear_func_operated_mapping_matrix_dict

    @property
    def data_linear_func_matrix_dict(self):
        """
        Returns a matrix that for each data pixel, maps it to the sum of the values of a linear object function
        convolved with the PSF kernel at the data pixel.

        If a linear function in an inversion is fixed, its values can be evaluated and preloaded beforehand. For every
        data pixel, the PSF convolution with this preloaded linear function can also be preloaded, in a matrix of
        shape [data_pixels, 1].

        Given that multiple linear functions can be used and fixed in an inversion, this matrix is extended to have
        dimensions [data_pixels, total_fixed_linear_functions].

        When mapper objects and linear functions are used simultaneously in an inversion, this preloaded matrix
        significantly speed up the computation of their off-diagonal terms in the curvature matrix.

        This is similar to the preloading performed via the w-tilde formalism, except that there it is the PSF convolved
        values of each noise-map value pair that are preloaded.

        In **PyAutoGalaxy** and **PyAutoLens**, this preload is used when linear light profiles are fixed in the model.
        For example, when using a multi Gaussian expansion, the values defining how those Gaussians are evaluated
        (e.g. `centre`, `ell_comps` and `sigma`) are often fixed in a model, meaning this matrix can be preloaded and
        used for speed up.

        Returns
        -------
        ndarray
            A matrix of shape [data_pixels, total_fixed_linear_functions] that for each data pixel, maps it to the sum
            of the values of a linear object function convolved with the PSF kernel at the data pixel.
        """
        if self.psf.convolve_over_sample_size > 1:
            raise exc.InversionException(
                "The data_linear_func_matrix preload consumes the PSF kernel at "
                "image resolution and is incompatible with an oversampled PSF "
                "(convolve_over_sample_size > 1)."
            )

        linear_func_list = self.cls_list_from(cls=AbstractLinearObjFuncList)

        data_linear_func_matrix_dict = {}

        for func_index, linear_func in enumerate(linear_func_list):
            curvature_weights = (
                self.linear_func_operated_mapping_matrix_dict[linear_func]
                / self.noise_map[:, None] ** 2
            )

            data_linear_func_matrix = (
                inversion_imaging_util.data_linear_func_matrix_from(
                    curvature_weights_matrix=curvature_weights,
                    kernel_native=self.psf.kernel.native,
                    mask=self.mask,
                )
            )

            data_linear_func_matrix_dict[linear_func] = data_linear_func_matrix

        return data_linear_func_matrix_dict

    @property
    def mapper_operated_mapping_matrix_dict(self) -> Dict:
        """
        The `operated_mapping_matrix` of a `Mapper` object describes the mappings between the observed data's values
        and the mapper's mesh pixels after a 2D convolution operation. It is described fully in the method
        `operated_mapping_matrix`.

        This property returns a dictionary mapping every mapper object to its corresponded operated mapping
        matrix, which is used for constructing the matrices that perform the linear inversion in an efficent way
        for the psf precision operator calculation.

        Returns
        -------
        A dictionary mapping every mapper object to its operated mapping matrix.
        """
        mapper_operated_mapping_matrix_dict = {}

        for mapper in self.cls_list_from(cls=Mapper):
            operated_mapping_matrix = self.psf.convolved_mapping_matrix_from(
                mapping_matrix=self._mapping_matrix_for_convolution_from(mapper),
                mask=self.mask,
                xp=self._xp,
            )

            mapper_operated_mapping_matrix_dict[mapper] = operated_mapping_matrix

        return mapper_operated_mapping_matrix_dict
