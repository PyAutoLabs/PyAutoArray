import copy
import hashlib
import warnings

import numpy as np
from typing import Dict, List, Optional, Type, Union

from autonerves import cached_property, is_test_mode

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
        preloads=None,
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
        dataset
            The dataset being reconstructed (e.g. an `Imaging` or `Interferometer` dataset, or a `DatasetInterface`
            whose attributes like `data` and `noise_map` may have been modified before being passed in).
        linear_obj_list
            The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
            input dataset's data and whose values are solved for via the inversion.
        settings
            Settings controlling how an inversion is fitted, for example which linear algebra formalism is used.
        xp
            The array module to use (`numpy` by default; pass `jax.numpy` for JAX support).
        preloads
            An optional `AbstractPreloads` (e.g. a `PreloadsInterferometer`) carrying pre-computed
            inversion quantities (e.g. the `curvature_matrix` `F`). When a quantity is invariant
            across the evaluations reusing this inversion — a fixed pixelization across a search, or
            the channel-invariant quantities of a datacube `FactorGraphModel` — it is computed once
            and preloaded here so the inversion reuses it instead of rebuilding the dominant
            inversion-setup cost. `None` (the default) leaves the standard behaviour unchanged, as
            does any individual preload field left `None`.
        """

        self.dataset = dataset

        self.linear_obj_list = linear_obj_list

        self.settings = settings or Settings()

        self.use_jax = xp is not np

        self._preloads = preloads

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
        cls
            The type of class whose presence is checked among the linear objects and regularizations in this inversion.
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

        - If `cls=aa.mesh.Mesh` and `cls_filtered=aa.mesh.RectangularRTUAdaptDensity`, a list of all pixelizations
        excluding those which are `RectangularRTUAdaptDensity` pixelizations will be returned.

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

        # Restrict to the mapper parameters. This is NOT edge zeroing -- `mapper_indices` drops the linear
        # objects that carry no regularization (light profiles and the like), which is a different index set
        # from `zeroed_ids_to_keep` and applies for a different reason. The two were conflated by an earlier
        # version of this comment.
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

        # Restrict to the mapper parameters. This is NOT edge zeroing -- `mapper_indices` drops the linear
        # objects that carry no regularization (light profiles and the like), which is a different index set
        # from `zeroed_ids_to_keep` and applies for a different reason. The two were conflated by an earlier
        # version of this comment.
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

    @property
    def solve_ids_to_keep(self) -> Optional[np.ndarray]:
        """
        The global parameter indices the reconstruction actually solves for, or `None` when it solves the full
        system.

        This is the single answer to "which parameters did the solve include?", and it exists so that every
        quantity derived from the solve agrees with the solve about what it did. `reconstruction` subsets the
        linear system by these indices and scatters its result back with exact zeros elsewhere;
        `reconstruction_covariance_matrix` forms the covariance on the same submatrix and scatters back `NaN`.

        Before this property the predicate below lived inline in `reconstruction` and nowhere else, so the
        covariance had no way to know the solve had been subset -- it inverted the full `curvature_reg_matrix`
        and reported a noise value for pixels that were never solved for. Keep the two readers pointed here
        rather than re-deriving the condition, or they can drift apart again.

        `None` rather than "every index" is deliberate: the full-system path must stay byte-identical to what it
        was, and an `arange` would route it through indexing and scatter-back code it never used before.

        Note that `use_edge_zeroed_pixels` is consulted only when `use_positive_only_solver` is `True`, mirroring
        the nesting in `reconstruction`. That scoping is deliberate -- see the comment there and
        `Settings.use_edge_zeroed_pixels`.

        Returns
        -------
        The global indices kept by the solve, or `None` if the full system was solved.
        """
        if not self.settings.use_positive_only_solver:
            return None

        if not self.settings.use_edge_zeroed_pixels:
            return None

        if not self.has(cls=Mapper):
            return None

        return self.zeroed_ids_to_keep

    def _nnls_warm_start_fingerprint(self, ids_to_keep=None) -> Optional[str]:
        """
        Identify the index space this inversion's positive-only solve works in, so the
        cross-evaluation passive-set memo (`autoarray.inversion.inversion.nnls_memo`) only
        reuses a seed across evaluations whose reconstruction entries mean the same thing.

        Deliberately built from shapes, NOT from the curvature matrix or data vector: the
        memo exists to carry a passive set between *neighbouring* parameter points, whose
        matrices differ. What must not change is the mapping from index to source pixel --
        the mesh sizes and the parameter-vector length. Under edge zeroing the passive set
        lives in the subset index space, so `ids_to_keep` is part of the identity too.

        Returns `None` on the JAX path, which does not run fnnls at all.
        """
        if self._xp.__name__.startswith("jax"):
            return None

        parts = [str(tuple(np.shape(self.data_vector)))]

        for mapper in self.cls_list_from(cls=Mapper):
            parts.append(str(tuple(np.shape(mapper.source_plane_mesh_grid))))

        if ids_to_keep is not None:
            parts.append(
                hashlib.sha256(
                    np.ascontiguousarray(ids_to_keep, dtype=np.int64).tobytes()
                ).hexdigest()
            )

        return "|".join(parts)

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

            # `use_edge_zeroed_pixels` is deliberately nested here rather than checked alongside
            # `use_positive_only_solver`: edge-zeroing is scoped to the positive-only solver, and the
            # positive-negative branch below solves the full system regardless of its value. This is
            # intended, not an oversight -- do not "fix" it by hoisting the check out of this branch.
            # `solve_ids_to_keep` encodes that nesting (it returns None unless BOTH settings are on),
            # so it is safe to consult here and nowhere higher up.
            ids_to_keep = self.solve_ids_to_keep

            if ids_to_keep is not None:

                # Use advanced indexing to select rows/columns
                data_vector = self.data_vector[ids_to_keep]
                curvature_reg_matrix = self.curvature_reg_matrix[ids_to_keep][
                    :, ids_to_keep
                ]

                # Perform reconstruction via fnnls
                reconstruction_partial = (
                    inversion_util.reconstruction_positive_only_from(
                        data_vector=data_vector,
                        curvature_reg_matrix=curvature_reg_matrix,
                        settings=self.settings,
                        xp=self._xp,
                        fingerprint=self._nnls_warm_start_fingerprint(
                            ids_to_keep=ids_to_keep
                        ),
                    )
                )

                # Allocate full solution array
                reconstruction = self._xp.zeros(self.data_vector.shape[0])

                # Scatter the partial solution back to the full shape
                if self._xp.__name__.startswith("jax"):
                    reconstruction = reconstruction.at[ids_to_keep].set(
                        reconstruction_partial
                    )
                else:
                    reconstruction[ids_to_keep] = reconstruction_partial

                return reconstruction

            else:

                return inversion_util.reconstruction_positive_only_from(
                    data_vector=self.data_vector,
                    curvature_reg_matrix=self.curvature_reg_matrix,
                    settings=self.settings,
                    xp=self._xp,
                    fingerprint=self._nnls_warm_start_fingerprint(),
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

        # Restrict to the mapper parameters. This is NOT edge zeroing -- `mapper_indices` drops the linear
        # objects that carry no regularization (light profiles and the like), which is a different index set
        # from `zeroed_ids_to_keep` and applies for a different reason. The two were conflated by an earlier
        # version of this comment.
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

        Under ``regularization_term_method == "cho_solve"`` (opt-in, default off), regularization schemes
        which know a factorization of their own matrix may instead supply their contribution directly via
        :meth:`AbstractRegularization.regularization_term_from` — the kernel schemes (``MaternKernel`` etc.)
        return ``coefficient * s^T C^-1 s`` from a single Cholesky solve of their covariance ``C``, avoiding
        the round-off of contracting the explicitly formed inverse (whose error is amplified by ``cond(C)``,
        ~1e9 on clustered traced mesh vertices). Because ``regularization_matrix_reduced`` is the block
        diagonal of the per-object matrices when every linear object is regularized, the term is the sum of
        the per-object terms; if any scheme has no shortcut (returns ``None``) the whole computation falls
        back to the formed matrix. The default ``"matmul"`` path never consults the shortcut, so default
        evidence values are unchanged.

        Returns
        -------
        float
            The regularization term of the inversion.
        """
        if not self.has(cls=AbstractRegularization):
            return 0.0

        if (
            self.settings.regularization_term_method == "cho_solve"
            and self.all_linear_obj_have_regularization
        ):
            # `reconstruction_reduced` is the full reconstruction here (the guard above is exactly the
            # no-reduction case), so the per-object slices index it directly.
            reconstruction = self.reconstruction_reduced

            term_list = [
                regularization.regularization_term_from(
                    linear_obj=linear_obj,
                    reconstruction=reconstruction[param_range[0] : param_range[1]],
                    xp=self._xp,
                )
                for linear_obj, regularization, param_range in zip(
                    self.linear_obj_list,
                    self.regularization_list,
                    self.param_range_list_from(cls=LinearObj),
                )
            ]

            if all(term is not None for term in term_list):
                return sum(term_list)

        return self._xp.matmul(
            self.reconstruction_reduced.T,
            self._xp.matmul(
                self.regularization_matrix_reduced, self.reconstruction_reduced
            ),
        )

    def _log_det_symmetric_from(self, matrix) -> float:
        """
        Log determinant of a symmetric positive-(semi)definite ``matrix``, used by the two evidence log-det terms.

        The method is selected by ``self.settings.log_det_method`` (see :class:`Settings`):

        - ``"cholesky"`` (default) — ``2 * sum(log(diag(cholesky(matrix))))``. This is the historical
          computation and is byte-for-byte unchanged, including the test-mode ``LinAlgError`` guard. On the
          NumPy backend a non-positive-definite matrix raises; on the JAX backend the Cholesky returns NaN,
          which propagates as a non-finite figure of merit and can stall gradient-based searches
          (autolens_workspace_developer#104).
        - ``"slogdet"`` — the ``logabsdet`` from ``xp.linalg.slogdet(matrix)``. Wherever ``matrix`` is
          positive-definite this equals the Cholesky value exactly (the sign is +1), but where the Cholesky
          would NaN it returns a finite, differentiable value instead, so it never stalls a gradient search.
          It is **opt-in and non-default** — intended for gradient-based work and for comparison against the
          Cholesky evidence, not as a replacement for it. See PyAutoArray#391.
        """
        if self.settings.log_det_method == "slogdet":
            return self._xp.linalg.slogdet(matrix)[1]

        try:
            return 2.0 * self._xp.sum(
                self._xp.log(self._xp.diag(self._xp.linalg.cholesky(matrix)))
            )
        except np.linalg.LinAlgError:
            if is_test_mode():
                # Singular matrix from a fabricated test-mode model; this evidence term is discarded.
                # Return a benign 0.0 so unguarded test-mode paths do not crash (matches the reconstruction
                # guard in inversion_util). Normal runs re-raise unchanged. numpy-only: the JAX cholesky
                # returns NaN rather than raising (which is exactly what "slogdet" exists to avoid).
                return 0.0
            raise

    @property
    def log_det_curvature_reg_matrix_term(self) -> float:
        """
        The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

        This uses the Cholesky decomposition which is already computed before solving the reconstruction
        (or ``slogdet`` when ``Settings.log_det_method == "slogdet"`` — see :meth:`_log_det_symmetric_from`).
        """
        if not self.has(cls=AbstractRegularization):
            return 0.0

        return self._log_det_symmetric_from(self.curvature_reg_matrix_reduced)

    @property
    def log_det_regularization_matrix_term(self) -> float:
        """
        The Bayesian evidence of an inversion which quantifies its overall goodness-of-fit uses the log determinant
        of regularization matrix, Log[Det[Lambda*H]].

        Unlike the determinant of the curvature reg matrix, which uses an existing preloading Cholesky decomposition
        used for the source reconstruction, this uses scipy sparse linear algebra to solve the determinant efficiently
        (or ``slogdet`` when ``Settings.log_det_method == "slogdet"`` — see :meth:`_log_det_symmetric_from`).

        Under ``log_det_method == "slogdet"`` (opt-in, default off — PyAutoArray#391), regularization schemes
        which know a factorization of their own matrix may additionally supply the term directly via
        :meth:`AbstractRegularization.log_det_regularization_matrix_term_from` — the kernel schemes
        (``MaternKernel`` etc.) return the analytically exact ``pixels * log(coeff) - log det C`` from a single
        Cholesky of their covariance ``C``, avoiding the round-off of factorizing the explicitly formed inverse
        (which reaches ~1e-6 absolute in the evidence at cond(C) ~ 1e9 on clustered traced mesh vertices).
        Because ``regularization_matrix_reduced`` is the block diagonal of the per-object matrices when every
        linear object is regularized, the term is the sum of the per-object terms; if any scheme has no
        shortcut (returns ``None``) the whole computation falls back to ``slogdet`` of the formed matrix.
        The default ``"cholesky"`` path never consults the shortcut, so default evidence values are unchanged.

        Returns
        -------
        float
            The log determinant of the regularization matrix.
        """
        if not self.has(cls=AbstractRegularization):
            return 0.0

        if (
            self.settings.log_det_method == "slogdet"
            and self.all_linear_obj_have_regularization
        ):
            term_list = [
                regularization.log_det_regularization_matrix_term_from(
                    linear_obj=linear_obj, xp=self._xp
                )
                for linear_obj, regularization in zip(
                    self.linear_obj_list, self.regularization_list
                )
            ]

            if all(term is not None for term in term_list):
                return sum(term_list)

        return self._log_det_symmetric_from(self.regularization_matrix_reduced)

    @property
    def reconstruction_covariance_matrix(self) -> np.ndarray:
        """
        Returns the covariance matrix of the reconstruction, ``C = [F + reg_coeff*H]^-1``.

        This is the inverse of the curvature matrix with regularization -- the same matrix used to solve for the
        reconstruction via the linear inversion. Its diagonal holds the variance of each reconstructed pixel and
        its off-diagonal entries the covariances between pixels; the off-diagonals are routinely negative, which is
        a property of the covariance and not an error.

        For the RMS standard deviation of each pixel (the quantity used for scientific analysis) use
        `reconstruction_noise_map`, which takes the square root of this matrix's diagonal.

        Formed on the parameters the solve actually solved for
        -----------------------------------------------------
        When `use_edge_zeroed_pixels` applies (see `solve_ids_to_keep`), `reconstruction` does not solve the
        full system: it subsets `curvature_reg_matrix` to `zeroed_ids_to_keep`, solves the reduced problem and
        scatters the answer back with **exact zeros** at the excluded pixels. Those pixels are the mesh's
        poorly-constrained boundary vertices, zeroed precisely to keep the inversion stable.

        This matrix is formed on that same index set and scattered back the same way, so it describes the
        estimator that was actually computed. The excluded rows and columns are `NaN`, not zero: zero is a
        covariance value ("known exactly"), whereas these parameters were never estimated at all. The returned
        shape is always `[total_params, total_params]`, so callers do not have to branch on the settings.

        Previously the full matrix was inverted regardless, which re-admitted into an explicit inverse the very
        rows the solve dropped to stay stable, and reported a noise value for a pixel whose reconstruction reads
        exactly `0` because it was never solved for.

        Note this also changes the values on the parameters that ARE solved. Inverting the submatrix is not the
        corresponding block of the full inverse: for a symmetric positive-definite `A`,
        `[A^-1]_keep >= (A_keep)^-1` in the positive-semidefinite ordering, so every kept pixel's variance is
        lower here than it was. That is the correct quantity -- the excluded parameters are held at zero by
        construction, so conditioning on them is exact, not an approximation. It is unrelated to the
        NNLS active-set caveat documented on `reconstruction_noise_map`, where the pixels held at zero are
        chosen by the data rather than fixed in advance.

        Measured on real ray-traced fits (Isothermal `einstein_radius=1.6` + shear, compact Sersic source,
        `r=3.0"` mask, `over_sample_size_pixelization=4`, PSF and Poisson noise), at the regularization
        coefficient the Bayesian evidence selects:

        - `RectangularBilinearAdaptDensity(28, 28)`, 108 of 784 parameters zeroed: the old value was
          overstated by a median factor of 1.0002, but 21% of solved pixels move by more than 1%, 9% by more
          than 10%, and the worst by 1.8x. The change is concentrated near the mesh edge rather than spread.
        - `Delaunay` set up as the workspace's own example does (an `Overlay` image mesh plus a ring of edge
          points, `zeroed_pixels=30`): median 1.0045, 99th percentile 1.90, worst 2.37. Delaunay is affected
          too whenever `zeroed_pixels > 0`; with the default `zeroed_pixels=0` this property is unchanged to
          the bit.
        - Downstream source flux and magnification through the workspace's `S/N >= 5` cut moved by 0.00% in
          every case: the pixels whose noise changes most are near the edge, where the reconstruction has no
          flux to move.

        The inverse is formed from a Cholesky factorization rather than `np.linalg.inv`, for two reasons:

        - `cho_factor` raises `LinAlgError` when the matrix is not positive-definite. `np.linalg.inv` raises only
          on an exactly singular matrix, so an indefinite `curvature_reg_matrix` -- which the inversion does
          encounter, hence `Settings.no_regularization_add_to_curvature_diag_value` -- previously returned a
          plausible-looking covariance with no error and no warning. A covariance is only defined for a
          positive-definite matrix, so failing here is correct and is handled by the callers in
          `inversion/plot/inversion_plots.py`.
        - `np.linalg.inv` is LU-based and exploits neither the symmetry nor the positive-definiteness this matrix
          has. Its output drifts out of symmetry as conditioning worsens (measured at ~5e-7 absolute at
          `cond ~ 1e12`, against ~3e-16 for the Cholesky solve), while a covariance matrix is symmetric by
          definition.

        Every failure mode raises `LinAlgError`, including a non-finite `curvature_reg_matrix`. That case is
        checked explicitly because scipy would otherwise raise `ValueError`, which the callers -- here and
        downstream -- do not catch; the previous implementation returned a silently all-`NaN` matrix instead.

        The matrix is symmetrized on input. `cho_factor` reads only the upper triangle, so an asymmetric input
        would otherwise be inverted as though its lower triangle matched, silently and with no diagnostic.
        `curvature_reg_matrix` is `F + H` and symmetric by construction, so this is defensive only.

        This property is NumPy-only: the input is coerced with `np.asarray`, so a JAX `curvature_reg_matrix`
        forces a device-to-host transfer. It is a post-fit diagnostic, not part of the likelihood, so it is not on
        the JIT path.

        Returns
        -------
        The covariance matrix of the reconstruction, of shape [total_params, total_params].
        """
        from scipy.linalg import cho_factor, cho_solve

        full_matrix = np.asarray(self.curvature_reg_matrix)

        # Form the covariance on exactly the parameters the solve solved for. `solve_ids_to_keep` is None when
        # the solve used the full system, in which case this path is unchanged.
        ids_to_keep = self.solve_ids_to_keep

        if ids_to_keep is None:
            matrix = full_matrix
        else:
            ids_to_keep = np.asarray(ids_to_keep)
            matrix = full_matrix[ids_to_keep][:, ids_to_keep]

        # The guard runs on the SUBMATRIX, not the full one. A non-finite entry in a row the solve excluded
        # cannot reach the factorization, and the reconstruction does not fail on it either -- failing here
        # would make the covariance stricter than the solve it describes. It must also run before the
        # scatter-back below, which fills the excluded entries with NaN deliberately.
        if not np.isfinite(matrix).all():
            raise np.linalg.LinAlgError(
                "The curvature_reg_matrix contains non-finite entries (NaN or inf), so the reconstruction "
                "covariance is undefined. Raised as LinAlgError so the plotting and CSV callers, which guard "
                "on LinAlgError, degrade gracefully rather than aborting the model-fit."
            )

        # cho_factor reads only the upper triangle; symmetrize so an asymmetric input cannot be silently
        # inverted as though its lower triangle matched its upper.
        matrix = 0.5 * (matrix + matrix.T)

        covariance = cho_solve(
            cho_factor(matrix, check_finite=False),
            np.eye(matrix.shape[0], dtype=matrix.dtype),
            check_finite=False,
        )

        # cho_solve is accurate but not bitwise symmetric; a covariance matrix is symmetric by definition.
        covariance = 0.5 * (covariance + covariance.T)

        if ids_to_keep is None:
            return covariance

        # Scatter back to the full parameter shape, so this property's shape does not depend on the settings.
        # The excluded entries are NaN ("never estimated"), which is what the solve says about them -- their
        # reconstruction is an exact structural zero, not a fitted value.
        full_covariance = np.full(full_matrix.shape, np.nan, dtype=covariance.dtype)
        full_covariance[np.ix_(ids_to_keep, ids_to_keep)] = covariance

        return full_covariance

    @property
    def reconstruction_noise_map_with_covariance(self) -> np.ndarray:
        """
        Deprecated alias of `reconstruction_covariance_matrix`.

        This property previously returned ``np.sqrt(np.linalg.inv(curvature_reg_matrix))`` -- an elementwise square
        root of the whole covariance matrix. Because the off-diagonal entries of a covariance matrix are routinely
        negative, every such entry was `NaN` by construction, for any input matrix however well-conditioned, and
        each call emitted `RuntimeWarning: invalid value encountered in sqrt`.

        It now returns the covariance matrix itself, so the values differ: the diagonal holds variances rather
        than standard deviations, and the off-diagonals hold covariances rather than `NaN`.

        Note it also inherits `reconstruction_covariance_matrix`'s index set: under `use_edge_zeroed_pixels` the
        rows and columns of parameters the solve excluded are `NaN` (never estimated), and the entries that
        remain are the inverse of the submatrix rather than a block of the full inverse. See that property.

        Returns
        -------
        The covariance matrix of the reconstruction (see `reconstruction_covariance_matrix`).
        """
        warnings.warn(
            "`reconstruction_noise_map_with_covariance` is deprecated; use "
            "`reconstruction_covariance_matrix` instead. Note the values have changed: it now returns the "
            "covariance matrix, so its diagonal holds variances rather than standard deviations (the previous "
            "elementwise square root made every off-diagonal NaN). For the RMS noise of each pixel use "
            "`reconstruction_noise_map`.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.reconstruction_covariance_matrix

    @property
    def reconstruction_noise_map(self):
        """
        Returns the noise-map of the reconstruction as a one dimensional ndarray, which does not account for the
        covariance of the noise between pixels.

        This matrix is representative of the noise properties of the fit and should be used for any scientific
        analysis (e.g. source reconstructions of strong lenses).

        The noise-map of the reconstruction is the RMS standard deviation of the noise in every pixel of the
        reconstruction. This definition is identical to the `noise_map` attributes of dataset objects.

        It is computed as the square root of the diagonal of `reconstruction_covariance_matrix`, which is the
        inverse of the same matrix used to solve for the reconstruction via the linear inversion.

        Pixels the solve never estimated are `NaN`
        ------------------------------------------
        Under `use_edge_zeroed_pixels` (the shipped default, and unconditional for the
        `Rectangular*AdaptDensity` mesh family, whose `zeroed_pixels` is the whole edge ring) the solve excludes
        the mesh's boundary vertices and writes an exact `0.0` into `reconstruction` for them. This noise map
        reports **`NaN`** at exactly those pixels, meaning "never estimated" -- they have no uncertainty because
        they have no fitted value.

        So the two arrays agree on which pixels were solved. The implication runs one way, and the direction
        matters: **`NaN` implies the reconstruction is exactly `0.0` there, but not the converse.** The
        non-negative solver also pins pixels it DID solve for at exactly `0.0`, and those keep a finite,
        meaningful noise value. On a representative fit -- `RectangularBilinearAdaptDensity(28, 28)`, 784
        parameters -- 603 pixels read `0.0` while only the 108 structurally excluded ones are `NaN`; the other
        495 were solved and pinned by the constraint. `np.isnan(reconstruction_noise_map)` is therefore the way
        to identify the never-estimated pixels; `reconstruction == 0.0` is not, and conflates the two.

        Previously the excluded pixels carried a finite noise value computed as though they had been solved,
        so there was no way to tell them apart from the pinned ones at all.

        `NaN` propagates rather than raising, and the consumers handle it: the colour scales derive their
        limits with `np.nanmax` (`plot/utils.py:norm_from`), and `save_reconstruction_csv` already writes `nan`
        into this column when the covariance cannot be computed. A signal-to-noise map formed as
        `reconstruction / reconstruction_noise_map` gives `0.0 / NaN = NaN` at these pixels, silently, and a
        `NaN >= threshold` comparison is `False` -- so they fall outside any signal-to-noise cut rather than
        being counted as significant.

        This previously took the diagonal of an elementwise-square-rooted matrix. The two are algebraically
        identical -- `np.sqrt` is elementwise, so it commutes with taking the diagonal -- but only numerically
        equivalent, since the covariance is now formed by Cholesky rather than LU. The difference is
        conditioning-limited roundoff, measured at ~7e-15 relative at `cond ~ 1e3` rising to ~4e-5 at
        `cond ~ 1e13`; neither result is the more correct one.

        Caveat -- this is the uncertainty of the UNCONSTRAINED solve
        ------------------------------------------------------------
        `reconstruction_covariance_matrix` is `[F + reg_coeff*H]^-1`, the posterior covariance of the
        positive-negative (unconstrained) solution of Warren & Dye (2003) eq. 12. But
        `Settings.use_positive_only_solver` defaults to `True`, so the reconstruction is normally a
        non-negative least-squares solve. Constraining `s >= 0` truncates the posterior, and a truncated
        Gaussian's covariance is not the untruncated one, so this **overstates** the per-pixel uncertainty --
        by more for pixels near the `s = 0` boundary, and it is not meaningful at all for pixels the solver
        pinned at exactly zero.

        The size of the discrepancy was measured on real ray-traced lens fits (`Isothermal` + shear,
        `RectangularBilinearAdaptDensity`, `Constant` regularization). It depends strongly on the
        regularization coefficient, and the Bayesian evidence -- which is what a model-fit maximises when
        choosing that coefficient -- happens to select the regime where it is small:

        - At the evidence-optimal coefficient, this noise map is overstated by a median factor of ~1.01
          (extended source) to ~1.26 (very compact source), with individual pixels up to ~2x. Source flux
          and magnification computed through a `S/N >= 5` cut moved by 0.0% to -1.3%.
        - At coefficients well below the evidence optimum (an under-regularized fit) the factor grows to
          ~2.8 median and ~10x on individual pixels.

        So for most fits this is a small, one-directional (conservative) bias. Treat per-pixel error bars on
        a very compact source as good to a few tens of percent rather than exact, and be more careful if the
        regularization coefficient came out well below its evidence optimum.

        Note that restricting the covariance to the solver's free set is **not** the correction: that treats
        the active set as known and so understates. The two bracket the true truncated-Gaussian posterior.

        Returns
        -------
        The noise-map of the reconstruction as a one dimensional ndarray, which does not account for the covariance
        of the noise between pixels.
        """
        return np.sqrt(np.diag(self.reconstruction_covariance_matrix))

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

        for mapper in self.cls_list_from(cls=Mapper):
            index = self.linear_obj_list.index(mapper)
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
