import numpy as np
from typing import Dict, List, Optional, Union

from autonerves import cached_property

from autoarray import exc
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.interferometer.abstract import (
    AbstractInversionInterferometer,
)
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.mappers import mapper_util
from autoarray.settings import Settings
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.visibilities import Visibilities

from autoarray.inversion.inversion import inversion_util


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
        for linear_obj in linear_obj_list:
            if linear_obj.operated_mapping_matrix_override is not None:
                raise exc.InversionException(
                    "A linear object with an `operated_mapping_matrix_override` was passed to the sparse "
                    "(w-tilde) interferometer inversion, which constructs its linear algebra without an "
                    "explicit operated mapping matrix and therefore cannot apply the override.\n\n"
                    "Use the mapping formalism instead (e.g. do not call `apply_sparse_operator` on the "
                    "interferometer dataset)."
                )

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            preloads=preloads,
        )

    @cached_property
    def data_vector(self) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects the `data_vectors` are concatenated ensuring their values are solved
        for simultaneously.

        The `data_vector` is computed as `Lᵀ d~`, where `L` is the real-space mapping matrix of every
        linear object stacked horizontally and `d~ = Re(Fᴴ W d)` is the dirty image cached on the
        dataset's `sparse_operator`. Each linear object contributes its own block, concatenated in
        `linear_obj_list` order (matching `param_range_list_from(cls=LinearObj)`):

        - For a `Mapper` the block `Aᵀ d~` is computed from the mapper's sparse COO triplets
          (`_data_vector_mapper_from`), as a segment-sum of `vals * d~[rows]` over source pixel
          columns. The dense (masked pixels x source pixels) mapping matrix is therefore never
          built in the likelihood, which at 91,507 masked pixels and 5,746 source pixels is 4.2 GB.
        - For an `AbstractLinearObjFuncList` column `b`, `bᵀ d~ = bᵀ Re(Fᴴ W d) = Re((F b)ᴴ W d)`,
          which is exactly the entry the mapping (dense) formalism computes via the linear
          function's transformed mapping matrix. Its mapping matrix is small (one column per
          linear function), so it is used directly.

        The cached dirty image is that of the visibilities the `sparse_operator` was built from. When
        the inversion's data differs from them (e.g. a `DatasetInterface` whose data has the visibilities
        of ordinary light profiles subtracted), the dataset's `sparse_dirty_image` of that data is used
        instead, so that `D` is consistent with the data the chi-squared is computed from.

        If the dataset's `sparse_operator` carries a `primary_beam_slim` `P`, every linear object's
        real-space mapping matrix `L` is replaced by `P L` (the mapper triplets carry `P` in their
        `vals`), giving `D = Lᵀ P d~`.
        """
        dirty_image = getattr(self.dataset, "sparse_dirty_image", None)

        if dirty_image is None:
            dirty_image = self.dataset.sparse_operator.dirty_image

        dirty_image = self._xp.asarray(
            getattr(dirty_image, "array", dirty_image), dtype=self._xp.float64
        )

        data_vector_list = []

        for linear_obj in self.linear_obj_list:
            if isinstance(linear_obj, Mapper):
                data_vector_list.append(
                    self._data_vector_mapper_from(
                        mapper=linear_obj, dirty_image=dirty_image
                    )
                )
            else:
                data_vector_list.append(
                    self._xp.dot(
                        self._mapping_matrix_beamed_from(linear_obj.mapping_matrix).T,
                        dirty_image,
                    )
                )

        if len(data_vector_list) == 1:
            return data_vector_list[0]

        return self._xp.concatenate(data_vector_list)

    @property
    def _primary_beam_slim(self):
        """
        The primary beam `P` on the slim masked grid carried by the dataset's `sparse_operator`
        (`None` if it has none). Read via `getattr` so a `DatasetInterface` (e.g. PyAutoGalaxy's),
        whose `sparse_operator` is the dataset's own, works unchanged.
        """
        return getattr(self.dataset.sparse_operator, "primary_beam_slim", None)

    def _mapping_matrix_beamed_from(self, mapping_matrix):
        """
        Returns `diag(P) @ mapping_matrix` for a linear function's real-space mapping matrix, or the
        mapping matrix unchanged when there is no primary beam.
        """
        primary_beam_slim = self._primary_beam_slim

        if primary_beam_slim is None:
            return mapping_matrix

        return mapping_matrix * self._xp.asarray(primary_beam_slim)[:, None]

    def _sparse_triplets_slim_from(self, mapper: Mapper, apply_primary_beam: bool = True):
        """
        Returns the sparse COO triplets `(rows, cols, vals)` of a mapper's real-space mapping
        operator `A`, with `rows` indexed on the slim (masked-pixel) grid the dirty image lives on.

        This is the sibling of `_sparse_triplets_curvature_from`, whose rows are extent-flat
        indices on the `W~` operator's rectangular grid and therefore cannot index the slim dirty
        image.

        The triplets reproduce `mapper_util.mapping_matrix_from` exactly: `vals` carry the
        `sub_fraction` normalisation of each sub-pixel's slim pixel, and stencil slots that
        `mapping_matrix_from` drops (slot index `>= pix_sizes_for_sub_slim_index`, or a negative
        source pixel index) are given zero weight and column 0, so they contribute nothing.

        Parameters
        ----------
        mapper
            The mapper whose mapping operator is expressed in sparse triplet form.
        apply_primary_beam
            If `True` (default) and the dataset has a primary beam `P`, the triplets are those of the
            beamed operator `P A` (used by the data vector); `False` gives those of `A` (used for the
            true-sky reconstructed image).
        """
        return mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
            slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=self.mask.extent_index_for_masked_pixel,
            sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
            return_rows_slim=True,
            slim_weights=self._primary_beam_slim if apply_primary_beam else None,
            pix_sizes_for_sub=mapper.pix_sizes_for_sub_slim_index,
            xp=self._xp,
        )

    def _data_vector_mapper_from(self, mapper: Mapper, dirty_image) -> np.ndarray:
        """
        Returns a mapper's block of the `data_vector`, `Aᵀ d~`, computed from the sparse triplets
        of its mapping operator `A` without building the dense mapping matrix.

        Entry `j` is the sum over the non-zeros of `A`'s column `j` of `val * d~[row]`, i.e. a
        segment-sum of `vals * dirty_image[rows]` over the source pixel columns.

        Parameters
        ----------
        mapper
            The mapper whose data vector block is computed.
        dirty_image
            The slim (masked-pixel indexed) dirty image `d~`.
        """
        rows, cols, vals = self._sparse_triplets_slim_from(mapper=mapper)

        weights = vals * dirty_image[rows]

        if self._xp is np:
            return np.bincount(cols, weights=weights, minlength=mapper.params)

        return self._xp.zeros(mapper.params, dtype=weights.dtype).at[cols].add(weights)

    def _sparse_triplets_curvature_from(self, mapper: Mapper):
        """
        Returns the sparse COO triplets `(rows, cols, vals)` of a mapper's real-space mapping
        operator `A`, with `rows` indexed on the grid the interferometer `W~` operator lives on.

        The interferometer `W~` operator lives on the unmasked-extent rectangular grid
        (`shape_native_masked_pixels`), not the full native grid used by the imaging path.
        The triplets are therefore built with extent-flat row indices so that they match the
        operator's (M = extent_y * extent_x, B) scatter buffer.

        This is the interferometer counterpart of `Mapper.sparse_triplets_curvature`, which
        indexes rows on the imaging FFT grid and therefore cannot be reused here.

        If the dataset has a primary beam `P` the triplets are those of the beamed operator `P A`,
        so every curvature block built from them is `Aᵀ P W~ P A` with the FFT kernel untouched.

        Parameters
        ----------
        mapper
            The mapper whose mapping operator is expressed in sparse triplet form.
        """
        return mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
            slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=self.mask.extent_index_for_masked_pixel,
            sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
            return_rows_slim=False,
            slim_weights=self._primary_beam_slim,
            xp=self._xp,
        )

    @cached_property
    def curvature_matrix(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        If there are multiple linear objects their contributions are combined ensuring their `curvature_matrix`
        values are solved for simultaneously. This includes all diagonal and off-diagonal terms describing the
        covariances between linear objects, whether those objects are `Mapper`s, `AbstractLinearObjFuncList`s
        (e.g. linear light profiles), or a mixture of both.

        Every block is formed with the same operator `W~ = Re(Fᴴ W F)`, applied by the dataset's
        `sparse_operator` on the unmasked-extent rectangular grid:

        - mapper–mapper diagonal:     `A_iᵀ W~ A_i`
        - mapper–mapper off-diagonal: `A_iᵀ W~ A_j`
        - mapper–function:            `A_iᵀ W~ B_k`
        - function–function:          `B_kᵀ W~ B_l`

        Only the upper blocks are computed, with `curvature_matrix_mirrored_from` filling the lower ones.

        If a `preloads.curvature_matrix` was injected (e.g. the datacube shared-state path, where `F` is
        identical for every channel) it is returned directly, so the dominant `F = LᵀW̃L` build is skipped.
        This is the only invariant inversion-setup quantity that must be preloaded explicitly: the mapper
        (and therefore `mapping_matrix` and `regularization_matrix`) is already reused by passing the same
        `linear_obj_list` to every channel's inversion.
        """
        if self._preloads is not None and self._preloads.curvature_matrix is not None:
            return self._preloads.curvature_matrix

        if not self.has(cls=AbstractLinearObjFuncList) and self.total(cls=Mapper) == 1:
            # The single-mapper case is the performance-critical one and its matrix is already
            # square, symmetric and complete, so it bypasses the block assembly and mirroring.
            curvature_matrix = self.curvature_matrix_diag
        else:
            if self.has(cls=AbstractLinearObjFuncList):
                curvature_matrix = self._curvature_matrix_func_list_and_mapper
            else:
                curvature_matrix = self._curvature_matrix_multi_mapper

            curvature_matrix = inversion_util.curvature_matrix_mirrored_from(
                curvature_matrix=curvature_matrix,
                xp=self._xp,
            )

        if len(self.no_regularization_index_list) > 0:
            curvature_matrix = inversion_util.curvature_matrix_with_added_to_diag_from(
                curvature_matrix=curvature_matrix,
                value=self.settings.no_regularization_add_to_curvature_diag_value,
                no_regularization_index_list=self.no_regularization_index_list,
                xp=self._xp,
            )

        return curvature_matrix

    @cached_property
    def curvature_matrix_diag(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes the diagonal terms of F of the inversion's first `Mapper` using the sparse
        linear algebra formalism, returning a matrix of shape [mapper.params, mapper.params].
        """
        mapper = self.cls_list_from(cls=Mapper)[0]

        rows, cols, vals = self._sparse_triplets_curvature_from(mapper=mapper)

        return self.dataset.sparse_operator.curvature_matrix_diag_from(
            rows=rows,
            cols=cols,
            vals=vals,
            S=mapper.params,
            xp=self._xp,
        )

    @property
    def _curvature_matrix_mapper_diag(self) -> Optional[np.ndarray]:
        """
        Returns the diagonal regions of the `curvature_matrix`, a 2D matrix which uses the mappings between the data
        and the linear objects to construct the simultaneous linear equations. The object is described in full in
        the method `curvature_matrix`.

        This method computes the diagonal entries of all mapper objects in the `curvature_matrix`, placing each
        one in the parameter range of its mapper in the full [total_params, total_params] matrix.
        """
        if not self.has(cls=Mapper):
            return None

        curvature_matrix = self._xp.zeros((self.total_params, self.total_params))

        mapper_list = self.cls_list_from(cls=Mapper)
        mapper_param_range_list = self.param_range_list_from(cls=Mapper)

        for mapper_index, mapper in enumerate(mapper_list):
            rows, cols, vals = self._sparse_triplets_curvature_from(mapper=mapper)

            diag = self.dataset.sparse_operator.curvature_matrix_diag_from(
                rows=rows,
                cols=cols,
                vals=vals,
                S=mapper.params,
                xp=self._xp,
            )

            start, end = mapper_param_range_list[mapper_index]

            if self._xp is np:
                curvature_matrix[start:end, start:end] = diag
            else:
                curvature_matrix = curvature_matrix.at[start:end, start:end].set(diag)

        return curvature_matrix

    def _curvature_matrix_off_diag_from(
        self, mapper_0: Mapper, mapper_1: Mapper
    ) -> np.ndarray:
        """
        Returns the off-diagonal block `A_0ᵀ W~ A_1` of the `curvature_matrix` describing the covariance
        between two mappers, of shape [mapper_0.params, mapper_1.params].
        """
        rows_0, cols_0, vals_0 = self._sparse_triplets_curvature_from(mapper=mapper_0)
        rows_1, cols_1, vals_1 = self._sparse_triplets_curvature_from(mapper=mapper_1)

        return self.dataset.sparse_operator.curvature_matrix_off_diag_from(
            rows0=rows_0,
            cols0=cols_0,
            vals0=vals_0,
            rows1=rows_1,
            cols1=cols_1,
            vals1=vals_1,
            S0=mapper_0.params,
            S1=mapper_1.params,
            xp=self._xp,
        )

    @property
    def _curvature_matrix_multi_mapper(self) -> np.ndarray:
        """
        Returns the `curvature_matrix`, a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations. The object is described in full in the method `curvature_matrix`.

        This method computes the mapper entries of the `curvature_matrix` when there are multiple mapper objects in
        the `Inversion`, filling in each mapper's diagonal block and the upper off-diagonal blocks describing the
        covariance between every pair of mappers. The lower blocks are filled in by the mirroring performed in
        `curvature_matrix`.
        """
        curvature_matrix = self._curvature_matrix_mapper_diag

        if self.total(cls=Mapper) == 1:
            return curvature_matrix

        mapper_list = self.cls_list_from(cls=Mapper)
        mapper_param_range_list = self.param_range_list_from(cls=Mapper)

        for i in range(len(mapper_list)):
            mapper_param_range_i = mapper_param_range_list[i]

            for j in range(i + 1, len(mapper_list)):
                mapper_param_range_j = mapper_param_range_list[j]

                off_diag = self._curvature_matrix_off_diag_from(
                    mapper_0=mapper_list[i], mapper_1=mapper_list[j]
                )

                if self._xp is np:
                    curvature_matrix[
                        mapper_param_range_i[0] : mapper_param_range_i[1],
                        mapper_param_range_j[0] : mapper_param_range_j[1],
                    ] = off_diag
                else:
                    curvature_matrix = curvature_matrix.at[
                        mapper_param_range_i[0] : mapper_param_range_i[1],
                        mapper_param_range_j[0] : mapper_param_range_j[1],
                    ].set(off_diag)

        return curvature_matrix

    @property
    def _curvature_matrix_func_list_and_mapper(self) -> np.ndarray:
        """
        Returns the `curvature_matrix`, a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations. The object is described in full in the method `curvature_matrix`.

        This method computes the `curvature_matrix` when one or more `AbstractLinearObjFuncList` objects (e.g. linear
        light profiles) are fitted, optionally simultaneously with one or more `Mapper` objects.

        All mapper blocks are computed first, then the mapper–function off-diagonal blocks `A_iᵀ W~ B_k`, then the
        function–function blocks `B_kᵀ W~ B_l`. Only the upper blocks are filled in, with the lower ones filled in
        by the mirroring performed in `curvature_matrix`.

        Unlike the imaging sparse inversion, the linear function's `mapping_matrix` is passed to the operator
        without any noise weighting or forward operation applied, because the interferometer operator
        `W~ = Re(Fᴴ W F)` already contains both.
        """
        if self.has(cls=Mapper):
            curvature_matrix = self._curvature_matrix_multi_mapper
        else:
            curvature_matrix = self._xp.zeros((self.total_params, self.total_params))

        sparse_operator = self.dataset.sparse_operator
        extent_index_for_masked_pixel = self.mask.extent_index_for_masked_pixel

        mapper_list = self.cls_list_from(cls=Mapper)
        mapper_param_range_list = self.param_range_list_from(cls=Mapper)

        linear_func_list = self.cls_list_from(cls=AbstractLinearObjFuncList)
        linear_func_param_range_list = self.param_range_list_from(
            cls=AbstractLinearObjFuncList
        )

        # With a primary beam `P` each linear function's response is `P B_k`, matching the beamed
        # mapper triplets, so every block is formed with the beamed operator.
        mapping_matrix_list = [
            self._mapping_matrix_beamed_from(linear_func.mapping_matrix)
            for linear_func in linear_func_list
        ]

        for mapper_index, mapper in enumerate(mapper_list):
            mapper_param_range = mapper_param_range_list[mapper_index]

            rows, cols, vals = self._sparse_triplets_curvature_from(mapper=mapper)

            for func_index in range(len(linear_func_list)):
                linear_func_param_range = linear_func_param_range_list[func_index]

                off_diag = sparse_operator.curvature_matrix_off_diag_func_list_from(
                    curvature_weights=mapping_matrix_list[func_index],
                    extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                    rows=rows,
                    cols=cols,
                    vals=vals,
                    S=mapper.params,
                    xp=self._xp,
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

        # The linear func x linear func block is symmetric, so only the upper triangle of blocks is
        # computed, with the mirrored block set from the transpose.
        for index_0 in range(len(linear_func_list)):
            linear_func_param_range_0 = linear_func_param_range_list[index_0]

            for index_1 in range(index_0, len(linear_func_list)):
                linear_func_param_range_1 = linear_func_param_range_list[index_1]

                diag = sparse_operator.curvature_matrix_func_list_from(
                    curvature_weights_0=mapping_matrix_list[index_0],
                    curvature_weights_1=mapping_matrix_list[index_1],
                    extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                    xp=self._xp,
                )

                if self._xp is np:
                    curvature_matrix[
                        linear_func_param_range_0[0] : linear_func_param_range_0[1],
                        linear_func_param_range_1[0] : linear_func_param_range_1[1],
                    ] = diag

                    if index_1 != index_0:
                        curvature_matrix[
                            linear_func_param_range_1[0] : linear_func_param_range_1[1],
                            linear_func_param_range_0[0] : linear_func_param_range_0[1],
                        ] = diag.T
                else:
                    curvature_matrix = curvature_matrix.at[
                        linear_func_param_range_0[0] : linear_func_param_range_0[1],
                        linear_func_param_range_1[0] : linear_func_param_range_1[1],
                    ].set(diag)

                    if index_1 != index_0:
                        curvature_matrix = curvature_matrix.at[
                            linear_func_param_range_1[0] : linear_func_param_range_1[1],
                            linear_func_param_range_0[0] : linear_func_param_range_0[1],
                        ].set(diag.T)

        return curvature_matrix

    @property
    def mapped_reconstructed_data_dict(
        self,
    ) -> Dict[LinearObj, Array2D]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays. This does not track which
        quantities belong to which linear objects, therefore the linear equation's solutions (which are returned as
        ndarrays) do not contain information on which linear object(s) they correspond to.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed image (in real space, before the non-uniform fast Fourier transform), where the keys
        are the instances of each linear object in the inversion.

        Each linear object's reconstructed image is `L s`, where `L` is its real-space mapping matrix and `s` its
        block of the `reconstruction`. This is the mirror of the sparse `data_vector` (`Lᵀ d~`):

        - For a `Mapper` the image `A s` is computed from the mapper's sparse COO triplets
          (`_sparse_triplets_slim_from`), as a scatter-add of `vals * s[cols]` over the slim image pixel
          rows. The dense (masked pixels x source pixels) mapping matrix is therefore never built, which
          on a 384 x 384 unmasked grid with a 48² source mesh is 2.7 GB.
        - For an `AbstractLinearObjFuncList` the mapping matrix is small (one column per linear function),
          so the dense `mapping_matrix @ reconstruction` product is used directly, as in the inherited
          implementation.

        The result is used by `mapped_reconstructed_operated_data_dict` (model visibilities) and by fit
        outputs such as galaxy image dictionaries.
        """
        mapped_reconstructed_data_dict = {}

        reconstruction_dict = self.source_quantity_dict_from(
            source_quantity=self.reconstruction
        )

        for linear_obj in self.linear_obj_list:
            reconstruction = reconstruction_dict[linear_obj]

            if isinstance(linear_obj, Mapper):
                mapped_reconstructed_data = self._mapped_reconstructed_data_mapper_from(
                    mapper=linear_obj, reconstruction=reconstruction
                )
            else:
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

    def _mapped_reconstructed_data_mapper_from(
        self, mapper: Mapper, reconstruction
    ) -> np.ndarray:
        """
        Returns a mapper's reconstructed image, `A s`, computed from the sparse triplets of its mapping
        operator `A` without building the dense mapping matrix.

        Entry `i` (a slim image pixel) is the sum over the non-zeros of `A`'s row `i` of `val * s[col]`,
        i.e. a scatter-add of `vals * reconstruction[cols]` over the slim image pixel rows. This is the
        transpose of the segment-sum performed by `_data_vector_mapper_from`.

        Parameters
        ----------
        mapper
            The mapper whose reconstructed image is computed.
        reconstruction
            The mapper's block of the `reconstruction` (its source pixel values `s`).
        """
        # The reconstructed image is the true sky `A s`, so the primary beam is not applied; the
        # model visibilities `F P A s` pick it up in the transformer.
        rows, cols, vals = self._sparse_triplets_slim_from(
            mapper=mapper, apply_primary_beam=False
        )

        contrib = vals * reconstruction[cols]

        n_slim = self.mask.pixels_in_mask

        if self._xp is np:
            return np.bincount(rows, weights=contrib, minlength=n_slim)

        return self._xp.zeros(n_slim, dtype=contrib.dtype).at[rows].add(contrib)

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
        if self.transformer is None:
            raise exc.InversionException(
                "This interferometer inversion's dataset is array-free (an `Interferometer` "
                "built by from_stream / from_sparse_terms) and has no transformer, so the "
                "reconstruction cannot be mapped to visibilities "
                "(`mapped_reconstructed_operated_data_dict`). The `log_evidence` and "
                "`fast_chi_squared` do not need it; use the in-memory constructor if you need "
                "model visibilities."
            )

        mapped_reconstructed_operated_data_dict = {}

        image_dict = self.mapped_reconstructed_data_dict

        for linear_obj in self.linear_obj_list:
            visibilities = self.transformer.visibilities_from(
                image=image_dict[linear_obj], xp=self._xp
            )

            visibilities = Visibilities(visibilities=visibilities)

            mapped_reconstructed_operated_data_dict[linear_obj] = visibilities

        return mapped_reconstructed_operated_data_dict
