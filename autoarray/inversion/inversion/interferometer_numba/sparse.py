import numpy as np
from typing import List, Union

from autonerves import cached_property, conf

from autoarray import exc
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.interferometer.sparse import (
    InversionInterferometerSparse,
)
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.settings import Settings

from autoarray.inversion.inversion.interferometer_numba import (
    inversion_interferometer_numba_util,
)


class InversionInterferometerSparseNumba(InversionInterferometerSparse):
    def __init__(
        self,
        dataset: Union[Interferometer, DatasetInterface],
        linear_obj_list: List[LinearObj],
        settings: Settings = None,
        xp=np,
        preloads=None,
    ):
        """
        The single-mapper interferometer inversion with its curvature matrix `F` assembled
        by a numba CPU kernel instead of the FFT route.

        `InversionInterferometerSparse` forms `F = Aᵀ W~ A` by applying `W~` as an FFT
        convolution to blocks of `A`'s columns. That cost is set by the number of source
        columns, whatever their density. The `direct_conv` kernel instead convolves each
        source column over the `(ny, nx)` extent rectangle directly, at a cost that scales
        with the column's non-zeros -- so it wins while the mapping operator stays sparse
        (2-7x below ~60 non-zeros per source column on Delaunay meshes, ~77 on rectangular
        ones; autolens_profiling issue #226) and loses above it.

        Everything except `curvature_matrix_diag` is inherited: the data vector,
        regularization, the reconstruction and the evidence terms are the parent's, and
        `curvature_matrix` keeps the parent's no-regularization diagonal handling. The
        kernel returns a complete symmetric `F`, so no mirroring pass is needed (and the
        parent's single-mapper branch does not apply one).

        The class raises on every configuration the kernel cannot represent rather than
        silently working around it -- see `_check_preconditions`. The factory
        (`inversion_interferometer_from`) checks the same conditions before it routes
        here and falls through to `InversionInterferometerSparse` when any fails; a
        direct construction with bad inputs still raises.

        Parameters
        ----------
        dataset
            The interferometer dataset (or `DatasetInterface`) being reconstructed. It
            must carry a `sparse_operator`, whose `nufft_precision_operator` is the `W~`
            preload the kernel convolves with.
        linear_obj_list
            The linear objects reconstructing the data. Exactly one `Mapper`, and nothing
            else.
        settings
            The inversion settings (`autoarray.settings.Settings`).
        xp
            The array module. Must be `numpy`: the kernel is numba, with no JAX path.
        preloads
            Optional `AbstractPreloads`, forwarded to the parent unchanged.
        """
        if xp is not np:
            raise exc.InversionException(
                "`InversionInterferometerSparseNumba` was passed a non-NumPy array module "
                f"({xp!r}). The `direct_conv` curvature kernel is a numba `@jit` function "
                "with no JAX path; use `InversionInterferometerSparse` for the JAX/FFT "
                "route."
            )

        try:
            import numba  # noqa: F401
        except ModuleNotFoundError as error:
            raise exc.InversionException(
                "The numba interferometer inversion requires numba, which is not "
                "installed. Install it, or use `InversionInterferometerSparse`."
            ) from error

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            preloads=preloads,
        )

        self._check_preconditions()

    def _check_preconditions(self) -> None:
        """
        Raise on every configuration the `direct_conv` kernel cannot represent.

        These are raised, not worked around: each one would otherwise compute a different
        (and silently wrong) `F`, or none at all. The factory checks the same conditions
        up front and simply does not route here when one fails.
        """
        if self.has(cls=AbstractLinearObjFuncList):
            raise exc.InversionException(
                "A linear-function list (e.g. a linear light profile or MGE basis) was "
                "passed to `InversionInterferometerSparseNumba`. The `direct_conv` kernel "
                "assembles F only from a mapper's `pix_indexes/sizes/weights_for_sub_slim_index` "
                "triplets and has no mapper x function or function x function block.\n\n"
                "Use `InversionInterferometerSparse` (which does support mixed linear "
                "objects) for such a model."
            )

        total_mappers = self.total(cls=Mapper)

        if total_mappers != 1:
            raise exc.InversionException(
                f"`InversionInterferometerSparseNumba` was passed {total_mappers} mappers. "
                "The `direct_conv` kernel builds a single [pix_pixels, pix_pixels] "
                "curvature matrix from one mapper and has no off-diagonal mapper x mapper "
                "block.\n\n"
                "Use `InversionInterferometerSparse` for a multi-mapper model."
            )

        mapper = self.cls_list_from(cls=Mapper)[0]

        sub_fraction = np.asarray(mapper.over_sampler.sub_fraction.array)

        if not np.all(sub_fraction == 1.0):
            raise exc.InversionException(
                "`InversionInterferometerSparseNumba` was passed a mapper whose "
                f"over-sampler has `sub_fraction != 1` (minimum {float(sub_fraction.min())}, "
                "i.e. `over_sample_size` up to "
                f"{int(round(1.0 / float(sub_fraction.min())))}).\n\n"
                "The sparse triplets fold `over_sampler.sub_fraction` into the mapping "
                "weights (`interferometer/sparse.py::_sparse_triplets_curvature_from`); "
                "the `direct_conv` kernel uses `pix_weights_for_sub_slim_index` as-is and "
                "its rows are indexed on the slim grid, so with over-sampling it would "
                "silently compute a different F.\n\n"
                "Apply `over_sample_size_pixelization=1` to the dataset, or use "
                "`InversionInterferometerSparse`."
            )

    @cached_property
    def kernel_index_arrays(self) -> dict:
        """
        The mapper's triplets in the flat CSR / CSC / extent-grid layout the kernel takes.

        The extent implied by the mask is checked against the preload's `(2ny, 2nx)`
        shape, because the kernel indexes the preload by extent offsets with wrapped
        (negative) indices -- a mismatch would wrap silently rather than fail.
        """
        mapper = self.cls_list_from(cls=Mapper)[0]

        extent_index_for_masked_pixel = np.asarray(
            self.mask.extent_index_for_masked_pixel
        )[np.asarray(mapper.slim_index_for_sub_slim_index)]

        inputs = inversion_interferometer_numba_util.kernel_inputs_from(
            pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
            extent_shape=self.mask.shape_native_masked_pixels,
            pix_pixels=int(mapper.params),
        )

        preload_shape = np.asarray(
            self.dataset.sparse_operator.nufft_precision_operator
        ).shape

        if (2 * inputs["ny"], 2 * inputs["nx"]) != preload_shape:
            raise exc.InversionException(
                "The unmasked extent implied by the mask "
                f"({inputs['ny']} x {inputs['nx']}) does not match the sparse operator's "
                f"`nufft_precision_operator` shape {preload_shape}, which must be "
                "(2ny, 2nx). The `direct_conv` kernel indexes the preload by extent "
                "offsets, so a mismatch would wrap silently instead of failing."
            )

        return inputs

    @property
    def curvature_matrix_diag(self) -> np.ndarray:
        """
        `F = Aᵀ W~ A` for the inversion's single mapper, from the `direct_conv` numba
        kernel.

        The returned matrix is complete and symmetric (the kernel loops the full source
        column x masked pixel space rather than halving on symmetry), so it drops straight
        into the parent's single-mapper branch with no mirroring pass.
        """
        inputs = self.kernel_index_arrays

        preload = np.ascontiguousarray(
            np.asarray(
                self.dataset.sparse_operator.nufft_precision_operator, dtype=np.float64
            )
        )

        if _numba_parallel():
            kernel = inversion_interferometer_numba_util.direct_conv_parallel_kernel()
        else:
            kernel = inversion_interferometer_numba_util.curvature_direct_conv

        return kernel(
            preload,
            inputs["iy"],
            inputs["ix"],
            inputs["flat"],
            inputs["indptr"],
            inputs["col"],
            inputs["val"],
            inputs["cscptr"],
            inputs["csc_row"],
            inputs["csc_val"],
            inputs["ny"],
            inputs["nx"],
            inputs["pix_pixels"],
        )


def _numba_parallel() -> bool:
    """
    Whether the parallel (`prange`) kernel is used, read from the same
    `general.yaml -> numba -> parallel` flag the shared `numba_util.jit` decorator reads,
    with the same fallback when no config supplies it.
    """
    try:
        return bool(conf.instance["general"]["numba"]["parallel"])
    except Exception:
        return False
