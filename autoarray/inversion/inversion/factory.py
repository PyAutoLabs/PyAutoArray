import numpy as np
from typing import List, Optional, Tuple, Union

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping

from autoarray.inversion.inversion.interferometer.mapping import (
    InversionInterferometerMapping,
)
from autoarray.inversion.inversion.interferometer.sparse import (
    InversionInterferometerSparse,
)
from autoarray.inversion.inversion.interferometer.abstract import (
    real_space_mask_from,
)
from autoarray.inversion.inversion.interferometer_numba.sparse import (
    InversionInterferometerSparseNumba,
)
from autoarray.inversion.inversion.interferometer_numba import (
    sparse as interferometer_numba_sparse,
)
from autoarray.inversion.inversion.interferometer_numba import (
    inversion_interferometer_numba_util,
)
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util import (
    SparseLinAlgImagingNumba,
)
from autoarray.inversion.inversion.imaging_numba.sparse import (
    InversionImagingSparseNumba,
)
from autoarray.inversion.inversion.imaging.sparse import (
    InversionImagingSparse,
)
from autoarray.settings import Settings
from autoarray.structures.arrays.uniform_2d import Array2D


def inversion_from(
    dataset: Union[Imaging, Interferometer, DatasetInterface],
    linear_obj_list: List[LinearObj],
    settings: Settings = None,
    xp=np,
    preloads=None,
):
    """
    Factory which given an input dataset and list of linear objects, creates an `Inversion`.

    An `Inversion` reconstructs the input dataset using a list of linear objects (e.g. a list of analytic functions
    or a pixelized grid). The inversion solves for the values of these linear objects that best reconstruct the
    dataset, via linear matrix algebra.

    Different `Inversion` objects are used for different dataset types (e.g. `Imaging`, `Interferometer`) and
    for different linear algebra formalisms (determined via the input `settings`) which solve for the linear object
    parameters in different ways.

    This factory inspects the type of dataset input and settings of the inversion in order to create the appropriate
    inversion object.

    Parameters
    ----------
    dataset
        The dataset (e.g. `Imaging`, `Interferometer`) whose data is reconstructed via the `Inversion`.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    if isinstance(dataset.data, Array2D):
        return inversion_imaging_from(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
        )

    return inversion_interferometer_from(
        dataset=dataset,
        linear_obj_list=linear_obj_list,
        settings=settings,
        xp=xp,
        preloads=preloads,
    )


def inversion_imaging_from(
    dataset,
    linear_obj_list: List[LinearObj],
    settings: Settings = None,
    xp=np,
):
    """
    Factory which given an input `Imaging` dataset and list of linear objects, creates an `InversionImaging`.

    Unlike the `inversion_from` factory this function takes the `data` and `noise_map` objects as separate
    inputs, which facilitates certain computations where the `dataset` object is unpacked before the `Inversion` is
    performed (for example if the noise-map is scaled before the inversion to downweight certain regions of the
    data).

    An `Inversion` reconstructs the input dataset using a list of linear objects (e.g. a list of analytic functions
    or a pixelized grid). The inversion solves for the values of these linear objects that best reconstruct the
    dataset, via linear matrix algebra.

    Different `Inversion` objects are used for different linear algebra formalisms (determined via the
    input `settings`) which solve for the linear object parameters in different ways.

    This factory inspects the type of dataset input and settings of the inversion in order to create the appropriate
    inversion object.

    Parameters
    ----------
    dataset
        The dataset (e.g. `Imaging`) whose data is reconstructed via the `Inversion`.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """

    # Unlike `inversion_interferometer_from`, imaging keeps func-list-only inversions (e.g. an MGE with no
    # pixelization) on the mapping formalism. A func-list-only imaging mapping inversion is cheap (one PSF
    # convolution per basis function), and the imaging sparse data vector's handling of profile-subtracted
    # data is tracked separately, so the interferometer routing change is deliberately not mirrored here.
    use_sparse_operator = True

    if all(
        isinstance(linear_obj, AbstractLinearObjFuncList)
        for linear_obj in linear_obj_list
    ):
        use_sparse_operator = False

    if dataset.sparse_operator is not None and use_sparse_operator:

        if isinstance(dataset.sparse_operator, SparseLinAlgImagingNumba):

            return InversionImagingSparseNumba(
                dataset=dataset,
                linear_obj_list=linear_obj_list,
                settings=settings,
                xp=xp,
            )

        return InversionImagingSparse(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
        )

    return InversionImagingMapping(
        dataset=dataset,
        linear_obj_list=linear_obj_list,
        settings=settings,
        xp=xp,
    )


def primary_beam_is_trivial(primary_beam_slim) -> bool:
    """
    Whether a primary beam leaves the interferometer response unchanged: `None`, or every value exactly 1.
    """
    if primary_beam_slim is None:
        return True

    return bool(np.all(np.asarray(primary_beam_slim) == 1.0))


def inversion_interferometer_from(
    dataset: Union[Interferometer, DatasetInterface],
    linear_obj_list: List[LinearObj],
    settings: Settings = None,
    xp=np,
    preloads=None,
):
    """
    Factory which given an input `Interferometer` dataset and list of linear objects, creates
    an `InversionInterferometer`.

    Unlike the `inversion_from` factory this function takes the `data` and `noise_map` objects as separate
    inputs, which facilitates certain computations where the `dataset` object is unpacked before the `Inversion` is
    performed (for example if the noise-map is scaled before the inversion to downweight certain regions of the
    data).

    An `Inversion` reconstructs the input dataset using a list of linear objects (e.g. a list of analytic functions
    or a pixelized grid). The inversion solves for the values of these linear objects that best reconstruct the
    dataset, via linear matrix algebra.

    Different `Inversion` objects are used for different linear algebra formalisms (determined via the
    input `settings`) which solve for the linear object parameters in different ways.

    This factory inspects the type of dataset input and settings of the inversion in order to create the appropriate
    inversion object.

    Parameters
    ----------
    dataset
        The dataset (e.g. `Interferometer`) whose data is reconstructed via the `Inversion`.
    linear_obj_list
        The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
        input dataset's data and whose values are solved for via the inversion.
    settings
        Settings controlling how an inversion is fitted for example which linear algebra formalism is used.

    Returns
    -------
    An `Inversion` whose type is determined by the input `dataset` and `settings`.
    """
    # A sparse operator selects the w-tilde formalism for every linear object list, including func-list-only
    # ones (e.g. an MGE with no pixelization). For interferometer data this avoids the dense
    # `transform_mapping_matrix` of shape (N_vis, S), which dominates run time and memory at large N_vis, and
    # `InversionInterferometerSparse` handles zero mappers (the func-func curvature blocks and the data vector
    # need no mapper). `_use_interferometer_numba` returns False whenever a func-list is present.
    if dataset.sparse_operator is not None:

        # The numba `direct_conv` kernel uses the mapper weights as-is and cannot apply a primary
        # beam, so a dataset with a non-trivial one (any value other than 1) always takes the FFT
        # sparse route. `P = ones` keeps the numba route, so it is bit-for-bit `P = None`.
        if primary_beam_is_trivial(
            getattr(dataset.sparse_operator, "primary_beam_slim", None)
        ) and _use_interferometer_numba(
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            extent_shape=(
                dataset.sparse_operator.y_shape,
                dataset.sparse_operator.x_shape,
            ),
            # Only the NumPy cost model reads the mask, so the JAX path never resolves it.
            real_space_mask=(
                real_space_mask_from(dataset=dataset) if xp is np else None
            ),
        ):
            return InversionInterferometerSparseNumba(
                dataset=dataset,
                linear_obj_list=linear_obj_list,
                settings=settings,
                xp=xp,
                preloads=preloads,
            )

        return InversionInterferometerSparse(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            preloads=preloads,
        )

    return InversionInterferometerMapping(
        dataset=dataset,
        linear_obj_list=linear_obj_list,
        settings=settings,
        xp=xp,
    )


def _use_interferometer_numba(
    linear_obj_list: List[LinearObj],
    settings: Settings = None,
    xp=np,
    extent_shape: Optional[Tuple[int, int]] = None,
    real_space_mask=None,
) -> bool:
    """
    Whether an interferometer inversion is routed to the numba `direct_conv` curvature
    path (`InversionInterferometerSparseNumba`) rather than the FFT one
    (`InversionInterferometerSparse`).

    Every condition below is a routing decision, not an error: a model the kernel cannot
    represent, or a geometry where the FFT route is faster, simply falls through to the
    sparse path silently. Constructing `InversionInterferometerSparseNumba` directly with
    such inputs still raises -- the class checks the same preconditions itself, so the two
    cannot drift apart in meaning, only in whether they are fatal.

    The conditions are, in order of cost to evaluate:

    - `xp is np` -- the kernel is numba, with no JAX path.
    - `settings.interferometer_route` -- `"fft"` never routes to numba; `"numba"` forces it
      whenever the kernel can represent the inversion (the conditions below that are not
      about speed), bypassing the geometry gate and the kill switch. For benchmarking.
    - `settings.interferometer_numba_nnz_per_source_max` is not `0` -- `0` is the kill
      switch.
    - exactly one `Mapper` and no `AbstractLinearObjFuncList` -- the kernel builds a single
      mapper-mapper block and has no off-diagonal or function blocks.
    - no over-sampling (`sub_fraction == 1`) -- the kernel uses the mapper's weights as-is.
    - the geometry gate, which picks the faster route:

      - `interferometer_numba_nnz_per_source_max` positive -- a hard gate: numba at or
        below that many mean non-zeros per source column (the pre-#628 behaviour, kept
        so explicit A/B settings are honoured).
      - `interferometer_numba_nnz_per_source_max` `None` (the packaged default) -- the
        cost model `inversion_interferometer_numba_util.curvature_route_seconds_from`,
        which predicts both routes' time from the source pixels, the mapping operator's
        non-zeros and the extent rectangle `extent_shape`, and routes to numba when
        `direct_conv` is predicted no slower. Without an `extent_shape` the model cannot
        be evaluated and the FFT route is used. When numba is serial and
        `interferometer_direct_conv_symmetric_max_work_fraction` is positive, the
        symmetric kernel's work fraction is computed from the mapper's triplets and
        `real_space_mask` (`symmetric_work_fraction_from`), and if it is at or below that
        threshold -- i.e. `InversionInterferometerSparseNumba` will run the cheaper
        symmetric kernel -- it is passed to the cost model, which scales the predicted
        `direct_conv` time by it. Without a `real_space_mask` whose extent matches
        `extent_shape`, the full kernel's cost is predicted.
    - `import numba` succeeds.

    NaN mapping weights (an overflowing or degenerate `DelaunayNN` Sibson row) are
    deliberately not a condition: the kernel propagates them into `F` exactly as the FFT
    and mapping routes do, so every route returns the same NaN evidence and the sample is
    rejected identically (pinned by the NaN-weight tests in
    `test_interferometer_numba.py` and `test_factory.py`).

    Parameters
    ----------
    linear_obj_list
        The linear objects reconstructing the data.
    settings
        The inversion settings, whose `interferometer_route` and
        `interferometer_numba_nnz_per_source_max` select the route. `None` uses the
        packaged defaults.
    xp
        The array module the inversion runs on.
    extent_shape
        The `(ny, nx)` extent rectangle the `W~` operator lives on, i.e. the dataset's
        `sparse_operator.y_shape, sparse_operator.x_shape`
        (`= mask.shape_native_masked_pixels`). Needed only by the default cost model.
    real_space_mask
        The real-space mask the inversion reconstructs on (`real_space_mask_from`), used
        only by the default cost model for the symmetric kernel's work fraction.
    """
    if xp is not np:
        return False

    settings = settings if settings is not None else Settings()

    route = settings.interferometer_route

    if route == "fft":
        return False

    nnz_max = settings.interferometer_numba_nnz_per_source_max

    if route is None and nnz_max is not None and nnz_max <= 0:
        return False

    if any(
        isinstance(linear_obj, AbstractLinearObjFuncList)
        for linear_obj in linear_obj_list
    ):
        return False

    mapper_list = [
        linear_obj for linear_obj in linear_obj_list if isinstance(linear_obj, Mapper)
    ]

    if len(mapper_list) != 1 or len(mapper_list) != len(linear_obj_list):
        return False

    mapper = mapper_list[0]

    sub_fraction = np.asarray(mapper.over_sampler.sub_fraction.array)

    if not np.all(sub_fraction == 1.0):
        return False

    if route is None:
        if nnz_max is not None:
            nnz_per_source_column = (
                inversion_interferometer_numba_util.nnz_per_source_column_from(
                    mapper=mapper
                )
            )

            if nnz_per_source_column > nnz_max:
                return False

        else:
            if extent_shape is None:
                return False

            t_fft, t_direct_conv = (
                inversion_interferometer_numba_util.curvature_route_seconds_from(
                    source_pixels=int(mapper.params),
                    nnz=int(np.asarray(mapper.pix_sizes_for_sub_slim_index).sum()),
                    extent_shape=extent_shape,
                    work_fraction=_symmetric_work_fraction_for_route(
                        mapper=mapper,
                        settings=settings,
                        extent_shape=extent_shape,
                        real_space_mask=real_space_mask,
                    ),
                )
            )

            if t_direct_conv > t_fft:
                return False

    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        return False

    return True


def _symmetric_work_fraction_for_route(
    mapper, settings: Settings, extent_shape, real_space_mask
) -> Optional[float]:
    """
    The work fraction to give the route cost model: the symmetric kernel's
    `work_fraction` when `InversionInterferometerSparseNumba` would run it (serial numba,
    the gate positive, finite weights and the fraction at or below it), else `None`
    (predict the full kernel).

    Computed from the mapper's triplets and the mask's extent indices, with the same
    statistics the inversion's `symmetric_restriction_inputs` uses, so the gate here and
    the kernel choice there cannot disagree.
    """
    threshold = settings.interferometer_direct_conv_symmetric_max_work_fraction

    if (
        threshold <= 0.0
        or real_space_mask is None
        or interferometer_numba_sparse._numba_parallel()
    ):
        return None

    if tuple(int(n) for n in real_space_mask.shape_native_masked_pixels) != tuple(
        int(n) for n in extent_shape
    ):
        return None

    # Non-finite weights make the inversion fall back to the full kernel.
    sizes = np.asarray(mapper.pix_sizes_for_sub_slim_index)
    weights = np.asarray(mapper.pix_weights_for_sub_slim_index)
    valid = np.arange(weights.shape[1])[None, :] < sizes[:, None]

    if not np.all(np.isfinite(weights[valid])):
        return None

    extent_index_for_masked_pixel = np.asarray(
        real_space_mask.extent_index_for_masked_pixel
    )[np.asarray(mapper.slim_index_for_sub_slim_index)]

    work_fraction = inversion_interferometer_numba_util.symmetric_work_fraction_from(
        pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
        extent_index_for_masked_pixel=extent_index_for_masked_pixel,
        extent_shape=extent_shape,
        pix_pixels=int(mapper.params),
    )

    if work_fraction > threshold:
        return None

    return work_fraction
