import numpy as np
from typing import List, Union

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping

from autoarray.inversion.inversion.interferometer.mapping import (
    InversionInterferometerMapping,
)
from autoarray.inversion.inversion.interferometer.sparse import (
    InversionInterferometerSparse,
)
from autoarray.inversion.inversion.interferometer_numba.sparse import (
    InversionInterferometerSparseNumba,
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
    use_sparse_operator = True

    if all(
        isinstance(linear_obj, AbstractLinearObjFuncList)
        for linear_obj in linear_obj_list
    ):
        use_sparse_operator = False

    if dataset.sparse_operator is not None and use_sparse_operator:

        if _use_interferometer_numba(
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
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
    - `settings.interferometer_numba_nnz_per_source_max > 0` -- `0` is the kill switch.
    - exactly one `Mapper` and no `AbstractLinearObjFuncList` -- the kernel builds a single
      mapper-mapper block and has no off-diagonal or function blocks.
    - no over-sampling (`sub_fraction == 1`) -- the kernel uses the mapper's weights as-is.
    - the mapper's mean non-zeros per source column is at or below the gate -- above it the
      FFT route is faster (see `Settings.interferometer_numba_nnz_per_source_max`).
    - `import numba` succeeds.

    Parameters
    ----------
    linear_obj_list
        The linear objects reconstructing the data.
    settings
        The inversion settings, whose `interferometer_numba_nnz_per_source_max` is the
        geometry gate. `None` uses the packaged defaults.
    xp
        The array module the inversion runs on.
    """
    if xp is not np:
        return False

    settings = settings if settings is not None else Settings()

    nnz_max = settings.interferometer_numba_nnz_per_source_max

    if nnz_max is None or nnz_max <= 0:
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

    nnz_per_source_column = (
        inversion_interferometer_numba_util.nnz_per_source_column_from(mapper=mapper)
    )

    if nnz_per_source_column > nnz_max:
        return False

    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        return False

    return True
