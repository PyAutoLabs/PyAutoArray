"""
The off-diagonal blocks of the `curvature_matrix` in the sparse imaging inversions, asserted
against brute-force references.

The linear-func x linear-func block is computed from the upper triangle of blocks only, with
the mirrored block set from the transpose; the mapper x linear-func block of the numba
inversion correlates the curvature weights with the PSF via an FFT convolution with the
reversed kernel and scatters the result in numba.
"""

import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.inversion.imaging.sparse import InversionImagingSparse
from autoarray.inversion.inversion.imaging_numba import inversion_imaging_numba_util
from autoarray.inversion.inversion.imaging_numba.sparse import (
    InversionImagingSparseNumba,
)
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.linear_obj.unique_mappings import UniqueMappings
from autoarray.inversion.mappers.abstract import Mapper

ASYMMETRIC_KERNEL = np.arange(1.0, 16.0).reshape(3, 5)


class FakeLinearFunc:
    def __init__(self, params):
        self.params = params


class StubMixin:
    """Bypasses the real constructor: the property under test only needs the linear
    func list, their param ranges, the noise map and an empty starting matrix."""

    def __init__(self, operated_matrix_list, noise_map):
        self._func_list = [
            FakeLinearFunc(params=matrix.shape[1]) for matrix in operated_matrix_list
        ]

        param_range_list = []
        total_params = 0
        for linear_func in self._func_list:
            param_range_list.append([total_params, total_params + linear_func.params])
            total_params += linear_func.params

        self._param_range_list = param_range_list
        self._total_params = total_params
        self._noise_map = noise_map

        self.linear_func_operated_mapping_matrix_dict = {
            linear_func: matrix
            for linear_func, matrix in zip(self._func_list, operated_matrix_list)
        }

    @property
    def _xp(self):
        return np

    @property
    def noise_map(self):
        return self._noise_map

    def cls_list_from(self, cls):
        if cls is Mapper:
            return []
        return self._func_list

    def param_range_list_from(self, cls):
        if cls is Mapper:
            return []
        return self._param_range_list

    @property
    def _curvature_matrix_multi_mapper(self):
        return np.zeros((self._total_params, self._total_params))


class StubInversionSparse(StubMixin, InversionImagingSparse):
    pass


class StubInversionSparseNumba(StubMixin, InversionImagingSparseNumba):
    pass


def curvature_matrix_brute_force_from(operated_matrix_list, noise_map):
    param_range_list = []
    total_params = 0
    for matrix in operated_matrix_list:
        param_range_list.append([total_params, total_params + matrix.shape[1]])
        total_params += matrix.shape[1]

    curvature_matrix = np.zeros((total_params, total_params))

    for index_0, matrix_0 in enumerate(operated_matrix_list):
        for index_1, matrix_1 in enumerate(operated_matrix_list):
            curvature_matrix[
                param_range_list[index_0][0] : param_range_list[index_0][1],
                param_range_list[index_1][0] : param_range_list[index_1][1],
            ] = np.dot((matrix_0 / noise_map[:, None]).T, matrix_1 / noise_map[:, None])

    return curvature_matrix


@pytest.fixture
def operated_matrix_list_and_noise_map():
    rng = np.random.default_rng(7)

    data_pixels = 37

    operated_matrix_list = [
        rng.normal(size=(data_pixels, params)) for params in (3, 2, 4)
    ]

    # Spatially varying, non-constant, non-symmetric noise map.
    noise_map = 0.5 + rng.random(data_pixels) * 2.0

    return operated_matrix_list, noise_map


@pytest.mark.parametrize("cls", [StubInversionSparse, StubInversionSparseNumba])
def test__curvature_matrix_func_list_blocks__matches_brute_force(
    cls, operated_matrix_list_and_noise_map
):
    operated_matrix_list, noise_map = operated_matrix_list_and_noise_map

    inversion = cls(operated_matrix_list=operated_matrix_list, noise_map=noise_map)

    curvature_matrix = inversion._curvature_matrix_func_list_and_mapper

    brute_force = curvature_matrix_brute_force_from(
        operated_matrix_list=operated_matrix_list, noise_map=noise_map
    )

    assert curvature_matrix == pytest.approx(brute_force, abs=1.0e-12)


# The mapper x linear-func block of the numba sparse imaging inversion no longer correlates
# the curvature weights with the PSF inside its numba kernel: the correlation runs as a
# batched FFT convolution with the reversed PSF (`Convolver.reversed_kernel`) and only the
# scatter onto source pixels stays in numba. The block is also written together with its
# transpose, since the global mirroring pass over F was removed.
#
# The test below asserts both against the retained dense sliding-window kernel, with an
# asymmetric, non-square PSF so a missing reversal or a transposed axis cannot pass.


class FakeMapper:
    def __init__(self, params, unique_mappings):
        self.params = params
        self.unique_mappings = unique_mappings


class StubInversionMapperAndFunc(InversionImagingSparseNumba):
    """Bypasses the real constructor: `_curvature_matrix_mapper_func_blocks_from` only needs
    the mapper and linear func lists, their param ranges, the noise map, the mask and the
    PSF."""

    def __init__(self, mapper, operated_matrix, noise_map, mask, psf):
        self._mapper = mapper
        self._func = FakeLinearFunc(params=operated_matrix.shape[1])
        self._noise_map = noise_map
        self._mask = mask
        self._psf = psf

        self.linear_func_operated_mapping_matrix_dict = {self._func: operated_matrix}

        self._total_params = mapper.params + self._func.params

    @property
    def _xp(self):
        return np

    @property
    def total_params(self):
        return self._total_params

    @property
    def noise_map(self):
        return self._noise_map

    @property
    def mask(self):
        return self._mask

    @property
    def psf(self):
        return self._psf

    def cls_list_from(self, cls):
        if cls is Mapper:
            return [self._mapper]
        return [self._func]

    def param_range_list_from(self, cls):
        if cls is Mapper:
            return [[0, self._mapper.params]]
        return [[self._mapper.params, self.total_params]]


@pytest.fixture
def mapper_and_func_inversion():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
            ]
        ),
        pixel_scales=1.0,
    )

    data_pixels = int(mask.pixels_in_mask)
    pix_pixels = 5
    n_funcs = 3

    rng = np.random.default_rng(505)

    max_lengths = 2
    unique_mappings = UniqueMappings(
        data_to_pix_unique=rng.integers(
            0, pix_pixels, size=(data_pixels, max_lengths)
        ).astype("int"),
        data_weights=rng.random(size=(data_pixels, max_lengths)),
        pix_lengths=rng.integers(1, max_lengths + 1, size=data_pixels).astype("int"),
    )

    mapper = FakeMapper(params=pix_pixels, unique_mappings=unique_mappings)

    operated_matrix = rng.normal(size=(data_pixels, n_funcs))
    noise_map = 0.5 + rng.random(data_pixels) * 2.0

    psf = aa.Convolver(
        kernel=aa.Array2D.no_mask(values=ASYMMETRIC_KERNEL, pixel_scales=1.0),
    )

    return StubInversionMapperAndFunc(
        mapper=mapper,
        operated_matrix=operated_matrix,
        noise_map=noise_map,
        mask=mask,
        psf=psf,
    )


def test__curvature_matrix_mapper_func_blocks__matches_dense_kernel_and_places_transpose(
    mapper_and_func_inversion,
):
    inversion = mapper_and_func_inversion

    total_params = inversion.total_params

    curvature_matrix = inversion._curvature_matrix_mapper_func_blocks_from(
        curvature_matrix=np.zeros((total_params, total_params))
    )

    mapper = inversion._mapper
    curvature_weights = np.array(
        list(inversion.linear_func_operated_mapping_matrix_dict.values())[0]
        / inversion.noise_map[:, None] ** 2
    )

    off_diag = inversion_imaging_numba_util.curvature_matrix_off_diags_via_mapper_and_linear_func_curvature_vector_from(
        data_to_pix_unique=mapper.unique_mappings.data_to_pix_unique,
        data_weights=mapper.unique_mappings.data_weights,
        pix_lengths=mapper.unique_mappings.pix_lengths,
        pix_pixels=mapper.params,
        curvature_weights=curvature_weights,
        mask=np.array(inversion.mask),
        psf_kernel=ASYMMETRIC_KERNEL,
    )

    assert curvature_matrix[: mapper.params, mapper.params :] == pytest.approx(
        off_diag, rel=1.0e-6
    )

    # The global mirroring pass is gone, so the transpose must be written here.
    assert curvature_matrix[mapper.params :, : mapper.params] == pytest.approx(
        off_diag.T, rel=1.0e-6
    )

    # The mapper x mapper and linear-func x linear-func blocks are not this helper's to write.
    assert curvature_matrix[: mapper.params, : mapper.params] == pytest.approx(
        np.zeros((mapper.params, mapper.params)), abs=1.0e-12
    )
