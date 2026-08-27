"""
The linear-func x linear-func block of the `curvature_matrix` in the sparse imaging
inversions is computed from the upper triangle of blocks only, with the mirrored block
set from the transpose. This asserts the result matches a brute-force full double loop
for a random, spatially varying noise map.
"""

import numpy as np
import pytest

from autoarray.inversion.inversion.imaging.sparse import InversionImagingSparse
from autoarray.inversion.inversion.imaging_numba.sparse import (
    InversionImagingSparseNumba,
)
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.mappers.abstract import Mapper


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
