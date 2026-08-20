"""
The cross-evaluation memo for linear-func operated mapping matrices in the
numba CPU sparse inversion (`imaging_numba/sparse.py`).

The memo must: reuse the matrix when a fresh linear-func object fingerprints
identically to a previous evaluation's (the fixed-MGE campaign case); recompute
when any state differs (free profile parameters); fall back to the uncached
parent computation when an object cannot be fingerprinted or the memo is
disabled; and never hand out writeable buffers.
"""

import numpy as np
import pytest

from autoarray.inversion.inversion.imaging_numba import sparse as sparse_module
from autoarray.inversion.inversion.imaging_numba.sparse import (
    InversionImagingSparseNumba,
    _operated_mapping_matrix_memo,
    _operated_mapping_matrix_memo_key,
)


class FakeLinearFunc:
    """Stands in for an MGE linear-func bundle: `values` plays the role of the
    profile parameters, and computing the override is counted class-wide so
    tests can assert whether the convolution work actually ran."""

    compute_count = 0

    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    @property
    def operated_mapping_matrix_override(self):
        type(self).compute_count += 1
        return np.outer(self.values, np.arange(1.0, 4.0))


class UnpicklableLinearFunc(FakeLinearFunc):
    def __init__(self, values):
        super().__init__(values)
        self.blocker = lambda: None  # lambdas cannot be pickled


class StubInversion(InversionImagingSparseNumba):
    """Bypasses the real constructor; the property under test only needs
    `cls_list_from` (and instance-dict storage for its cached_property)."""

    def __init__(self, linear_func_list):
        self._stub_linear_func_list = list(linear_func_list)

    def cls_list_from(self, cls):
        return self._stub_linear_func_list


@pytest.fixture(autouse=True)
def _clean_memo():
    _operated_mapping_matrix_memo.clear()
    FakeLinearFunc.compute_count = 0
    yield
    _operated_mapping_matrix_memo.clear()


def test__memo_key__stable_for_equal_state__distinct_for_different_state():
    key_a = _operated_mapping_matrix_memo_key(FakeLinearFunc([1.0, 2.0]))
    key_b = _operated_mapping_matrix_memo_key(FakeLinearFunc([1.0, 2.0]))
    key_c = _operated_mapping_matrix_memo_key(FakeLinearFunc([1.0, 2.5]))

    assert key_a == key_b
    assert key_a != key_c


def test__memo_key__unpicklable_state_returns_none():
    assert _operated_mapping_matrix_memo_key(UnpicklableLinearFunc([1.0])) is None


def test__identical_state_across_fresh_objects__computes_once():
    func_eval_0 = FakeLinearFunc([1.0, 2.0])
    dict_0 = StubInversion([func_eval_0]).linear_func_operated_mapping_matrix_dict

    # A sampler's next evaluation builds a FRESH object with identical state.
    func_eval_1 = FakeLinearFunc([1.0, 2.0])
    dict_1 = StubInversion([func_eval_1]).linear_func_operated_mapping_matrix_dict

    assert FakeLinearFunc.compute_count == 1
    assert np.array_equal(dict_0[func_eval_0], dict_1[func_eval_1])
    assert not dict_1[func_eval_1].flags.writeable


def test__changed_state__recomputes_and_matches_uncached_result():
    StubInversion([FakeLinearFunc([1.0, 2.0])]).linear_func_operated_mapping_matrix_dict

    func_changed = FakeLinearFunc([1.0, 3.0])
    result = StubInversion([func_changed]).linear_func_operated_mapping_matrix_dict[
        func_changed
    ]

    assert FakeLinearFunc.compute_count == 2
    assert np.array_equal(result, np.outer([1.0, 3.0], np.arange(1.0, 4.0)))


def test__cached_property__single_dict_build_per_inversion():
    inversion = StubInversion([FakeLinearFunc([1.0, 2.0])])

    dict_first = inversion.linear_func_operated_mapping_matrix_dict
    dict_second = inversion.linear_func_operated_mapping_matrix_dict

    assert dict_first is dict_second


def test__unpicklable_func__falls_back_to_uncached_parent_and_stores_nothing():
    func = UnpicklableLinearFunc([1.0, 2.0])

    result = StubInversion([func]).linear_func_operated_mapping_matrix_dict[func]

    assert np.array_equal(result, np.outer([1.0, 2.0], np.arange(1.0, 4.0)))
    assert len(_operated_mapping_matrix_memo) == 0


def test__env_var_disables_memo(monkeypatch):
    monkeypatch.setenv("AUTOARRAY_NUMBA_OPERATED_MEMO", "0")

    func = FakeLinearFunc([1.0, 2.0])
    result = StubInversion([func]).linear_func_operated_mapping_matrix_dict[func]

    assert np.array_equal(result, np.outer([1.0, 2.0], np.arange(1.0, 4.0)))
    assert len(_operated_mapping_matrix_memo) == 0


def test__memo_eviction__bounded_size():
    for value in range(sparse_module._OPERATED_MAPPING_MATRIX_MEMO_MAX_ENTRIES + 3):
        func = FakeLinearFunc([float(value)])
        StubInversion([func]).linear_func_operated_mapping_matrix_dict

    assert (
        len(_operated_mapping_matrix_memo)
        == sparse_module._OPERATED_MAPPING_MATRIX_MEMO_MAX_ENTRIES
    )
