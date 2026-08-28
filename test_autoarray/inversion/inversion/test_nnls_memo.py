"""
The cross-evaluation memo for the positive-only (fnnls) solve's passive set
(`nnls_memo.py`, wired in through `reconstruction_positive_only_from` and
`AbstractInversion.reconstruction`).

The memo only ever changes how many active-set iterations the solve takes: the
NNLS optimum is unique, so a memoized reconstruction must equal an un-memoized
one to round-off, including in the edge-zeroed subset branch where the passive
set lives in the subset index space.
"""

import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.inversion import nnls_memo
from autoarray.inversion.inversion.nnls_memo import (
    _NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES,
    _nnls_passive_set_memo,
    memo_key,
    passive_set_get,
    passive_set_put,
)


@pytest.fixture(autouse=True)
def _clean_memo():
    _nnls_passive_set_memo.clear()
    yield
    _nnls_passive_set_memo.clear()


def _normal_equations(seed, n=8, n_data=20):
    """A system whose unconstrained solution has negative components, so the
    passive set is a strict subset and a seed can be wrong about it."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_data, n))
    x = Z @ rng.normal(size=n) + rng.normal(size=n_data)
    return Z.T @ Z, Z.T @ x


class SubsetInversion(aa.m.MockInversion):
    """Pins the edge-zeroed subset branch of `reconstruction` without building a
    real mesh: `solve_ids_to_keep` is the only thing that branch consults."""

    def __init__(self, ids_to_keep, **kwargs):
        super().__init__(**kwargs)
        self._ids_to_keep = ids_to_keep

    @property
    def solve_ids_to_keep(self):
        return self._ids_to_keep


def _inversion_from(curvature_reg_matrix, data_vector, memo, ids_to_keep=None):
    n = data_vector.shape[0]

    kwargs = dict(
        linear_obj_list=[
            aa.m.MockMapper(source_plane_mesh_grid=np.zeros((n, 2)), parameters=n)
        ],
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=aa.Settings(
            use_positive_only_solver=True,
            use_edge_zeroed_pixels=False,
            nnls_warm_start_memo=memo,
        ),
    )

    if ids_to_keep is None:
        return aa.m.MockInversion(**kwargs)

    return SubsetInversion(ids_to_keep=ids_to_keep, **kwargs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test__memoized_reconstruction__matches_unmemoized(seed):
    curvature_reg_matrix, data_vector = _normal_equations(seed)

    expected = _inversion_from(
        curvature_reg_matrix, data_vector, memo=False
    ).reconstruction

    # The first memoized solve populates the memo; the second consumes it.
    for _ in range(2):
        reconstruction = _inversion_from(
            curvature_reg_matrix, data_vector, memo=True
        ).reconstruction

        assert reconstruction == pytest.approx(expected, rel=1e-10, abs=1e-12)

    assert len(_nnls_passive_set_memo) == 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test__memoized_reconstruction__subset_branch__matches_unmemoized(seed):
    curvature_reg_matrix, data_vector = _normal_equations(seed)

    ids_to_keep = np.array([0, 2, 3, 5, 6, 7])

    expected = _inversion_from(
        curvature_reg_matrix, data_vector, memo=False, ids_to_keep=ids_to_keep
    ).reconstruction

    for _ in range(2):
        reconstruction = _inversion_from(
            curvature_reg_matrix, data_vector, memo=True, ids_to_keep=ids_to_keep
        ).reconstruction

        assert reconstruction == pytest.approx(expected, rel=1e-10, abs=1e-12)

    # The subset solve is of size len(ids_to_keep), and its passive set indexes
    # the subset -- not the full parameter vector.
    (key,) = _nnls_passive_set_memo
    assert key.startswith(f"{len(ids_to_keep)}:")
    assert np.all(_nnls_passive_set_memo[key] < len(ids_to_keep))


def test__fingerprint__changes_with_ids_to_keep():
    # The subset passive set indexes the subset, so a different `ids_to_keep`
    # is a different index space and must not reuse the seed.
    curvature_reg_matrix, data_vector = _normal_equations(0)

    fingerprint_a = _inversion_from(
        curvature_reg_matrix, data_vector, memo=True
    )._nnls_warm_start_fingerprint(ids_to_keep=np.array([0, 2, 3]))

    fingerprint_b = _inversion_from(
        curvature_reg_matrix, data_vector, memo=True
    )._nnls_warm_start_fingerprint(ids_to_keep=np.array([0, 2, 4]))

    fingerprint_full = _inversion_from(
        curvature_reg_matrix, data_vector, memo=True
    )._nnls_warm_start_fingerprint()

    assert fingerprint_a != fingerprint_b
    assert fingerprint_a != fingerprint_full


def test__passive_set_put__evicts_the_oldest_entry_when_full():
    for i in range(_NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES + 2):
        passive_set_put(key=f"key_{i}", passive_set=np.array([i]))

    assert len(_nnls_passive_set_memo) == _NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES
    assert "key_0" not in _nnls_passive_set_memo
    assert "key_1" not in _nnls_passive_set_memo
    assert f"key_{_NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES + 1}" in _nnls_passive_set_memo


def test__passive_set_put__stores_a_read_only_copy():
    passive_set = np.array([0, 3, 4])

    passive_set_put(key="key", passive_set=passive_set)

    passive_set[0] = 99

    stored = passive_set_get(key="key", n=5)

    assert np.array_equal(stored, np.array([0, 3, 4]))
    with pytest.raises(ValueError):
        stored[0] = 1


def test__passive_set_get__miss_on_out_of_range_indices_and_unknown_key():
    passive_set_put(key="key", passive_set=np.array([0, 3, 4]))

    assert passive_set_get(key="key", n=5) is not None
    assert passive_set_get(key="key", n=4) is None
    assert passive_set_get(key="other", n=5) is None


def test__memo_enabled__reads_the_environment(monkeypatch):
    monkeypatch.delenv("AUTOARRAY_NNLS_WARM_START", raising=False)
    assert nnls_memo.memo_enabled() is True

    monkeypatch.setenv("AUTOARRAY_NNLS_WARM_START", "0")
    assert nnls_memo.memo_enabled() is False

    monkeypatch.setenv("AUTOARRAY_NNLS_WARM_START", "1")
    assert nnls_memo.memo_enabled() is True


def test__memo_key__separates_solve_sizes():
    assert memo_key(n=3, fingerprint="mesh") != memo_key(n=4, fingerprint="mesh")
