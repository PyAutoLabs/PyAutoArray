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
    _NNLS_BACKOFF_AFTER_FALLBACKS,
    _NNLS_BACKOFF_MAX_SKIP,
    _NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES,
    _nnls_backoff,
    _nnls_passive_set_memo,
    backoff_end_skip,
    backoff_record_accept,
    backoff_record_fallback,
    backoff_should_skip,
    memo_drop,
    memo_clear,
    memo_key,
    passive_set_get,
    passive_set_put,
    seed_error_fraction,
)


@pytest.fixture(autouse=True)
def _clean_memo():
    _nnls_passive_set_memo.clear()
    _nnls_backoff.clear()
    yield
    _nnls_passive_set_memo.clear()
    _nnls_backoff.clear()


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


def _inversion_from(
    curvature_reg_matrix, data_vector, memo, ids_to_keep=None, tolerance=None
):
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
            nnls_warm_start_error_tolerance=tolerance,
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
    assert np.all(_nnls_passive_set_memo[key].passive_set < len(ids_to_keep))


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
        passive_set_put(
            key=f"key_{i}", passive_set=np.array([i]), dense_error_fraction=0.1
        )

    assert len(_nnls_passive_set_memo) == _NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES
    assert "key_0" not in _nnls_passive_set_memo
    assert "key_1" not in _nnls_passive_set_memo
    assert f"key_{_NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES + 1}" in _nnls_passive_set_memo


def test__passive_set_put__stores_a_read_only_copy_and_the_reference_fraction():
    passive_set = np.array([0, 3, 4])

    passive_set_put(key="key", passive_set=passive_set, dense_error_fraction=0.25)

    passive_set[0] = 99

    entry = passive_set_get(key="key", n=5)

    assert np.array_equal(entry.passive_set, np.array([0, 3, 4]))
    assert entry.dense_error_fraction == 0.25
    with pytest.raises(ValueError):
        entry.passive_set[0] = 1


def test__passive_set_get__miss_on_out_of_range_indices_and_unknown_key():
    passive_set_put(
        key="key", passive_set=np.array([0, 3, 4]), dense_error_fraction=0.0
    )

    assert passive_set_get(key="key", n=5) is not None
    assert passive_set_get(key="key", n=4) is None
    assert passive_set_get(key="other", n=5) is None


def test__memo_drop__forgets_the_key_and_is_a_no_op_when_absent():
    passive_set_put(key="key", passive_set=np.array([0, 2]), dense_error_fraction=0.1)

    memo_drop(key="key")

    assert passive_set_get(key="key", n=5) is None

    memo_drop(key="key")


def test__memo_enabled__reads_the_environment(monkeypatch):
    monkeypatch.delenv("AUTOARRAY_NNLS_WARM_START", raising=False)
    assert nnls_memo.memo_enabled() is True

    monkeypatch.setenv("AUTOARRAY_NNLS_WARM_START", "0")
    assert nnls_memo.memo_enabled() is False

    monkeypatch.setenv("AUTOARRAY_NNLS_WARM_START", "1")
    assert nnls_memo.memo_enabled() is True


def test__memo_key__separates_solve_sizes():
    assert memo_key(n=3, fingerprint="mesh") != memo_key(n=4, fingerprint="mesh")



# ===================================================================
# Per-key back-off on scattered evaluation streams (PyAutoArray#613)
# ===================================================================


def _skips_until_probe(key):
    """How many consecutive would-be-seeded solves the back-off skips for `key`."""
    skipped = 0

    while backoff_should_skip(key):
        skipped += 1

    return skipped


def test__backoff__engages_after_consecutive_fallbacks_and_grows_to_the_cap():
    assert _NNLS_BACKOFF_AFTER_FALLBACKS == 2

    backoff_record_fallback("key")
    assert _skips_until_probe("key") == 0

    skips = []
    for _ in range(9):
        backoff_record_fallback("key")
        skips.append(_skips_until_probe("key"))

    assert skips == [1, 2, 4, 8, 16, 32, 32, 32, 32]
    assert max(skips) == _NNLS_BACKOFF_MAX_SKIP


def test__backoff__accept_resets_the_streak_and_end_skip_keeps_it():
    for _ in range(3):
        backoff_record_fallback("key")

    backoff_end_skip("key")
    assert _skips_until_probe("key") == 0

    # end_skip kept the streak: the next fallback escalates rather than restarting.
    backoff_record_fallback("key")
    assert _skips_until_probe("key") == 4

    backoff_record_accept("key")
    assert "key" not in _nnls_backoff

    backoff_record_fallback("key")
    assert _skips_until_probe("key") == 0

    # Unknown keys are no-ops.
    backoff_end_skip("other")
    backoff_record_accept("other")
    assert backoff_should_skip("other") is False


def test__backoff__table_is_bounded():
    for i in range(_NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES + 3):
        backoff_record_fallback(f"key{i}")

    assert len(_nnls_backoff) == _NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES


def test__memo_clear__forgets_entries_and_backoff():
    passive_set_put(key="key", passive_set=np.array([0]), dense_error_fraction=0.1)
    for _ in range(3):
        backoff_record_fallback("key")

    memo_clear()

    assert _nnls_passive_set_memo == {}
    assert _nnls_backoff == {}


def test__seed_error_fraction__counts_the_symmetric_difference():
    assert seed_error_fraction(np.array([0, 2]), np.array([2, 0]), n=5) == 0.0
    assert seed_error_fraction(np.array([0, 1]), np.array([0, 2, 3]), n=5) == 0.6
    assert seed_error_fraction(np.array([], dtype=int), np.array([4]), n=5) == 0.2


_N_STREAM = 40


def _stream_system(Z, coeffs):
    """Normal equations whose NNLS passive set follows the signs of `coeffs`."""
    x = Z @ coeffs
    return Z.T @ Z + 0.1 * np.eye(Z.shape[1]), Z.T @ x


def _scattered_stream(seed, length=40):
    """iid draws: every solve is an unrelated system, so no seed is any good."""
    rng = np.random.default_rng(seed)
    return [
        _stream_system(rng.normal(size=(90, _N_STREAM)), rng.normal(size=_N_STREAM))
        for _ in range(length)
    ]


def _walk_stream(seed, length=40, step=1e-3):
    """A local walk: each system is a small perturbation of the previous one."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(90, _N_STREAM))
    coeffs = rng.normal(size=_N_STREAM)
    systems = []
    for _ in range(length):
        systems.append(_stream_system(Z, coeffs))
        Z = Z + step * rng.normal(size=Z.shape)
        coeffs = coeffs + step * rng.normal(size=coeffs.shape)
    return systems


def _replay(monkeypatch, systems, memo=True):
    """
    Solve `systems` in order through `reconstruction_positive_only_from` with one
    memo key, returning the reconstructions and each solve's stats dict.
    """
    import autoarray.util.fnnls as fnnls_mod

    original = fnnls_mod.fnnls_cholesky
    captured = []

    def _wrapped(ZTZ, ZTx, P_initial=np.zeros(0, dtype=int), stats=None, factor=None):
        captured.append(stats)
        return original(ZTZ, ZTx, P_initial, stats=stats, factor=factor)

    monkeypatch.setattr(fnnls_mod, "fnnls_cholesky", _wrapped)

    settings = aa.Settings(
        use_positive_only_solver=True,
        nnls_warm_start_memo=memo,
        nnls_warm_start_error_tolerance=1.5,
    )

    reconstructions = [
        aa.util.inversion.reconstruction_positive_only_from(
            data_vector=q,
            curvature_reg_matrix=Q,
            settings=settings,
            fingerprint="stream",
        )
        for Q, q in systems
    ]

    monkeypatch.setattr(fnnls_mod, "fnnls_cholesky", original)

    # One stats dict per solve: the seeded attempt and the solve that returned
    # share the dict (a raising seeded attempt never reaches the end).
    stats = list({id(d): d for d in captured}.values())

    return reconstructions, stats


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test__backoff__scattered_stream_stops_reseeding_and_keeps_the_answer(
    monkeypatch, seed
):
    systems = _scattered_stream(seed)

    expected, _ = _replay(monkeypatch, systems, memo=False)

    def bad_seeds(stats):
        return sum(s["warm_start_fallback"] for s in stats)

    # Reference: the same replay with the back-off switched off. Every other
    # solve is seeded and (bar a chance hit) every seed falls back.
    with monkeypatch.context() as m:
        m.setattr(nnls_memo, "_NNLS_BACKOFF_AFTER_FALLBACKS", 10**9)
        _, stats_without = _replay(monkeypatch, systems)

    _nnls_passive_set_memo.clear()
    _nnls_backoff.clear()

    reconstructions, stats = _replay(monkeypatch, systems)

    for reconstruction, reference in zip(reconstructions, expected):
        assert reconstruction == pytest.approx(reference, rel=1e-9, abs=1e-11)

    assert bad_seeds(stats_without) >= 18
    assert not any(s["warm_start_backoff"] for s in stats_without)

    # With the back-off the probes thin out exponentially (solves 1, 3, 6, 10,
    # 16, 26 on a stream with no chance hits): far fewer bad seeds are paid for.
    assert bad_seeds(stats) <= 0.6 * bad_seeds(stats_without)
    assert sum(s["warm_start_backoff"] for s in stats) >= 10


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test__backoff__local_walk_is_untouched_bit_for_bit(monkeypatch, seed):
    systems = _walk_stream(seed)

    reconstructions, stats = _replay(monkeypatch, systems)

    # The walk is what the memo is for: every solve after the first is seeded,
    # no seed falls back, and the back-off never engages.
    assert [s["seed_source"] for s in stats] == ["dense"] + ["memo"] * (
        len(systems) - 1
    )
    assert not any(s["warm_start_fallback"] for s in stats)
    assert not any(s["warm_start_backoff"] for s in stats)
    assert _nnls_backoff == {}

    # Bit-identical to the same replay with the back-off switched off entirely.
    _nnls_passive_set_memo.clear()
    monkeypatch.setattr(nnls_memo, "_NNLS_BACKOFF_AFTER_FALLBACKS", 10**9)

    reference, _ = _replay(monkeypatch, systems)

    for reconstruction, expected in zip(reconstructions, reference):
        assert np.array_equal(reconstruction, expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test__backoff__scattered_then_local_regains_the_memo_after_one_solve(
    monkeypatch, seed
):
    scattered = _scattered_stream(seed, length=30)
    systems = scattered + _walk_stream(seed + 100, length=20)

    # The walk starts from an unrelated system, so its first solve is dense
    # (backed off or fallen back); the free check on that dense solve sees the
    # skipped seed would have passed, and from the second walk solve on every
    # solve is seeded again -- the long skip scheduled by the scattered phase is
    # not waited out.
    reconstructions, stats = _replay(monkeypatch, systems)

    walk_stats = stats[len(scattered) :]

    assert all(s["seed_source"] == "memo" for s in walk_stats[2:])
    assert not any(s["warm_start_fallback"] for s in walk_stats[2:])
