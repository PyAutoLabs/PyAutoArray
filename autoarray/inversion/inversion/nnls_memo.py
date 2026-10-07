import os

import numpy as np
from typing import Dict, NamedTuple, Optional

# Cross-evaluation memo for the positive-only (fnnls) solve's passive set.
#
# The Bro & De Jong active set iteration is warm-started from a guess at which
# reconstruction entries are non-zero. The production guess -- the sign of the
# unconstrained dense solve -- gets ~150 of ~1560 entries wrong on a Delaunay
# euclid fit, and each wrong entry costs an active-set iteration in a solve
# that is ~70% of the whole numba CPU likelihood evaluation. Successive
# sampler evaluations sit close together in parameter space and share nearly
# all of their passive set, so seeding from the previous evaluation's FINAL
# passive set is a much better guess.
#
# Entries are keyed by the index space the passive set lives in (mesh/data
# shapes, plus the edge-zeroed subset when one is in use) -- never by the
# matrix values, since the whole point is to hit across nearby parameter
# points. A wrong seed cannot corrupt the answer: the NNLS optimum is unique
# and the solver adds and removes entries until the KKT conditions hold, so a
# stale entry costs iterations, not correctness. Forked pool workers inherit a
# copy at fork and diverge from there, which is fine for the same reason.
#
# An entry therefore carries TWO things: the passive set to seed from, and
# `dense_error_fraction` -- the error fraction of the most recent solve for
# that key that started from the DENSE-SIGN guess. That number is the
# reference the fallback guard in `reconstruction_positive_only_from` measures
# a seed against: the PyAutoArray#498 robustness matrix showed the absolute
# error fraction of a seed does NOT separate seeds that save iterations from
# seeds that cost them (helpful and unhelpful cells overlap at 0.048-0.138),
# but the ratio of the seed's fraction to the dense-sign start's does (helpful
# cells never exceed 0.89, the worst seed reaches 1.42). The reference is
# per-key and self-calibrating, so a solve regime far outside anything the
# matrix probed cannot drag a stale seed through a whole run: once a seed is
# that much worse than the dense-sign start, the entry is dropped
# (`memo_drop`) and the next solve for that key restarts dense, refreshing the
# reference.
#
# That guard judges a seed only AFTER the seeded solve has run. On a
# scattered evaluation stream (iid draws, e.g. a nested sampler's early live
# points) every seed is bad, so the stream alternates dense solve -> bad
# seeded solve -> drop -> dense solve ...: half the solves pay for a bad seed.
# autolens_profiling#332 measured the alma interferometer Delaunay solve 2.17x
# slower memo-on than memo-off on such a stream. The per-key back-off below
# stops that: after `_NNLS_BACKOFF_AFTER_FALLBACKS` seeded solves for a key
# have fallen back in a row, the next solve(s) for it skip the seed and start
# dense (still refreshing the entry), for 1, 2, 4, ... up to
# `_NNLS_BACKOFF_MAX_SKIP` solves, then the seed is probed again. One accepted
# seed resets the streak, and a stream that never falls back twice in a row (a
# local walk) never meets the back-off at all.
#
# A backed-off solve also judges, for free, the seed it skipped: the dense
# solve's final passive set is the unique optimum's, so the seed's error count
# on this system is just the size of its disagreement with that set
# (`seed_error_fraction`) -- exactly what `warm_start_errors` would have
# reported had the seed been used. If the skipped seed would have passed the
# fallback guard against this dense solve, the remaining skips are cancelled
# and the next solve is seeded again: a stream that turns local regains the
# memo after one solve, without waiting out the schedule. The streak is kept
# (only an accepted REAL seeded solve resets it), so a shadow check that passes
# by chance on a scattered stream costs one probe and lengthens the next skip.
#
# Disable with AUTOARRAY_NNLS_WARM_START=0.


class MemoEntry(NamedTuple):
    """
    One memoized solve: the passive set to seed the next solve for this key
    from, and the error fraction of the most recent dense-sign-started solve
    for the same key (the reference the fallback guard compares a seed to).
    """

    passive_set: np.ndarray
    dense_error_fraction: float


_nnls_passive_set_memo: Dict[str, MemoEntry] = {}

_NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES = 8

# Consecutive seeded-solve fallbacks for one key before the back-off engages.
# 2, not 1: a single fallback already costs only one bad solve (the next solve
# restarts dense anyway), and a local walk that crosses one sharp change must
# keep the memo on the very next probe.
_NNLS_BACKOFF_AFTER_FALLBACKS = 2

# Cap on the number of solves skipped between probes of a backed-off key. On a
# stream where every seed is bad, one probe in (cap + 1) solves still pays for
# a bad seed; on a stream that turns local the memo is back within cap solves.
_NNLS_BACKOFF_MAX_SKIP = 32


class BackoffState(NamedTuple):
    """
    Per-key back-off bookkeeping: how many seeded solves in a row fell back
    (`fallback_streak`) and how many upcoming solves still skip the seed
    (`skip_remaining`).
    """

    fallback_streak: int
    skip_remaining: int


_nnls_backoff: Dict[str, BackoffState] = {}


def memo_enabled() -> bool:
    """
    Whether the passive-set memo is active in this process.
    """
    return os.environ.get("AUTOARRAY_NNLS_WARM_START", "1") != "0"


def memo_key(n: int, fingerprint) -> str:
    """
    The memo key for a solve of size `n` whose index space is described by
    `fingerprint` (see `AbstractInversion._nnls_warm_start_fingerprint`).
    """
    return f"{n}:{fingerprint}"


def passive_set_get(key: str, n: int) -> Optional[MemoEntry]:
    """
    The memoized entry for `key` -- its passive set and dense-sign reference
    error fraction -- or None on a miss.

    An entry whose indices do not all fit a size-`n` solve is a miss, not a
    hit: `n` is already part of the key, so this only fires if a caller
    fingerprints two different index spaces identically, and a miss is always
    a safe outcome.
    """
    entry = _nnls_passive_set_memo.get(key)

    if entry is None:
        return None

    if entry.passive_set.size and entry.passive_set.max() >= n:
        return None

    return entry


def passive_set_put(
    key: str, passive_set: np.ndarray, dense_error_fraction: float
) -> None:
    """
    Store a solve's final passive set alongside the dense-sign reference error
    fraction to carry forward, evicting the oldest entry once the memo is full
    (FIFO; the memo tracks one inversion's recent history, not a working set
    worth ranking).
    """
    stored = np.asarray(passive_set, dtype=int).copy()
    stored.setflags(write=False)

    if (
        key not in _nnls_passive_set_memo
        and len(_nnls_passive_set_memo) >= _NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES
    ):
        _nnls_passive_set_memo.pop(next(iter(_nnls_passive_set_memo)))

    _nnls_passive_set_memo[key] = MemoEntry(
        passive_set=stored, dense_error_fraction=float(dense_error_fraction)
    )


def memo_drop(key: str) -> None:
    """
    Forget `key`, so the next solve for it restarts from the dense-sign guess
    and refreshes the reference error fraction. A no-op if the key is absent.
    """
    _nnls_passive_set_memo.pop(key, None)


def memo_clear() -> None:
    """
    Forget every memo entry AND every back-off state, returning the process to
    a cold start. Harnesses that compare memo arms must use this rather than
    clearing `_nnls_passive_set_memo` alone: back-off state left by one arm
    (e.g. an iid stream) would otherwise carry into the next.
    """
    _nnls_passive_set_memo.clear()
    _nnls_backoff.clear()


def backoff_should_skip(key: str) -> bool:
    """
    Whether the solve for `key` about to run should skip the memo seed and
    start dense, consuming one skip if so. False for a key with no back-off.
    """
    state = _nnls_backoff.get(key)

    if state is None or state.skip_remaining <= 0:
        return False

    _nnls_backoff[key] = state._replace(skip_remaining=state.skip_remaining - 1)

    return True


def backoff_record_fallback(key: str) -> None:
    """
    Record that a seeded solve for `key` breached the fallback guard. Once
    `_NNLS_BACKOFF_AFTER_FALLBACKS` have happened in a row, schedule the next
    1, 2, 4, ... (capped at `_NNLS_BACKOFF_MAX_SKIP`) solves to skip the seed.
    """
    state = _nnls_backoff.get(key, BackoffState(0, 0))

    streak = state.fallback_streak + 1

    excess = streak - _NNLS_BACKOFF_AFTER_FALLBACKS

    skip = 0 if excess < 0 else min(2**excess, _NNLS_BACKOFF_MAX_SKIP)

    if (
        key not in _nnls_backoff
        and len(_nnls_backoff) >= _NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES
    ):
        _nnls_backoff.pop(next(iter(_nnls_backoff)))

    _nnls_backoff[key] = BackoffState(fallback_streak=streak, skip_remaining=skip)


def backoff_end_skip(key: str) -> None:
    """
    Cancel the remaining skips for `key` -- the next solve is seeded again --
    while keeping its fallback streak, so a failed probe resumes the back-off
    at the escalated length. A no-op if the key has no back-off.
    """
    state = _nnls_backoff.get(key)

    if state is not None:
        _nnls_backoff[key] = state._replace(skip_remaining=0)


def backoff_record_accept(key: str) -> None:
    """
    Record that a seeded solve for `key` passed the fallback guard, resetting
    its back-off entirely. A no-op if the key has no back-off.
    """
    _nnls_backoff.pop(key, None)


def seed_error_fraction(seed_passive_set: np.ndarray, passive_set, n: int) -> float:
    """
    The fraction of a size-`n` solve's entries that the warm-start passive set
    `seed_passive_set` gets wrong relative to the solve's final `passive_set`
    -- the quantity `fnnls_cholesky` reports as `warm_start_errors / n` when it
    is seeded from `seed_passive_set`. Used to judge a seed the back-off skipped
    against the dense solve that ran instead.
    """
    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[np.asarray(seed_passive_set, dtype=int)] = True

    final_mask = np.zeros(n, dtype=bool)
    final_mask[np.asarray(passive_set, dtype=int)] = True

    return np.count_nonzero(seed_mask != final_mask) / max(n, 1)
