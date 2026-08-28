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
