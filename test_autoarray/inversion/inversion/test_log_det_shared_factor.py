"""
``log det(F + lambda*H)`` read off the NNLS Cholesky factor, on the NumPy path.

One positive-only likelihood evaluation used to factorize ``F + lambda*H`` twice.
``fnnls_cholesky`` finishes the non-negative least squares solve holding a Cholesky factor of
``M[P][:, P]``, where ``M`` is ``curvature_reg_matrix_reduced`` and ``P`` is the solve's final
passive set -- 1485 of the 1500 columns on the production HST Delaunay ``pixels=1500``
``AdaptSplit`` system -- and then threw it away; ``log_det_curvature_reg_matrix_term`` did a
fresh full Cholesky of the same matrix, ~40 ms of a ~270 ms NumPy likelihood call.

``fnnls_cholesky`` now publishes that factor through an optional ``factor`` out-dict (the solve
itself is byte-identical -- nothing reads the dict), and the term reads the full log determinant
off it with the block-determinant identity

    log det M = log det M_PP + log det(M_AA - M_AP M_PP^-1 M_PA)

(:func:`inversion_util.log_det_from_passive_cholesky_from`): a triangular solve with ``|A|``
right-hand sides plus an ``|A| x |A|`` Cholesky, where ``A`` is the active set.

These tests pin four things: the **identity** (the Schur route against a dense Cholesky of the
same matrix), the **factor contract** (what ``fnnls_cholesky`` publishes really is a factor of
the passive submatrix, and publishing it does not perturb the solution), the **end-to-end value
and the fact that the fast path is what produced it**, and every **fallback** -- because a
log determinant read off the wrong factor would be a silently wrong Bayesian evidence, not a
crash.

On tolerances
-------------
Unlike the sparse regularization route (see ``test_log_det_sparse.py``, where two different
fp64 factorizations of an ill-conditioned matrix agree only to its conditioning), this is the
*same* factorization arithmetic rearranged, so it is pinned at ``rtol=1e-12`` throughout. The
production system measures <= 1.8e-12 nats from the dense ``potrf`` across six instances and a
bit-identical log evidence on the fiducial one.
"""

import numpy as np
import pytest
import scipy.linalg

import autoarray as aa

from autoarray import fixtures
from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.util.fnnls import fnnls_cholesky


def _dense_log_det(matrix):
    """The pre-lever computation: ``2 * sum(log(diag(cholesky(M))))``."""
    return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))


def _random_spd(n, seed):
    """A well-conditioned symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)

    matrix = rng.normal(size=(n, n))
    matrix = matrix @ matrix.T + n * np.eye(n)

    return 0.5 * (matrix + matrix.T)


def _factor_of_passive_set(matrix, passive_set):
    """
    The buffer `fnnls_cholesky` would have published for this passive set: an ``(n, n)`` array
    whose leading ``k x k`` upper triangle is `cholesky(matrix[P][:, P])` and whose every other
    entry is zero, exactly as the solver's preallocated buffer is.
    """
    n = matrix.shape[0]

    upper = scipy.linalg.cholesky(matrix[np.ix_(passive_set, passive_set)], lower=False)

    U_buffer = np.zeros((n, n))
    U_buffer[: upper.shape[0], : upper.shape[1]] = upper

    return U_buffer


# ===================================================================
# The identity: log_det_from_passive_cholesky_from vs a dense Cholesky
# ===================================================================


@pytest.mark.parametrize("n", [64, 256, 512])
@pytest.mark.parametrize("n_active", [0, 1, 5, 40])
def test__log_det_from_passive_cholesky_from__matches_dense_cholesky(n, n_active):
    """
    The passive set is deliberately **neither sorted nor contiguous**: the active-set solver
    appends and deletes columns, so the factor's row order is the order indices entered the
    passive set and the off-diagonal block ``M_PA`` has to be taken in that same order. Sorting
    ``P`` here would pass while hiding a route that only works for sorted sets.
    """
    matrix = _random_spd(n=n, seed=n + n_active)

    rng = np.random.default_rng(n * 1000 + n_active)

    permutation = rng.permutation(n)

    passive_set = permutation[: n - n_active]

    log_det = inversion_util.log_det_from_passive_cholesky_from(
        matrix=matrix,
        U_buffer=_factor_of_passive_set(matrix=matrix, passive_set=passive_set),
        k_active=passive_set.size,
        passive_set=passive_set,
    )

    assert log_det == pytest.approx(_dense_log_det(matrix), rel=1.0e-12)


def test__log_det_from_passive_cholesky_from__empty_active_set__is_the_passive_factor_alone():
    """
    Every column passive: the published factor is already a factor of the whole matrix, so the
    Schur term is skipped entirely (there is no ``|A| x |A|`` block to form).
    """
    matrix = _random_spd(n=128, seed=11)

    passive_set = np.random.default_rng(11).permutation(128)

    U_buffer = _factor_of_passive_set(matrix=matrix, passive_set=passive_set)

    log_det = inversion_util.log_det_from_passive_cholesky_from(
        matrix=matrix,
        U_buffer=U_buffer,
        k_active=128,
        passive_set=passive_set,
    )

    assert log_det == pytest.approx(
        2.0 * np.sum(np.log(np.diag(U_buffer[:128, :128]))), rel=1.0e-12
    )
    assert log_det == pytest.approx(_dense_log_det(matrix), rel=1.0e-12)


def test__log_det_from_passive_cholesky_from__indefinite_schur_block__raises():
    """
    A matrix which is not positive-definite must raise, exactly as `np.linalg.cholesky` does on
    it, so that the caller's test-mode guard and `FitException` resampling behave as they always
    have. Returning 0.0 or a NaN here would put a fabricated number into the Bayesian evidence.

    The passive block is left positive-definite and the *active* block is broken, so it is the
    Schur complement that fails rather than the published factor.
    """
    matrix = _random_spd(n=64, seed=5)

    passive_set = np.arange(60)

    U_buffer = _factor_of_passive_set(matrix=matrix, passive_set=passive_set)

    matrix[60:, 60:] -= 10.0 * np.eye(4) * np.max(np.abs(matrix))

    with pytest.raises(np.linalg.LinAlgError):
        inversion_util.log_det_from_passive_cholesky_from(
            matrix=matrix,
            U_buffer=U_buffer,
            k_active=passive_set.size,
            passive_set=passive_set,
        )


# ===================================================================
# The factor contract: what fnnls_cholesky publishes
# ===================================================================


def _nnls_problem_with_negatives(n=40, n_data=90, seed=0):
    """
    Normal equations whose unconstrained solution has negative components, so the solve's
    passive set is a strict subset and the published factor is of a genuine submatrix.
    """
    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(n_data, n))
    x = Z @ rng.normal(size=n) + 3.0 * rng.normal(size=n_data)

    return Z.T @ Z, Z.T @ x


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test__fnnls_cholesky__published_factor_is_a_factor_of_the_passive_submatrix(seed):
    """
    The real solver, not a reconstruction of it: `U_buffer[:k, :k]` transposed-squared must
    reproduce `ZTZ[P][:, P]` in the published order, and `k_active` must be the passive count.
    """
    ZTZ, ZTx = _nnls_problem_with_negatives(seed=seed)

    factor = {}

    reconstruction = fnnls_cholesky(ZTZ, ZTx, factor=factor)

    passive_set = factor["passive_set"]
    k_active = factor["k_active"]

    # a strict subset, or the test is not exercising the thing it is about
    assert 0 < k_active < ZTZ.shape[0]
    assert passive_set.shape == (k_active,)
    assert factor["matrix_shape"] == ZTZ.shape

    # the solution's support is exactly the passive set
    assert set(np.flatnonzero(reconstruction).tolist()) == set(passive_set.tolist())

    upper = factor["U_buffer"][:k_active, :k_active]

    assert upper.T @ upper == pytest.approx(
        ZTZ[np.ix_(passive_set, passive_set)], abs=1.0e-10
    )

    # the buffer outside the active corner is zero and meaningless, as documented
    assert np.count_nonzero(np.tril(upper, k=-1)) == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test__fnnls_cholesky__publishing_the_factor_does_not_change_the_solution(seed):
    """
    Bit-identical, not approximately equal: the `factor` out-dict adds a publish after the
    solve and changes no arithmetic in it, and that is the property the whole lever rests on.
    """
    ZTZ, ZTx = _nnls_problem_with_negatives(seed=seed)

    assert np.array_equal(
        fnnls_cholesky(ZTZ, ZTx),
        fnnls_cholesky(ZTZ, ZTx, factor={}),
    )


def test__reconstruction_positive_only_from__clears_a_stale_factor_on_entry():
    """
    A dict handed in carrying another matrix's factor must not survive the call, or a caller
    could read a factor belonging to a system it is not asking about.
    """
    ZTZ, ZTx = _nnls_problem_with_negatives(seed=0)

    factor = {
        "U_buffer": np.eye(3),
        "k_active": 3,
        "passive_set": np.arange(3),
        "matrix_shape": (3, 3),
        "stale": True,
    }

    inversion_util.reconstruction_positive_only_from(
        data_vector=ZTx,
        curvature_reg_matrix=ZTZ,
        settings=aa.Settings(use_positive_only_solver=True),
        fingerprint=None,
        factor=factor,
    )

    assert "stale" not in factor
    assert factor["matrix_shape"] == ZTZ.shape


# ===================================================================
# End to end on a real NumPy-path inversion
# ===================================================================


def _split_mapper(regularization, zeroed_pixels=0):
    """
    The `delaunay_mapper_9_3x3` fixture's mesh, carrying a real split regularization scheme (the
    production family) rather than `Constant`.
    """
    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[
            [0.6, -0.3],
            [0.5, -0.8],
            [0.2, 0.1],
            [0.0, 0.5],
            [-0.3, -0.8],
            [-0.6, -0.5],
            [-0.4, -1.1],
            [-1.2, 0.8],
            [-1.5, 0.9],
        ],
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    interpolator = aa.mesh.Delaunay(
        pixels=9, zeroed_pixels=zeroed_pixels
    ).interpolator_from(
        source_plane_data_grid=fixtures.make_grid_2d_sub_2_7x7(),
        source_plane_mesh_grid=source_plane_mesh_grid,
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=regularization,
        image_plane_mesh_grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1),
    )


def _dataset(seed):
    """
    The `masked_imaging_7x7` fixture with seeded random data in place of its flat ones.

    The flat fixture's reconstruction is entirely positive, so its solve leaves an *empty*
    active set and the Schur term is never formed. Signed data drives some source pixels to
    zero, which is what puts columns in the active set -- the regime the fast path is for.
    """
    dataset = fixtures.make_masked_imaging_7x7()

    data = aa.Array2D(
        values=np.random.default_rng(seed).normal(0.0, 1.0, size=9),
        mask=dataset.mask,
    )

    return DatasetInterface(
        data=data,
        noise_map=dataset.noise_map,
        grids=dataset.grids,
        psf=dataset.psf,
    )


def _counting_schur(monkeypatch):
    """Count the fast path's calls without changing what it returns."""
    original = inversion_util.log_det_from_passive_cholesky_from

    calls = []

    def _wrapped(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(inversion_util, "log_det_from_passive_cholesky_from", _wrapped)

    return calls


@pytest.mark.parametrize(
    "regularization",
    [
        aa.reg.ConstantSplit(coefficient=1.0),
        aa.reg.AdaptSplit(
            inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1
        ),
    ],
)
@pytest.mark.parametrize("seed", [1, 2, 5])
def test__log_det_curvature_reg_matrix_term__real_inversion__fast_path_matches_dense(
    monkeypatch, regularization, seed
):
    """
    The value, and the fact that the fast path is what produced it. Asserting only the value
    would pass with the fast path silently never firing, which is exactly how this lever would
    fail: it would be correct and do nothing.
    """
    calls = _counting_schur(monkeypatch)

    inversion = aa.Inversion(
        dataset=_dataset(seed),
        linear_obj_list=[_split_mapper(regularization)],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    matrix = np.asarray(inversion.curvature_reg_matrix_reduced)

    term = inversion.log_det_curvature_reg_matrix_term

    assert len(calls) == 1
    assert term == pytest.approx(_dense_log_det(matrix), rel=1.0e-12)

    # the solve really did leave columns in the active set, so the Schur block was formed
    k_active = inversion._nnls_factor["k_active"]

    assert 0 < k_active < matrix.shape[0]
    assert np.count_nonzero(np.asarray(inversion.reconstruction)) == k_active


def test__log_det_curvature_reg_matrix_term__edge_zeroing_with_no_edge_pixels__fast_path_fires(
    monkeypatch,
):
    """
    The index-coverage trap. With `use_edge_zeroed_pixels` on and no edge pixels to zero,
    `solve_ids_to_keep` returns `arange(n)` rather than `None` -- which is the configuration of
    the production HST Delaunay system this lever was measured on. A coverage guard written as
    "the solve took no subset" (`ids is None`) would decline every production evaluation and
    save nothing, while still passing a value-only assertion.
    """
    calls = _counting_schur(monkeypatch)

    inversion = aa.Inversion(
        dataset=_dataset(2),
        linear_obj_list=[_split_mapper(aa.reg.ConstantSplit(coefficient=1.0))],
        settings=aa.Settings(
            use_positive_only_solver=True, use_edge_zeroed_pixels=True
        ),
    )

    assert np.array_equal(inversion.solve_ids_to_keep, np.arange(9))

    matrix = np.asarray(inversion.curvature_reg_matrix_reduced)

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        _dense_log_det(matrix), rel=1.0e-12
    )
    assert len(calls) == 1


def test__curvature_reg_matrix__is_cached__same_object_on_repeat_access():
    """
    Reached more than once per likelihood evaluation (the solve and the reduced matrix), and
    each access re-ran an out-of-place `(n, n)` add -- ~2 ms per access at n = 1500. It was a
    `cached_property` until the blanket `e819fa12` sweep.
    """
    inversion = aa.Inversion(
        dataset=_dataset(1),
        linear_obj_list=[_split_mapper(aa.reg.ConstantSplit(coefficient=1.0))],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    matrix = inversion.curvature_reg_matrix

    assert inversion.curvature_reg_matrix is matrix

    # and it is the out-of-place sum, leaving the cached `curvature_matrix` intact
    assert np.asarray(matrix) == pytest.approx(
        np.asarray(inversion.curvature_matrix)
        + np.asarray(inversion.regularization_matrix),
        rel=1.0e-12,
    )


# ===================================================================
# Fallbacks: everything that must take the unchanged dense route
# ===================================================================


def test__log_det_curvature_reg_matrix_term__positive_negative_solver__dense_route(
    monkeypatch,
):
    """
    Route (c): no positive-only solve, so no factor exists. The term must not reach the fast
    path -- and must not evaluate the reconstruction it would not otherwise need.
    """
    calls = _counting_schur(monkeypatch)

    inversion = aa.Inversion(
        dataset=_dataset(2),
        linear_obj_list=[_split_mapper(aa.reg.ConstantSplit(coefficient=1.0))],
        settings=aa.Settings(use_positive_only_solver=False),
    )

    matrix = np.asarray(inversion.curvature_reg_matrix_reduced)

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        _dense_log_det(matrix), rel=1.0e-12
    )
    assert calls == []
    assert inversion._nnls_factor is None


def test__log_det_curvature_reg_matrix_term__slogdet_method__unchanged(monkeypatch):
    """The ``"slogdet"`` opt-in path is a different value by construction and is untouched."""
    calls = _counting_schur(monkeypatch)

    inversion = aa.Inversion(
        dataset=_dataset(2),
        linear_obj_list=[_split_mapper(aa.reg.ConstantSplit(coefficient=1.0))],
        settings=aa.Settings(use_positive_only_solver=True, log_det_method="slogdet"),
    )

    matrix = np.asarray(inversion.curvature_reg_matrix_reduced)

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        np.linalg.slogdet(matrix)[1], rel=1.0e-12
    )
    assert calls == []


def test__log_det_curvature_reg_matrix_term__fast_path_raises__falls_back_to_dense(
    monkeypatch,
):
    """
    A `LinAlgError` out of the Schur route means this matrix is at the edge of what fp64
    resolves as positive-definite. The dense factorization below is then the authority on it
    (it either returns its own value or raises with the test-mode guard in place); the fast path
    must never answer with a fabricated 0.0.
    """

    def raise_lin_alg_error(**kwargs):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(
        inversion_util, "log_det_from_passive_cholesky_from", raise_lin_alg_error
    )

    inversion = aa.Inversion(
        dataset=_dataset(2),
        linear_obj_list=[_split_mapper(aa.reg.ConstantSplit(coefficient=1.0))],
        settings=aa.Settings(use_positive_only_solver=True),
    )

    matrix = np.asarray(inversion.curvature_reg_matrix_reduced)

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        _dense_log_det(matrix), rel=1.0e-12
    )


def test__factor_published_is_the_fallback_solve_s__when_a_memo_seed_raises():
    """
    A memo-seeded attempt that raises is retried from the dense-sign start
    (`warm_start_fallback`), and it is the *retry's* factor that must be published: the failed
    attempt's factor belongs to a solve whose result was discarded.

    The seeded attempt is forced to raise by patching the solver for the first call only, which
    is how `reconstruction_positive_only_from`'s own retry is reached.
    """
    from autoarray.inversion.inversion import nnls_memo
    import autoarray.util.fnnls as fnnls_mod

    nnls_memo._nnls_passive_set_memo.clear()

    ZTZ, ZTx = _nnls_problem_with_negatives(seed=1)

    key = nnls_memo.memo_key(n=ZTx.shape[0], fingerprint="mesh")

    nnls_memo.passive_set_put(
        key=key, passive_set=np.arange(ZTx.shape[0]), dense_error_fraction=0.5
    )

    original = fnnls_mod.fnnls_cholesky
    calls = []

    def _raise_first(
        ZTZ, ZTx, P_initial=np.zeros(0, dtype=int), stats=None, factor=None
    ):
        calls.append(P_initial)

        if len(calls) == 1:
            raise np.linalg.LinAlgError("forced failure of the seeded attempt")

        return original(ZTZ, ZTx, P_initial, stats=stats, factor=factor)

    factor = {}

    try:
        fnnls_mod.fnnls_cholesky = _raise_first

        reconstruction = inversion_util.reconstruction_positive_only_from(
            data_vector=ZTx,
            curvature_reg_matrix=ZTZ,
            settings=aa.Settings(
                use_positive_only_solver=True, nnls_warm_start_memo=True
            ),
            fingerprint="mesh",
            factor=factor,
        )
    finally:
        fnnls_mod.fnnls_cholesky = original
        nnls_memo._nnls_passive_set_memo.clear()

    # the seeded attempt raised and the dense-seed retry ran
    assert len(calls) == 2

    # the published factor is the retry's, and it describes the solution that was returned
    assert factor["k_active"] == np.count_nonzero(reconstruction)
    assert set(factor["passive_set"].tolist()) == set(
        np.flatnonzero(reconstruction).tolist()
    )

    upper = factor["U_buffer"][: factor["k_active"], : factor["k_active"]]

    assert upper.T @ upper == pytest.approx(
        ZTZ[np.ix_(factor["passive_set"], factor["passive_set"])], abs=1.0e-10
    )
