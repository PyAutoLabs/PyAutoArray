"""
One Cholesky factor of ``F + lambda*H`` shared by the positive-negative solve and its log determinant.

On the NumPy backend with ``use_positive_only_solver=False`` a likelihood call factorized the same
symmetric positive-definite matrix twice: ``np.linalg.solve`` (an LU) for ``reconstruction`` and
``np.linalg.cholesky`` for ``log_det_curvature_reg_matrix_term`` (PyAutoArray#633). Both now read
``AbstractInversion._curvature_reg_matrix_cho_factor``.

These tests pin, against the two-factorization values computed here independently:

- the reconstruction (rtol 1e-10) and the log determinant (rtol 1e-9), on imaging and
  interferometer inversions, and that one factorization serves both;
- the failure semantics: an indefinite-but-nonsingular ``F + H`` is still solved by the LU and still
  makes the log determinant raise (or return 0.0 in test mode); a singular one still makes the solve
  raise (or return its test-mode dummy);
- the routes that must not use the factor: ``"slogdet"``, the positive-only solver, and a partially
  regularized system's log determinant.
"""

import numpy as np
import pytest
import scipy.linalg

import autoarray as aa

from autoarray import fixtures


def _two_factorization_values(inversion):
    """The pre-#633 computation: an LU solve plus a separate dense Cholesky."""
    matrix = np.asarray(inversion.curvature_reg_matrix)

    reconstruction = np.linalg.solve(matrix, np.asarray(inversion.data_vector))

    reduced = np.asarray(inversion.curvature_reg_matrix_reduced)

    log_det = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(reduced))))

    return reconstruction, log_det


def _count_cho_factor(monkeypatch):
    original = scipy.linalg.cho_factor

    calls = []

    def _wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "cho_factor", _wrapped)

    return calls


def _imaging_inversion(settings, linear_obj_list=None):
    return aa.Inversion(
        dataset=fixtures.make_masked_imaging_7x7(),
        linear_obj_list=linear_obj_list or [fixtures.make_rectangular_mapper_7x7_3x3()],
        settings=settings,
    )


def _interferometer_inversion(settings):
    return aa.Inversion(
        dataset=fixtures.make_interferometer_7(),
        linear_obj_list=[fixtures.make_rectangular_mapper_7x7_3x3()],
        settings=settings,
    )


def _with_matrix(inversion, matrix, data_vector):
    """Seed the cached `curvature_reg_matrix` / `data_vector` with a chosen system."""
    inversion.__dict__["curvature_reg_matrix"] = matrix
    inversion.__dict__["data_vector"] = data_vector
    return inversion


@pytest.mark.parametrize(
    "inversion_from",
    [_imaging_inversion, _interferometer_inversion],
    ids=["imaging", "interferometer"],
)
def test__shared_factor__matches_two_factorizations__one_factorization(
    monkeypatch, inversion_from
):
    calls = _count_cho_factor(monkeypatch)

    inversion = inversion_from(aa.Settings(use_positive_only_solver=False))

    reconstruction_ref, log_det_ref = _two_factorization_values(inversion)

    # log det first: it must read the factor without evaluating the reconstruction.
    log_det = inversion.log_det_curvature_reg_matrix_term

    assert "reconstruction" not in inversion.__dict__

    reconstruction = np.asarray(inversion.reconstruction)

    assert len(calls) == 1
    assert inversion._curvature_reg_matrix_cho_factor is not None

    assert reconstruction == pytest.approx(reconstruction_ref, rel=1.0e-10, abs=0.0)
    assert log_det == pytest.approx(log_det_ref, rel=1.0e-9)

    # the factor is a copy: `F + H` and `F` are left intact for the reduced matrix / covariance
    assert np.asarray(inversion.curvature_reg_matrix) == pytest.approx(
        np.asarray(inversion.curvature_matrix)
        + np.asarray(inversion.regularization_matrix),
        rel=1.0e-12,
    )


def test__shared_factor__random_spd_system__matches_solve_and_dense_cholesky():
    rng = np.random.default_rng(3)

    n = 9  # the rectangular 3x3 mapper's parameter count

    a = rng.normal(size=(n, n))
    matrix = a @ a.T + 1.0e-3 * np.eye(n)
    data_vector = rng.normal(size=n)

    inversion = _with_matrix(
        _imaging_inversion(aa.Settings(use_positive_only_solver=False)),
        matrix=matrix,
        data_vector=data_vector,
    )

    assert np.asarray(inversion.reconstruction) == pytest.approx(
        np.linalg.solve(matrix, data_vector), rel=1.0e-10, abs=0.0
    )
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix)))), rel=1.0e-9
    )


def _indefinite_nonsingular(n=9):
    matrix = np.eye(n)
    matrix[0, 0] = -1.0
    return matrix


def test__indefinite_matrix__lu_solve_kept__log_det_raises(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    matrix = _indefinite_nonsingular()
    data_vector = np.arange(1.0, 10.0)

    inversion = _with_matrix(
        _imaging_inversion(aa.Settings(use_positive_only_solver=False)),
        matrix=matrix,
        data_vector=data_vector,
    )

    assert inversion._curvature_reg_matrix_cho_factor is None

    # as before: the LU solves an indefinite nonsingular system ...
    assert np.asarray(inversion.reconstruction) == pytest.approx(
        np.linalg.solve(matrix, data_vector), rel=1.0e-12
    )

    # ... and the dense Cholesky log determinant rejects it
    with pytest.raises(np.linalg.LinAlgError):
        inversion.log_det_curvature_reg_matrix_term


def test__indefinite_matrix__test_mode__log_det_guard_returns_zero(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    inversion = _with_matrix(
        _imaging_inversion(aa.Settings(use_positive_only_solver=False)),
        matrix=_indefinite_nonsingular(),
        data_vector=np.ones(9),
    )

    assert inversion.log_det_curvature_reg_matrix_term == 0.0


def test__singular_matrix__solve_raises__test_mode_dummy(monkeypatch):
    matrix = np.ones((9, 9))

    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    inversion = _with_matrix(
        _imaging_inversion(aa.Settings(use_positive_only_solver=False)),
        matrix=matrix,
        data_vector=np.arange(1.0, 10.0),
    )

    with pytest.raises(np.linalg.LinAlgError):
        inversion.reconstruction

    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    inversion = _with_matrix(
        _imaging_inversion(aa.Settings(use_positive_only_solver=False)),
        matrix=matrix,
        data_vector=np.arange(1.0, 10.0),
    )

    assert (np.asarray(inversion.reconstruction) == 1.0).all()


def test__non_finite_matrix__no_factor__unchanged_routes(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    matrix = np.eye(9)
    matrix[4, 4] = np.nan

    inversion = _with_matrix(
        _imaging_inversion(aa.Settings(use_positive_only_solver=False)),
        matrix=matrix,
        data_vector=np.ones(9),
    )

    assert inversion._curvature_reg_matrix_cho_factor is None


def test__slogdet_method__no_factor__unchanged(monkeypatch):
    calls = _count_cho_factor(monkeypatch)

    inversion = _imaging_inversion(
        aa.Settings(use_positive_only_solver=False, log_det_method="slogdet")
    )

    matrix = np.asarray(inversion.curvature_reg_matrix_reduced)

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        np.linalg.slogdet(matrix)[1], rel=1.0e-12
    )
    assert np.asarray(inversion.reconstruction) == pytest.approx(
        np.linalg.solve(matrix, np.asarray(inversion.data_vector)), rel=1.0e-12
    )
    assert inversion._curvature_reg_matrix_cho_factor is None
    assert calls == []


def test__positive_only_solver__no_factor(monkeypatch):
    calls = _count_cho_factor(monkeypatch)

    inversion = _imaging_inversion(aa.Settings(use_positive_only_solver=True))

    _ = inversion.reconstruction
    _ = inversion.log_det_curvature_reg_matrix_term

    assert inversion._curvature_reg_matrix_cho_factor is None
    assert calls == []


def test__partially_regularized__solve_uses_factor__log_det_dense_reduced():
    dataset = fixtures.make_masked_imaging_7x7()

    grid = aa.Grid2D.from_mask(mask=dataset.mask)

    mapping_matrix = np.full(fill_value=0.5, shape=(9, 2))
    mapping_matrix[0, 0] = 0.8
    mapping_matrix[1, 1] = 0.4

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion = _imaging_inversion(
        aa.Settings(use_positive_only_solver=False),
        linear_obj_list=[linear_obj, fixtures.make_rectangular_mapper_7x7_3x3()],
    )

    assert not inversion.all_linear_obj_have_regularization

    reconstruction_ref, log_det_ref = _two_factorization_values(inversion)

    assert np.asarray(inversion.reconstruction) == pytest.approx(
        reconstruction_ref, rel=1.0e-10, abs=0.0
    )
    assert inversion._curvature_reg_matrix_cho_factor is not None

    # the factor is of the full matrix; the log det is of the mapper-only reduced matrix
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        log_det_ref, rel=1.0e-9
    )
