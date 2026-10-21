"""
Parity tests for the matrix-free interferometer utilities (phases 1-3 of PyAutoArray #638).

The matrix-free sparse mapping matrix, curvature matrix action, data vector and PCG solve are held
against the dense quantities of `InversionInterferometerSparse` on a small synthetic dataset, and
(phase 3) the exact preconditioner log-determinant, the stochastic Lanczos quadrature estimate of
`ln det(F + H)` and the assembled log evidence against their dense counterparts.

These tests deliberately carry no `importorskip("jax")`: the module is NumPy / scipy only, so the
`unittest-nojax` CI leg runs them.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse

import autoarray as aa
from autoarray.inversion.inversion.interferometer import (
    inversion_interferometer_matrix_free_util as mf,
)
from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
    overlay_grid_from,
)


def _dataset_sparse_from(seed=1, n_visibilities=40, uv_scale=1.0e4):
    """
    A circular-masked dataset, so the slim grid is a strict subset of its unmasked extent and the
    scatter / gather between them is exercised, with random visibilities of unit noise.

    `uv_scale` sets the baseline lengths in wavelengths. At the default `1e4` the fringes are far
    broader than the mask, so `W~` is close to a constant and nowhere near the Powell preconditioner's
    `κ₀ I`; with many visibilities on long baselines (`500`, `1e6`) the dirty beam is compact, as for
    real data, and `P⁻¹(F + H)` has eigenvalues near one, the regime the stochastic log-determinant
    is built for.
    """
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=0.5, radius=2.5)

    rng = np.random.default_rng(seed=seed)

    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=rng.normal(size=(n_visibilities, 2))),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((n_visibilities, 2))),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)) * uv_scale,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    return mask, dataset.apply_sparse_operator(use_jax=False)


def _rectangular_mapper_from(mask, coefficient=1.0, shape=(6, 6), regularization=None):
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)

    mesh = aa.mesh.RectangularUniform(shape=shape)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(
            overlay_grid_from(shape_native=shape, grid=grid.over_sampled)
        ),
        adapt_data=aa.Array2D.ones(shape, pixel_scales=0.1),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=regularization or aa.reg.Constant(coefficient=coefficient),
    )


def _delaunay_mapper_from(mask, coefficient=1.0):
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)

    image_mesh_grid = aa.image_mesh.Overlay(shape=(5, 5)).image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    interpolator = aa.mesh.Delaunay(pixels=image_mesh_grid.shape[0]).interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=aa.reg.Constant(coefficient=coefficient),
    )


def _inversion_from(dataset_sparse, mapper):
    """
    The sparse-path inversion with the positive-negative solve, so its `reconstruction` is the
    unconstrained solution of `curvature_reg_matrix s = data_vector`.
    """
    inversion = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=aa.Settings(use_positive_only_solver=False),
    )

    assert isinstance(inversion, aa.InversionInterferometerSparse)

    return inversion


@pytest.mark.parametrize(
    "mapper_from", [_rectangular_mapper_from, _delaunay_mapper_from]
)
def test__sparse_mapping_matrix__equals_dense_mapping_matrix_exactly(mapper_from):
    mask, dataset_sparse = _dataset_sparse_from()
    mapper = mapper_from(mask=mask)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    mapping_matrix = mf.sparse_mapping_matrix_from(inversion=inversion, mapper=mapper)

    assert isinstance(mapping_matrix, scipy.sparse.csr_matrix)
    assert np.array_equal(mapping_matrix.toarray(), np.asarray(mapper.mapping_matrix))


@pytest.mark.parametrize(
    "mapper_from", [_rectangular_mapper_from, _delaunay_mapper_from]
)
def test__curvature_matvec_and_data_vector__match_sparse_path(mapper_from):
    mask, dataset_sparse = _dataset_sparse_from()
    mapper = mapper_from(mask=mask)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    operator = mf.MatrixFreeDataOperator.from_inversion(
        inversion=inversion, mapper=mapper
    )

    v = np.random.default_rng(seed=2).normal(size=operator.params)

    curvature_matvec = np.asarray(inversion.curvature_matrix) @ v

    assert np.max(
        np.abs(operator.curvature_matvec(v) - curvature_matvec)
    ) <= 1.0e-12 * np.max(np.abs(curvature_matvec))

    data_vector = np.asarray(inversion.data_vector)

    assert np.max(np.abs(operator.data_vector - data_vector)) <= 1.0e-12 * np.max(
        np.abs(data_vector)
    )

    assert operator.kappa0 == pytest.approx(
        float(dataset_sparse.sparse_operator.nufft_precision_operator[0, 0]), 1.0e-14
    )


def test__wtilde_image_from__equals_sparse_operator_apply_operator():
    mask, dataset_sparse = _dataset_sparse_from()
    mapper = _rectangular_mapper_from(mask=mask)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    operator = mf.MatrixFreeDataOperator.from_inversion(
        inversion=inversion, mapper=mapper
    )

    image = np.random.default_rng(seed=3).normal(
        size=(operator.y_shape, operator.x_shape)
    )

    wtilde_image = dataset_sparse.sparse_operator.apply_operator(
        image.reshape(-1, 1), xp=np
    )[:, 0]

    assert np.array_equal(operator.wtilde_image_from(image).ravel(), wtilde_image)


@pytest.mark.parametrize("coefficient", [1.0, 0.01])
def test__reconstruction_via_pcg__matches_unconstrained_sparse_path(coefficient):
    mask, dataset_sparse = _dataset_sparse_from()
    mapper = _rectangular_mapper_from(mask=mask, coefficient=coefficient)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    operator = mf.MatrixFreeDataOperator.from_inversion(
        inversion=inversion, mapper=mapper
    )

    # The autoarray regularization matrix already carries the Constant coefficient, so lambda = 1.
    regularization_matrix = scipy.sparse.csr_matrix(
        np.asarray(inversion.regularization_matrix)
    )

    reconstruction, info = mf.reconstruction_via_pcg_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        coefficient=1.0,
        rtol=1.0e-12,
    )

    reconstruction_sparse_path = np.asarray(inversion.reconstruction)

    assert info == 0
    assert np.max(
        np.abs(reconstruction - reconstruction_sparse_path)
    ) <= 1.0e-8 * np.max(np.abs(reconstruction_sparse_path))


@pytest.mark.parametrize("coefficient", [1.0, 0.01])
def test__reconstruction_via_pcg__matern_sparse_matrix__matches_unconstrained_sparse_path(
    coefficient,
):
    """
    Phase 2: the natively sparse Matérn precision of `MaternSparse` is the `H` of the matrix-free solve.
    The dense-path inversion uses the same `MaternSparse` instance as the pixelization's regularization
    (through its dense `regularization_matrix_from`), and since `H` already carries the coefficient the
    matrix-free `λ` is 1.
    """
    mask, dataset_sparse = _dataset_sparse_from()

    regularization = aa.reg.MaternSparse(coefficient=coefficient, scale=1.0)

    mapper = _rectangular_mapper_from(mask=mask, regularization=regularization)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    operator = mf.MatrixFreeDataOperator.from_inversion(
        inversion=inversion, mapper=mapper
    )

    regularization_matrix = regularization.regularization_matrix_sparse_from(mapper)

    assert np.array_equal(
        regularization_matrix.toarray(), np.asarray(inversion.regularization_matrix)
    )

    preconditioner, _ = mf.powell_preconditioner_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        coefficient=1.0,
    )

    assert scipy.sparse.issparse(preconditioner)

    reconstruction, info = mf.reconstruction_via_pcg_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        coefficient=1.0,
        rtol=1.0e-12,
    )

    reconstruction_sparse_path = np.asarray(inversion.reconstruction)

    assert info == 0
    assert np.max(
        np.abs(reconstruction - reconstruction_sparse_path)
    ) <= 1.0e-8 * np.max(np.abs(reconstruction_sparse_path))


def test__reconstruction_via_pcg__coefficient_scales_regularization_matrix():
    mask, dataset_sparse = _dataset_sparse_from()
    mapper = _rectangular_mapper_from(mask=mask)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    operator = mf.MatrixFreeDataOperator.from_inversion(
        inversion=inversion, mapper=mapper
    )

    regularization_matrix = np.asarray(inversion.regularization_matrix)

    coefficient = 30.0

    reconstruction, info = mf.reconstruction_via_pcg_from(
        operator=operator,
        regularization_matrix_sparse=scipy.sparse.csr_matrix(regularization_matrix),
        coefficient=coefficient,
        rtol=1.0e-12,
    )

    reconstruction_dense = np.linalg.solve(
        np.asarray(inversion.curvature_matrix) + coefficient * regularization_matrix,
        np.asarray(inversion.data_vector),
    )

    assert info == 0
    assert np.max(np.abs(reconstruction - reconstruction_dense)) <= 1.0e-8 * np.max(
        np.abs(reconstruction_dense)
    )


def test__powell_preconditioner__equals_kappa0_mtm_plus_coefficient_regularization():
    mask, dataset_sparse = _dataset_sparse_from()
    mapper = _rectangular_mapper_from(mask=mask)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    operator = mf.MatrixFreeDataOperator.from_inversion(
        inversion=inversion, mapper=mapper
    )

    regularization_matrix = np.asarray(inversion.regularization_matrix)
    mapping_matrix = np.asarray(mapper.mapping_matrix)

    preconditioner, factor = mf.powell_preconditioner_from(
        operator=operator,
        regularization_matrix_sparse=scipy.sparse.csr_matrix(regularization_matrix),
        coefficient=2.0,
    )

    preconditioner_dense = (
        operator.kappa0 * mapping_matrix.T @ mapping_matrix
        + 2.0 * regularization_matrix
    )

    assert preconditioner.toarray() == pytest.approx(preconditioner_dense, 1.0e-12)

    b = np.random.default_rng(seed=4).normal(size=operator.params)

    assert factor.solve(b) == pytest.approx(
        np.linalg.solve(preconditioner_dense, b), 1.0e-8
    )


def _matern_sparse_system_from(coefficient=1.0):
    """
    The phase-3 case: a `MaternSparse`-regularized rectangular mesh on the data-dominated dataset
    (500 visibilities on long baselines), returning the dense-path inversion and the matrix-free
    operator, `H` and its factor `R` with `H = RᵀR` (the coefficient folded into `R`).
    """
    mask, dataset_sparse = _dataset_sparse_from(n_visibilities=500, uv_scale=1.0e6)

    regularization = aa.reg.MaternSparse(coefficient=coefficient, scale=1.0)

    mapper = _rectangular_mapper_from(mask=mask, regularization=regularization)
    inversion = _inversion_from(dataset_sparse=dataset_sparse, mapper=mapper)

    operator = mf.MatrixFreeDataOperator.from_inversion(
        inversion=inversion, mapper=mapper
    )

    regularization_matrix = regularization.regularization_matrix_sparse_from(mapper)
    regularization_factor = np.sqrt(
        regularization.coefficient
    ) * regularization.regularization_factor_sparse_from(mapper)

    return (
        dataset_sparse,
        regularization,
        mapper,
        inversion,
        operator,
        regularization_matrix,
        regularization_factor,
    )


@pytest.mark.parametrize("coefficient", [1.0, 0.01])
def test__log_det_preconditioner_from__equals_dense_slogdet(coefficient):
    _, _, _, _, operator, regularization_matrix, _ = _matern_sparse_system_from(
        coefficient=coefficient
    )

    preconditioner, factor = mf.powell_preconditioner_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        coefficient=1.0,
    )

    sign, log_det = np.linalg.slogdet(preconditioner.toarray())

    assert sign == 1.0
    assert mf.log_det_preconditioner_from(factor) == pytest.approx(log_det, abs=1.0e-10)


@pytest.mark.parametrize("coefficient", [1.0, 0.01])
def test__log_det_curvature_reg_via_slq__within_half_percent_of_dense_and_deterministic(
    coefficient,
):
    (
        _,
        _,
        _,
        inversion,
        operator,
        regularization_matrix,
        regularization_factor,
    ) = _matern_sparse_system_from(coefficient=coefficient)

    sign, log_det_dense = np.linalg.slogdet(np.asarray(inversion.curvature_reg_matrix))

    assert sign == 1.0

    estimate, log_det_probes = mf.log_det_curvature_reg_via_slq_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        regularization_factor_sparse=regularization_factor,
        coefficient=1.0,
        n_probes=10,
        n_steps=40,
    )

    assert log_det_probes.shape == (10,)
    assert estimate == pytest.approx(np.mean(log_det_probes), 1.0e-14)
    assert abs(estimate - log_det_dense) <= 0.005 * abs(log_det_dense)

    estimate_repeat, log_det_probes_repeat = mf.log_det_curvature_reg_via_slq_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        regularization_factor_sparse=regularization_factor,
        coefficient=1.0,
        n_probes=10,
        n_steps=40,
    )

    assert estimate_repeat == estimate
    assert np.array_equal(log_det_probes_repeat, log_det_probes)


def test__log_det_curvature_reg_via_slq__error_shrinks_with_probes():
    """
    Loose convergence check: over a handful of seeds the mean error at 10 probes is below that at 2,
    and the Monte-Carlo standard error of the mean shrinks.
    """
    (
        _,
        _,
        _,
        inversion,
        operator,
        regularization_matrix,
        regularization_factor,
    ) = _matern_sparse_system_from()

    _, log_det_dense = np.linalg.slogdet(np.asarray(inversion.curvature_reg_matrix))

    def errors_and_standard_errors(n_probes):
        errors, standard_errors = [], []

        for seed in range(5):
            estimate, log_det_probes = mf.log_det_curvature_reg_via_slq_from(
                operator=operator,
                regularization_matrix_sparse=regularization_matrix,
                regularization_factor_sparse=regularization_factor,
                coefficient=1.0,
                n_probes=n_probes,
                seed=seed,
            )
            errors.append(abs(estimate - log_det_dense))
            standard_errors.append(np.std(log_det_probes) / np.sqrt(n_probes))

        return np.mean(errors), np.mean(standard_errors)

    error_2, standard_error_2 = errors_and_standard_errors(n_probes=2)
    error_10, standard_error_10 = errors_and_standard_errors(n_probes=10)

    assert error_10 < error_2
    assert standard_error_10 < standard_error_2


def test__log_det_curvature_reg_via_slq__factor_not_matching_regularization_matrix__raises():
    _, regularization, mapper, _, operator, regularization_matrix, _ = (
        _matern_sparse_system_from(coefficient=0.01)
    )

    with pytest.raises(ValueError):
        mf.log_det_curvature_reg_via_slq_from(
            operator=operator,
            regularization_matrix_sparse=regularization_matrix,
            regularization_factor_sparse=regularization.regularization_factor_sparse_from(
                mapper
            ),
            coefficient=1.0,
        )


@pytest.mark.parametrize("coefficient", [1.0, 0.01])
def test__log_evidence_matrix_free_from__matches_dense_sparse_path_evidence(
    coefficient,
):
    """
    The dense reference is `FitInterferometer.log_evidence`'s assembly, `fit_util.log_evidence_from`,
    of the unconstrained sparse-path inversion's terms. With the dense `ln det(F + H)` substituted the
    other four terms agree to round-off; with the SLQ estimate the total agrees within the SLQ
    tolerance (0.5% of `ln det(F + H)`).
    """
    from autoarray.fit import fit_util

    (
        dataset_sparse,
        regularization,
        mapper,
        inversion,
        operator,
        regularization_matrix,
        regularization_factor,
    ) = _matern_sparse_system_from(coefficient=coefficient)

    noise_normalization = dataset_sparse.sparse_operator.noise_normalization

    log_evidence_dense = fit_util.log_evidence_from(
        chi_squared=inversion.fast_chi_squared,
        regularization_term=inversion.regularization_term,
        log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
        log_regularization_term=inversion.log_det_regularization_matrix_term,
        noise_normalization=noise_normalization,
    )

    reconstruction, info = mf.reconstruction_via_pcg_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        coefficient=1.0,
        rtol=1.0e-12,
    )

    assert info == 0

    log_det_regularization = regularization.log_det_regularization_matrix_term_from(
        mapper
    )

    assert log_det_regularization == pytest.approx(
        inversion.log_det_regularization_matrix_term, abs=1.0e-10
    )

    kwargs = dict(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix,
        regularization_factor_sparse=regularization_factor,
        coefficient=1.0,
        reconstruction=reconstruction,
        log_det_regularization=log_det_regularization,
        data_term=dataset_sparse.sparse_operator.data_term,
        noise_normalization=noise_normalization,
    )

    log_evidence_exact_log_det = mf.log_evidence_matrix_free_from(
        **kwargs, log_det_curvature_reg=inversion.log_det_curvature_reg_matrix_term
    )

    assert log_evidence_exact_log_det == pytest.approx(
        log_evidence_dense, abs=1.0e-10 * abs(log_evidence_dense)
    )

    log_evidence_slq = mf.log_evidence_matrix_free_from(**kwargs)

    _, log_det_dense = np.linalg.slogdet(np.asarray(inversion.curvature_reg_matrix))

    assert abs(log_evidence_slq - log_evidence_dense) <= 0.5 * 0.005 * abs(
        log_det_dense
    )


def test__matrix_free_util__imports_and_runs_with_jax_blocked():
    """
    The module must work in the `unittest-nojax` leg. JAX is in `sys.modules` for the rest of a
    normal test session, so a subprocess hides `jax`, `jaxlib` and `nufftax` from the import system,
    imports the module and runs a PCG solve and the phase-3 log-determinant / evidence functions.
    """
    script = textwrap.dedent(
        """
        import importlib.machinery
        import sys

        # Emulate an environment where JAX is not installed: the path finder reports the blocked
        # packages as absent, so `importlib.util.find_spec("jax")` is None and importing them
        # raises `ModuleNotFoundError`, exactly as without the packages.
        _find_spec = importlib.machinery.PathFinder.find_spec.__func__


        def _find_spec_blocking_jax(cls, name, path=None, target=None):
            if name.split(".")[0] in ("jax", "jaxlib", "nufftax"):
                return None
            return _find_spec(cls, name, path, target)


        importlib.machinery.PathFinder.find_spec = classmethod(_find_spec_blocking_jax)

        # Guard the guard: if the block stopped working this test would pass vacuously.
        try:
            import jax
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("jax was not blocked")

        import numpy as np
        import scipy.sparse

        import autoarray as aa
        from autoarray.inversion.inversion.interferometer import (
            inversion_interferometer_matrix_free_util as mf,
        )

        mask = aa.Mask2D.circular(shape_native=(8, 8), pixel_scales=1.0, radius=3.0)

        rng = np.random.default_rng(seed=1)

        dataset = aa.Interferometer(
            data=aa.Visibilities(visibilities=rng.normal(size=(5, 2))),
            noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((5, 2))),
            uv_wavelengths=rng.normal(size=(5, 2)),
            real_space_mask=mask,
            transformer_class=aa.TransformerDFT,
        ).apply_sparse_operator(method="numpy")

        grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)
        image_mesh_grid = aa.image_mesh.Overlay(shape=(4, 4)).image_plane_mesh_grid_from(
            mask=mask, adapt_data=None
        )
        mapper = aa.Mapper(
            interpolator=aa.mesh.Delaunay(
                pixels=image_mesh_grid.shape[0]
            ).interpolator_from(
                source_plane_data_grid=grid, source_plane_mesh_grid=image_mesh_grid
            ),
            regularization=aa.reg.Constant(coefficient=1.0),
        )

        inversion = aa.Inversion(
            dataset=dataset,
            linear_obj_list=[mapper],
            settings=aa.Settings(use_positive_only_solver=False),
        )

        operator = mf.MatrixFreeDataOperator.from_inversion(inversion, mapper)

        reconstruction, info = mf.reconstruction_via_pcg_from(
            operator,
            scipy.sparse.csr_matrix(np.asarray(inversion.regularization_matrix)),
            1.0,
            rtol=1.0e-12,
        )

        assert info == 0
        assert np.allclose(reconstruction, np.asarray(inversion.reconstruction))

        regularization_matrix = scipy.sparse.csr_matrix(
            np.asarray(inversion.regularization_matrix)
        )
        regularization_factor = scipy.sparse.csr_matrix(
            np.linalg.cholesky(regularization_matrix.toarray()).T
        )

        _, factor = mf.powell_preconditioner_from(operator, regularization_matrix, 1.0)
        assert np.isfinite(mf.log_det_preconditioner_from(factor))

        log_det, _ = mf.log_det_curvature_reg_via_slq_from(
            operator, regularization_matrix, regularization_factor, 1.0, n_probes=2
        )
        assert np.isfinite(log_det)

        log_evidence = mf.log_evidence_matrix_free_from(
            operator,
            regularization_matrix,
            regularization_factor,
            1.0,
            reconstruction,
            np.linalg.slogdet(regularization_matrix.toarray())[1],
            1.0,
            1.0,
            n_probes=2,
        )
        assert np.isfinite(log_evidence)

        assert not any(
            name.split(".")[0] in ("jax", "jaxlib", "nufftax") for name in sys.modules
        )
        """
    )

    environment = dict(os.environ)
    environment["PYAUTO_DISABLE_JAX"] = "1"

    subprocess.run(
        [sys.executable, "-c", script], check=True, env=environment, timeout=600
    )
