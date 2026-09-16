"""
The sparse log determinant of the regularization matrix on the NumPy path.

``AbstractInversion.log_det_regularization_matrix_term`` used to dense-factorize
``regularization_matrix_reduced`` whatever it looked like. The neighbour and split
regularization schemes build ``H`` from a mesh's adjacency, so it carries ``O(1)`` non-zeros
per row however many pixels there are (8.45 on the production HST Delaunay ``pixels=1500``
``AdaptSplit`` matrix) and the dense ``O(pixels^3)`` Cholesky spent nearly all of its work on
structural zeros -- ~37 ms of a ~300 ms NumPy likelihood call. On the NumPy path the term now
goes through ``inversion_util.log_det_sparse_spd_from``, a sparse SuperLU factorization, which
is 6.1x faster on that matrix (36.9 ms -> 6.1 ms at 1 thread).

These tests pin three things: the **value**, the **exceptions**, and the two regime switches
that keep the dense Cholesky where it is the faster route (matrices below
``SPARSE_LOG_DET_MIN_PIXELS``, and dense kernel matrices above
``SPARSE_LOG_DET_MAX_NNZ_PER_ROW`` non-zeros per row). The floors also keep every small
fixture -- including the NumPy/JAX parity fixtures at ``pixels=9`` and ``16``, whose log
determinants are pinned across backends at 1e-10 -- on the historical computation exactly.

On tolerances
-------------
The sparse and dense routes are two different fp64 factorizations of the same matrix, so they
agree to the matrix's conditioning, not to machine precision. The split family's ``H`` is
ill-conditioned -- ``cond ~ 6.4e12`` on the production HST Delaunay ``pixels=1500``
``AdaptSplit`` matrix, where the two differ by **7.4e-10 relative** (5.7e-6 nats on a log
determinant of 7756.6, which is 2.9e-6 nats and 9.8e-11 relative on that fit's log evidence of
29140.3). On a deliberately degenerate random mesh the disagreement reaches 2.8e-9, and
NumPy's own ``slogdet`` differs from the dense Cholesky by up to 2.8e-8 on matrices of this
conditioning. No fp64 route is the truth to 1e-10 there, so:

- the **well-conditioned** synthetic matrices, where the route's own accuracy is what is being
  measured, are pinned at ``rtol=1e-12``;
- the **real** ``ConstantSplit`` matrix (``cond ~ 1e8``) at ``rtol=1e-10``;
- the **real** ``AdaptSplit`` matrix (``cond ~ 4e12``) at ``rtol=1e-7``, alongside an assertion
  that ``slogdet`` and the dense Cholesky disagree by no more than the same amount -- so the
  tolerance is pinned to the matrix's fp64 ambiguity rather than chosen to pass.
"""

import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.inversion import inversion_util


def _random_sparse_spd(pixels, band, seed):
    """
    A well-conditioned (``cond <= 1e6``) symmetric positive-definite matrix with a banded
    sparsity pattern, standing in for a mesh's adjacency.
    """
    rng = np.random.default_rng(seed)

    matrix = np.zeros((pixels, pixels))

    for offset in range(1, band + 1):
        values = rng.uniform(-1.0, -0.1, size=pixels - offset)
        matrix += np.diag(values, offset) + np.diag(values, -offset)

    np.fill_diagonal(matrix, np.abs(matrix).sum(axis=1) + 1.0)

    return matrix


def _delaunay_split_mapper(pixels, regularization, seed=1):
    """
    A real Delaunay mapper carrying a real split regularization scheme.

    The mesh vertices are a jittered uniform lattice rather than a Gaussian blob. A blob of
    random vertices produces Delaunay slivers whose interpolation weights push ``cond(H)`` past
    1e13, where the two fp64 factorizations disagree at 1e-8 and the test would be measuring
    the fixture's degeneracy rather than the route. A spread mesh is also the realistic one:
    production meshes are Hilbert- or KNN-derived and cover the traced source plane.
    """
    rng = np.random.default_rng(seed)

    side = int(np.ceil(np.sqrt(pixels)))

    lattice = np.stack(
        np.meshgrid(np.linspace(-1.0, 1.0, side), np.linspace(-1.0, 1.0, side)), axis=-1
    ).reshape(-1, 2)[:pixels]

    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=lattice + rng.normal(0.0, 0.15 / side, size=(pixels, 2)),
        shape_native=(pixels, 1),
        pixel_scales=1.0,
    )

    data_pixels = 4 * pixels

    source_plane_data_grid = aa.Grid2D.no_mask(
        values=rng.normal(0.0, 0.6, size=(data_pixels, 2)),
        shape_native=(data_pixels, 1),
        pixel_scales=1.0,
    )

    interpolator = aa.mesh.Delaunay(pixels=pixels).interpolator_from(
        source_plane_data_grid=source_plane_data_grid,
        source_plane_mesh_grid=source_plane_mesh_grid,
        adapt_data=aa.Array2D.ones(
            shape_native=(4, data_pixels // 4), pixel_scales=0.1
        ),
    )

    return aa.Mapper(interpolator=interpolator, regularization=regularization)


def _kernel_regularization_matrix(pixels, seed=2):
    """
    A real kernel-scheme ``H``, which is ``coefficient * C^-1`` and therefore fully dense.

    ``pixels`` must clear ``SPARSE_LOG_DET_MIN_PIXELS`` so that the **density** regime switch
    is what declines the matrix, rather than the size floor declining it first.
    """
    rng = np.random.default_rng(seed)

    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=aa.Grid2D.no_mask(
            values=rng.normal(0.0, 1.0, size=(pixels, 2)),
            shape_native=(pixels, 1),
            pixel_scales=1.0,
        )
    )

    regularization = aa.reg.MaternKernel(coefficient=3.0, scale=2.0, nu=2.0)

    return np.asarray(regularization.regularization_matrix_from(linear_obj=mapper))


def _dense_log_det(matrix):
    """The pre-lever computation: ``2 * sum(log(diag(cholesky(H))))``."""
    return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))


@pytest.mark.parametrize(
    "pixels, band",
    [(256, 1), (300, 2), (400, 3), (512, 5)],
)
def test__log_det_sparse_spd_from__matches_slogdet_and_dense_cholesky(pixels, band):
    matrix = _random_sparse_spd(pixels=pixels, band=band, seed=pixels)

    assert np.linalg.cond(matrix) < 1.0e6

    log_det = inversion_util.log_det_sparse_spd_from(matrix=matrix)

    assert log_det == pytest.approx(np.linalg.slogdet(matrix)[1], rel=1.0e-12)
    assert log_det == pytest.approx(_dense_log_det(matrix), rel=1.0e-12)


@pytest.mark.parametrize(
    "regularization, rtol",
    [
        # cond(H) ~ 1e8: the log determinant is well determined in fp64
        (aa.reg.ConstantSplit(coefficient=1.0), 1.0e-10),
        # cond(H) ~ 4e12 (the lambda^4 coefficient convention against the stencil's smooth null
        # modes): it is not. See the tolerance note below.
        (
            aa.reg.AdaptSplit(
                inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1
            ),
            1.0e-7,
        ),
    ],
)
def test__log_det_sparse_spd_from__matches_dense_cholesky_on_real_split_matrix(
    regularization, rtol
):
    """
    The real thing: the split-stencil ``H`` of a Delaunay mesh.

    The ``AdaptSplit`` tolerance is 1e-7 because at ``cond(H) ~ 4e12`` fp64 does not determine
    this log determinant more tightly than that: NumPy's own ``slogdet`` of the same matrix
    differs from the dense Cholesky by up to 2.8e-8 relative, which the assertion below pins as
    the reference scale. The sparse factorization's disagreement (1e-11 to 1.7e-8 across sizes
    and seeds; 7.4e-10 on the production ``pixels=1500`` matrix) is the same size as that
    ambiguity, not an error on top of it.
    """
    mapper = _delaunay_split_mapper(pixels=300, regularization=regularization)

    matrix = np.asarray(regularization.regularization_matrix_from(linear_obj=mapper))

    # the stencil is O(1) per row, which is what makes the sparse route the right one
    assert np.count_nonzero(matrix) / matrix.shape[0] < 20.0

    dense = _dense_log_det(matrix)

    # the reference scale: two *existing* fp64 routes over the same matrix
    assert np.linalg.slogdet(matrix)[1] == pytest.approx(dense, rel=rtol)

    log_det = inversion_util.log_det_sparse_spd_from(matrix=matrix)

    assert log_det is not None
    assert log_det == pytest.approx(dense, rel=rtol)


def test__log_det_sparse_spd_from__matrix_below_the_size_floor__returns_none():
    """
    The size regime switch. A sparse factorization carries ~0.14 ms of fixed setup, so below
    ``SPARSE_LOG_DET_MIN_PIXELS`` it is *slower* than the dense Cholesky (0.07x at
    ``pixels=9``, 0.38x at 128, 1.8x at 256) and the function declines the matrix.
    """
    assert inversion_util.SPARSE_LOG_DET_MIN_PIXELS == 256

    for pixels in (9, 16, 64, 128, 255):
        matrix = _random_sparse_spd(pixels=pixels, band=2, seed=pixels)

        assert inversion_util.log_det_sparse_spd_from(matrix=matrix) is None


def test__log_det_regularization_matrix_term__small_split_matrix__never_reaches_splu(
    monkeypatch,
):
    """
    The size floor asserted from the inversion, on a **real** small split-stencil ``H``: with
    SuperLU broken the term still evaluates, to the historical dense value.

    SuperLU is what is broken here rather than ``log_det_sparse_spd_from`` itself, because the
    regime switch lives inside that function -- patching it out would remove the thing under
    test along with the factorization.
    """
    import scipy.sparse.linalg

    def raise_splu(*args, **kwargs):
        raise AssertionError("splu must not be reached below the size floor")

    monkeypatch.setattr(scipy.sparse.linalg, "splu", raise_splu)

    regularization = aa.reg.ConstantSplit(coefficient=1.0)

    mapper = _delaunay_split_mapper(pixels=64, regularization=regularization)

    matrix = np.asarray(regularization.regularization_matrix_from(linear_obj=mapper))

    inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        regularization_matrix=matrix,
    )

    assert inversion.log_det_regularization_matrix_term == pytest.approx(
        _dense_log_det(matrix), rel=1.0e-12
    )


def test__log_det_sparse_spd_from__dense_kernel_matrix__returns_none():
    """
    The density regime switch. A kernel scheme's ``H`` is fully dense, where the sparse
    factorization is ~7x slower than the dense Cholesky at ``pixels=1500``, so the function
    declines it and the caller's dense Cholesky runs.
    """
    matrix = _kernel_regularization_matrix(pixels=300)

    assert np.count_nonzero(matrix) == matrix.size

    assert inversion_util.log_det_sparse_spd_from(matrix=matrix) is None


def test__log_det_regularization_matrix_term__dense_kernel_matrix__never_reaches_splu(
    monkeypatch,
):
    """
    The regime switch asserted from the inversion rather than the helper: with SuperLU broken,
    a dense kernel ``H`` still evaluates its term (it never gets there) while a sparse split
    ``H`` fails loudly.
    """
    import scipy.sparse.linalg

    def raise_splu(*args, **kwargs):
        raise AssertionError("splu must not be reached for a dense matrix")

    monkeypatch.setattr(scipy.sparse.linalg, "splu", raise_splu)

    kernel_matrix = _kernel_regularization_matrix(pixels=300)

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        regularization_matrix=kernel_matrix,
    )

    assert inversion.log_det_regularization_matrix_term == pytest.approx(
        _dense_log_det(kernel_matrix), rel=1.0e-12
    )

    # the same broken splu is reached by a sparse matrix, confirming the patch is live
    with pytest.raises(AssertionError):
        inversion_util.log_det_sparse_spd_from(
            matrix=_random_sparse_spd(pixels=300, band=2, seed=0)
        )


def test__log_det_regularization_matrix_term__real_split_inversion__matches_dense_value():
    """
    The inversion's term, end to end on the NumPy path, against the pre-lever dense value
    computed independently in the test.
    """
    regularization = aa.reg.AdaptSplit(
        inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1
    )

    mapper = _delaunay_split_mapper(pixels=300, regularization=regularization)

    matrix = np.asarray(regularization.regularization_matrix_from(linear_obj=mapper))

    inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        regularization_matrix=matrix,
    )

    assert inversion.regularization_matrix_reduced.shape == matrix.shape

    assert inversion.log_det_regularization_matrix_term == pytest.approx(
        _dense_log_det(matrix), rel=1.0e-7
    )
    assert inversion.log_det_regularization_matrix_term == pytest.approx(
        np.linalg.slogdet(matrix)[1], rel=1.0e-7
    )


def test__log_det_regularization_matrix_term__slogdet_method__unchanged():
    """The ``"slogdet"`` opt-in path is untouched by the sparse route."""
    matrix = _random_sparse_spd(pixels=300, band=2, seed=7)

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        regularization_matrix=matrix,
        settings=aa.Settings(log_det_method="slogdet"),
    )

    assert inversion.log_det_regularization_matrix_term == pytest.approx(
        np.linalg.slogdet(matrix)[1], rel=1.0e-12
    )


@pytest.mark.parametrize(
    "matrix",
    [
        # exactly singular: SuperLU's RuntimeError, np.linalg.cholesky's LinAlgError
        np.zeros((300, 300)),
        # rank deficient: two identical rows in an otherwise banded matrix
        np.diag(np.arange(1.0, 301.0)),
    ],
)
def test__log_det_sparse_spd_from__singular_matrix__raises_lin_alg_error(matrix):
    matrix = np.array(matrix)
    matrix[0, 0] = 0.0  # a zero pivot in both parametrizations

    with pytest.raises(np.linalg.LinAlgError):
        inversion_util.log_det_sparse_spd_from(matrix=matrix)

    # the dense Cholesky the route replaces raises the same error on the same matrix
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(matrix)


def test__log_det_sparse_spd_from__non_positive_definite_matrix__raises_lin_alg_error():
    """
    SuperLU factorizes an indefinite matrix happily and would return ``log|det|``. The
    diagonal pivots are checked so it raises where the dense Cholesky raises instead.
    """
    matrix = _random_sparse_spd(pixels=300, band=2, seed=3)
    matrix[5, 5] = -matrix[5, 5]

    with pytest.raises(np.linalg.LinAlgError):
        inversion_util.log_det_sparse_spd_from(matrix=matrix)

    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(matrix)


def test__log_det_regularization_matrix_term__singular_matrix__test_mode_returns_zero(
    monkeypatch,
):
    """
    The test-mode guard, unchanged: a fabricated test-mode model's singular ``H`` discards
    this evidence term rather than crashing, and a normal run re-raises.
    """
    matrix = np.zeros((300, 300))

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        regularization_matrix=matrix,
    )

    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    with pytest.raises(np.linalg.LinAlgError):
        inversion.log_det_regularization_matrix_term

    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")
    assert inversion.log_det_regularization_matrix_term == 0.0


def test__log_det_curvature_reg_matrix_term__is_not_routed_sparsely(monkeypatch):
    """
    ``log_det_curvature_reg_matrix_term`` factorizes ``F + lambda H``, which is dense. It must
    keep the dense Cholesky: with SuperLU broken it still evaluates.
    """
    import scipy.sparse.linalg

    def raise_splu(*args, **kwargs):
        raise AssertionError("splu must not be reached by the curvature reg term")

    monkeypatch.setattr(scipy.sparse.linalg, "splu", raise_splu)

    matrix = _random_sparse_spd(pixels=300, band=2, seed=11)

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=aa.m.MockRegularization())],
        curvature_reg_matrix=matrix,
    )

    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        _dense_log_det(matrix), rel=1.0e-12
    )
