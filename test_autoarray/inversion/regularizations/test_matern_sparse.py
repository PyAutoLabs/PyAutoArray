"""
Tests of the sparse Matérn (SPDE) precision regularization `MaternSparse` (phase 2 of PyAutoArray #638).

These carry no `importorskip("jax")`: the module is NumPy / scipy only, so the `unittest-nojax` CI leg
runs them.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse

import autoarray as aa


def _rectangular_mapper_from(shape=(4, 3), mesh_cls=None):
    """
    A mapper with a rectangular mesh laid over a circular-masked grid. The mesh grid's y and x spacings
    differ (1.1875 and 1.5833 for the 4x3 mesh), so the per-axis Laplacian scaling is exercised.
    """
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=0.5, radius=2.5)
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=2)

    mesh = (mesh_cls or aa.mesh.RectangularUniform)(shape=shape)

    return aa.Mapper(
        interpolator=mesh.interpolator_from(
            source_plane_data_grid=grid, source_plane_mesh_grid=None
        ),
        regularization=aa.reg.Constant(),
    )


def _laplacian_operator_by_hand_from(mapper, scale, nu=1.0):
    """
    `L = kappa^2 I - lap` built pixel by pixel from the mesh's (y, x) coordinates: neighbours are found
    geometrically (same row and one x-spacing apart, or same column and one y-spacing apart) rather than
    from the row-major index arithmetic the class uses, so this also checks the mesh ordering.
    """
    grid = np.asarray(mapper.source_plane_mesh_grid.array)
    pixels = grid.shape[0]

    ys = np.unique(grid[:, 0])
    xs = np.unique(grid[:, 1])
    dy = ys[1] - ys[0]
    dx = xs[1] - xs[0]

    kappa = np.sqrt(2.0 * nu) / scale

    L = kappa**2 * np.eye(pixels)

    for k in range(pixels):
        for m in range(pixels):
            if k == m:
                continue
            same_row = np.isclose(grid[k, 0], grid[m, 0])
            same_col = np.isclose(grid[k, 1], grid[m, 1])
            if same_row and np.isclose(abs(grid[k, 1] - grid[m, 1]), dx):
                L[k, m] -= 1.0 / dx**2
                L[k, k] += 1.0 / dx**2
            if same_col and np.isclose(abs(grid[k, 0] - grid[m, 0]), dy):
                L[k, m] -= 1.0 / dy**2
                L[k, k] += 1.0 / dy**2

    return L


def _c_by_hand_from(L, shape):
    centre = (shape[0] // 2) * shape[1] + shape[1] // 2
    # The covariance (c L^2)^-1 has centre variance [L^-2]_cc / c, so c = [L^-2]_cc makes it one.
    return np.linalg.inv(L @ L)[centre, centre]


def test__mesh_ordering__source_plane_mesh_grid_is_row_major_from_top_left():
    mapper = _rectangular_mapper_from(shape=(4, 3))

    grid = np.asarray(mapper.source_plane_mesh_grid.array).reshape(4, 3, 2)

    # y is constant along a row and decreases down the rows; x increases along a row.
    assert np.allclose(grid[:, :, 0], grid[:, :1, 0])
    assert np.all(np.diff(grid[:, 0, 0]) < 0.0)
    assert np.allclose(grid[:, :, 1], grid[:1, :, 1])
    assert np.all(np.diff(grid[0, :, 1]) > 0.0)

    shape_native, pixel_scales = aa.reg.MaternSparse.mesh_shape_and_pixel_scales_from(
        mapper
    )

    assert shape_native == (4, 3)
    assert pixel_scales == pytest.approx(mapper.mesh_geometry.pixel_scales, 1.0e-12)


def test__regularization_matrix__4x3_mesh__equals_hand_built_coefficient_c_l_squared():
    mapper = _rectangular_mapper_from(shape=(4, 3))

    reg = aa.reg.MaternSparse(coefficient=1.7, scale=2.0)

    L = _laplacian_operator_by_hand_from(mapper=mapper, scale=2.0)
    c = _c_by_hand_from(L=L, shape=(4, 3))

    regularization_matrix = reg.regularization_matrix_sparse_from(mapper)

    assert isinstance(regularization_matrix, scipy.sparse.csr_matrix)
    assert regularization_matrix.toarray() == pytest.approx(
        1.7 * c * L @ L, rel=1.0e-12, abs=1.0e-14
    )

    # The dense-path matrix is the same array.
    assert np.array_equal(
        reg.regularization_matrix_from(mapper), regularization_matrix.toarray()
    )

    # R^T R is the unscaled precision.
    factor = reg.regularization_factor_sparse_from(mapper)
    assert (factor.T @ factor).toarray() == pytest.approx(
        c * L @ L, rel=1.0e-12, abs=1.0e-14
    )


def test__regularization_matrix__symmetric_positive_definite():
    mapper = _rectangular_mapper_from(shape=(5, 6))

    regularization_matrix = aa.reg.MaternSparse(
        coefficient=0.3, scale=1.5
    ).regularization_matrix_from(mapper)

    assert np.allclose(
        regularization_matrix, regularization_matrix.T, rtol=0.0, atol=0.0
    )
    assert np.min(np.linalg.eigvalsh(regularization_matrix)) > 0.0
    np.linalg.cholesky(regularization_matrix)


@pytest.mark.parametrize("shape", [(4, 3), (7, 8)])
def test__log_det__matches_slogdet(shape):
    mapper = _rectangular_mapper_from(shape=shape)

    reg = aa.reg.MaternSparse(coefficient=2.5, scale=1.2)

    sign, log_det = np.linalg.slogdet(reg.regularization_matrix_from(mapper))

    assert sign == 1.0
    assert reg.log_det_regularization_matrix_term_from(mapper) == pytest.approx(
        log_det, abs=1.0e-10
    )


def test__coefficient_scaling__matrix_doubles_and_log_det_shifts_by_n_log_2():
    mapper = _rectangular_mapper_from(shape=(4, 3))

    reg = aa.reg.MaternSparse(coefficient=1.3, scale=2.0)
    reg_double = aa.reg.MaternSparse(coefficient=2.6, scale=2.0)

    assert reg_double.regularization_matrix_from(mapper) == pytest.approx(
        2.0 * reg.regularization_matrix_from(mapper), rel=1.0e-14
    )
    assert reg_double.log_det_regularization_matrix_term_from(
        mapper
    ) - reg.log_det_regularization_matrix_term_from(mapper) == pytest.approx(
        mapper.params * np.log(2.0), abs=1.0e-10
    )


def test__weights__equal_hand_built_w_inv_c_l_squared_w_inv():
    mapper = _rectangular_mapper_from(shape=(4, 3))

    weights = np.random.default_rng(seed=1).uniform(0.5, 2.0, size=mapper.params)

    reg = aa.reg.MaternSparse(coefficient=0.8, scale=2.0, weights=weights)

    L = _laplacian_operator_by_hand_from(mapper=mapper, scale=2.0)
    c = _c_by_hand_from(L=L, shape=(4, 3))
    W_inv = np.diag(1.0 / weights)

    regularization_matrix = reg.regularization_matrix_from(mapper)

    assert regularization_matrix == pytest.approx(
        0.8 * W_inv @ (c * L @ L) @ W_inv, rel=1.0e-12, abs=1.0e-14
    )

    sign, log_det = np.linalg.slogdet(regularization_matrix)
    assert reg.log_det_regularization_matrix_term_from(mapper) == pytest.approx(
        log_det, abs=1.0e-10
    )

    with pytest.raises(ValueError):
        aa.reg.MaternSparse(weights=np.ones(3)).regularization_matrix_from(mapper)


def test__regularization_term__equals_s_h_s():
    mapper = _rectangular_mapper_from(shape=(4, 3))

    weights = np.random.default_rng(seed=2).uniform(0.5, 2.0, size=mapper.params)
    reg = aa.reg.MaternSparse(coefficient=0.8, scale=2.0, weights=weights)

    s = np.random.default_rng(seed=3).normal(size=mapper.params)

    assert reg.regularization_term_from(mapper, s) == pytest.approx(
        s @ reg.regularization_matrix_from(mapper) @ s, rel=1.0e-12
    )


def test__regularization_weights__coefficient_times_ones():
    mapper = _rectangular_mapper_from(shape=(4, 3))

    assert np.array_equal(
        aa.reg.MaternSparse(coefficient=3.0).regularization_weights_from(mapper),
        3.0 * np.ones(12),
    )


def test__adaptive_rectangular_mesh__supported():
    mapper = _rectangular_mapper_from(
        shape=(4, 3), mesh_cls=aa.mesh.RectangularRTUAdaptDensity
    )

    reg = aa.reg.MaternSparse(coefficient=1.0, scale=2.0)

    L = _laplacian_operator_by_hand_from(mapper=mapper, scale=2.0)
    c = _c_by_hand_from(L=L, shape=(4, 3))

    assert reg.regularization_matrix_from(mapper) == pytest.approx(
        c * L @ L, rel=1.0e-12, abs=1.0e-14
    )


def test__non_rectangular_mesh__raises_not_implemented():
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=0.5, radius=2.5)
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)

    image_mesh_grid = aa.image_mesh.Overlay(shape=(5, 5)).image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    mapper = aa.Mapper(
        interpolator=aa.mesh.Delaunay(
            pixels=image_mesh_grid.shape[0]
        ).interpolator_from(
            source_plane_data_grid=grid, source_plane_mesh_grid=image_mesh_grid
        ),
        regularization=aa.reg.Constant(),
    )

    reg = aa.reg.MaternSparse(coefficient=1.0, scale=1.0)

    with pytest.raises(NotImplementedError, match="rectangular"):
        reg.regularization_matrix_from(mapper)

    with pytest.raises(NotImplementedError, match="rectangular"):
        reg.log_det_regularization_matrix_term_from(mapper)


def test__nu_not_one__raises_not_implemented():
    with pytest.raises(NotImplementedError, match="nu = 1"):
        aa.reg.MaternSparse(nu=2.0)

    with pytest.raises(NotImplementedError):
        aa.reg.MaternSparse(nu=0.5)


def test__covariance__unit_variance_at_centre_and_decays_along_row():
    """
    A loose sanity check of the prior the precision encodes, not an exactness test: `C = (c L^2)^-1`
    has unit variance at the mesh centre by construction (`c`), and the correlation with the centre
    falls monotonically with distance along the centre row, as a Matérn kernel does. Edge effects of
    the reflecting boundary mean `C` is not exactly the stationary Matérn kernel, so no values beyond
    these are asserted.
    """
    shape = (11, 11)
    mapper = _rectangular_mapper_from(shape=shape)

    reg = aa.reg.MaternSparse(coefficient=1.0, scale=1.5)

    covariance = np.linalg.inv(reg.regularization_matrix_from(mapper))

    centre_row, centre_col = shape[0] // 2, shape[1] // 2
    centre = centre_row * shape[1] + centre_col

    assert covariance[centre, centre] == pytest.approx(1.0, rel=1.0e-10)

    along_row = covariance[
        centre, centre_row * shape[1] + np.arange(centre_col, shape[1])
    ]

    assert np.all(np.diff(along_row) < 0.0)
    assert np.all(along_row > 0.0)


def test__matern_sparse__imports_and_runs_with_jax_blocked():
    """
    The module must work in the `unittest-nojax` leg; a subprocess hides `jax`, `jaxlib` and `nufftax`
    from the import system (the same pattern as `test_matrix_free_util.py`), imports the class and
    builds the matrix and its log-det.
    """
    script = textwrap.dedent(
        """
        import importlib.machinery
        import sys

        _find_spec = importlib.machinery.PathFinder.find_spec.__func__


        def _find_spec_blocking_jax(cls, name, path=None, target=None):
            if name.split(".")[0] in ("jax", "jaxlib", "nufftax"):
                return None
            return _find_spec(cls, name, path, target)


        importlib.machinery.PathFinder.find_spec = classmethod(_find_spec_blocking_jax)

        try:
            import jax
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("jax was not blocked")

        import numpy as np

        import autoarray as aa
        from autoarray.inversion.regularization.matern_sparse import MaternSparse

        mask = aa.Mask2D.circular(shape_native=(8, 8), pixel_scales=1.0, radius=3.0)
        grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)
        mapper = aa.Mapper(
            interpolator=aa.mesh.RectangularUniform(shape=(4, 4)).interpolator_from(
                source_plane_data_grid=grid, source_plane_mesh_grid=None
            ),
            regularization=aa.reg.Constant(),
        )

        reg = MaternSparse(coefficient=1.0, scale=1.0)
        H = reg.regularization_matrix_from(mapper)
        assert np.isclose(
            reg.log_det_regularization_matrix_term_from(mapper), np.linalg.slogdet(H)[1]
        )

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
