"""
Which positive-only solver `AbstractInversion.reconstruction` dispatches to (PyAutoArray#566).

`Settings.positive_only_solver = "certified"` selects the certified active-set solve only on the JAX
backend for mapper-only inversions; inversions with linear light-profile / MGE coefficients
(`AbstractLinearObjFuncList`) and every NumPy inversion keep today's solver.
"""

import importlib.util

import numpy as np
import pytest

import autoarray as aa


requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


def _system(n, seed=3):
    """An SPD system whose unconstrained solution has negative entries, so positivity binds."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3 * n, n))
    b = rng.normal(size=3 * n)
    curvature_reg_matrix = A.T @ A
    data_vector = A.T @ b
    assert (np.linalg.solve(curvature_reg_matrix, data_vector) < 0.0).any()
    return data_vector, curvature_reg_matrix


def _mapper(pixels=16):
    return aa.m.MockMapper(
        mesh=aa.mesh.RectangularUniform(shape=(4, 4)),
        parameters=pixels,
        regularization=aa.reg.Constant(),
        # Only its shape is read, by the NumPy warm-start memo's fingerprint.
        source_plane_mesh_grid=np.zeros((pixels, 2)),
    )


def _inversion(
    linear_obj_list,
    data_vector,
    curvature_reg_matrix,
    use_jax,
    positive_only_solver="certified",
    use_edge_zeroed_pixels=False,
):
    if use_jax:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        data_vector = jnp.asarray(data_vector)
        curvature_reg_matrix = jnp.asarray(curvature_reg_matrix)

    inversion = aa.m.MockInversion(
        linear_obj_list=linear_obj_list,
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=aa.Settings(
            use_positive_only_solver=True,
            use_edge_zeroed_pixels=use_edge_zeroed_pixels,
            positive_only_solver=positive_only_solver,
            # Off so two NumPy solves of the same system are bit-comparable: with the memo on, the second
            # solve starts from the first's passive set and can differ in the last ulp.
            nnls_warm_start_memo=False,
        ),
    )
    # `MockInversion` does not take `xp`; the backend flag is what `AbstractInversion(xp=jnp)` sets.
    inversion.use_jax = use_jax
    return inversion


def test__positive_only_solver_used__default_is_pdip():
    data_vector, curvature_reg_matrix = _system(16)

    inversion = _inversion(
        [_mapper()],
        data_vector,
        curvature_reg_matrix,
        use_jax=False,
        positive_only_solver=None,
    )

    assert inversion.settings.positive_only_solver == "pdip"
    assert inversion.positive_only_solver_used == "pdip"


def test__positive_only_solver_used__numpy_backend_keeps_pdip_and_fnnls():
    data_vector, curvature_reg_matrix = _system(16)

    certified = _inversion(
        [_mapper()], data_vector, curvature_reg_matrix, use_jax=False
    )
    default = _inversion(
        [_mapper()],
        data_vector,
        curvature_reg_matrix,
        use_jax=False,
        positive_only_solver="pdip",
    )

    assert certified.positive_only_solver_used == "pdip"
    assert np.array_equal(certified.reconstruction, default.reconstruction)


@requires_jax
def test__positive_only_solver_used__mapper_only_jax_is_certified_and_matches_pdip():
    data_vector, curvature_reg_matrix = _system(16)

    certified = _inversion([_mapper()], data_vector, curvature_reg_matrix, use_jax=True)
    pdip = _inversion(
        [_mapper()],
        data_vector,
        curvature_reg_matrix,
        use_jax=True,
        positive_only_solver="pdip",
    )

    assert certified.positive_only_solver_used == "certified"
    assert pdip.positive_only_solver_used == "pdip"

    x_certified = np.asarray(certified.reconstruction)
    x_pdip = np.asarray(pdip.reconstruction)

    assert (x_certified >= 0.0).all()
    assert (x_certified == 0.0).any()
    assert x_certified == pytest.approx(
        x_pdip, rel=1.0e-8, abs=1.0e-8 * np.max(np.abs(x_certified))
    )


@requires_jax
def test__positive_only_solver_used__linear_func_list_present_keeps_pdip():
    data_vector, curvature_reg_matrix = _system(17)

    func_list = aa.m.MockLinearObjFuncList(parameters=1)

    certified_setting = _inversion(
        [func_list, _mapper()], data_vector, curvature_reg_matrix, use_jax=True
    )
    pdip = _inversion(
        [func_list, _mapper()],
        data_vector,
        curvature_reg_matrix,
        use_jax=True,
        positive_only_solver="pdip",
    )

    assert certified_setting.positive_only_solver_used == "pdip"
    assert np.array_equal(
        np.asarray(certified_setting.reconstruction), np.asarray(pdip.reconstruction)
    )


@requires_jax
def test__positive_only_solver_used__no_mapper_keeps_pdip():
    data_vector, curvature_reg_matrix = _system(4)

    inversion = _inversion(
        [aa.m.MockLinearObjFuncList(parameters=4)],
        data_vector,
        curvature_reg_matrix,
        use_jax=True,
    )

    assert inversion.positive_only_solver_used == "pdip"


@requires_jax
def test__positive_only_solver_used__edge_zeroed_subset_is_preserved():
    # The 4x4 rectangular mesh's 12 edge pixels are zeroed and the 4 interior pixels [5, 6, 9, 10] solved.
    data_vector, curvature_reg_matrix = _system(16)

    certified = _inversion(
        [_mapper()],
        data_vector,
        curvature_reg_matrix,
        use_jax=True,
        use_edge_zeroed_pixels=True,
    )
    pdip = _inversion(
        [_mapper()],
        data_vector,
        curvature_reg_matrix,
        use_jax=True,
        positive_only_solver="pdip",
        use_edge_zeroed_pixels=True,
    )

    assert certified.positive_only_solver_used == "certified"
    assert np.asarray(certified.solve_ids_to_keep) == pytest.approx(
        np.array([5, 6, 9, 10])
    )

    x_certified = np.asarray(certified.reconstruction)
    x_pdip = np.asarray(pdip.reconstruction)

    keep = np.array([5, 6, 9, 10])
    edge = np.setdiff1d(np.arange(16), keep)

    assert (x_certified[edge] == 0.0).all()
    assert (x_pdip[edge] == 0.0).all()
    assert x_certified == pytest.approx(
        x_pdip, rel=1.0e-8, abs=1.0e-8 * np.max(np.abs(x_certified))
    )
