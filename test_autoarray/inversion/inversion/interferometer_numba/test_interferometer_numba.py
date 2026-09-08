import numpy as np
import pytest
import types

import autoarray as aa

from autoarray import exc
from autoarray.inversion.inversion.factory import _use_interferometer_numba
from autoarray.inversion.inversion.interferometer.sparse import (
    InversionInterferometerSparse,
)
from autoarray.inversion.inversion.interferometer_numba.sparse import (
    InversionInterferometerSparseNumba,
)
from autoarray.inversion.inversion.interferometer_numba import (
    inversion_interferometer_numba_util as numba_util,
)

pytest.importorskip("numba")


def _dataset_from(seed=0, n_visibilities=5):
    """
    The 7x7 / `TransformerDFT` interferometer dataset the sparse-operator parity tests in
    `interferometer/test_interferometer.py::_sparse_parity_setup` use, plus its
    sparse-operator form. Duplicated here rather than imported so the numba tests do not
    depend on a sibling test module's private helper.
    """
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=2.0,
    )

    rng = np.random.default_rng(seed=seed)

    data = aa.Visibilities(
        visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
    )
    noise_map = aa.VisibilitiesNoiseMap(
        visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
    )
    uv_wavelengths = rng.normal(size=(n_visibilities, 2)).astype(np.float64)

    dataset = aa.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    return mask, dataset, dataset.apply_sparse_operator(use_jax=False)


def _delaunay_mapper_from(mask, pixels=9, shape=(3, 3), over_sample_size=1):
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=over_sample_size)

    mesh = aa.mesh.Delaunay(pixels=pixels)
    image_mesh = aa.image_mesh.Overlay(shape=shape)
    image_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=None)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=aa.reg.Constant(coefficient=1.0),
    )


def _rectangular_mapper_from(mask, shape=(3, 3), over_sample_size=1):
    from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
        overlay_grid_from,
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=over_sample_size)

    source_plane_mesh_grid = overlay_grid_from(
        shape_native=shape, grid=grid.over_sampled
    )

    mesh = aa.mesh.RectangularUniform(shape=shape)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(source_plane_mesh_grid),
        adapt_data=aa.Array2D.ones(shape, pixel_scales=0.1),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=aa.reg.Constant(coefficient=1.0),
    )


def _assert_numba_matches_sparse(dataset_sparse, mapper):
    """
    Pin the numba `direct_conv` inversion against the NumPy FFT (`xp=np`) inversion, which
    is the path it replaces, at the repo's exact-parity idiom
    `rtol=1e-10, atol=1e-10 * max|reference|`.
    """
    inversion_sparse = InversionInterferometerSparse(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        xp=np,
    )
    inversion_numba = InversionInterferometerSparseNumba(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        xp=np,
    )

    curvature_matrix = np.asarray(inversion_sparse.curvature_matrix)
    atol = 1.0e-10 * np.abs(curvature_matrix).max()

    np.testing.assert_allclose(
        np.asarray(inversion_numba.curvature_matrix),
        curvature_matrix,
        rtol=1.0e-10,
        atol=atol,
    )

    data_vector = np.asarray(inversion_sparse.data_vector)

    np.testing.assert_allclose(
        np.asarray(inversion_numba.data_vector),
        data_vector,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(data_vector).max(),
    )

    reconstruction = np.asarray(inversion_sparse.reconstruction)

    np.testing.assert_allclose(
        np.asarray(inversion_numba.reconstruction),
        reconstruction,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(reconstruction).max(),
    )

    assert inversion_numba.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_sparse.log_det_curvature_reg_matrix_term, rel=1.0e-10
    )
    assert inversion_numba.log_det_regularization_matrix_term == pytest.approx(
        inversion_sparse.log_det_regularization_matrix_term, rel=1.0e-10
    )

    return inversion_sparse, inversion_numba


def test__numba_inversion__delaunay__matches_sparse_numpy_inversion():
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    _assert_numba_matches_sparse(dataset_sparse=dataset_sparse, mapper=mapper)


def test__numba_inversion__rectangular__matches_sparse_numpy_inversion():
    mask, _, dataset_sparse = _dataset_from(seed=1)
    mapper = _rectangular_mapper_from(mask=mask)

    _assert_numba_matches_sparse(dataset_sparse=dataset_sparse, mapper=mapper)


def test__numba_inversion__control__one_percent_scale_of_curvature_matrix_fails_the_pin():
    """
    The parity pin above is only meaningful if it can fail: a 1% rescaling of `F` — far
    smaller than any real kernel bug — must break it.
    """
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    inversion_sparse, inversion_numba = _assert_numba_matches_sparse(
        dataset_sparse=dataset_sparse, mapper=mapper
    )

    curvature_matrix = np.asarray(inversion_sparse.curvature_matrix)
    atol = 1.0e-10 * np.abs(curvature_matrix).max()

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            1.01 * np.asarray(inversion_numba.curvature_matrix),
            curvature_matrix,
            rtol=1.0e-10,
            atol=atol,
        )


def _kernel_args_from(dataset_sparse, mapper):
    inversion_numba = InversionInterferometerSparseNumba(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        xp=np,
    )

    inputs = inversion_numba.kernel_index_arrays

    preload = np.ascontiguousarray(
        np.asarray(
            dataset_sparse.sparse_operator.nufft_precision_operator, dtype=np.float64
        )
    )

    return (
        preload,
        inputs["iy"],
        inputs["ix"],
        inputs["flat"],
        inputs["indptr"],
        inputs["col"],
        inputs["val"],
        inputs["cscptr"],
        inputs["csc_row"],
        inputs["csc_val"],
        inputs["ny"],
        inputs["nx"],
        inputs["pix_pixels"],
    )


def test__parallel_kernel_equals_serial_kernel():
    """
    Both kernels are called through the module functions directly rather than through the
    `general.yaml -> numba -> parallel` flag, so the comparison does not depend on which
    one the config would have selected.
    """
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    args = _kernel_args_from(dataset_sparse=dataset_sparse, mapper=mapper)

    curvature_matrix_serial = numba_util.curvature_direct_conv(*args)
    curvature_matrix_parallel = numba_util.direct_conv_parallel_kernel()(*args)

    np.testing.assert_allclose(
        curvature_matrix_parallel,
        curvature_matrix_serial,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(curvature_matrix_serial).max(),
    )


def test__kernel_inputs_from__matches_a_hand_built_expectation():
    """
    A three-pixel, two-source-pixel mapper whose CSR / CSC / extent arrays are small
    enough to write out by hand.

    The extent rectangle is 2 x 3, so the mask's flat extent indices `[0, 2, 4]` are rows
    `[0, 0, 1]` and columns `[0, 2, 1]`. The triplet rows are padded to width 2, and the
    second pixel maps to a single source pixel, so its padding entry must be dropped.
    """
    pix_indexes = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int64)
    pix_sizes = np.array([2, 1, 2], dtype=np.int64)
    pix_weights = np.array([[0.25, 0.75], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)

    inputs = numba_util.kernel_inputs_from(
        pix_indexes_for_sub_slim_index=pix_indexes,
        pix_sizes_for_sub_slim_index=pix_sizes,
        pix_weights_for_sub_slim_index=pix_weights,
        extent_index_for_masked_pixel=np.array([0, 2, 4], dtype=np.int64),
        extent_shape=(2, 3),
        pix_pixels=2,
    )

    assert inputs["ny"] == 2
    assert inputs["nx"] == 3
    assert inputs["n_pix"] == 3
    assert inputs["nnz"] == 5
    assert inputs["pix_pixels"] == 2

    assert inputs["iy"].tolist() == [0, 0, 1]
    assert inputs["ix"].tolist() == [0, 2, 1]
    assert inputs["flat"].tolist() == [0, 2, 4]

    # CSR: the padding entry of row 1 is dropped, so `col`/`val` hold five entries.
    assert inputs["indptr"].tolist() == [0, 2, 3, 5]
    assert inputs["col"].tolist() == [0, 1, 1, 0, 1]
    assert inputs["val"].tolist() == [0.25, 0.75, 1.0, 0.5, 0.5]

    # CSC: source 0 is hit by rows 0 and 2, source 1 by rows 0, 1 and 2.
    assert inputs["cscptr"].tolist() == [0, 2, 5]
    assert inputs["csc_row"].tolist() == [0, 2, 0, 1, 2]
    assert inputs["csc_val"].tolist() == [0.25, 0.5, 0.75, 1.0, 0.5]


def test__nnz_per_source_column_from():
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    expected = float(np.asarray(mapper.pix_sizes_for_sub_slim_index).sum()) / float(
        mapper.params
    )

    assert numba_util.nnz_per_source_column_from(mapper=mapper) == pytest.approx(
        expected, 1.0e-12
    )


def test__factory__routes_to_numba_below_the_gate_and_sparse_above_it():
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    nnz_per_source_column = numba_util.nnz_per_source_column_from(mapper=mapper)

    inversion = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=aa.Settings(
            interferometer_numba_nnz_per_source_max=nnz_per_source_column + 1.0
        ),
    )

    assert isinstance(inversion, InversionInterferometerSparseNumba)

    inversion = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=aa.Settings(
            interferometer_numba_nnz_per_source_max=nnz_per_source_column - 1.0
        ),
    )

    assert isinstance(inversion, InversionInterferometerSparse)
    assert not isinstance(inversion, InversionInterferometerSparseNumba)


def test__factory__gate_of_zero_disables_the_numba_path():
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    inversion = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=aa.Settings(interferometer_numba_nnz_per_source_max=0),
    )

    assert isinstance(inversion, InversionInterferometerSparse)
    assert not isinstance(inversion, InversionInterferometerSparseNumba)


def test__routing_predicate__falls_through_on_every_unsupported_configuration():
    """
    The routing predicate is exercised directly, because a routing miss is silent by
    design: it returns the sparse inversion rather than raising, so a test that only
    looked at the returned type could not tell *which* condition rejected it.

    `xp` is checked with a stand-in module rather than `jax.numpy`, so this test carries
    no JAX dependency and runs on the no-JAX CI leg.
    """
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)
    settings = aa.Settings(interferometer_numba_nnz_per_source_max=1.0e6)

    assert _use_interferometer_numba(linear_obj_list=[mapper], settings=settings, xp=np)

    # A non-NumPy array module never routes to numba.
    assert not _use_interferometer_numba(
        linear_obj_list=[mapper],
        settings=settings,
        xp=types.SimpleNamespace(__name__="jax.numpy"),
    )

    # A linear function list has no block in the kernel.
    func_list = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=np.ones((mask.pixels_in_mask, 1)),
    )

    assert not _use_interferometer_numba(
        linear_obj_list=[func_list, mapper], settings=settings, xp=np
    )

    # More than one mapper has no off-diagonal block in the kernel.
    mapper_1 = _delaunay_mapper_from(mask=mask, pixels=4, shape=(2, 2))

    assert not _use_interferometer_numba(
        linear_obj_list=[mapper, mapper_1], settings=settings, xp=np
    )

    # Over-sampling is folded into the sparse weights but not the kernel's.
    mapper_over_sampled = _delaunay_mapper_from(mask=mask, over_sample_size=2)

    assert not _use_interferometer_numba(
        linear_obj_list=[mapper_over_sampled], settings=settings, xp=np
    )

    # The geometry gate.
    assert not _use_interferometer_numba(
        linear_obj_list=[mapper],
        settings=aa.Settings(interferometer_numba_nnz_per_source_max=0.5),
        xp=np,
    )


def test__precondition__non_numpy_array_module_raises():
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    with pytest.raises(exc.InversionException, match="non-NumPy array module"):
        InversionInterferometerSparseNumba(
            dataset=dataset_sparse,
            linear_obj_list=[mapper],
            xp=types.SimpleNamespace(__name__="jax.numpy"),
        )


def test__precondition__linear_func_list_raises():
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    func_list = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=np.ones((mask.pixels_in_mask, 1)),
    )

    with pytest.raises(exc.InversionException, match="linear-function list"):
        InversionInterferometerSparseNumba(
            dataset=dataset_sparse,
            linear_obj_list=[func_list, mapper],
            xp=np,
        )


def test__precondition__multiple_mappers_raise():
    mask, _, dataset_sparse = _dataset_from()
    mapper_0 = _delaunay_mapper_from(mask=mask)
    mapper_1 = _delaunay_mapper_from(mask=mask, pixels=4, shape=(2, 2))

    with pytest.raises(exc.InversionException, match="was passed 2 mappers"):
        InversionInterferometerSparseNumba(
            dataset=dataset_sparse,
            linear_obj_list=[mapper_0, mapper_1],
            xp=np,
        )


def test__precondition__over_sampling_raises():
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask, over_sample_size=2)

    with pytest.raises(exc.InversionException, match="sub_fraction"):
        InversionInterferometerSparseNumba(
            dataset=dataset_sparse,
            linear_obj_list=[mapper],
            xp=np,
        )


def test__kernel_index_arrays__preload_shape_mismatch_raises(monkeypatch):
    """
    The kernel indexes the preload with wrapped (negative) extent offsets, so a preload
    whose shape is not `(2ny, 2nx)` would wrap silently rather than fail. The check names
    both shapes.
    """
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    inversion_numba = InversionInterferometerSparseNumba(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        xp=np,
    )

    preload = np.asarray(dataset_sparse.sparse_operator.nufft_precision_operator)

    monkeypatch.setattr(
        type(dataset_sparse.sparse_operator),
        "nufft_precision_operator",
        property(lambda self: preload[:-2, :]),
        raising=False,
    )

    with pytest.raises(exc.InversionException, match="does not match the sparse"):
        _ = inversion_numba.kernel_index_arrays


def test__existing_sparse_path_is_unchanged_when_the_gate_rejects_the_geometry():
    """
    The default interferometer route must be untouched by the new class: with the gate
    closed, the inversion the factory builds is the sparse one and its matrices are
    identical to those of a directly constructed `InversionInterferometerSparse`.
    """
    mask, _, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    inversion_factory = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=aa.Settings(interferometer_numba_nnz_per_source_max=0),
    )
    inversion_direct = InversionInterferometerSparse(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        xp=np,
    )

    assert type(inversion_factory) is InversionInterferometerSparse

    np.testing.assert_allclose(
        np.asarray(inversion_factory.curvature_matrix),
        np.asarray(inversion_direct.curvature_matrix),
        rtol=1.0e-12,
        atol=0.0,
    )
