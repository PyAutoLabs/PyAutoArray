import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.inversion.factory import _use_interferometer_numba
from autoarray.inversion.inversion.interferometer_numba import (
    inversion_interferometer_numba_util as numba_util,
)
from autoarray.inversion.inversion.interferometer_numba import sparse as numba_sparse
from autoarray.inversion.inversion.interferometer_numba.sparse import (
    InversionInterferometerSparseNumba,
)

pytest.importorskip("numba")

RESTRICTION_KEYS = list(numba_util.SYMMETRIC_KERNEL_KEYS)


def _prep_reference(iy, ix, cscptr, csc_row, ny, nx, pix_pixels):
    """
    The straightforward (Python-loop) set-up of the PyAutoArray#635 prototype, ranking
    columns by their top row `ymin` (stable, empty columns last), kept as the oracle for
    the vectorised `symmetric_restriction_inputs_from`.
    """
    S = pix_pixels
    ymin_by_column = np.array(
        [
            (
                iy[csc_row[cscptr[s] : cscptr[s + 1]]].min()
                if cscptr[s + 1] > cscptr[s]
                else ny
            )
            for s in range(S)
        ]
    )
    order = np.argsort(ymin_by_column, kind="stable").astype(np.int64)
    rank = np.empty(S, np.int64)
    rank[order] = np.arange(S)
    ymin = ymin_by_column[order]
    ylo_suffix = np.minimum.accumulate(ymin[::-1])[::-1].astype(np.int64)

    # In ymin order the suffix minimum is each column's own top row.
    assert np.array_equal(ylo_suffix, ymin)
    xlo = np.full(ny, nx, np.int64)
    xhi = np.zeros(ny, np.int64)
    for y, x in zip(iy, ix):
        xlo[y] = min(xlo[y], x)
        xhi[y] = max(xhi[y], x + 1)

    width = np.maximum(xhi - xlo, 0)
    cum = np.concatenate([np.cumsum(width[::-1])[::-1], [0]])
    nnz_col = np.diff(cscptr)[order]
    work_fraction = float((nnz_col * cum[ylo_suffix]).sum() / (nnz_col.sum() * ny * nx))

    return [order, rank, ylo_suffix, xlo, xhi], work_fraction


def _even_preload_from(ny, nx, seed=1):
    """
    A random `(2ny, 2nx)` operator that is even in the wrapped offset, as `W~` is, so
    `F = Aᵀ W~ A` is symmetric (to round-off) and the mirror pass is valid.
    """
    P = np.random.default_rng(seed).standard_normal((2 * ny, 2 * nx))
    return np.ascontiguousarray(0.5 * (P + np.roll(P[::-1, ::-1], 1, axis=(0, 1))))


def _mask_from(kind):
    if kind == "square":
        ny = nx = 9
        m = np.ones((ny, nx), dtype=bool)
    elif kind == "circular":
        ny = nx = 15
        yy, xx = np.mgrid[:ny, :nx]
        m = (yy - 7) ** 2 + (xx - 7) ** 2 <= 7.5**2
    elif kind == "irregular":
        # Two offset blobs with ragged rows and two empty rows between them.
        ny, nx = 14, 12
        rng = np.random.default_rng(5)
        m = np.zeros((ny, nx), dtype=bool)
        m[0:6, 0:7] = rng.random((6, 7)) < 0.7
        m[8:14, 4:12] = rng.random((6, 8)) < 0.7
        m[0, 0] = m[13, 11] = True
    else:
        raise ValueError(kind)

    yy, xx = np.mgrid[:ny, :nx]
    iy = yy[m].astype(np.int64)
    ix = xx[m].astype(np.int64)

    return iy, ix, ny, nx


def _triplets_from(kind, iy, ix, ny, nx, seed=0):
    """
    Mapper-like `[N_pix, P_max]` triplets: 3 nearest of a random vertex set (Delaunay-like)
    or the 4 corners of a coarse grid cell (rectangular-like), with a few extra source
    pixels no masked pixel maps to (empty columns), shuffled so the column index order is
    not the ranked (`ymin`) order.
    """
    rng = np.random.default_rng(seed)
    n_pix = iy.shape[0]
    points = np.c_[iy, ix].astype(float)

    if kind == "delaunay":
        S_used = 17
        vertices = np.c_[rng.uniform(0, ny, S_used), rng.uniform(0, nx, S_used)]
        d = ((points[:, None, :] - vertices[None, :, :]) ** 2).sum(-1)
        indexes = np.argsort(d, axis=1)[:, :3]
        sizes = np.full(n_pix, 3)
        # Ragged rows: some pixels map to 1 or 2 vertices only (padded with junk).
        sizes[::5] = 2
        sizes[::7] = 1
    elif kind == "rectangular":
        cell = 3
        gy, gx = ny // cell + 2, nx // cell + 2
        S_used = gy * gx
        cy, cx = iy // cell, ix // cell
        indexes = np.c_[
            cy * gx + cx,
            cy * gx + cx + 1,
            (cy + 1) * gx + cx,
            (cy + 1) * gx + cx + 1,
        ]
        sizes = np.full(n_pix, 4)
    else:
        raise ValueError(kind)

    weights = rng.random(indexes.shape)
    weights /= weights.sum(axis=1, keepdims=True)

    S = S_used + 3
    relabel = rng.permutation(S)
    indexes = relabel[indexes]
    # Padding entries beyond `sizes` hold junk, as real mapper rows may.
    pad = np.arange(indexes.shape[1])[None, :] >= sizes[:, None]
    indexes = np.where(pad, S - 1, indexes)

    return indexes, sizes, weights, S


def _kernel_args_from(mask_kind, triplet_kind, seed=0):
    iy, ix, ny, nx = _mask_from(mask_kind)
    indexes, sizes, weights, S = _triplets_from(triplet_kind, iy, ix, ny, nx, seed)

    inputs = numba_util.kernel_inputs_from(
        pix_indexes_for_sub_slim_index=indexes,
        pix_sizes_for_sub_slim_index=sizes,
        pix_weights_for_sub_slim_index=weights,
        extent_index_for_masked_pixel=iy * nx + ix,
        extent_shape=(ny, nx),
        pix_pixels=S,
    )

    args = [_even_preload_from(ny, nx)] + [
        inputs[k]
        for k in [
            "iy",
            "ix",
            "flat",
            "indptr",
            "col",
            "val",
            "cscptr",
            "csc_row",
            "csc_val",
            "ny",
            "nx",
            "pix_pixels",
        ]
    ]

    restriction = numba_util.symmetric_restriction_inputs_from(
        iy=inputs["iy"],
        ix=inputs["ix"],
        cscptr=inputs["cscptr"],
        csc_row=inputs["csc_row"],
        ny=ny,
        nx=nx,
        pix_pixels=S,
    )

    return args, inputs, restriction, (indexes, sizes, iy * nx + ix, (ny, nx), S)


GEOMETRIES = [
    ("square", "delaunay"),
    ("square", "rectangular"),
    ("circular", "delaunay"),
    ("circular", "rectangular"),
    ("irregular", "delaunay"),
    ("irregular", "rectangular"),
]


@pytest.mark.parametrize("mask_kind, triplet_kind", GEOMETRIES)
def test__symmetric_kernel__matches_full_kernel(mask_kind, triplet_kind):
    args, inputs, restriction, _ = _kernel_args_from(mask_kind, triplet_kind)

    assert np.any(np.diff(inputs["cscptr"]) == 0)  # some empty source columns

    F_full = numba_util.curvature_direct_conv(*args)
    F_sym = numba_util.curvature_direct_conv_symmetric(
        *args, *[restriction[k] for k in RESTRICTION_KEYS]
    )

    # The entries the kernel computes (rank(col) >= rank(row)) are bit-identical to the
    # full kernel's; the mirrored ones equal it to the full kernel's own asymmetry.
    rank = restriction["rank"]
    computed = rank[None, :] >= rank[:, None]

    assert np.array_equal(F_sym[computed], F_full[computed])

    scale = np.abs(F_full).max()

    assert np.abs(F_sym - F_full).max() <= 1.0e-14 * scale
    assert np.array_equal(F_sym, F_sym.T)


def test__symmetric_kernel__computed_entries_bit_identical_for_shuffled_pixel_order():
    """
    The column-major gather adds a column's non-zeros in increasing pixel index, as the
    full kernel's CSR sweep does, so exactness does not need row-major pixels.
    """
    iy, ix, ny, nx = _mask_from("irregular")
    perm = np.random.default_rng(7).permutation(iy.shape[0])
    iy, ix = iy[perm], ix[perm]
    indexes, sizes, weights, S = _triplets_from("delaunay", iy, ix, ny, nx)

    inputs = numba_util.kernel_inputs_from(
        pix_indexes_for_sub_slim_index=indexes,
        pix_sizes_for_sub_slim_index=sizes,
        pix_weights_for_sub_slim_index=weights,
        extent_index_for_masked_pixel=iy * nx + ix,
        extent_shape=(ny, nx),
        pix_pixels=S,
    )
    args = [_even_preload_from(ny, nx)] + [
        inputs[k]
        for k in [
            "iy",
            "ix",
            "flat",
            "indptr",
            "col",
            "val",
            "cscptr",
            "csc_row",
            "csc_val",
            "ny",
            "nx",
            "pix_pixels",
        ]
    ]
    restriction = numba_util.symmetric_restriction_inputs_from(
        inputs["iy"],
        inputs["ix"],
        inputs["cscptr"],
        inputs["csc_row"],
        ny,
        nx,
        S,
    )

    F_full = numba_util.curvature_direct_conv(*args)
    F_sym = numba_util.curvature_direct_conv_symmetric(
        *args, *[restriction[k] for k in RESTRICTION_KEYS]
    )

    rank = restriction["rank"]
    computed = rank[None, :] >= rank[:, None]

    assert np.array_equal(F_sym[computed], F_full[computed])
    assert np.abs(F_sym - F_full).max() <= 1.0e-14 * np.abs(F_full).max()


def test__symmetric_kernel__upper_triangle_bit_identical_when_columns_are_in_ymin_order():
    """
    With the source columns already in `ymin` order the ranked triangle is the index
    upper triangle, so `np.triu` of the two kernels is bit-identical.
    """
    args, inputs, restriction, triplets = _kernel_args_from("circular", "delaunay")

    indexes, sizes, flat, extent_shape, S = triplets
    relabel = np.empty(S, dtype=np.int64)
    relabel[restriction["order"]] = np.arange(S)

    rng = np.random.default_rng(0)
    weights = rng.random(indexes.shape)

    inputs = numba_util.kernel_inputs_from(
        pix_indexes_for_sub_slim_index=relabel[indexes],
        pix_sizes_for_sub_slim_index=sizes,
        pix_weights_for_sub_slim_index=weights,
        extent_index_for_masked_pixel=flat,
        extent_shape=extent_shape,
        pix_pixels=S,
    )
    args = [args[0]] + [
        inputs[k]
        for k in [
            "iy",
            "ix",
            "flat",
            "indptr",
            "col",
            "val",
            "cscptr",
            "csc_row",
            "csc_val",
            "ny",
            "nx",
            "pix_pixels",
        ]
    ]
    restriction = numba_util.symmetric_restriction_inputs_from(
        inputs["iy"],
        inputs["ix"],
        inputs["cscptr"],
        inputs["csc_row"],
        inputs["ny"],
        inputs["nx"],
        S,
    )

    assert np.array_equal(restriction["order"], np.arange(S))

    F_full = numba_util.curvature_direct_conv(*args)
    F_sym = numba_util.curvature_direct_conv_symmetric(
        *args, *[restriction[k] for k in RESTRICTION_KEYS]
    )

    assert np.array_equal(np.triu(F_sym), np.triu(F_full))


def test__symmetric_kernel__control__wrong_restriction_fails():
    """
    Control: dropping the first extent row from the restriction of the first-ranked
    column must change `F`, so the equality tests above can fail.
    """
    args, _, restriction, _ = _kernel_args_from("circular", "delaunay")

    ylo_suffix = restriction["ylo_suffix"].copy()
    ylo_suffix[0] += 1
    extra = [restriction[k] for k in RESTRICTION_KEYS]
    extra[2] = ylo_suffix

    F_full = numba_util.curvature_direct_conv(*args)
    F_bad = numba_util.curvature_direct_conv_symmetric(*args, *extra)

    assert np.abs(F_bad - F_full).max() > 1.0e-6 * np.abs(F_full).max()


@pytest.mark.parametrize("mask_kind, triplet_kind", GEOMETRIES)
def test__symmetric_restriction_inputs_from__matches_reference(mask_kind, triplet_kind):
    _, inputs, restriction, triplets = _kernel_args_from(mask_kind, triplet_kind)

    reference, work_fraction = _prep_reference(
        inputs["iy"],
        inputs["ix"],
        inputs["cscptr"],
        inputs["csc_row"],
        inputs["ny"],
        inputs["nx"],
        inputs["pix_pixels"],
    )

    assert len(reference) == len(RESTRICTION_KEYS)

    for key, expected in zip(RESTRICTION_KEYS, reference):
        assert restriction[key].dtype == np.int64
        assert np.array_equal(restriction[key], expected), key

    assert restriction["work_fraction"] == pytest.approx(work_fraction, rel=1.0e-14)

    indexes, sizes, flat, extent_shape, S = triplets

    assert (
        numba_util.symmetric_work_fraction_from(
            pix_indexes_for_sub_slim_index=indexes,
            pix_sizes_for_sub_slim_index=sizes,
            extent_index_for_masked_pixel=flat,
            extent_shape=extent_shape,
            pix_pixels=S,
        )
        == restriction["work_fraction"]
    )


def test__work_fraction__hand_checked_geometry():
    """
    Extent 4x4. Masked pixels: row 0 x = 0, 1; row 1 none; row 2 x = 1, 2, 3; row 3 x = 2.
    Column 0 maps pixel (3, 2); column 1 maps (0, 0) and (2, 1); column 2 is empty.

    Top rows (ymin) 3 / 0 / empty -> order [1, 0, 2]; ylo_suffix = [0, 3, 4]. Row widths
    2 / 0 / 3 / 1, so the rows from 0 sum to 6 and from 3 to 1:
    work_fraction = (2 * 6 + 1 * 1 + 0) / (3 * 16) = 13 / 48.
    """
    iy = np.array([0, 0, 2, 2, 2, 3])
    ix = np.array([0, 1, 1, 2, 3, 2])
    # Pixel 0 -> column 1, pixel 2 -> column 1, pixel 5 -> column 0.
    indexes = np.array([[1], [0], [1], [0], [0], [0]])
    sizes = np.array([1, 0, 1, 0, 0, 1])
    weights = np.ones((6, 1))

    inputs = numba_util.kernel_inputs_from(
        pix_indexes_for_sub_slim_index=indexes,
        pix_sizes_for_sub_slim_index=sizes,
        pix_weights_for_sub_slim_index=weights,
        extent_index_for_masked_pixel=iy * 4 + ix,
        extent_shape=(4, 4),
        pix_pixels=3,
    )

    restriction = numba_util.symmetric_restriction_inputs_from(
        inputs["iy"], inputs["ix"], inputs["cscptr"], inputs["csc_row"], 4, 4, 3
    )

    assert restriction["order"].tolist() == [1, 0, 2]
    assert restriction["rank"].tolist() == [1, 0, 2]
    assert restriction["ylo_suffix"].tolist() == [0, 3, 4]
    assert restriction["xlo_row"].tolist() == [0, 4, 1, 2]
    assert restriction["xhi_row"].tolist() == [2, 0, 4, 3]
    assert restriction["work_fraction"] == pytest.approx(13.0 / 48.0, rel=1.0e-15)

    args = [_even_preload_from(4, 4)] + [
        inputs[k]
        for k in [
            "iy",
            "ix",
            "flat",
            "indptr",
            "col",
            "val",
            "cscptr",
            "csc_row",
            "csc_val",
            "ny",
            "nx",
            "pix_pixels",
        ]
    ]

    F_full = numba_util.curvature_direct_conv(*args)
    F_sym = numba_util.curvature_direct_conv_symmetric(
        *args, *[restriction[k] for k in RESTRICTION_KEYS]
    )

    assert np.abs(F_sym - F_full).max() <= 1.0e-14 * np.abs(F_full).max()


def test__curvature_route_seconds_from__work_fraction():
    extent_shape = (341, 341)

    t_fft, t_dc = numba_util.curvature_route_seconds_from(
        source_pixels=1459, nnz=273661, extent_shape=extent_shape
    )

    assert numba_util.curvature_route_seconds_from(
        source_pixels=1459, nnz=273661, extent_shape=extent_shape, work_fraction=None
    ) == (t_fft, t_dc)

    # Aris 512 S = 1,459 measured FFT 28.3 s vs full direct_conv 38.4 s: the full kernel is
    # predicted slower ...
    assert t_dc > t_fft

    # ... and the symmetric kernel, at the work fraction of Aris's real 512 Delaunay mesh
    # (0.511, multiply imaged tall columns; the synthetic geometry gives 0.414), faster.
    t_fft_sym, t_dc_sym = numba_util.curvature_route_seconds_from(
        source_pixels=1459,
        nnz=273661,
        extent_shape=extent_shape,
        work_fraction=0.511,
    )

    assert t_fft_sym == t_fft
    assert t_dc_sym == pytest.approx(
        t_dc * numba_util.DIRECT_CONV_SYMMETRIC_WORK_FACTOR * 0.511, rel=1.0e-12
    )
    assert t_dc_sym < t_fft

    # The factor never makes the prediction slower than the full kernel.
    assert numba_util.curvature_route_seconds_from(
        source_pixels=1459, nnz=273661, extent_shape=extent_shape, work_fraction=0.9
    ) == (t_fft, t_dc)


def test__settings__symmetric_max_work_fraction():
    assert aa.Settings().interferometer_direct_conv_symmetric_max_work_fraction == 0.7
    assert (
        aa.Settings(
            interferometer_direct_conv_symmetric_max_work_fraction=0
        ).interferometer_direct_conv_symmetric_max_work_fraction
        == 0.0
    )
    assert (
        aa.Settings(
            interferometer_direct_conv_symmetric_max_work_fraction=0.7
        ).interferometer_direct_conv_symmetric_max_work_fraction
        == 0.7
    )

    with pytest.raises(ValueError):
        aa.Settings(interferometer_direct_conv_symmetric_max_work_fraction=-0.1)


def _dataset_from(seed=0, n_visibilities=7):
    """
    A 13x13 interferometer dataset with a circular mask (extent 9x9) and its
    sparse-operator form, large enough for a Delaunay mesh to give a work fraction below
    one. Built here rather than imported, as in `test_interferometer_numba.py`.
    """
    yy, xx = np.mgrid[:13, :13]
    mask = aa.Mask2D(
        mask=(yy - 6) ** 2 + (xx - 6) ** 2 > 4.5**2,
        pixel_scales=1.0,
    )

    rng = np.random.default_rng(seed=seed)

    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=rng.normal(size=(n_visibilities, 2))),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((n_visibilities, 2))),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    return mask, dataset.apply_sparse_operator(use_jax=False)


def _delaunay_mapper_from(mask, pixels=16, seed=3):
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)

    mesh = aa.mesh.Delaunay(pixels=pixels)

    rng = np.random.default_rng(seed=seed)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(
            rng.uniform(-4.5, 4.5, size=(pixels, 2))
        ),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=aa.reg.Constant(coefficient=1.0),
    )


def _log_evidence_terms_from(inversion):
    return (
        float(inversion.fast_chi_squared)
        + float(inversion.regularization_term)
        + float(inversion.log_det_curvature_reg_matrix_term)
        - float(inversion.log_det_regularization_matrix_term)
    )


def test__inversion__gate_selects_the_kernel_and_leaves_the_likelihood_unchanged(
    monkeypatch,
):
    monkeypatch.setattr(numba_sparse, "_numba_parallel", lambda: False)

    calls = {"symmetric": 0}
    symmetric = numba_util.curvature_direct_conv_symmetric

    def spy(*args):
        calls["symmetric"] += 1
        return symmetric(*args)

    monkeypatch.setattr(numba_util, "curvature_direct_conv_symmetric", spy)

    mask, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    def inversion_from(threshold):
        return InversionInterferometerSparseNumba(
            dataset=dataset_sparse,
            linear_obj_list=[mapper],
            settings=aa.Settings(
                interferometer_direct_conv_symmetric_max_work_fraction=threshold
            ),
            xp=np,
        )

    inversion_off = inversion_from(threshold=0.0)
    work_fraction = inversion_off.symmetric_restriction_inputs["work_fraction"]

    assert 0.0 < work_fraction < 1.0
    assert not inversion_off.use_symmetric_kernel

    curvature_off = np.asarray(inversion_off.curvature_matrix)
    evidence_off = _log_evidence_terms_from(inversion_off)

    assert calls["symmetric"] == 0

    inversion_below = inversion_from(threshold=0.999 * work_fraction)
    assert not inversion_below.use_symmetric_kernel
    np.asarray(inversion_below.curvature_matrix)
    assert calls["symmetric"] == 0

    inversion_on = inversion_from(threshold=work_fraction)
    assert inversion_on.use_symmetric_kernel

    curvature_on = np.asarray(inversion_on.curvature_matrix)
    evidence_on = _log_evidence_terms_from(inversion_on)

    assert calls["symmetric"] == 1

    np.testing.assert_allclose(
        curvature_on,
        curvature_off,
        rtol=0.0,
        atol=1.0e-13 * np.abs(curvature_off).max(),
    )
    assert evidence_on == pytest.approx(evidence_off, rel=1.0e-12)

    # The parallel path always uses the full parallel kernel.
    monkeypatch.setattr(numba_sparse, "_numba_parallel", lambda: True)
    assert not inversion_from(threshold=1.0).use_symmetric_kernel


def test__factory__passes_the_work_fraction_to_the_cost_model(monkeypatch):
    mask, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    seen = []
    route_seconds = numba_util.curvature_route_seconds_from

    def spy(**kwargs):
        seen.append(kwargs.get("work_fraction"))
        return route_seconds(**kwargs)

    monkeypatch.setattr(numba_util, "curvature_route_seconds_from", spy)
    monkeypatch.setattr(numba_sparse, "_numba_parallel", lambda: False)

    work_fraction = InversionInterferometerSparseNumba(
        dataset=dataset_sparse, linear_obj_list=[mapper], xp=np
    ).symmetric_restriction_inputs["work_fraction"]

    extent_shape = (
        dataset_sparse.sparse_operator.y_shape,
        dataset_sparse.sparse_operator.x_shape,
    )

    def route(threshold, real_space_mask=mask):
        _use_interferometer_numba(
            linear_obj_list=[mapper],
            settings=aa.Settings(
                interferometer_direct_conv_symmetric_max_work_fraction=threshold
            ),
            xp=np,
            extent_shape=extent_shape,
            real_space_mask=real_space_mask,
        )
        return seen[-1]

    assert route(threshold=1.0) == work_fraction
    assert route(threshold=work_fraction) == work_fraction
    assert route(threshold=0.999 * work_fraction) is None
    assert route(threshold=0.0) is None
    assert route(threshold=1.0, real_space_mask=None) is None

    monkeypatch.setattr(numba_sparse, "_numba_parallel", lambda: True)
    assert route(threshold=1.0) is None


def test__inversion__non_finite_weights_use_the_full_kernel(monkeypatch):
    """
    NaN mapping weights must give the full kernel's NaN pattern (and so a NaN evidence),
    which the symmetric kernel does not reproduce on empty source columns' rows.
    """
    monkeypatch.setattr(numba_sparse, "_numba_parallel", lambda: False)

    mask, dataset_sparse = _dataset_from()
    mapper = _delaunay_mapper_from(mask=mask)

    inversion = InversionInterferometerSparseNumba(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=aa.Settings(
            interferometer_direct_conv_symmetric_max_work_fraction=1.0
        ),
        xp=np,
    )

    assert inversion.use_symmetric_kernel

    inversion.kernel_index_arrays["val"][0] = np.nan

    assert not inversion.use_symmetric_kernel
