"""
Tests of `InversionInterferometerMatrixFree`, the fit-selectable matrix-free interferometer inversion
(`Settings(interferometer_route="matrix_free")`, PyAutoArray #657, phase 5a of #638).

Every fit quantity is held against the FFT sparse route (`InversionInterferometerSparse`) on the
`MaternSparse` rectangular-mesh cases of `test_matrix_free_util.py`, whose fixture helpers are reused.
Like that module, these tests carry no module-level JAX dependency, so the `unittest-nojax` leg runs
them.
"""

import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray.fit import fit_util
from autoarray.inversion.inversion.factory import inversion_interferometer_from

from test_autoarray.inversion.inversion.interferometer.test_matrix_free_util import (
    _dataset_sparse_from,
    _delaunay_mapper_from,
    _rectangular_mapper_from,
)


def _settings_from(route, use_positive_only_solver=False):
    return aa.Settings(
        interferometer_route=route,
        use_positive_only_solver=use_positive_only_solver,
        use_edge_zeroed_pixels=False,
    )


def _matern_case_from(coefficient=1.0, mesh_shape=(6, 6)):
    """
    The util tests' data-dominated dataset (500 visibilities on long baselines) and a `MaternSparse`
    rectangular mapper (over-sampled, `over_sample_size=2`).
    """
    mask, dataset_sparse = _dataset_sparse_from(n_visibilities=500, uv_scale=1.0e6)

    mapper = _rectangular_mapper_from(
        mask=mask,
        regularization=aa.reg.MaternSparse(coefficient=coefficient, scale=1.0),
        shape=mesh_shape,
    )

    return mask, dataset_sparse, mapper


def _inversion_pair_from(
    dataset_sparse, mapper, use_positive_only_solver=False, linear_obj_list=None
):
    linear_obj_list = linear_obj_list or [mapper]

    inversion_matrix_free = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=linear_obj_list,
        settings=_settings_from("matrix_free", use_positive_only_solver),
    )
    inversion_fft = aa.Inversion(
        dataset=dataset_sparse,
        linear_obj_list=linear_obj_list,
        settings=_settings_from("fft", use_positive_only_solver),
    )

    return inversion_matrix_free, inversion_fft


def _assert_relative_close(value, reference, rtol):
    value = np.asarray(value)
    reference = np.asarray(reference)

    assert np.max(np.abs(value - reference)) <= rtol * np.max(np.abs(reference))


def test__factory__routes_by_interferometer_route():
    _, dataset_sparse, mapper = _matern_case_from()

    inversion = inversion_interferometer_from(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=_settings_from("matrix_free"),
    )

    assert isinstance(inversion, aa.InversionInterferometerMatrixFree)

    inversion = inversion_interferometer_from(
        dataset=dataset_sparse,
        linear_obj_list=[mapper],
        settings=_settings_from("fft"),
    )

    assert type(inversion) is aa.InversionInterferometerSparse

    inversion_matrix_free, _ = _inversion_pair_from(
        dataset_sparse=dataset_sparse, mapper=mapper
    )

    assert isinstance(inversion_matrix_free, aa.InversionInterferometerMatrixFree)


def test__factory__matrix_free_without_sparse_operator__raises():
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=0.5, radius=2.5)

    rng = np.random.default_rng(seed=1)

    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=rng.normal(size=(20, 2))),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((20, 2))),
        uv_wavelengths=rng.normal(size=(20, 2)) * 1.0e4,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    mapper = _rectangular_mapper_from(
        mask=mask, regularization=aa.reg.MaternSparse(coefficient=1.0, scale=1.0)
    )

    with pytest.raises(exc.InversionException, match="sparse_operator"):
        aa.Inversion(
            dataset=dataset,
            linear_obj_list=[mapper],
            settings=_settings_from("matrix_free"),
        )


@pytest.mark.parametrize("coefficient", [1.0, 0.01])
def test__fit_quantities__match_fft_route(coefficient):
    """
    The reconstruction (PCG, rtol 1e-10) matches the dense solve to ~1e-11 of the max; the
    chi-squared, regularization term and exact `ln det H` to round-off; `ln det(F + H)` is the SLQ
    estimate, within its documented 0.5% tolerance, and so is the assembled evidence (the tolerance
    logic of `test__log_evidence_matrix_free_from__matches_dense_sparse_path_evidence`).
    """
    _, dataset_sparse, mapper = _matern_case_from(coefficient=coefficient)

    inversion_matrix_free, inversion_fft = _inversion_pair_from(
        dataset_sparse=dataset_sparse, mapper=mapper
    )

    reconstruction = inversion_matrix_free.reconstruction

    assert isinstance(reconstruction, np.ndarray)
    assert reconstruction.shape == (inversion_matrix_free.operator.params,)

    _assert_relative_close(reconstruction, inversion_fft.reconstruction, rtol=1.0e-6)

    _assert_relative_close(
        inversion_matrix_free.data_vector, inversion_fft.data_vector, rtol=1.0e-10
    )

    for name in [
        "regularization_term",
        "fast_chi_squared",
        "log_det_regularization_matrix_term",
    ]:
        assert float(getattr(inversion_matrix_free, name)) == pytest.approx(
            float(getattr(inversion_fft, name)), rel=1.0e-8
        ), name

    log_det_curvature_reg_dense = float(inversion_fft.log_det_curvature_reg_matrix_term)

    assert inversion_matrix_free.log_det_curvature_reg_matrix_term == pytest.approx(
        log_det_curvature_reg_dense, rel=0.005
    )

    noise_normalization = dataset_sparse.sparse_operator.noise_normalization

    def log_evidence_from(inversion):
        return fit_util.log_evidence_from(
            chi_squared=inversion.fast_chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=noise_normalization,
        )

    log_evidence_dense = log_evidence_from(inversion_fft)
    log_evidence_matrix_free = log_evidence_from(inversion_matrix_free)

    assert abs(log_evidence_matrix_free - log_evidence_dense) <= 0.5 * 0.005 * abs(
        log_det_curvature_reg_dense
    )


def test__log_det_curvature_reg_matrix_term__is_deterministic():
    _, dataset_sparse, mapper = _matern_case_from()

    inversion_0, _ = _inversion_pair_from(dataset_sparse=dataset_sparse, mapper=mapper)
    inversion_1, _ = _inversion_pair_from(dataset_sparse=dataset_sparse, mapper=mapper)

    assert (
        inversion_0.log_det_curvature_reg_matrix_term
        == inversion_1.log_det_curvature_reg_matrix_term
    )


def test__mapped_reconstructed_data_and_image__match_fft_route():
    _, dataset_sparse, mapper = _matern_case_from()

    inversion_matrix_free, inversion_fft = _inversion_pair_from(
        dataset_sparse=dataset_sparse, mapper=mapper
    )

    _assert_relative_close(
        inversion_matrix_free.mapped_reconstructed_data.array,
        inversion_fft.mapped_reconstructed_data.array,
        rtol=1.0e-6,
    )
    _assert_relative_close(
        inversion_matrix_free.mapped_reconstructed_operated_data.array,
        inversion_fft.mapped_reconstructed_operated_data.array,
        rtol=1.0e-6,
    )
    _assert_relative_close(
        inversion_matrix_free.reconstruction_dict[mapper],
        inversion_fft.reconstruction_dict[mapper],
        rtol=1.0e-6,
    )


@pytest.mark.parametrize("coefficient", [0.01, 1.0e4])
def test__positive_only__matches_fft_route_positive_only(coefficient):
    """
    The setup of `test__reconstruction_positive_via_interior_point__matches_sparse_path_positive_only`
    (10 x 10 `MaternSparse` mesh, edge-zeroing off), through the factory.
    """
    _, dataset_sparse, mapper = _matern_case_from(
        coefficient=coefficient, mesh_shape=(10, 10)
    )

    inversion_matrix_free, inversion_fft = _inversion_pair_from(
        dataset_sparse=dataset_sparse, mapper=mapper, use_positive_only_solver=True
    )

    reconstruction = inversion_matrix_free.reconstruction
    reconstruction_fft = np.asarray(inversion_fft.reconstruction)

    assert np.mean(reconstruction_fft == 0.0) >= 0.25
    assert np.all(reconstruction >= 0.0)
    assert np.array_equal(reconstruction == 0.0, reconstruction_fft == 0.0)
    assert inversion_matrix_free.positive_result.converged

    _assert_relative_close(reconstruction, reconstruction_fft, rtol=1.0e-6)


def test__primary_beam__matches_fft_route():
    """
    The sparse mapping matrix is built from the parent's beamed triplets, so a non-trivial primary beam
    is supported and matches the FFT route.
    """
    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=0.5, radius=2.5)

    rng = np.random.default_rng(seed=1)

    n_visibilities = 500

    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=rng.normal(size=(n_visibilities, 2))),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((n_visibilities, 2))),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)) * 1.0e6,
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
        primary_beam=aa.Array2D(
            values=np.random.default_rng(seed=2).uniform(
                0.2, 1.0, size=mask.pixels_in_mask
            ),
            mask=mask,
        ),
    )

    dataset_sparse = dataset.apply_sparse_operator(use_jax=False)

    mapper = _rectangular_mapper_from(
        mask=mask, regularization=aa.reg.MaternSparse(coefficient=1.0, scale=1.0)
    )

    inversion_matrix_free, inversion_fft = _inversion_pair_from(
        dataset_sparse=dataset_sparse, mapper=mapper
    )

    _assert_relative_close(
        inversion_matrix_free.reconstruction, inversion_fft.reconstruction, rtol=1.0e-6
    )
    assert inversion_matrix_free.fast_chi_squared == pytest.approx(
        float(inversion_fft.fast_chi_squared), rel=1.0e-8
    )
    _assert_relative_close(
        inversion_matrix_free.mapped_reconstructed_data.array,
        inversion_fft.mapped_reconstructed_data.array,
        rtol=1.0e-6,
    )


def test__unsupported_configurations__raise():
    mask, dataset_sparse, mapper = _matern_case_from()

    def inversion_from(linear_obj_list, settings=None):
        return aa.Inversion(
            dataset=dataset_sparse,
            linear_obj_list=linear_obj_list,
            settings=settings or _settings_from("matrix_free"),
        )

    # Two mappers.
    mapper_1 = _rectangular_mapper_from(
        mask=mask, regularization=aa.reg.MaternSparse(coefficient=1.0, scale=1.0)
    )

    with pytest.raises(exc.InversionException, match="2 mappers"):
        inversion_from([mapper, mapper_1])

    # A linear function list.
    func_list = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=np.ones((mask.pixels_in_mask, 1)),
    )

    with pytest.raises(exc.InversionException, match="linear-function list"):
        inversion_from([func_list, mapper])

    # A Delaunay mesh.
    with pytest.raises(exc.InversionException, match="Delaunay mesh"):
        inversion_from([_delaunay_mapper_from(mask=mask)])

    # A regularization without a sparse form.
    mapper_constant = _rectangular_mapper_from(mask=mask, coefficient=1.0)

    with pytest.raises(exc.InversionException, match="Constant regularization"):
        inversion_from([mapper_constant])

    # Edge-zeroing with the positive-only solver.
    with pytest.raises(exc.InversionException, match="use_edge_zeroed_pixels"):
        inversion_from(
            [mapper],
            settings=aa.Settings(
                interferometer_route="matrix_free",
                use_positive_only_solver=True,
                use_edge_zeroed_pixels=True,
            ),
        )

    # Every message names the route and points at the FFT route.
    with pytest.raises(exc.InversionException) as error:
        inversion_from([mapper_constant])

    assert "matrix_free" in str(error.value)
    assert 'interferometer_route="fft"' in str(error.value)


def test__jax_array_module__raises():
    jnp = pytest.importorskip("jax.numpy")

    _, dataset_sparse, mapper = _matern_case_from()

    with pytest.raises(exc.InversionException, match="non-NumPy array module"):
        aa.InversionInterferometerMatrixFree(
            dataset=dataset_sparse,
            linear_obj_list=[mapper],
            settings=_settings_from("matrix_free"),
            xp=jnp,
        )


@pytest.mark.parametrize(
    "name",
    [
        "curvature_matrix",
        "curvature_matrix_diag",
        "curvature_reg_matrix",
        "curvature_reg_matrix_reduced",
        "reconstruction_covariance_matrix",
        "reconstruction_noise_map",
        "reconstruction_noise_map_with_covariance",
        "reconstruction_noise_map_dict",
    ],
)
def test__dense_attributes__raise(name):
    _, dataset_sparse, mapper = _matern_case_from()

    inversion, _ = _inversion_pair_from(dataset_sparse=dataset_sparse, mapper=mapper)

    with pytest.raises(exc.InversionException, match=name):
        getattr(inversion, name)


def test__save_reconstruction_csv__writes_nan_noise_map(tmp_path):
    import autoarray.plot  # noqa: F401  (initialises the plot package before its submodule)
    from autoarray.inversion.plot.inversion_plots import save_reconstruction_csv

    _, dataset_sparse, mapper = _matern_case_from()

    inversion, _ = _inversion_pair_from(dataset_sparse=dataset_sparse, mapper=mapper)

    save_reconstruction_csv(inversion=inversion, output_path=tmp_path)

    lines = (tmp_path / "source_plane_reconstruction_0.csv").read_text().splitlines()

    assert lines[0] == "y,x,reconstruction,noise_map"
    assert len(lines) == 1 + mapper.params
    assert all(line.endswith(",nan") for line in lines[1:])


def test__settings__interferometer_route_validation():
    assert (
        aa.Settings(interferometer_route="matrix_free").interferometer_route
        == "matrix_free"
    )

    with pytest.raises(ValueError):
        aa.Settings(interferometer_route="typo")
