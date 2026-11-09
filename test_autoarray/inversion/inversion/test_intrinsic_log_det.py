"""
PyAutoArray#682: the evidence's ``log det H`` term for intrinsic regularization schemes.

The smoothness schemes build an ``H`` whose constant vector is an exact null mode, held up only by the 1e-8
diagonal jitter. The evidence therefore normalises them with the pseudo-determinant (the product of the
non-zero eigenvalues), and ``AbstractRegularization.has_constant_null_mode`` says which schemes those are.
"""

import numpy as np
import pytest

import autoarray as aa
from autoarray import fixtures


def _mappers():
    return {
        "delaunay": fixtures.make_delaunay_mapper_9_3x3(),
        "rectangular": fixtures.make_rectangular_mapper_7x7_3x3(),
    }


SCHEMES = [
    aa.reg.Constant(coefficient=3.0),
    aa.reg.ConstantSplit(coefficient=3.0),
    aa.reg.Adapt(inner_coefficient=0.1, outer_coefficient=10.0),
    aa.reg.AdaptPower(inner_coefficient=0.1, outer_coefficient=10.0),
    aa.reg.AdaptSplit(inner_coefficient=0.1, outer_coefficient=10.0),
    aa.reg.AdaptSplitPower(inner_coefficient=0.1, outer_coefficient=10.0),
    aa.reg.Zeroth(coefficient=3.0),
    aa.reg.ConstantZeroth(coefficient_neighbor=3.0, coefficient_zeroth=1.0),
    aa.reg.AdaptSplitZeroth(inner_coefficient=0.1, outer_coefficient=10.0),
    aa.reg.AdaptSplitZerothPower(inner_coefficient=0.1, outer_coefficient=10.0),
    aa.reg.GaussianKernel(coefficient=3.0, scale=0.5),
    aa.reg.ExponentialKernel(coefficient=3.0, scale=0.5),
]


def _cases(schemes):
    """
    Every (scheme, mesh) pair the scheme applies to: split schemes need an adaptive mesh's split tables,
    which the rectangular mesh does not provide, so they are paired with the Delaunay mesh only.
    """
    return [
        pytest.param(scheme, mesh, id=f"{type(scheme).__name__}-{mesh}")
        for scheme in schemes
        for mesh in ("delaunay", "rectangular")
        if mesh == "delaunay" or not scheme.is_split_regularization
    ]


def _matrix_from(regularization, mapper):
    return np.asarray(regularization.regularization_matrix_from(linear_obj=mapper))


@pytest.mark.parametrize("regularization, mesh", _cases(SCHEMES))
def test__has_constant_null_mode__matches_measured_spectrum(regularization, mesh):
    """
    The flag must describe the matrix: an intrinsic scheme's smallest eigenvalue is the jitter with the
    constant vector as its eigenvector and an O(1) gap above it; every other scheme is positive-definite
    without the jitter.
    """
    matrix = _matrix_from(regularization, _mappers()[mesh])

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    pixels = matrix.shape[0]
    overlap = abs(eigenvectors[:, 0] @ np.ones(pixels) / np.sqrt(pixels))

    intrinsic = eigenvalues[0] < 1.0e-6 and overlap > 1.0 - 1.0e-9

    assert regularization.has_constant_null_mode == intrinsic

    if intrinsic:
        assert eigenvalues[1] > 1.0e-3


def _inversion_from(mapper_list):
    return aa.m.MockInversion(
        linear_obj_list=mapper_list,
        regularization_matrix=__import__("scipy.linalg").linalg.block_diag(
            *[np.asarray(mapper.regularization_matrix) for mapper in mapper_list]
        ),
    )


def _mapper_with(regularization, mesh):
    mapper = _mappers()[mesh]
    return type(mapper)(
        interpolator=mapper.interpolator,
        regularization=regularization,
        image_plane_mesh_grid=mapper.image_plane_mesh_grid,
    )


@pytest.mark.parametrize(
    "regularization, mesh",
    _cases([scheme for scheme in SCHEMES if scheme.has_constant_null_mode]),
)
def test__log_det_regularization_matrix_term__intrinsic__is_pseudo_determinant(
    regularization, mesh
):
    """
    The term equals an independent eigenvalue pseudo-determinant (log of the non-zero eigenvalues), not the
    full determinant, which differs from it by the log of the ~1e-8 jitter.
    """
    mapper = _mapper_with(regularization, mesh)
    matrix = np.asarray(mapper.regularization_matrix)

    eigenvalues = np.linalg.eigvalsh(matrix)

    log_det = _inversion_from([mapper]).log_det_regularization_matrix_term

    assert log_det == pytest.approx(np.sum(np.log(eigenvalues[1:])), abs=1.0e-7)

    # The full determinant differs by the log of the constant-mode eigenvalue: the 1e-8 jitter plus round-off
    # of order eps * ||H||. That round-off is what the pseudo-determinant removes; at `Adapt` outer=10 it is
    # ~2e-4 in the log here, hence the loose tolerance on this illustrative gap only.
    assert np.sum(np.log(eigenvalues)) - log_det == pytest.approx(
        np.log(1.0e-8), abs=1.0e-3
    )


@pytest.mark.parametrize("mesh", ["delaunay", "rectangular"])
def test__log_det_regularization_matrix_term__intrinsic__independent_of_constant_mode(
    mesh,
):
    """
    The pseudo-determinant does not see the constant mode at all: moving that mode's eigenvalue from the
    jitter to 1e-5 leaves the term unchanged, where the full determinant would shift by ``log(1e3)``.
    """
    mapper = _mapper_with(
        aa.reg.Adapt(inner_coefficient=0.1, outer_coefficient=10.0), mesh
    )
    matrix = np.asarray(mapper.regularization_matrix)
    pixels = matrix.shape[0]

    lifted = matrix + 1.0e-5 * np.ones((pixels, pixels)) / pixels

    log_det = aa.m.MockInversion(
        linear_obj_list=[mapper], regularization_matrix=matrix
    ).log_det_regularization_matrix_term
    log_det_lifted = aa.m.MockInversion(
        linear_obj_list=[mapper], regularization_matrix=lifted
    ).log_det_regularization_matrix_term

    assert log_det_lifted == pytest.approx(log_det, abs=1.0e-6)


def test__log_det_regularization_matrix_term__non_intrinsic__unchanged():
    """A positive-definite scheme keeps the full determinant."""
    mapper = _mapper_with(aa.reg.Zeroth(coefficient=3.0), "delaunay")
    matrix = np.asarray(mapper.regularization_matrix)

    log_det = _inversion_from([mapper]).log_det_regularization_matrix_term

    assert log_det == pytest.approx(np.linalg.slogdet(matrix)[1], rel=1.0e-12)


def test__log_det_regularization_matrix_term__multiple_mappers__sum_of_blocks():
    """
    Each regularized mapper is its own block: an intrinsic block contributes its pseudo-determinant and a
    positive-definite block its determinant, so the term is the sum of the per-mapper terms.
    """
    mapper_0 = _mapper_with(aa.reg.Constant(coefficient=3.0), "delaunay")
    mapper_1 = _mapper_with(aa.reg.Zeroth(coefficient=2.0), "rectangular")
    mapper_2 = _mapper_with(
        aa.reg.Adapt(inner_coefficient=0.1, outer_coefficient=10.0), "rectangular"
    )

    combined = _inversion_from([mapper_0, mapper_1, mapper_2])

    separate = sum(
        _inversion_from([mapper]).log_det_regularization_matrix_term
        for mapper in (mapper_0, mapper_1, mapper_2)
    )

    assert combined.log_det_regularization_matrix_term == pytest.approx(
        separate, rel=1.0e-12
    )
