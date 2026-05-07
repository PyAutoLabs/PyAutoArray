import autoarray as aa

from autoarray.inversion.inversion.imaging.inversion_imaging_util import (
    ImagingSparseOperator,
)
from autoarray.inversion.inversion.imaging.sparse import (
    InversionImagingSparse,
)

from autoarray import exc

import numpy as np
import pytest
from pathlib import Path

directory = Path(__file__).resolve().parent


def test__operated_mapping_matrix_property(psf_3x3, rectangular_mapper_7x7_3x3):

    inversion = aa.m.MockInversionImaging(
        mask=rectangular_mapper_7x7_3x3.mask,
        psf=psf_3x3,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
    )

    assert inversion.operated_mapping_matrix_list[0][0, 0] == pytest.approx(
        1.61999997, 1e-4
    )
    assert inversion.operated_mapping_matrix[0, 0] == pytest.approx(1.61999997408, 1e-4)

    mask = aa.Mask2D(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=1.0,
    )
    psf = aa.m.MockPSF(operated_mapping_matrix=np.ones((2, 2)))

    inversion = aa.m.MockInversionImaging(
        mask=mask,
        psf=psf,
        linear_obj_list=[rectangular_mapper_7x7_3x3, rectangular_mapper_7x7_3x3],
    )

    operated_mapping_matrix_0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix_1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    assert inversion.operated_mapping_matrix_list[0] == pytest.approx(
        operated_mapping_matrix_0, 1.0e-4
    )
    assert inversion.operated_mapping_matrix_list[1] == pytest.approx(
        operated_mapping_matrix_1, 1.0e-4
    )
    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )


def test__operated_mapping_matrix_property__with_operated_mapping_matrix_override(
    psf_3x3, rectangular_mapper_7x7_3x3
):
    psf = aa.m.MockPSF(operated_mapping_matrix=np.ones((2, 2)))

    operated_mapping_matrix_override = np.array([[1.0, 2.0], [3.0, 4.0]])

    linear_obj = aa.m.MockLinearObjFuncList(
        mapping_matrix=None,
        operated_mapping_matrix_override=operated_mapping_matrix_override,
    )

    inversion = aa.m.MockInversionImaging(
        mask=rectangular_mapper_7x7_3x3.mask,
        psf=psf,
        linear_obj_list=[rectangular_mapper_7x7_3x3, linear_obj],
    )

    operated_mapping_matrix_0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix = np.array([[1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 3.0, 4.0]])

    assert inversion.operated_mapping_matrix_list[0] == pytest.approx(
        operated_mapping_matrix_0, 1.0e-4
    )
    assert inversion.operated_mapping_matrix_list[1] == pytest.approx(
        operated_mapping_matrix_override, 1.0e-4
    )
    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )


def test__curvature_matrix(rectangular_mapper_7x7_3x3):
    noise_map = np.ones(2)
    psf = aa.m.MockPSF(operated_mapping_matrix=np.ones((2, 10)))

    operated_mapping_matrix_override = np.array([[1.0, 2.0], [3.0, 4.0]])

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=None,
        operated_mapping_matrix_override=operated_mapping_matrix_override,
        regularization=None,
    )

    dataset = aa.DatasetInterface(
        data=aa.Array2D.ones(shape_native=(2, 10), pixel_scales=1.0),
        noise_map=noise_map,
        psf=psf,
    )

    inversion = aa.InversionImagingMapping(
        dataset=dataset,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(no_regularization_add_to_curvature_diag_value=False),
    )

    assert inversion.curvature_matrix[0:2, 0:2] == pytest.approx(
        np.array([[10.0, 14.0], [14.0, 20.0]]), 1.0e-4
    )

    assert inversion.curvature_matrix[0, 0] - 10.0 < 1.0e-12
    assert inversion.curvature_matrix[3, 3] - 2.0 < 1.0e-12

    inversion = aa.InversionImagingMapping(
        dataset=dataset,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.Settings(no_regularization_add_to_curvature_diag_value=True),
    )

    assert inversion.curvature_matrix[0, 0] - 10.0 > 0.0
    assert inversion.curvature_matrix[3, 3] - 2.0 < 1.0e-12
