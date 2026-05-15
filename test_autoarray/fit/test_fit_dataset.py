import pytest

import autoarray as aa


def test__figure_of_merit__with_inversion_and_regularization__equals_log_evidence(
    masked_imaging_7x7, model_image_7x7
):
    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(regularization=aa.m.MockRegularization())],
        data_vector=1,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        inversion=inversion,
    )

    assert fit.figure_of_merit == fit.log_evidence


def test__figure_of_merit__with_inversion_and_no_regularization__equals_log_likelihood(
    masked_imaging_7x7, model_image_7x7
):
    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=None)], data_vector=1
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        inversion=inversion,
    )

    assert fit.figure_of_merit == fit.log_likelihood


def test__chi_squared__with_noise_covariance_matrix__uses_covariance_weighted_formula(
    masked_imaging_covariance_7x7, model_image_7x7
):
    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_covariance_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
    )

    chi_squared = aa.util.fit.chi_squared_with_noise_covariance_from(
        residual_map=fit.residual_map,
        noise_covariance_matrix_inv=masked_imaging_covariance_7x7.noise_covariance_matrix_inv,
    )

    assert fit.chi_squared == chi_squared


def test__figure_of_merit__with_noise_covariance_matrix__equals_negative_half_chi_squared_plus_noise_normalization(
    masked_imaging_covariance_7x7, model_image_7x7
):
    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_covariance_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
    )

    assert fit.figure_of_merit == pytest.approx(
        -0.5 * (fit.chi_squared + fit.noise_normalization), 1.0e-4
    )


def test__chi_squared__without_noise_covariance_matrix__differs_from_covariance_weighted_result(
    masked_imaging_covariance_7x7, masked_imaging_7x7, model_image_7x7
):
    fit_with_covariance = aa.m.MockFitImaging(
        dataset=masked_imaging_covariance_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
    )

    chi_squared_covariance = aa.util.fit.chi_squared_with_noise_covariance_from(
        residual_map=fit_with_covariance.residual_map,
        noise_covariance_matrix_inv=masked_imaging_covariance_7x7.noise_covariance_matrix_inv,
    )

    fit_without_covariance = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
    )

    assert fit_without_covariance.chi_squared != pytest.approx(
        chi_squared_covariance, 1.0e-4
    )


def test__grids__with_dataset_model_grid_offset__lp_and_pixelization_grids_offset_correctly(
    imaging_7x7, mask_2d_7x7, model_image_7x7
):
    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        dataset_model=aa.DatasetModel(grid_offset=(1.0, 2.0)),
    )

    assert fit.dataset_model.grid_offset == (1.0, 2.0)
    assert fit.grids.lp[0] == pytest.approx((0.0, -3.0), 1.0e-4)
    assert fit.grids.pixelization[0] == pytest.approx((0.0, -3.0), 1.0e-4)


def test__grids__with_dataset_model_grid_rotation_angle__lp_grid_rotated_correctly(
    imaging_7x7, mask_2d_7x7, model_image_7x7
):
    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    # Rotation by 90 degrees CCW about the origin maps (y, x) -> (x, -y).
    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        dataset_model=aa.DatasetModel(grid_rotation_angle=90.0),
    )

    assert fit.dataset_model.grid_rotation_angle == 90.0
    # 90 deg CCW rotation in (y, x) order maps (y, x) -> (x, -y).
    original = masked_imaging_7x7.grids.lp[0]
    rotated = fit.grids.lp[0]
    assert rotated[0] == pytest.approx(original[1], 1.0e-4)
    assert rotated[1] == pytest.approx(-original[0], 1.0e-4)


def test__grids__with_grid_offset_and_grid_rotation_angle__shift_then_rotate(
    imaging_7x7, mask_2d_7x7, model_image_7x7
):
    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        dataset_model=aa.DatasetModel(
            grid_offset=(1.0, 2.0), grid_rotation_angle=90.0
        ),
    )

    # First subtract the offset, then rotate 90deg CCW: (y, x) -> (x, -y).
    original = masked_imaging_7x7.grids.lp[0]
    shifted_y = original[0] - 1.0
    shifted_x = original[1] - 2.0
    expected = (shifted_x, -shifted_y)

    assert fit.grids.lp[0] == pytest.approx(expected, 1.0e-4)


def test__grids__with_grid_rotation_angle_zero__matches_subtracted_from(
    imaging_7x7, mask_2d_7x7, model_image_7x7
):
    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    fit_rotated = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        dataset_model=aa.DatasetModel(grid_offset=(1.0, 2.0), grid_rotation_angle=0.0),
    )

    # angle=0 is identity rotation, so the result must equal the offset-only path.
    assert fit_rotated.grids.lp[0] == pytest.approx((0.0, -3.0), 1.0e-4)
    assert fit_rotated.grids.pixelization[0] == pytest.approx((0.0, -3.0), 1.0e-4)
