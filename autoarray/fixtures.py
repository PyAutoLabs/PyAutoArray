import numpy as np

import autoarray as aa


def make_mask_1d_7():
    mask = np.array([True, True, False, False, False, True, True])

    return aa.Mask1D(mask=mask, pixel_scales=(1.0,))


def make_mask_2d_7x7():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))


def make_mask_2d_7x7_1_pix():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))


def make_blurring_mask_2d_7x7():
    blurring_mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, False, False, False, False, False, True],
            [True, False, True, True, True, False, True],
            [True, False, True, True, True, False, True],
            [True, False, True, True, True, False, True],
            [True, False, False, False, False, False, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask2D(mask=blurring_mask, pixel_scales=(1.0, 1.0))


### arrays ###


def make_array_1d_7():
    return aa.Array1D.ones(shape_native=(7,), pixel_scales=(1.0,))


def make_array_2d_7x7():
    return aa.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_array_2d_rgb_7x7():
    return aa.Array2DRGB(values=np.ones((7, 7, 3)), mask=make_mask_2d_7x7())


def make_layout_2d_7x7():
    return aa.Layout2D(
        shape_2d=(7, 7),
        original_roe_corner=(1, 0),
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


# GRIDS #


def make_grid_1d_7():
    return aa.Grid1D.from_mask(mask=make_mask_1d_7())


def make_grid_2d_7x7():
    return aa.Grid2D.from_mask(mask=make_mask_2d_7x7())


def make_grid_2d_sub_1_7x7():
    return aa.Grid2D.from_mask(mask=make_mask_2d_7x7(), over_sample_size=1)


def make_grid_2d_sub_2_7x7():
    return aa.Grid2D.from_mask(mask=make_mask_2d_7x7(), over_sample_size=2)


def make_grid_2d_7x7_simple():
    grid_2d_7x7 = make_grid_2d_7x7()
    grid_2d_7x7[0] = np.array([1.0, 1.0])
    grid_2d_7x7[1] = np.array([1.0, 0.0])
    grid_2d_7x7[2] = np.array([1.0, 1.0])
    grid_2d_7x7[3] = np.array([1.0, 0.0])
    return grid_2d_7x7


def make_blurring_grid_2d_7x7():
    return aa.Grid2D.from_mask(mask=make_blurring_mask_2d_7x7())


def make_image_7x7():
    return aa.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_psf_3x3():
    psf = aa.Array2D.no_mask(
        values=[[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]],
        pixel_scales=(1.0, 1.0),
    )

    return aa.Convolver(kernel=psf)


def make_psf_3x3_no_blur():
    return aa.Convolver.no_blur(pixel_scales=(1.0, 1.0))


def make_noise_map_7x7():
    return aa.Array2D.full(fill_value=2.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_noise_covariance_matrix_7x7():
    noise_covariance_matrix_7x7 = np.eye(N=49, M=49)

    noise_covariance_matrix_7x7[:, 24] = 1.0
    noise_covariance_matrix_7x7[24, :] = 1.0

    return noise_covariance_matrix_7x7


def make_grid_2d_irregular_7x7():
    return aa.Grid2DIrregular(values=[(0.1, 0.1), (0.2, 0.2)])


def make_grid_2d_irregular_7x7_list():
    return [
        aa.Grid2DIrregular(values=[(0.1, 0.1), (0.2, 0.2)]),
        aa.Grid2DIrregular(values=[(0.3, 0.3)]),
    ]


def make_imaging_7x7():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_map=make_noise_map_7x7(),
        over_sample_size_lp=1,
    )


def make_imaging_7x7_sub_2():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_map=make_noise_map_7x7(),
        over_sample_size_lp=2,
    )


def make_imaging_covariance_7x7():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_covariance_matrix=make_noise_covariance_matrix_7x7(),
        over_sample_size_lp=1,
    )


def make_imaging_7x7_no_blur():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3_no_blur(),
        noise_map=make_noise_map_7x7(),
        over_sample_size_lp=1,
    )


def make_imaging_7x7_no_blur_sub_2():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3_no_blur(),
        noise_map=make_noise_map_7x7(),
        over_sample_size_lp=2,
    )


def make_visibilities_7():
    return aa.Visibilities.full(shape_slim=(7,), fill_value=1.0)


def make_visibilities_noise_map_7():
    return aa.VisibilitiesNoiseMap.full(shape_slim=(7,), fill_value=2.0)


def make_uv_wavelengths_7x2():
    return np.array(
        [
            [-55636.4609375, 171376.90625],
            [-6903.21923828, 51155.578125],
            [-63488.4140625, 4141.28369141],
            [55502.828125, 47016.7265625],
            [54160.75390625, -99354.1796875],
            [-9327.66308594, -95212.90625],
            [0.0, 0.0],
        ]
    )


def make_uv_wavelengths_7x2_no_fft():
    return np.ones(shape=(7, 2))


def make_interferometer_7():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerDFT,
    )


def make_interferometer_7_no_fft():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2_no_fft(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerDFT,
    )


def make_interferometer_7_grid():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerDFT,
    )


def make_interferometer_7_lop():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerNUFFT,
    )


def make_transformer_7x7_7():
    return aa.TransformerDFT(
        uv_wavelengths=make_uv_wavelengths_7x2(), real_space_mask=make_mask_2d_7x7()
    )


### MASKED DATA ###


def make_masked_imaging_7x7():
    imaging_7x7 = make_imaging_7x7()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_covariance_7x7():
    imaging_7x7 = make_imaging_covariance_7x7()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_7x7_no_blur():

    imaging_7x7 = make_imaging_7x7_no_blur()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_7x7_no_blur_sub_2():
    imaging_7x7 = make_imaging_7x7_no_blur_sub_2()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_model_image_7x7():
    imaging_7x7 = make_masked_imaging_7x7()

    return 5.0 * imaging_7x7.data


def make_imaging_fit_x1_plane_7x7():
    imaging_7x7 = make_masked_imaging_7x7()

    model_data = 5.0 * imaging_7x7.data

    return aa.m.MockFitImaging(
        dataset=imaging_7x7, use_mask_in_fit=False, model_data=model_data
    )


def make_fit_interferometer_7():
    interferometer_7 = make_interferometer_7()

    model_data = 5.0 * interferometer_7.data

    return aa.m.MockFitInterferometer(
        dataset=interferometer_7, use_mask_in_fit=False, model_data=model_data
    )


def make_regularization_constant():
    return aa.reg.Constant(coefficient=1.0)


def make_regularization_constant_split():
    return aa.reg.ConstantSplit(coefficient=1.0)


def make_regularization_adaptive_brightness():
    return aa.reg.Adapt(inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.5)


def make_regularization_adaptive_brightness_split():
    return aa.reg.AdaptSplit(
        inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.5
    )


def make_regularization_gaussian_kernel():
    return aa.reg.GaussianKernel(coefficient=1.0, scale=0.5)


def make_regularization_exponential_kernel():
    return aa.reg.ExponentialKernel(coefficient=1.0, scale=0.5)


def make_regularization_matern_kernel():
    return aa.reg.MaternKernel(coefficient=1.0, scale=0.5, nu=0.7)


def make_over_sampler_2d_7x7():
    return aa.OverSampler(mask=make_mask_2d_7x7(), sub_size=2)


def make_border_relocator_2d_7x7():
    return aa.BorderRelocator(
        mask=make_mask_2d_7x7(), sub_size=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
    )


def make_rectangular_mapper_7x7_3x3():

    from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
        overlay_grid_from,
    )

    shape_native = (3, 3)

    source_plane_mesh_grid = overlay_grid_from(
        shape_native=shape_native, grid=make_grid_2d_sub_2_7x7().over_sampled
    )

    mesh = aa.mesh.RectangularUniform(shape=shape_native)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=make_grid_2d_sub_2_7x7(),
        source_plane_mesh_grid=aa.Grid2DIrregular(source_plane_mesh_grid),
        adapt_data=aa.Array2D.ones(shape_native, pixel_scales=0.1),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=make_regularization_constant(),
    )


def make_delaunay_mapper_9_3x3():

    grid_9 = aa.Grid2D.no_mask(
        values=[
            [0.6, -0.3],
            [0.5, -0.8],
            [0.2, 0.1],
            [0.0, 0.5],
            [-0.3, -0.8],
            [-0.6, -0.5],
            [-0.4, -1.1],
            [-1.2, 0.8],
            [-1.5, 0.9],
        ],
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    mesh = aa.mesh.Delaunay(pixels=9)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=make_grid_2d_sub_2_7x7(),
        source_plane_mesh_grid=grid_9,
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=make_regularization_constant(),
        image_plane_mesh_grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1),
    )


def make_knn_mapper_9_3x3():

    grid_9 = aa.Grid2D.no_mask(
        values=[
            [0.6, -0.3],
            [0.5, -0.8],
            [0.2, 0.1],
            [0.0, 0.5],
            [-0.3, -0.8],
            [-0.6, -0.5],
            [-0.4, -1.1],
            [-1.2, 0.8],
            [-1.5, 0.9],
        ],
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    mesh = aa.mesh.KNearestNeighbor(pixels=9, split_neighbor_division=1)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=make_grid_2d_sub_2_7x7(),
        source_plane_mesh_grid=grid_9,
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )

    return aa.Mapper(
        interpolator=interpolator,
        regularization=make_regularization_constant(),
        image_plane_mesh_grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1),
    )


def make_rectangular_inversion_7x7_3x3():
    return aa.Inversion(
        dataset=make_masked_imaging_7x7(),
        linear_obj_list=[make_rectangular_mapper_7x7_3x3()],
    )


def make_delaunay_inversion_9_3x3():
    return aa.Inversion(
        dataset=make_masked_imaging_7x7(),
        linear_obj_list=[make_delaunay_mapper_9_3x3()],
    )


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2128))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
