from astropy import units
from astropy.modeling import functional_models
from astropy.coordinates import Angle
import numpy as np
import pytest

import autoarray as aa
from pathlib import Path

test_data_path = Path(Path(__file__).resolve().parent) / "files"


@pytest.mark.parametrize(
    "pixel_scales, expected_pixel_scales",
    [
        (1.0, (1.0, 1.0)),
        (2.0, (2.0, 2.0)),
    ],
)
def test__no_blur__identity_kernel_with_correct_pixel_scales(
    pixel_scales, expected_pixel_scales
):
    convolver = aa.Convolver.no_blur(pixel_scales=pixel_scales)

    assert (
        convolver.kernel.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert convolver.kernel.pixel_scales == expected_pixel_scales


def test__from_gaussian__normalized__kernel_values_match_expected():

    convolver = aa.Convolver.from_gaussian(
        shape_native=(3, 3),
        pixel_scales=1.0,
        centre=(0.1, 0.1),
        axis_ratio=0.9,
        angle=45.0,
        sigma=1.0,
        normalize=True,
    )

    assert convolver.kernel.native == pytest.approx(
        np.array(
            [
                [0.06281, 0.13647, 0.0970],
                [0.11173, 0.21589, 0.136477],
                [0.065026, 0.11173, 0.06281],
            ]
        ),
        1.0e-3,
    )


def test__normalize__ones_kernel__each_element_equals_one_ninth():

    kernel_data = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    convolver = aa.Convolver(kernel=kernel_data, normalize=True)

    assert convolver.kernel.native == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)


def test__convolved_image_from__matches_scipy_convolve2d():

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), pixel_scales=(1.0, 1.0), radius=4.0
    )

    import scipy.signal

    kernel = aa.Array2D.no_mask(
        values=np.arange(49).reshape(7, 7), pixel_scales=mask.pixel_scales
    )
    image = aa.Array2D.no_mask(
        values=np.arange(900).reshape(30, 30), pixel_scales=mask.pixel_scales
    )

    blurred_image_via_scipy = scipy.signal.convolve2d(
        image.native, kernel.native, mode="same"
    )

    blurred_image_via_scipy = aa.Array2D.no_mask(
        values=blurred_image_via_scipy, pixel_scales=mask.pixel_scales
    )
    blurred_masked_image_via_scipy = aa.Array2D(
        values=blurred_image_via_scipy.native, mask=mask
    )

    # Now reproduce this data using the convolved_image_from function

    convolver = aa.Convolver(kernel=kernel)

    masked_image = aa.Array2D(values=image.native, mask=mask)

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=kernel.shape_native
    )

    blurring_image = aa.Array2D(values=image.native, mask=blurring_mask)

    blurred_masked_im_1 = convolver.convolved_image_from(
        image=masked_image, blurring_image=blurring_image
    )

    assert blurred_masked_image_via_scipy == pytest.approx(
        blurred_masked_im_1.array, 1e-4
    )


def test__convolve_imaged_from__no_blurring__matches_scipy_convolve2d_with_blurring_mask_zeroed():
    # Setup a blurred data, using the PSF to perform the convolution in 2D, then masks it to make a 1d array.

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), pixel_scales=(1.0, 1.0), radius=4.0
    )

    import scipy.signal

    kernel = aa.Array2D.no_mask(
        values=np.arange(49).reshape(7, 7), pixel_scales=mask.pixel_scales
    )
    image = aa.Array2D.no_mask(
        values=np.arange(900).reshape(30, 30), pixel_scales=mask.pixel_scales
    )

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=kernel.shape_native
    )

    blurred_image_via_scipy = scipy.signal.convolve2d(
        image.native * blurring_mask, kernel.native, mode="same"
    )
    blurred_image_via_scipy = aa.Array2D.no_mask(
        values=blurred_image_via_scipy, pixel_scales=mask.pixel_scales
    )
    blurred_masked_image_via_scipy = aa.Array2D(
        values=blurred_image_via_scipy.native, mask=mask
    )

    # Now reproduce this data using the frame convolver_image

    masked_image = aa.Array2D(values=image.native, mask=mask)

    convolver = aa.Convolver(kernel=kernel)

    blurred_masked_im_1 = convolver.convolved_image_from(
        image=masked_image, blurring_image=None
    )

    assert blurred_masked_image_via_scipy == pytest.approx(
        blurred_masked_im_1.array, 1e-4
    )


def test__convolved_mapping_matrix_from__single_source_pixel__blurred_values_match_expected():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, True, True, True, True, True],
            ]
        ),
        pixel_scales=1.0,
    )

    kernel = aa.Array2D.no_mask(
        values=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]],
        pixel_scales=mask.pixel_scales,
    )

    convolver = aa.Convolver(kernel=kernel)

    mapping = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [
                0,
                1,
                0,
            ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    blurred_mapping = convolver.convolved_mapping_matrix_from(mapping, mask)

    assert blurred_mapping == pytest.approx(
        np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0.4, 0],
                [0, 0.2, 0],
                [0.4, 0, 0],
                [0.2, 0, 0.4],
                [0.3, 0, 0.2],
                [0, 0.1, 0.3],
                [0, 0, 0],
                [0.1, 0, 0],
                [0, 0, 0.1],
                [0, 0, 0],
            ]
        ),
        rel=1.0e-4,
    )


def test__convolved_mapping_matrix_from__multiple_source_pixels__blurred_values_match_expected():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, True, True, True, True, True],
            ]
        ),
        pixel_scales=1.0,
    )

    kernel = aa.Array2D.no_mask(
        values=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]],
        pixel_scales=mask.pixel_scales,
    )

    convolver = aa.Convolver(kernel=kernel)

    mapping = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [
                0,
                1,
                0,
            ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    blurred_mapping = convolver.convolved_mapping_matrix_from(mapping, mask)

    assert blurred_mapping == pytest.approx(
        np.array(
            [
                [0, 0.6, 0],
                [0, 0.9, 0],
                [0, 0.5, 0],
                [0, 0.3, 0],
                [0, 0.1, 0],
                [0, 0.1, 0],
                [0, 0.5, 0],
                [0, 0.2, 0],
                [0.6, 0, 0],
                [0.5, 0, 0.4],
                [0.3, 0, 0.2],
                [0, 0.1, 0.3],
                [0.1, 0, 0],
                [0.1, 0, 0],
                [0, 0, 0.1],
                [0, 0, 0],
            ]
        ),
        abs=1e-4,
    )


def test__convolve_imaged_from__via_fft__sizes_not_precomputed__compare_numerical_value():

    # -------------------------------
    # Case 1: direct image convolution
    # -------------------------------
    mask = aa.Mask2D.circular(
        shape_native=(20, 20), pixel_scales=(1.0, 1.0), radius=5.0
    )

    image = aa.Array2D.no_mask(values=np.arange(400).reshape(20, 20), pixel_scales=1.0)
    masked_image = aa.Array2D(values=image.native, mask=mask)

    kernel = aa.Array2D.no_mask(
        values=np.arange(49).reshape(7, 7), pixel_scales=mask.pixel_scales
    )

    convolver_fft = aa.Convolver(
        kernel=kernel,
        use_fft=True,
        normalize=True,
    )

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=convolver_fft.kernel.shape_native
    )
    blurring_image = aa.Array2D(values=image.native, mask=blurring_mask)

    blurred_fft = convolver_fft.convolved_image_from(
        image=masked_image, blurring_image=blurring_image
    )

    assert blurred_fft.native.array[13, 13] == pytest.approx(249.5, abs=1e-6)
