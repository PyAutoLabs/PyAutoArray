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


def test__from_gaussian__small_datasets__kernel_keeps_full_requested_shape(monkeypatch):
    """
    The `PYAUTO_SMALL_DATASETS` fast-mode cap shrinks datasets (and the grids paired with
    them) to 16x16, but a convolution kernel's shape is intrinsic to the operator. If the
    cap reached the grid the Gaussian is evaluated on, its 256 values would be wrapped in an
    `Array2D` at the uncapped `shape_native` of 961, raising an `ArrayException`.
    """
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    convolver = aa.Convolver.from_gaussian(
        shape_native=(31, 31),
        pixel_scales=0.1,
        sigma=0.1,
    )

    assert convolver.kernel.shape_native == (31, 31)

    image = aa.Array2D.ones(shape_native=(16, 16), pixel_scales=0.1)

    blurred_image = convolver.convolved_image_from(image=image, blurring_image=None)

    assert blurred_image.shape_native == (16, 16)


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


def _ground_truth_centres(n, pixel_scale):
    return (np.arange(n) - (n - 1) / 2.0) * pixel_scale


def _ground_truth_gaussian(y, x, sigma, centre=(0.0, 0.0)):
    r2 = (y - centre[0]) ** 2 + (x - centre[1]) ** 2
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * r2 / sigma**2)


def _ground_truth_scene(over_sample_size):
    """
    The brute-force reference scene of PyAutoMind/feature/autoarray/oversampling_ground_truth.py:
    11x11 image at ps=1", circular mask r=3.5" (37 pixels), Gaussian source (sigma=1.2",
    centre (0.3, -0.4)"), Gaussian PSF (sigma=0.8") with fixed physical kernel radius 2.0".
    """
    s = over_sample_size

    mask = aa.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=3.5)

    kernel_n = int(2 * round(2.0 * s) + 1)
    kc = _ground_truth_centres(kernel_n, 1.0 / s)
    kyy, kxx = np.meshgrid(-kc, kc, indexing="ij")
    kernel = aa.Array2D.no_mask(
        values=_ground_truth_gaussian(kyy, kxx, 0.8), pixel_scales=1.0 / s
    )

    convolver = aa.Convolver(
        kernel=kernel, normalize=True, convolve_over_sample_size=s
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=s)

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=convolver.kernel_shape_image_resolution,
        allow_padding=True,
    )
    blurring_grid = aa.Grid2D.from_mask(mask=blurring_mask, over_sample_size=s)

    def eval_source(grid_over_sampled):
        arr = np.array(grid_over_sampled)
        return _ground_truth_gaussian(arr[:, 0], arr[:, 1], 1.2, (0.3, -0.4))

    image = eval_source(grid.over_sampled)
    blurring_image = eval_source(blurring_grid.over_sampled)

    return mask, convolver, image, blurring_image


def test__convolve_over_sample_size__matches_brute_force_ground_truth():
    mask, convolver, image, blurring_image = _ground_truth_scene(over_sample_size=2)

    convolved = convolver.convolved_image_from(
        image=image, blurring_image=blurring_image, mask=mask
    )
    convolved = np.array(convolved)

    assert convolved.sum() == pytest.approx(2.796562184524787e00, abs=1.0e-12)
    assert convolved[0] == pytest.approx(3.726289901353439e-02, abs=1.0e-12)
    assert convolved[17] == pytest.approx(2.025075336159483e-01, abs=1.0e-12)
    assert convolved[36] == pytest.approx(1.090767109119494e-02, abs=1.0e-12)


def test__convolve_over_sample_size_one__scene_matches_ground_truth():
    # The same scene through the unchanged s=1 path reproduces the brute-force
    # reference, pinning the test scene itself against the ground-truth script.
    mask, convolver, image, blurring_image = _ground_truth_scene(over_sample_size=1)

    image = aa.Array2D(values=image, mask=mask)
    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=(5, 5), allow_padding=True
    )
    blurring_image = aa.Array2D(values=blurring_image, mask=blurring_mask)

    convolved = convolver.convolved_image_via_real_space_np_from(
        image=image, blurring_image=blurring_image
    )
    convolved = np.array(convolved)

    assert convolved.sum() == pytest.approx(2.807349652595196e00, abs=1.0e-12)
    assert convolved[0] == pytest.approx(3.655472905370449e-02, abs=1.0e-12)
    assert convolved[17] == pytest.approx(2.069771979137382e-01, abs=1.0e-12)
    assert convolved[36] == pytest.approx(1.042470837248629e-02, abs=1.0e-12)


def test__convolve_over_sample_size__mapping_matrix__delta_kernel_bins_rows():
    # With a delta-function fine kernel, oversampled convolution reduces to
    # binning the sub-resolution mapping matrix rows by their mean, testing the
    # permutation and bin-down independently of convolution numerics.
    mask = aa.Mask2D.circular(shape_native=(7, 7), pixel_scales=1.0, radius=2.5)
    s = 2

    delta = np.zeros((3, 3))
    delta[1, 1] = 1.0
    kernel = aa.Array2D.no_mask(values=delta, pixel_scales=1.0 / s)

    convolver = aa.Convolver(kernel=kernel, convolve_over_sample_size=s)

    n_sub = mask.pixels_in_mask * s**2
    rng = np.random.default_rng(1)
    mapping_matrix = rng.random((n_sub, 3))

    convolved = convolver.convolved_mapping_matrix_from(
        mapping_matrix=mapping_matrix, mask=mask
    )

    binned = mapping_matrix.reshape(mask.pixels_in_mask, s**2, 3).mean(axis=1)

    assert convolved == pytest.approx(binned, abs=1.0e-14)


def test__convolve_over_sample_size__validation():
    kernel_native = aa.Array2D.no_mask(values=np.ones((3, 3)), pixel_scales=1.0)
    kernel_fine = aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=0.5)

    with pytest.raises(TypeError):
        aa.Convolver(kernel=kernel_native, convolve_over_sample_size=2.0)

    with pytest.raises(aa.exc.KernelException):
        aa.Convolver(kernel=kernel_native, convolve_over_sample_size=0)

    mask = aa.Mask2D.circular(shape_native=(7, 7), pixel_scales=1.0, radius=2.5)

    # Kernel at image resolution handed to an oversampled convolver.
    convolver = aa.Convolver(kernel=kernel_native, convolve_over_sample_size=2)
    with pytest.raises(aa.exc.KernelException):
        convolver.state_from(mask=mask)

    convolver = aa.Convolver(kernel=kernel_fine, convolve_over_sample_size=2)

    # Binned (image-resolution) input is a distinct, explicit error.
    with pytest.raises(aa.exc.KernelException):
        convolver.convolved_image_from(
            image=np.ones(mask.pixels_in_mask), blurring_image=None, mask=mask
        )

    # No precomputed state and no mask.
    with pytest.raises(aa.exc.KernelException):
        convolver.convolved_image_from(
            image=np.ones(mask.pixels_in_mask * 4), blurring_image=None
        )


def test__kernel_shape_image_resolution():
    kernel = aa.Array2D.no_mask(values=np.ones((5, 5)), pixel_scales=1.0)
    convolver = aa.Convolver(kernel=kernel)
    assert convolver.kernel_shape_image_resolution == (5, 5)

    kernel_fine = aa.Array2D.no_mask(values=np.ones((9, 9)), pixel_scales=0.5)
    convolver = aa.Convolver(kernel=kernel_fine, convolve_over_sample_size=2)
    assert convolver.kernel_shape_image_resolution == (5, 5)

    kernel_fine = aa.Array2D.no_mask(values=np.ones((15, 15)), pixel_scales=1.0 / 3.0)
    convolver = aa.Convolver(kernel=kernel_fine, convolve_over_sample_size=3)
    assert convolver.kernel_shape_image_resolution == (7, 7)


def test__convolve_over_sample_size__blurring_mask_padding__delta_kernel_identity():
    # A mask close to the image edge forces blurring_from to pad its output to a
    # larger frame; the fine-state geometry must embed the image mask in that same
    # frame. With a delta fine kernel, convolution then reduces to binning the
    # over-sampled input — any frame/permutation misalignment breaks the identity.
    import warnings

    mask = aa.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=5.0)
    s = 2

    delta = np.zeros((9, 9))
    delta[4, 4] = 1.0
    kernel = aa.Array2D.no_mask(values=delta, pixel_scales=1.0 / s)
    convolver = aa.Convolver(kernel=kernel, convolve_over_sample_size=s)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        state = convolver.state_from(mask=mask)
        assert any("Mask padded" in str(x.message) for x in w)

    assert state.image_mask.shape_native == (11, 11)

    rng = np.random.default_rng(2)
    values_sub = rng.random(mask.pixels_in_mask * s**2)

    convolver = aa.Convolver(kernel=kernel, state=state, convolve_over_sample_size=s)
    convolved = convolver.convolved_image_from(
        image=values_sub, blurring_image=None, mask=mask
    )

    binned = values_sub.reshape(mask.pixels_in_mask, s**2).mean(axis=1)

    assert np.array(convolved) == pytest.approx(binned, abs=1.0e-14)
