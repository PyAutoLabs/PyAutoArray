import importlib.util
import os
from os import path
import pytest
from matplotlib import pyplot

from autoarray import fixtures
from autoconf import conf

# Skip JAX-only tests when jax isn't installed. find_spec checks availability
# WITHOUT importing the module, so this conftest stays numpy-only per the
# "library unit tests stay numpy-only" rule.
collect_ignore_glob = []
if importlib.util.find_spec("jax") is None:
    collect_ignore_glob = ["test_jax_*.py", "**/test_jax_*.py"]


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "savefig", plot_patch)
    return plot_patch


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True, scope="session")
def remove_logs():
    yield
    for d, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".log"):
                os.remove(path.join(d, file))


@pytest.fixture(autouse=True)
def set_config_path(request):
    # if dirname(realpath(__file__)) in str(request.module):

    conf.instance.push(
        new_path=path.join(directory, "config"),
        output_path=path.join(directory, "output"),
    )


@pytest.fixture(name="mask_1d_7")
def make_mask_1d_7():
    return fixtures.make_mask_1d_7()


@pytest.fixture(name="mask_2d_7x7")
def make_mask_2d_7x7():
    return fixtures.make_mask_2d_7x7()


@pytest.fixture(name="mask_2d_7x7_1_pix")
def make_mask_2d_7x7_1_pix():
    return fixtures.make_mask_2d_7x7_1_pix()


@pytest.fixture(name="array_1d_7")
def make_array_1d_7():
    return fixtures.make_array_1d_7()


@pytest.fixture(name="array_2d_7x7")
def make_array_2d_7x7():
    return fixtures.make_array_2d_7x7()


@pytest.fixture(name="array_2d_rgb_7x7")
def make_array_2d_rgb_7x7():
    return fixtures.make_array_2d_rgb_7x7()


@pytest.fixture(name="layout_2d_7x7")
def make_layout_2d_7x7():
    return fixtures.make_layout_2d_7x7()


@pytest.fixture(name="grid_1d_7")
def make_grid_1d_7():
    return fixtures.make_grid_1d_7()


@pytest.fixture(name="grid_2d_7x7")
def make_grid_2d_7x7():
    return fixtures.make_grid_2d_7x7()


@pytest.fixture(name="grid_2d_sub_1_7x7")
def make_grid_2d_sub_1_7x7():
    return fixtures.make_grid_2d_sub_1_7x7()


@pytest.fixture(name="grid_2d_irregular_7x7")
def make_grid_2d_irregular_7x7():
    return fixtures.make_grid_2d_irregular_7x7()


@pytest.fixture(name="grid_2d_irregular_7x7_list")
def make_grid_2d_irregular_7x7_list():
    return fixtures.make_grid_2d_irregular_7x7_list()


@pytest.fixture(name="blurring_grid_2d_7x7")
def make_blurring_grid_2d_7x7():
    return fixtures.make_blurring_grid_2d_7x7()


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return fixtures.make_image_7x7()


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return fixtures.make_noise_map_7x7()


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return fixtures.make_psf_3x3()


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7():
    return fixtures.make_imaging_7x7()


@pytest.fixture(name="imaging_7x7_no_blur")
def make_imaging_7x7_no_blur():
    return fixtures.make_imaging_7x7_no_blur()


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7():
    return fixtures.make_masked_imaging_7x7()


@pytest.fixture(name="masked_imaging_covariance_7x7")
def make_masked_imaging_covariance_7x7():
    return fixtures.make_masked_imaging_covariance_7x7()


@pytest.fixture(name="masked_imaging_7x7_no_blur")
def make_masked_imaging_7x7_no_blur():
    return fixtures.make_masked_imaging_7x7_no_blur()


@pytest.fixture(name="masked_imaging_7x7_no_blur_sub_2")
def make_masked_imaging_7x7_no_blur_sub_2():
    return fixtures.make_masked_imaging_7x7_no_blur_sub_2()


@pytest.fixture(name="model_image_7x7")
def make_model_image_7x7():
    return fixtures.make_model_image_7x7()


@pytest.fixture(name="fit_imaging_7x7")
def make_imaging_fit_x1_plane_7x7():
    return fixtures.make_imaging_fit_x1_plane_7x7()


@pytest.fixture(name="visibilities_7")
def make_visibilities_7():
    return fixtures.make_visibilities_7()


@pytest.fixture(name="visibilities_noise_map_7")
def make_noise_map_7():
    return fixtures.make_visibilities_noise_map_7()


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7x2():
    return fixtures.make_uv_wavelengths_7x2()


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7():
    return fixtures.make_transformer_7x7_7()


@pytest.fixture(name="interferometer_7")
def make_interferometer_7():
    return fixtures.make_interferometer_7()


@pytest.fixture(name="interferometer_7_no_fft")
def make_interferometer_7_no_fft():
    return fixtures.make_interferometer_7_no_fft()


@pytest.fixture(name="interferometer_7_lop")
def make_interferometer_7_lop():
    return fixtures.make_interferometer_7_lop()


@pytest.fixture(name="fit_interferometer_7")
def make_fit_interferometer_7():
    return fixtures.make_fit_interferometer_7()


@pytest.fixture(name="regularization_constant")
def make_regularization_constant():
    return fixtures.make_regularization_constant()


@pytest.fixture(name="regularization_constant_split")
def make_regularization_constant_split():
    return fixtures.make_regularization_constant_split()


@pytest.fixture(name="regularization_adaptive_brightness")
def make_regularization_adaptive_brightness():
    return fixtures.make_regularization_adaptive_brightness()


@pytest.fixture(name="regularization_adaptive_brightness_split")
def make_regularization_adaptive_brightness_split():
    return fixtures.make_regularization_adaptive_brightness_split()


@pytest.fixture(name="regularization_gaussian_kernel")
def make_regularization_gaussian_kernel():
    return fixtures.make_regularization_gaussian_kernel()


@pytest.fixture(name="regularization_exponential_kernel")
def make_regularization_exponential_kernel():
    return fixtures.make_regularization_exponential_kernel()


@pytest.fixture(name="regularization_matern_kernel")
def make_regularization_matern_kernel():
    return fixtures.make_regularization_matern_kernel()


@pytest.fixture(name="rectangular_mapper_7x7_3x3")
def make_rectangular_mapper_7x7_3x3():
    return fixtures.make_rectangular_mapper_7x7_3x3()


@pytest.fixture(name="delaunay_mapper_9_3x3")
def make_delaunay_mapper_9_3x3():
    return fixtures.make_delaunay_mapper_9_3x3()


@pytest.fixture(name="knn_mapper_9_3x3")
def make_knn_mapper_9_3x3():
    return fixtures.make_knn_mapper_9_3x3()


@pytest.fixture(name="rectangular_inversion_7x7_3x3")
def make_rectangular_inversion_7x7_3x3():
    return fixtures.make_rectangular_inversion_7x7_3x3()


@pytest.fixture(name="delaunay_inversion_9_3x3")
def make_delaunay_inversion_9_3x3():
    return fixtures.make_delaunay_inversion_9_3x3()


@pytest.fixture(name="euclid_data")
def make_euclid_data():
    return fixtures.make_euclid_data()


@pytest.fixture(name="acs_ccd")
def make_acs_ccd():
    return fixtures.make_acs_ccd()


@pytest.fixture(name="acs_quadrant")
def make_acs_quadrant():
    return fixtures.make_acs_quadrant()
