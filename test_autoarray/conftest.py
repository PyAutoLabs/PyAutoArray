import os
from pathlib import Path
import pytest
from matplotlib import pyplot

from autoarray import fixtures
from autonerves import conf


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(str(path))


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "savefig", plot_patch)
    return plot_patch


directory = Path(__file__).resolve().parent


@pytest.fixture(autouse=True, scope="session")
def remove_logs():
    yield
    for d, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".log"):
                os.remove(Path(d) / file)


@pytest.fixture(autouse=True)
def set_config_path(request):
    # if dirname(realpath(__file__)) in str(request.module):

    conf.instance.push(
        new_path=str(directory / "config"),
        output_path=str(directory / "output"),
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


def pytest_collection_modifyitems(config, items):
    """Skip interferometer tests that need the default ``nufftax`` backend when
    it is not installed.

    The default ``Interferometer`` transformer is the JAX-native
    ``TransformerNUFFT``, backed by the optional ``nufftax`` dependency. When
    that backend is absent, any test that builds a default interferometer or
    transformer raises ``ModuleNotFoundError`` at construction. Skip exactly
    those tests while keeping the explicit ``DFT`` transformer tests, which
    have no ``nufftax`` dependency.
    """
    try:
        import nufftax  # noqa: F401

        return
    except ImportError:
        pass

    skip_nufftax = pytest.mark.skip(
        reason="nufftax (the default JAX interferometer backend) is not installed"
    )
    for item in items:
        name = item.nodeid.rsplit("::", 1)[-1]
        # Explicit DFT backend tests do not use nufftax — keep them.
        if "__dft__" in name:
            continue
        nodeid = item.nodeid.replace("\\", "/")
        needs_nufftax = (
            "fit/test_fit_interferometer.py" in nodeid
            or "dataset/interferometer/test_dataset.py" in nodeid
            or "inversion/inversion/interferometer/test_interferometer.py" in nodeid
            or ("operators/test_transformer.py" in nodeid and "__nufft__" in name)
        )
        if needs_nufftax:
            item.add_marker(skip_nufftax)


@pytest.fixture(autouse=True)
def _regime_independent_test_output(monkeypatch):
    """
    Clear ``PYAUTO_SMALL_DATASETS`` for every test unless the test sets it.

    Since PyAutoNerves#153 every FITS the stack writes carries a ``SMALLDAT``
    card whose value tracks this env var at write time. Several tests write into
    **tracked** fixture paths (14 of them across this repo and PyAutoNerves --
    a pre-existing pattern), so without this the bytes those tests produce
    depend on the ambient environment: run the suite in a shell exporting
    ``PYAUTO_SMALL_DATASETS=1`` -- which ``should_simulate``'s own docstring calls
    the default for most harness runs -- and the suite passes but leaves the
    working tree dirty.

    Pinning it here restores the property the stamp took away, that test output
    is a function of the test and not of the shell, and does so in one place
    rather than by rewriting every fixture-writing test. Tests that need a
    regime set it with ``monkeypatch.setenv`` in their body, which runs after
    this fixture and wins.
    """
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
