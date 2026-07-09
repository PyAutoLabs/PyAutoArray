import os
import shutil
from pathlib import Path


SMALL_DATASETS_SHAPE_NATIVE = (15, 15)
SMALL_DATASETS_PIXEL_SCALES = 0.6


def cap_array_2d_for_small_datasets(array_2d, pixel_scales):
    """
    Center-crop a 2D autoarray to the small-datasets cap when
    ``PYAUTO_SMALL_DATASETS=1`` is active.

    Returns ``(array_2d, pixel_scales)`` unchanged in any of these cases:

    - ``PYAUTO_SMALL_DATASETS`` is not set to ``"1"``.
    - ``array_2d.shape_native`` is already at-or-below the cap (15, 15).

    When the env var is set and the input shape exceeds (15, 15), returns a
    new ``Array2D`` center-cropped to (15, 15) with ``pixel_scales`` overridden
    to 0.6 — matching the convention used by ``Mask2D.circular`` and
    ``Grid2D.uniform`` so the loaded dataset stays shape-consistent with masks
    and grids built under the same env var.

    The same env var is honoured for shape construction in
    ``Mask2D.circular`` and ``Grid2D.uniform`` (and by ``should_simulate``
    for on-disk regeneration). This helper closes the gap for FITS loaders
    that read pre-committed datasets larger than the cap, which would
    otherwise broadcast-mismatch against capped masks/grids.

    Center-cropping (rather than downsampling/resampling) is intentional:
    smoke-mode tests don't require numerical correctness, and the simpler
    op avoids interpolation artifacts and a scipy dependency at this layer.
    """
    if os.environ.get("PYAUTO_SMALL_DATASETS") != "1":
        return array_2d, pixel_scales

    h, w = array_2d.shape_native
    cap_h, cap_w = SMALL_DATASETS_SHAPE_NATIVE
    if h <= cap_h and w <= cap_w:
        return array_2d, pixel_scales

    from autoarray.structures.arrays.uniform_2d import Array2D

    h0, w0 = (h - cap_h) // 2, (w - cap_w) // 2
    cropped = array_2d.native.array[h0:h0 + cap_h, w0:w0 + cap_w]
    return (
        Array2D.no_mask(values=cropped, pixel_scales=SMALL_DATASETS_PIXEL_SCALES),
        SMALL_DATASETS_PIXEL_SCALES,
    )


def should_simulate(dataset_path):
    """
    Returns True if the dataset at ``dataset_path`` needs to be simulated.

    When ``PYAUTO_SMALL_DATASETS=1`` is active, any existing dataset
    is deleted so the simulator re-creates it at the reduced resolution.  This
    avoids shape mismatches between full-resolution FITS files on disk and the
    15x15 mask/grid cap applied by the env var.

    Use this as a drop-in replacement for ``not path.exists(dataset_path)`` in
    the workspace auto-simulation pattern::

        if aa.util.dataset.should_simulate(dataset_path):
            subprocess.run([sys.executable, "scripts/.../simulator.py"], check=True)
    """
    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
        if Path(dataset_path).exists():
            shutil.rmtree(dataset_path)

    return not Path(dataset_path).exists()

SMALL_DATASETS_N_CATALOGUE = 25


def cap_catalogue_size_for_small_datasets(n: int) -> int:
    """
    Cap a catalogue-style dataset size (e.g. the number of weak-lensing background
    galaxies) to the small-datasets limit when ``PYAUTO_SMALL_DATASETS=1`` is active.

    Catalogue datasets have no pixel grid, so the (15, 15) array cap above does not
    apply to them; the number of catalogue entries is their size lever. Returns ``n``
    unchanged when the env var is not set to ``"1"`` or ``n`` is already at-or-below
    the cap (25).

    Simulators should apply this to *generated* catalogue sizes only (e.g. a
    requested number of random positions) — never to a user-provided grid or
    catalogue, which must round-trip unmodified.
    """
    if os.environ.get("PYAUTO_SMALL_DATASETS") != "1":
        return n

    return min(n, SMALL_DATASETS_N_CATALOGUE)
