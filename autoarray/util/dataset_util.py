import os
import shutil
from pathlib import Path


SMALL_DATASETS_SHAPE_NATIVE = (16, 16)
SMALL_DATASETS_PIXEL_SCALES = 0.6

# The FITS header card ``autonerves.fitsable.stamp_small_datasets_regime`` writes
# on every array the stack outputs. Deliberately duplicated here rather than
# imported: ``pyproject.toml`` floors autonerves at a release that predates the
# stamp, so an import would hard-fail against a legitimately-resolved older
# autonerves. Reading the card by name degrades to "absent" instead, which is
# exactly the fallback path below. Keep in sync with PyAutoNerves#153.
SMALL_DATASETS_HEADER_KEY = "SMALLDAT"


def cap_array_2d_for_small_datasets(array_2d, pixel_scales):
    """
    Center-crop a 2D autoarray to the small-datasets cap when
    ``PYAUTO_SMALL_DATASETS=1`` is active, and relabel it at the capped
    pixel scale.

    Returns ``(array_2d, pixel_scales)`` unchanged only when
    ``PYAUTO_SMALL_DATASETS`` is not set to ``"1"``.

    When the env var is set, ``pixel_scales`` is always overridden to 0.6 —
    matching the convention used by ``Mask2D.circular`` and ``Grid2D.uniform``
    so the loaded dataset stays shape-consistent with masks and grids built
    under the same env var — and the shape is handled per case:

    - Input shape exceeds (16, 16): center-cropped to (16, 16).
    - Input shape is already at-or-below the cap: kept as-is, because a capped
      simulator wrote it at 0.6 already. Only the scale is corrected.

    That second case must still rebuild the ``Array2D``, not just return a
    corrected scalar: the array is constructed by the caller before this call
    and carries its own geometry, so an uncorrected array would keep the
    caller's uncapped scale no matter what scalar is returned. Leaving it
    uncorrected mislabels the frame 6x (±0.8" instead of ±4.8" for a 16x16
    field), which pushes off-centre galaxies outside the frame; their
    non-negative linear intensity solve then correctly returns exactly 0.0 and
    the failure surfaces far downstream as a collapsed prior rather than as a
    geometry error (PyAutoArray #430).

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

    from autoarray.structures.arrays.uniform_2d import Array2D

    h, w = array_2d.shape_native
    cap_h, cap_w = SMALL_DATASETS_SHAPE_NATIVE
    if h <= cap_h and w <= cap_w:
        return (
            Array2D.no_mask(
                values=array_2d.native.array,
                pixel_scales=SMALL_DATASETS_PIXEL_SCALES,
            ),
            SMALL_DATASETS_PIXEL_SCALES,
        )

    h0, w0 = (h - cap_h) // 2, (w - cap_w) // 2
    cropped = array_2d.native.array[h0:h0 + cap_h, w0:w0 + cap_w]
    return (
        Array2D.no_mask(values=cropped, pixel_scales=SMALL_DATASETS_PIXEL_SCALES),
        SMALL_DATASETS_PIXEL_SCALES,
    )


def _on_disk_shape_native(data_path):
    """
    Returns the ``(rows, columns)`` shape of the first 2D image in the FITS file
    at ``data_path``, or ``None`` if that cannot be determined.

    Only the headers are read, never the pixel data, so this costs a single
    small read regardless of dataset size.

    ``None`` means "unknown", and every caller must treat it as "leave the
    dataset alone" — this function feeds a destructive predicate, so an
    unreadable or unconventional file must never be grounds for deleting it.
    """
    from astropy.io import fits

    try:
        with fits.open(data_path) as hdu_list:
            for hdu in hdu_list:
                header = hdu.header
                if header.get("NAXIS") == 2:
                    # NAXIS1 is the fastest-varying axis (columns), NAXIS2 the
                    # rows, so the numpy-order shape is (NAXIS2, NAXIS1).
                    return (header["NAXIS2"], header["NAXIS1"])
    except Exception:
        return None

    return None


def _small_datasets_stamp_on_disk(dataset_path):
    """
    Returns the small-datasets regime recorded in ``data.fits``'s header, as a
    tri-state: ``True`` (written by a capped run), ``False`` (written at full
    resolution), or ``None`` (no usable stamp -- unknown).

    ``None`` is returned for a missing file, an unreadable file, a file with no
    ``SMALLDAT`` card, **and** a card whose value is not a genuine FITS boolean.
    That last case is not pedantry: ``bool("F")`` is ``True`` in Python, so
    coercing a hand-edited or third-party string card would invert the regime
    and hand a ``True`` to a predicate that ends in ``shutil.rmtree``. A card
    this code did not write is not a card this code can trust.

    Callers must treat ``None`` as "leave the dataset alone" and fall back to
    :func:`_is_small_datasets_on_disk`. Unknown must never mean "full".
    """
    from astropy.io import fits

    data_path = Path(dataset_path) / "data.fits"

    if not data_path.exists():
        return None

    try:
        with fits.open(data_path) as hdu_list:
            for hdu in hdu_list:
                value = hdu.header.get(SMALL_DATASETS_HEADER_KEY)
                if isinstance(value, bool):
                    return value
    except Exception:
        return None

    return None


def _is_small_datasets_on_disk(dataset_path):
    """
    Returns True if the dataset on disk at ``dataset_path`` was written by a
    simulator running under ``PYAUTO_SMALL_DATASETS=1``.

    The regime is not recorded anywhere on disk, so it is inferred from the
    shape of ``data.fits``: the cap in ``Mask2D.circular`` / ``Grid2D.uniform``
    rewrites anything larger than ``SMALL_DATASETS_SHAPE_NATIVE`` to *exactly*
    that shape, so an on-disk ``data.fits`` at exactly (16, 16) can only have
    come from a capped run.

    Three deliberate narrownesses, all of them because this predicate ends in
    ``shutil.rmtree`` and a false positive silently deletes a user's data:

    - **Exactly** the cap shape, never "at or below" it. The cap cannot emit
      12x12, so widening the test buys no detection and only adds risk.
    - **``data.fits`` by name**, never "the first FITS in the directory". PSF
      kernels are legitimately tiny at full resolution (11x11 is common), and a
      glob would regenerate every dataset carrying one on every single run.
    - **Unknown means no.** A missing, unreadable or non-2D ``data.fits``
      returns False, preserving the existence-only behaviour for the dataset
      families this cannot speak about (see the caveat in ``should_simulate``).
    """
    data_path = Path(dataset_path) / "data.fits"

    if not data_path.exists():
        return False

    return _on_disk_shape_native(data_path) == SMALL_DATASETS_SHAPE_NATIVE


def should_simulate(dataset_path):
    """
    Returns True if the dataset at ``dataset_path`` needs to be simulated.

    A dataset is invalid when it was simulated under a different resolution
    regime than the one in force now, because ``PYAUTO_SMALL_DATASETS=1`` caps
    masks and grids to ``SMALL_DATASETS_SHAPE_NATIVE``. Both directions are
    checked:

    - Entering the **small** regime, any existing dataset is deleted so the
      simulator re-creates it at the reduced resolution, avoiding shape
      mismatches between full-resolution FITS on disk and the capped
      mask/grid.
    - Entering the **full** regime, a dataset left behind by an earlier capped
      run is likewise deleted. Existence alone cannot distinguish the two, so
      the regime is taken from the ``SMALLDAT`` header card that
      ``autonerves.fitsable`` stamps into every FITS the stack writes
      (``_small_datasets_stamp_on_disk``), falling back to inferring it from
      the data's shape (``_is_small_datasets_on_disk``) for datasets written
      before that stamp existed.

    That second check is what makes a local FAIL mean something. ``dataset/``
    is gitignored in the workspaces, so CI clones fresh and always simulates,
    while a local checkout keeps its dataset indefinitely — and since
    ``PYAUTO_SMALL_DATASETS=1`` is the default for most harness runs, a single
    earlier run would leave capped FITS that every later full-resolution run
    then loaded silently, producing deterministic, environment-only failures
    that could not be reproduced in CI (autolens_workspace_test#260).

    Use this as a drop-in replacement for ``not path.exists(dataset_path)`` in
    the workspace auto-simulation pattern::

        if aa.util.dataset.should_simulate(dataset_path):
            subprocess.run([sys.executable, "scripts/.../simulator.py"], check=True)

    Precedence
    ----------
    The stamp wins over the shape heuristic, and the three states are not
    interchangeable:

    - ``SMALLDAT = T`` -- delete. The writer said so.
    - ``SMALLDAT = F`` -- **keep**, unconditionally, without consulting shape.
      This also retires a false positive in the heuristic: a dataset that is
      legitimately 16x16 at full resolution used to be deleted on every run.
    - **absent** -- unknown, so fall back to the shape heuristic. Absence must
      never be read as "full resolution": every dataset written before the
      stamp landed is absent, and treating those as full would resurrect the
      original bug.

    Interferometer datasets are covered by the stamp and were not covered
    before: their visibility count is fixed by the committed uv file while the
    real-space grid behind it is capped, so a capped run writes a ``data.fits``
    with *identical* ``NAXIS`` and different values. That fails silently -- no
    shape mismatch, no assertion -- which is why a shape heuristic could never
    reach it (PyAutoNerves#153).

    Known gap
    ---------
    This reads ``<dataset_path>/data.fits`` and nothing else, which covers
    roughly 228 of the 253 ``should_simulate`` call sites in autolens_workspace.
    The rest have no file of that name at that level and so get no verdict:

    - interferometer **datacube** datasets, whose FITS sit in ``channel_XXX/``
      subdirectories;
    - **multi_dataset** datasets, which prefix the name (``{waveband}_data.fits``);
    - **sample** datasets, which nest under ``dataset_N/``;
    - the two FITS-less directories, ``dataset/weak/simple`` and
      ``dataset/point_source/multiple_sources``, which a FITS-header stamp
      cannot reach under any placement.

    All of those fail *safe*: no ``data.fits`` means no stamp, which means
    unknown, which means keep. Nothing is deleted that should not be.

    Of those, only the first three can actually harbour the stale-capped-dataset
    bug. The FITS-less pair, which look like the worst gap, are the least
    urgent: ``dataset/weak/simple`` is regime-**invariant** -- nothing in its
    write path reads ``PYAUTO_SMALL_DATASETS``, so a capped run and a full run
    produce an identical ``dataset.json`` and there is nothing to detect -- and
    ``dataset/point_source/multiple_sources``, which *is* regime-dependent, is
    excluded from harness execution by ``config/build/no_run.yaml`` pending
    PyAutoLens#480. Both of those facts will expire; see the follow-up.

    Widening the lookup is deliberately **not** done here. This predicate ends
    in ``shutil.rmtree``, and every trap recorded in autolens_workspace_test#260
    was about a widened match hitting a file it should not have -- a bare
    ``*.fits`` glob would delete every PSF-carrying dataset on every run. Growing
    the reach of a destructive predicate is its own change, with its own review,
    not a rider on the one that changes where its input comes from.

    Note that point-source datasets are **not** in this gap: they write a
    top-level ``data.fits`` alongside their JSON and are covered normally. The
    original issue text grouped them with weak lensing as "JSON with no FITS";
    that is true of weak lensing only.
    """
    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
        if Path(dataset_path).exists():
            shutil.rmtree(dataset_path)

        return not Path(dataset_path).exists()

    if Path(dataset_path).exists():
        stamp = _small_datasets_stamp_on_disk(dataset_path)

        if stamp is True:
            # Written by a capped run, on the writer's own authority.
            shutil.rmtree(dataset_path)
        elif stamp is None and _is_small_datasets_on_disk(dataset_path):
            # No stamp to trust, so fall back to inferring from shape.
            shutil.rmtree(dataset_path)
        # stamp is False -> known full resolution -> keep it, unconditionally.

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
