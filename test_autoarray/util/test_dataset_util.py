import numpy as np

import autoarray as aa

from autoarray.util.dataset_util import (
    cap_array_2d_for_small_datasets,
    SMALL_DATASETS_SHAPE_NATIVE,
    SMALL_DATASETS_PIXEL_SCALES,
)


def _array_2d(shape, fill=1.0, pixel_scales=0.08):
    return aa.Array2D.no_mask(
        values=fill * np.ones(shape), pixel_scales=pixel_scales
    )


def test__env_unset__returns_inputs_unchanged(monkeypatch):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    array = _array_2d((150, 150))
    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is array
    assert pixel_scales == 0.08


def test__env_set__shape_already_at_cap__relabels_pixel_scales_without_cropping(
    monkeypatch,
):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    array = _array_2d(SMALL_DATASETS_SHAPE_NATIVE, pixel_scales=0.08)
    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is not array
    assert result.shape_native == SMALL_DATASETS_SHAPE_NATIVE
    assert pixel_scales == SMALL_DATASETS_PIXEL_SCALES
    assert result.pixel_scales == (
        SMALL_DATASETS_PIXEL_SCALES,
        SMALL_DATASETS_PIXEL_SCALES,
    )
    assert (result.native.array == array.native.array).all()


def test__env_set__shape_below_cap__relabels_pixel_scales_without_cropping(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    raw = np.arange(10 * 10, dtype=float).reshape(10, 10)
    array = aa.Array2D.no_mask(values=raw, pixel_scales=0.08)

    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is not array
    # Shape is PRESERVED — the below-cap branch relabels, it must never crop.
    assert result.shape_native == (10, 10)
    assert pixel_scales == SMALL_DATASETS_PIXEL_SCALES
    assert result.pixel_scales == (
        SMALL_DATASETS_PIXEL_SCALES,
        SMALL_DATASETS_PIXEL_SCALES,
    )
    assert (result.native.array == raw).all()


def test__env_set__shape_above_cap__center_crops_and_overrides_pixel_scales(
    monkeypatch,
):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    raw = np.arange(150 * 150, dtype=float).reshape(150, 150)
    array = aa.Array2D.no_mask(values=raw, pixel_scales=0.08)

    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result is not array
    assert result.shape_native == SMALL_DATASETS_SHAPE_NATIVE
    assert pixel_scales == SMALL_DATASETS_PIXEL_SCALES

    cap_h, cap_w = SMALL_DATASETS_SHAPE_NATIVE
    h0, w0 = (150 - cap_h) // 2, (150 - cap_w) // 2
    expected = raw[h0:h0 + cap_h, w0:w0 + cap_w]
    assert (result.native.array == expected).all()


def test__env_set__non_square_above_cap__center_crops_to_16x16(monkeypatch):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    array = _array_2d((100, 50))
    result, pixel_scales = cap_array_2d_for_small_datasets(array, 0.08)

    assert result.shape_native == SMALL_DATASETS_SHAPE_NATIVE
    assert pixel_scales == SMALL_DATASETS_PIXEL_SCALES


"""
__should_simulate — regime transitions__

`should_simulate` must regenerate a dataset whenever the resolution regime on
disk differs from the one in force. Before autolens_workspace_test#260 only the
full->small transition was implemented; small->full silently reused capped FITS
at full resolution, producing deterministic failures that no CI run could
reproduce (CI clones fresh, so it never has a stale dataset).

All four transitions are covered below because the bug was precisely that one
of the four was never exercised.
"""

import json

from autoarray.util.dataset_util import (
    should_simulate,
    _is_small_datasets_on_disk,
    _on_disk_shape_native,
    _small_datasets_stamp_on_disk,
    _stamp_contradicted_by_shape,
    SMALL_DATASETS_HEADER_KEY,
)

def _set_stamp(file_path, stamp):
    """
    Force the ``SMALLDAT`` provenance card on an already-written FITS file.

    ``aa.output_to_fits`` stamps whatever regime is in force when it runs, which
    is not always the regime a test needs the file to claim. The three values:

    - ``True`` / ``False`` -- overwrite the card, so a test can write a dataset
      that *claims* a regime independently of the env it was written under.
    - ``None`` -- strip the card, producing a **legacy** file byte-equivalent to
      one written before PyAutoNerves#153. This is what keeps the shape-based
      fallback exercised; without it the fallback would silently become dead
      code and every pre-stamp dataset on disk would lose its protection.
    """
    from astropy.io import fits

    with fits.open(file_path, mode="update") as hdu_list:
        for hdu in hdu_list:
            if stamp is None:
                hdu.header.pop(SMALL_DATASETS_HEADER_KEY, None)
            else:
                hdu.header[SMALL_DATASETS_HEADER_KEY] = stamp


def _write_dataset(dataset_path, shape, extra_files=(), stamp="auto"):
    """
    Write a minimal dataset directory containing a `data.fits` of `shape`.

    ``stamp="auto"`` leaves whatever regime was in force at write time; any
    other value is forced onto the header via :func:`_set_stamp`.
    """
    dataset_path.mkdir(parents=True, exist_ok=True)

    aa.output_to_fits(
        values=np.ones(shape),
        file_path=str(dataset_path / "data.fits"),
        overwrite=True,
    )

    if stamp != "auto":
        _set_stamp(dataset_path / "data.fits", stamp)

    for name, file_shape in extra_files:
        aa.output_to_fits(
            values=np.ones(file_shape),
            file_path=str(dataset_path / name),
            overwrite=True,
        )

    return dataset_path


def test__small_regime__existing_full_dataset__is_deleted_and_resimulated(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")
    dataset_path = _write_dataset(tmp_path / "dataset", (180, 180))

    assert should_simulate(str(dataset_path)) is True
    assert not dataset_path.exists()


def test__small_regime__existing_small_dataset__is_still_deleted_and_resimulated(
    monkeypatch, tmp_path
):
    # The small path is unconditional by design: it cannot know the capped
    # dataset on disk was produced by the SAME cap, so it always regenerates.
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")
    dataset_path = _write_dataset(tmp_path / "dataset", SMALL_DATASETS_SHAPE_NATIVE)

    assert should_simulate(str(dataset_path)) is True
    assert not dataset_path.exists()


def test__full_regime__stale_small_dataset__is_deleted_and_resimulated(
    monkeypatch, tmp_path
):
    # THE REGRESSION TEST. Before autolens_workspace_test#260 this returned
    # False and the capped FITS were loaded at full resolution.
    #
    # The dataset is WRITTEN under the cap so it carries a truthful
    # ``SMALLDAT = T``, then read back in the full regime -- the actual
    # sequence that produced the bug. Writing it with the env unset would
    # stamp it ``F`` and describe a dataset that cannot exist.
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")
    dataset_path = _write_dataset(tmp_path / "dataset", SMALL_DATASETS_SHAPE_NATIVE)
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    assert should_simulate(str(dataset_path)) is True
    assert not dataset_path.exists()


def test__full_regime__stale_small_dataset__no_stamp__shape_fallback_still_deletes(
    monkeypatch, tmp_path
):
    # The fallback is NOT throwaway work: every dataset already on disk when
    # the stamp landed is unstamped, and the stamp can do nothing for those.
    # A legacy capped dataset must still be caught by its shape.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(
        tmp_path / "dataset", SMALL_DATASETS_SHAPE_NATIVE, stamp=None
    )

    assert _small_datasets_stamp_on_disk(str(dataset_path)) is None
    assert should_simulate(str(dataset_path)) is True
    assert not dataset_path.exists()


def test__full_regime__full_dataset_at_cap_shape__stamp_keeps_it(
    monkeypatch, tmp_path
):
    # The stamp RETIRES a false positive in the shape heuristic. A dataset that
    # is legitimately 16x16 at full resolution is indistinguishable from a
    # capped one by shape alone, so the heuristic deleted it on every single
    # run. A truthful ``SMALLDAT = F`` is the only thing that can save it --
    # and it must win WITHOUT the shape check getting a vote.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(
        tmp_path / "dataset", SMALL_DATASETS_SHAPE_NATIVE, stamp=False
    )

    assert _is_small_datasets_on_disk(str(dataset_path)) is True
    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()


def test__full_regime__full_dataset__is_kept(monkeypatch, tmp_path):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(tmp_path / "dataset", (180, 180))

    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()


def test__full_regime__absent_dataset__simulates(monkeypatch, tmp_path):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    assert should_simulate(str(tmp_path / "does_not_exist")) is True


"""
__should_simulate — false-positive guards__

The full-regime branch ends in `shutil.rmtree`, so every one of these asserts
that a dataset is PRESERVED. A regression here silently deletes real data.
"""


def test__full_regime__tiny_psf_alongside_full_data__is_kept(monkeypatch, tmp_path):
    # PSF kernels are legitimately tiny at full resolution (11x11 is the common
    # workspace value, and a 16x16 PSF is a plausible one). The check must key
    # on `data.fits` by name — a "first FITS in the directory" implementation
    # would delete this dataset on every run forever.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(
        tmp_path / "dataset",
        (180, 180),
        extra_files=(("psf.fits", (11, 11)), ("noise_map.fits", (180, 180))),
    )

    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()
    assert (dataset_path / "psf.fits").exists()


def test__full_regime__psf_at_exactly_the_cap_shape__does_not_trigger_deletion(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(
        tmp_path / "dataset",
        (180, 180),
        extra_files=(("psf.fits", SMALL_DATASETS_SHAPE_NATIVE),),
    )

    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()


def test__full_regime__below_cap_data__is_kept_because_the_cap_emits_exactly_16x16(
    monkeypatch, tmp_path
):
    # The cap rewrites anything larger to EXACTLY (16, 16) and never produces
    # 12x12, so a 12x12 dataset was not capped and must be left alone. This is
    # why the predicate is `==` and not `<=`.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(tmp_path / "dataset", (12, 12))

    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()


def test__full_regime__json_only_dataset__is_kept(monkeypatch, tmp_path):
    # Point-source and weak-lensing datasets carry no FITS at all. The check
    # cannot speak about them, so it must fall back to existence-only rather
    # than delete. (These remain exposed to the underlying bug — see the
    # "Known gap" section of should_simulate's docstring.)
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "point_dataset.json").write_text(json.dumps({"positions": []}))

    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "point_dataset.json").exists()


def test__full_regime__unreadable_data_fits__is_kept(monkeypatch, tmp_path):
    # "Unknown regime" must never mean "delete".
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "data.fits").write_bytes(b"not a fits file")

    assert _on_disk_shape_native(dataset_path / "data.fits") is None
    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()


def test__is_small_datasets_on_disk__reads_shape_from_header(tmp_path):
    small = _write_dataset(tmp_path / "small", SMALL_DATASETS_SHAPE_NATIVE)
    full = _write_dataset(tmp_path / "full", (180, 180))

    assert _is_small_datasets_on_disk(str(small)) is True
    assert _is_small_datasets_on_disk(str(full)) is False


def test__on_disk_shape_native__is_row_column_ordered(tmp_path):
    # NAXIS1 is columns and NAXIS2 is rows, so a non-square array must come
    # back in numpy (rows, columns) order rather than transposed.
    dataset_path = _write_dataset(tmp_path / "dataset", (30, 50))

    assert _on_disk_shape_native(dataset_path / "data.fits") == (30, 50)


def test__stamp_reader__non_boolean_card__is_unknown_not_true(monkeypatch, tmp_path):
    # bool("F") is True in Python. A string card left by a hand-edit or a
    # third-party tool must therefore NEVER be coerced: doing so would report
    # "capped" for a file claiming the opposite and feed that to shutil.rmtree.
    # Only a genuine FITS boolean counts; anything else is unknown.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(tmp_path / "dataset", (180, 180))
    _set_stamp(dataset_path / "data.fits", "F")

    assert _small_datasets_stamp_on_disk(str(dataset_path)) is None
    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()


def test__stamp_reader__unreadable_and_missing__are_unknown(monkeypatch, tmp_path):
    # "Unknown regime" must mean "leave it alone", never "delete".
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    missing = tmp_path / "no_such_dataset"
    assert _small_datasets_stamp_on_disk(str(missing)) is None

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "data.fits").write_text("this is not a FITS file")
    assert _small_datasets_stamp_on_disk(str(corrupt)) is None
    assert should_simulate(str(corrupt)) is False
    assert (corrupt / "data.fits").exists()


def test__interferometer_shaped_dataset__stamp_catches_what_shape_cannot(
    monkeypatch, tmp_path
):
    # THE CASE THIS TASK EXISTS FOR. An interferometer dataset's visibility
    # count is fixed by the committed uv file while the real-space grid behind
    # it is capped, so a capped run writes a data.fits with IDENTICAL NAXIS and
    # different values. There is no shape mismatch and no assertion to trip --
    # it fails silently, which is strictly worse than the loud imaging failure.
    #
    # Shape is provably blind to it; the stamp provably is not.
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")
    dataset_path = _write_dataset(tmp_path / "dataset", (360, 2))
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    # The heuristic that fixed the imaging case sees nothing wrong here.
    assert _is_small_datasets_on_disk(str(dataset_path)) is False
    # The stamp does.
    assert _small_datasets_stamp_on_disk(str(dataset_path)) is True
    assert should_simulate(str(dataset_path)) is True
    assert not dataset_path.exists()


def test__psf_carrying_dataset__is_not_deleted_by_a_glob(monkeypatch, tmp_path):
    # Guards the predecessor's trap: PSF kernels are legitimately tiny at full
    # resolution, so anything keying on "the first FITS in the directory"
    # instead of data.fits by name would delete every PSF-carrying dataset on
    # every run. The stamp path must not reintroduce that.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)
    dataset_path = _write_dataset(
        tmp_path / "dataset", (180, 180), extra_files=(("psf.fits", (11, 11)),)
    )

    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "psf.fits").exists()


def test__stamp_true_on_a_full_resolution_image__is_contradicted_and_kept(
    monkeypatch, tmp_path
):
    # THE DATA-LOSS REGRESSION. A user converting real 300x300 telescope data in
    # a shell exporting PYAUTO_SMALL_DATASETS=1 -- the documented harness
    # default -- stamps T on full-resolution data. Acting on that stamp alone
    # deletes their data, and the pre-stamp shape heuristic explicitly REFUSED
    # to: a stamp-preferring rule must not be a strict weakening of the safety
    # property PyAutoArray#471 established.
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")
    dataset_path = _write_dataset(tmp_path / "dataset", (300, 300))
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    assert _small_datasets_stamp_on_disk(str(dataset_path)) is True
    assert _is_small_datasets_on_disk(str(dataset_path)) is False  # #471 said keep
    assert _stamp_contradicted_by_shape(str(dataset_path)) is True

    assert should_simulate(str(dataset_path)) is False
    assert (dataset_path / "data.fits").exists()


def test__contradiction_guard__needs_BOTH_axes_over_the_cap(monkeypatch, tmp_path):
    # Both axes, never either. Interferometer data.fits is (n_visibilities, 2) --
    # 108384 x 2 for the committed sdp81 dataset -- so an "either axis" test
    # would refuse to delete the exact family the stamp exists to catch.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    interferometer = _write_dataset(tmp_path / "interf", (108384, 2), stamp=True)
    assert _stamp_contradicted_by_shape(str(interferometer)) is False

    imaging = _write_dataset(tmp_path / "imaging", (151, 151), stamp=True)
    assert _stamp_contradicted_by_shape(str(imaging)) is True

    at_cap = _write_dataset(tmp_path / "at_cap", SMALL_DATASETS_SHAPE_NATIVE, stamp=True)
    assert _stamp_contradicted_by_shape(str(at_cap)) is False


def test__contradiction_guard__unknown_shape_does_not_block_a_deletion(
    monkeypatch, tmp_path
):
    # The guard only ever BLOCKS a delete, so an unreadable file must not
    # silently protect a dataset the stamp correctly identified as stale.
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    missing = tmp_path / "gone"
    assert _stamp_contradicted_by_shape(str(missing)) is False
