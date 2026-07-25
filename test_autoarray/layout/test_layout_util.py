import numpy as np
import pytest
import autoarray as aa


def test__rotate_array_via_roe_corner_from():
    arr = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])

    arr_bl = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr, roe_corner=(1, 0)
    )

    assert arr_bl == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )

    arr_bl = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr_bl, roe_corner=(1, 0)
    )

    assert arr_bl == pytest.approx(np.array(arr), 1.0e-4)

    arr_br = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr, roe_corner=(1, 1)
    )

    assert arr_br == pytest.approx(
        np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 0.0]]), 1.0e-4
    )

    arr_br = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr_br, roe_corner=(1, 1)
    )

    assert arr_br == pytest.approx(np.array(arr), 1.0e-4)

    arr_tl = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr, roe_corner=(0, 0)
    )

    assert arr_tl == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]]), 1.0e-4
    )

    arr_tl = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr_tl, roe_corner=(0, 0)
    )

    assert arr_tl == pytest.approx(np.array(arr), 1.0e-4)

    arr_tr = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr, roe_corner=(0, 1)
    )

    assert arr_tr == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 0.0]]), 1.0e-4
    )

    arr_tr = aa.util.layout.rotate_array_via_roe_corner_from(
        array=arr_tr, roe_corner=(0, 1)
    )

    assert arr_tr == pytest.approx(np.array(arr), 1.0e-4)


def test__rotate_region_via_roe_corner_from():
    region = (0, 2, 1, 3)

    shape_native = (8, 10)

    region_bl = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region, shape_native=shape_native, roe_corner=(1, 0)
    )

    assert region_bl == (0, 2, 1, 3)

    region_bl = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region_bl, shape_native=shape_native, roe_corner=(1, 0)
    )

    assert region_bl == (0, 2, 1, 3)

    region_br = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region, shape_native=shape_native, roe_corner=(1, 1)
    )

    assert region_br == (0, 2, 7, 9)

    region_br = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region_br, shape_native=shape_native, roe_corner=(1, 1)
    )

    assert region_br == (0, 2, 1, 3)

    region_tl = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region, shape_native=shape_native, roe_corner=(0, 0)
    )

    assert region_tl == (6, 8, 1, 3)

    region_tl = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region_tl, shape_native=shape_native, roe_corner=(0, 0)
    )

    assert region_tl == (0, 2, 1, 3)

    region_tr = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region, shape_native=shape_native, roe_corner=(0, 1)
    )

    assert region_tr == (6, 8, 7, 9)

    region_tr = aa.util.layout.rotate_region_via_roe_corner_from(
        region=region_tr, shape_native=shape_native, roe_corner=(0, 1)
    )

    assert region_tr == (0, 2, 1, 3)


def test__region_after_extraction():
    region = aa.util.layout.region_after_extraction(
        original_region=(2, 4, 2, 4), extraction_region=(0, 6, 0, 6)
    )

    assert region == (2, 4, 2, 4)

    region = aa.util.layout.region_after_extraction(
        original_region=(2, 4, 2, 4), extraction_region=(3, 5, 3, 5)
    )

    assert region == (0, 1, 0, 1)

    region = aa.util.layout.region_after_extraction(
        original_region=(2, 4, 2, 4), extraction_region=(2, 5, 2, 5)
    )

    assert region == (0, 2, 0, 2)

    region = aa.util.layout.region_after_extraction(
        original_region=(2, 4, 2, 4), extraction_region=(0, 3, 0, 3)
    )

    assert region == (2, 3, 2, 3)


def test__region_after_extraction__regions_do_not_overlap__returns_none():
    region = aa.util.layout.region_after_extraction(
        original_region=(2, 4, 2, 4), extraction_region=(0, 6, 0, 1)
    )

    assert region == None

    region = aa.util.layout.region_after_extraction(
        original_region=(2, 4, 2, 4), extraction_region=(0, 1, 0, 6)
    )

    assert region == None

    region = aa.util.layout.region_after_extraction(
        original_region=(2, 4, 2, 4), extraction_region=(0, 1, 0, 1)
    )

    assert region == None

    region = aa.util.layout.region_after_extraction(
        original_region=None, extraction_region=(0, 6, 0, 1)
    )

    assert region == None


def test__x0x1_after_extraction():
    # Simple extractions

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=0, x1o=6, x0e=2, x1e=4)

    assert x0 == 0
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=0, x1o=6, x0e=3, x1e=5)

    assert x0 == 0
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=0, x1o=6, x0e=4, x1e=6)

    assert x0 == 0
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=0, x1o=6, x0e=5, x1e=6)

    assert x0 == 0
    assert x1 == 1

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=0, x1o=6, x0e=2, x1e=5)

    assert x0 == 0
    assert x1 == 3

    # 1d_extracted region is fully within original region

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=2, x1e=4)

    assert x0 == 0
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=3, x1e=5)

    assert x0 == 0
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=4, x1e=6)

    assert x0 == 0
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=5, x1e=6)

    assert x0 == 0
    assert x1 == 1

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=2, x1e=5)

    assert x0 == 0
    assert x1 == 3

    # 1d extracted region partly overlaps to left original region

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=1, x1e=3)

    assert x0 == 1
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=1, x1e=4)

    assert x0 == 1
    assert x1 == 3

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=3)

    assert x0 == 2
    assert x1 == 3

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=5)

    assert x0 == 2
    assert x1 == 5

    # 1D extracted region partly overlaps_to right original region

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=5, x1e=7)

    assert x0 == 0
    assert x1 == 1

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=5, x1e=8)

    assert x0 == 0
    assert x1 == 1

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=4, x1e=7)

    assert x0 == 0
    assert x1 == 2

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=2, x1e=8)

    assert x0 == 0
    assert x1 == 4

    # extraction region over full original region

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=8)

    assert x0 == 2
    assert x1 == 6

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=7)

    assert x0 == 2
    assert x1 == 6

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=1, x1e=8)

    assert x0 == 1
    assert x1 == 5

    # extraction region misses original region

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=7, x1e=8)

    assert x0 is None
    assert x1 is None

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=6, x1e=8)

    assert x0 is None
    assert x1 is None

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=1)

    assert x0 is None
    assert x1 is None

    x0, x1 = aa.util.layout.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=2)

    assert x0 is None
    assert x1 is None
