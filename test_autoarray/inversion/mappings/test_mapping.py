import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.mappings.mapping import (
    connected_components_from,
    contours_from_bool_native,
    image_regions_from,
    image_regions_from_slim_mask,
    pix_index_groups_from,
    source_contours_from,
)


class MockMapper:
    """A mapper stripped down to what the image-plane region helpers read from it."""

    def __init__(self, mapping_matrix, mask):
        self.mapping_matrix = mapping_matrix
        self.mask = mask


def make_mask_5x5():
    return aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)


def make_bimodal_mapper():
    """
    A mapper whose single mesh pixel maps to two disconnected 2x2 blocks of a 5x5 mask -- the
    two "multiple images" of one source region.
    """
    mask = make_mask_5x5()

    mapping_matrix = np.zeros(shape=(25, 2))
    mapping_matrix[[0, 1, 5, 6], 0] = 1.0
    mapping_matrix[[18, 19, 23, 24], 0] = 1.0
    mapping_matrix[[12], 1] = 1.0

    return MockMapper(mapping_matrix=mapping_matrix, mask=mask)


def test__connected_components_from__splits_a_disconnected_index_set():
    # A chain 0 - 1 - 2, plus an isolated pixel 3.
    neighbors = Neighbors(
        arr=np.array([[1, -1], [0, 2], [1, -1], [-1, -1]]),
        sizes=np.array([1, 2, 1, 0]),
    )

    components = connected_components_from(indexes=[0, 1, 2, 3], neighbors=neighbors)

    assert len(components) == 2
    assert (components[0] == np.array([0, 1, 2])).all()
    assert (components[1] == np.array([3])).all()


def test__connected_components_from__only_walks_through_indexes_in_the_input():
    neighbors = Neighbors(
        arr=np.array([[1, -1], [0, 2], [1, -1], [-1, -1]]),
        sizes=np.array([1, 2, 1, 0]),
    )

    # Pixel 1 is not in the input, so 0 and 2 are not connected through it.
    components = connected_components_from(indexes=[0, 2], neighbors=neighbors)

    assert len(components) == 2


def test__connected_components_from__ignores_entries_beyond_the_neighbor_size():
    # The rectangular mesh leaves stale (non -1) values beyond `sizes`, which must be ignored.
    neighbors = Neighbors(
        arr=np.array([[1, 3], [0, 3], [3, 3]]),
        sizes=np.array([1, 1, 0]),
    )

    components = connected_components_from(indexes=[0, 1, 2], neighbors=neighbors)

    assert len(components) == 2
    assert (components[0] == np.array([0, 1])).all()
    assert (components[1] == np.array([2])).all()


def test__pix_index_groups_from__flat_and_nested_inputs():
    assert len(pix_index_groups_from(pix_indexes=[0, 1, 2])) == 1
    assert len(pix_index_groups_from(pix_indexes=[[0, 1], [2]])) == 2
    assert pix_index_groups_from(pix_indexes=[]) == []


def test__contours_from_bool_native__single_block_is_one_closed_loop():
    bool_native = np.zeros(shape=(5, 5), dtype=bool)
    bool_native[0:2, 0:2] = True

    contours = contours_from_bool_native(
        bool_native=bool_native, geometry=make_mask_5x5().geometry
    )

    assert len(contours) == 1

    contour = contours[0]

    # One point per pixel edge of the block's boundary (8 edges), closed back onto its start.
    assert contour.shape == (9, 2)
    assert contour[0] == pytest.approx(contour[-1])

    # The block's corners in scaled units: y from +2.5 down to +0.5, x from -2.5 to -0.5.
    assert contour[:, 0].max() == pytest.approx(2.5)
    assert contour[:, 0].min() == pytest.approx(0.5)
    assert contour[:, 1].max() == pytest.approx(-0.5)
    assert contour[:, 1].min() == pytest.approx(-2.5)


def test__image_regions_from__bimodal_mapping_matrix_gives_two_regions():
    mapper = make_bimodal_mapper()

    regions = image_regions_from(mapper=mapper, pix_indexes=[0], min_pixels=1)

    assert len(regions) == 2

    assert (regions[0].slim_indexes == np.array([0, 1, 5, 6])).all()
    assert (regions[1].slim_indexes == np.array([18, 19, 23, 24])).all()

    for region in regions:
        assert len(region.contours) == 1
        assert region.contours[0][0] == pytest.approx(region.contours[0][-1])
        assert region.area() == pytest.approx(4.0)


def test__image_regions_from__min_pixels_discards_small_regions():
    mapper = make_bimodal_mapper()

    # Mesh pixel 1 maps to a single data pixel only.
    assert len(image_regions_from(mapper=mapper, pix_indexes=[1], min_pixels=1)) == 1
    assert len(image_regions_from(mapper=mapper, pix_indexes=[1], min_pixels=2)) == 0


def test__image_regions_from__weight_threshold_removes_weakly_mapped_pixels():
    mask = make_mask_5x5()

    mapping_matrix = np.zeros(shape=(25, 1))
    mapping_matrix[[0, 1, 5, 6], 0] = 1.0
    mapping_matrix[[24], 0] = 0.1

    mapper = MockMapper(mapping_matrix=mapping_matrix, mask=mask)

    assert len(image_regions_from(mapper=mapper, pix_indexes=[0], min_pixels=1)) == 2
    assert (
        len(
            image_regions_from(
                mapper=mapper, pix_indexes=[0], weight_threshold=0.5, min_pixels=1
            )
        )
        == 1
    )


def test__image_regions_from_slim_mask__region_centre_and_flux_methods():
    mask = make_mask_5x5()

    slim_bool = np.zeros(25, dtype=bool)
    slim_bool[[0, 1, 5, 6]] = True

    regions = image_regions_from_slim_mask(mask=mask, slim_bool=slim_bool, min_pixels=1)

    assert len(regions) == 1

    region = regions[0]

    assert region.centre == pytest.approx((1.5, -1.5))

    values = np.zeros(shape=(5, 5))
    values[0, 1] = 4.0
    values[1, 0] = 1.0

    array = aa.Array2D(values=values, mask=mask)

    assert region.flux_from(array=array) == pytest.approx(5.0)
    assert region.brightest_coordinate_from(array=array) == pytest.approx((2.0, -1.0))
    # Flux-weighted: (4 * (2.0, -1.0) + 1 * (1.0, -2.0)) / 5.
    assert region.centroid_from(array=array) == pytest.approx((1.8, -1.2))


def test__source_contours_from__rectangular_returns_one_closed_cell_per_pixel(
    rectangular_mapper_7x7_3x3,
):
    contours = source_contours_from(
        mapper=rectangular_mapper_7x7_3x3, pix_indexes=[0, 4]
    )

    assert len(contours) == 2

    for contour in contours:
        assert contour.shape == (5, 2)
        assert contour[0] == pytest.approx(contour[-1])

    # The central mesh pixel's cell is centred on the mesh origin.
    assert np.mean(contours[1][:-1, 0]) == pytest.approx(0.0, abs=1.0e-6)
    assert np.mean(contours[1][:-1, 1]) == pytest.approx(0.0, abs=1.0e-6)


def test__source_contours_from__delaunay_returns_closed_voronoi_cells(
    delaunay_mapper_9_3x3,
):
    contours = source_contours_from(
        mapper=delaunay_mapper_9_3x3, pix_indexes=list(range(9))
    )

    # Unbounded cells on the mesh's convex hull have no polygon, so fewer cells than pixels.
    assert 0 < len(contours) <= 9

    for contour in contours:
        assert contour.shape[1] == 2
        assert contour[0] == pytest.approx(contour[-1])


def test__mapper_mappings_from__pairs_source_pixels_with_image_regions(
    rectangular_mapper_7x7_3x3,
):
    mappings = rectangular_mapper_7x7_3x3.mappings_from(pix_indexes=[[0, 1], [8]])

    assert len(mappings) == 2

    assert (mappings[0].pix_indexes == np.array([0, 1])).all()
    assert mappings[0].peak_value is None
    assert len(mappings[0].source_contours) == 2
    assert len(mappings[0].image_regions) > 0
    assert len(mappings[0].image_contours) == sum(
        len(region.contours) for region in mappings[0].image_regions
    )


def test__mapper_mappings_from__flat_pix_indexes_is_one_mapping(
    rectangular_mapper_7x7_3x3,
):
    mappings = rectangular_mapper_7x7_3x3.mappings_from(pix_indexes=[0, 1])

    assert len(mappings) == 1
    assert (mappings[0].pix_indexes == np.array([0, 1])).all()
