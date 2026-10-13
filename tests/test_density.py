"""Region and density tests."""

import numpy as np
import pytest

from cdw.calibrate import GroundPlane
from cdw.density import DensityEstimator, Region
from cdw.detect import Detection

IMAGE_POINTS = [(380.0, 640.0), (900.0, 640.0), (760.0, 360.0), (520.0, 360.0)]
WORLD_POINTS = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]


@pytest.fixture
def ground():
    return GroundPlane(IMAGE_POINTS, WORLD_POINTS)


@pytest.fixture
def region(ground):
    return Region(IMAGE_POINTS, ground, name="corridor")


def box_at(x, y, w=20, h=60):
    """A box whose bottom-centre is (x, y)."""
    return Detection(x - w / 2, y - h, x + w / 2, y, conf=0.9)


def test_region_area(region):
    assert region.area_m2 == pytest.approx(6.0, abs=1e-9)


def test_interior_point_is_inside(region):
    assert region.contains((640, 500))


def test_exterior_point_is_outside(region):
    assert not region.contains((50, 50))
    assert not region.contains((640, 700))


def test_filter_uses_the_ground_reference_point(region):
    """A box whose feet are inside counts even if its head is outside.

    This is the whole reason the reference point is configurable: the box
    straddles the region boundary, and only one of its corners is the point
    that actually lies on the calibrated ground plane.
    """
    straddling = Detection(630, 300, 650, 500, conf=0.9)  # feet at (640, 500)
    assert len(region.filter([straddling], "box_bottom")) == 1
    assert len(region.filter([straddling], "box_top")) == 0


def test_degenerate_region_is_rejected(ground):
    with pytest.raises(ValueError, match="at least 3"):
        Region([(0, 0), (1, 1)], ground)


def test_mask_covers_roughly_the_polygon(region):
    mask = region.mask((720, 1280))
    assert mask[500, 640] == 255
    assert mask[50, 50] == 0


def test_density_is_count_over_area(region):
    est = DensityEstimator(region, window_s=3.0)
    sample = est.update(0.0, 12)
    assert sample.raw_density == pytest.approx(2.0)   # 12 / 6 m^2
    assert sample.density == pytest.approx(2.0)       # nothing to average yet


def test_smoothing_averages_over_the_window(region):
    """A one-frame dropout should barely move the smoothed value."""
    est = DensityEstimator(region, window_s=3.0)
    for i in range(30):                 # 2 s at 15 fps, steady 12 people
        est.update(i / 15.0, 12)
    dip = est.update(30 / 15.0, 0)      # detector loses everyone for one frame
    assert dip.raw_density == 0.0
    assert dip.density > 1.9            # smoothed value holds


def test_window_drops_stale_samples(region):
    est = DensityEstimator(region, window_s=1.0)
    for i in range(10):
        est.update(i / 10.0, 6)         # 1.0 people/m^2 for 1 s
    late = est.update(5.0, 30)          # long gap, then a spike
    # Everything older than 1 s is gone, so only the spike remains.
    assert late.window_frames == 1
    assert late.density == pytest.approx(5.0)


def test_window_is_in_seconds_not_frames(region):
    """Same real duration at different frame rates gives the same answer."""
    counts = lambda t: 6 if t < 1.0 else 18

    slow = DensityEstimator(region, window_s=2.0)
    for i in range(20):                 # 10 fps for 2 s
        s_slow = slow.update(i / 10.0, counts(i / 10.0))

    fast = DensityEstimator(region, window_s=2.0)
    for i in range(60):                 # 30 fps for 2 s
        s_fast = fast.update(i / 30.0, counts(i / 30.0))

    assert s_slow.density == pytest.approx(s_fast.density, rel=0.05)


def test_reset_clears_history(region):
    est = DensityEstimator(region, window_s=3.0)
    est.update(0.0, 24)
    est.reset()
    assert est.update(0.1, 6).density == pytest.approx(1.0)


def test_non_positive_window_is_rejected(region):
    with pytest.raises(ValueError, match="positive"):
        DensityEstimator(region, window_s=0.0)


def test_world_positions_land_inside_the_rectangle(region):
    dets = [box_at(640, 500), box_at(500, 620)]
    world = region.world_positions(dets, "box_bottom")
    assert world.shape == (2, 2)
    assert np.all(world[:, 0] >= -0.01) and np.all(world[:, 0] <= 3.01)
    assert np.all(world[:, 1] >= -0.01) and np.all(world[:, 1] <= 2.01)


# -- region with a head-height correction -----------------------------------


def test_region_area_is_unaffected_by_the_correction(ground):
    """The outline is drawn on the ground, so only people need correcting."""
    from cdw.calibrate import HeightCorrection

    hc = HeightCorrection((1.5, -4.0), 6.0, 1.7)
    plain = Region(IMAGE_POINTS, ground)
    corrected = Region(IMAGE_POINTS, ground, height_correction=hc)
    assert corrected.area_m2 == pytest.approx(plain.area_m2)
    assert corrected.area_m2 == pytest.approx(6.0, abs=1e-9)


def test_correction_changes_who_counts_as_inside(ground):
    """A head near the far edge projects outside the region without it.

    This is the failure the correction prevents: people at the far side of the
    region silently stop being counted, which deflates density exactly where
    the crowd is deepest.
    """
    from cdw.calibrate import HeightCorrection

    nadir, camera_h, head_h = (1.5, -4.0), 6.0, 1.7
    hc = HeightCorrection(nadir, camera_h, head_h)

    person = np.array([1.5, 1.85])          # inside, close to the far edge
    long_point = np.array(nadir) + (person - np.array(nadir)) * (camera_h / (camera_h - head_h))
    head_px = ground.to_image([long_point])[0]

    plain = Region(IMAGE_POINTS, ground)
    corrected = Region(IMAGE_POINTS, ground, height_correction=hc)

    assert not plain.contains(head_px)       # lost
    assert corrected.contains(head_px)       # counted


def test_to_ground_applies_the_correction(ground):
    from cdw.calibrate import HeightCorrection

    nadir, camera_h, head_h = (1.5, -4.0), 6.0, 1.7
    hc = HeightCorrection(nadir, camera_h, head_h)
    person = np.array([2.0, 1.0])
    long_point = np.array(nadir) + (person - np.array(nadir)) * (camera_h / (camera_h - head_h))
    head_px = ground.to_image([long_point])[0]

    region = Region(IMAGE_POINTS, ground, height_correction=hc)
    assert np.allclose(region.to_ground(head_px), person, atol=1e-9)
