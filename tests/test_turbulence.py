"""Turbulence / crowd-pressure tests.

The behaviour that has to hold is the discrimination the whole measure exists
for: a fast but orderly crowd must score low, a slow but conflicting crowd must
score high. If that inverts, the warning rule is worse than useless.
"""

import cv2
import numpy as np
import pytest

from cdw.calibrate import GroundPlane
from cdw.detect import Detection
from cdw.turbulence import TrackVelocities, crowd_pressure, local_velocity_variance

IMAGE_POINTS = [(380.0, 640.0), (900.0, 640.0), (760.0, 360.0), (520.0, 360.0)]
WORLD_POINTS = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]


@pytest.fixture
def ground():
    return GroundPlane(IMAGE_POINTS, WORLD_POINTS)


def grid_positions(n=16, spacing=0.5):
    side = int(np.sqrt(n))
    return np.array(
        [[i * spacing, j * spacing] for i in range(side) for j in range(side)],
        dtype=float,
    )


# -- local velocity variance ------------------------------------------------


def test_uniform_flow_has_zero_local_variance():
    pos = grid_positions()
    vel = np.tile([1.2, 0.0], (len(pos), 1))
    assert local_velocity_variance(pos, vel, k=5) == pytest.approx(0.0, abs=1e-12)


def test_fast_orderly_flow_scores_below_slow_conflicting_flow():
    """The measure must not be a proxy for speed."""
    pos = grid_positions()

    orderly = np.tile([1.5, 0.0], (len(pos), 1))            # everyone, 1.5 m/s
    conflicting = np.tile([0.2, 0.0], (len(pos), 1))
    conflicting[::2] *= -1                                   # alternating, 0.2 m/s

    assert local_velocity_variance(pos, orderly, k=5) < local_velocity_variance(
        pos, conflicting, k=5
    )


def test_variance_grows_with_disorder():
    rng = np.random.default_rng(0)
    pos = grid_positions()
    base = np.tile([1.0, 0.0], (len(pos), 1))
    low = local_velocity_variance(pos, base + rng.normal(0, 0.05, base.shape), k=5)
    high = local_velocity_variance(pos, base + rng.normal(0, 0.80, base.shape), k=5)
    assert high > low * 5


def test_local_beats_global_for_counterflow():
    """Two separated lanes moving oppositely are orderly, not turbulent.

    Globally the velocity variance is large; locally, each person's neighbours
    move the same way they do. The local measure should see that.
    """
    left = np.array([[0.2, y] for y in np.linspace(0, 2, 10)])
    right = np.array([[2.8, y] for y in np.linspace(0, 2, 10)])
    pos = np.vstack([left, right])
    vel = np.vstack([np.tile([1.0, 0.0], (10, 1)), np.tile([-1.0, 0.0], (10, 1))])

    local = local_velocity_variance(pos, vel, k=3, radius_m=0.5)
    global_var = float(vel.var(axis=0).sum())
    assert local < global_var * 0.1


def test_radius_cap_excludes_distant_neighbours():
    pos = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 0.0], [10.2, 0.0]])
    vel = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    assert local_velocity_variance(pos, vel, k=3, radius_m=1.0) == pytest.approx(0.0, abs=1e-12)
    assert local_velocity_variance(pos, vel, k=3, radius_m=None) > 0.1


def test_empty_and_tiny_inputs_are_safe():
    assert local_velocity_variance(np.empty((0, 2)), np.empty((0, 2))) == 0.0
    assert local_velocity_variance([[0.0, 0.0]], [[1.0, 0.0]]) == 0.0


# -- pressure ---------------------------------------------------------------


def test_pressure_is_density_times_variance():
    pos = grid_positions()
    rng = np.random.default_rng(1)
    vel = rng.normal(0, 0.5, (len(pos), 2))
    var = local_velocity_variance(pos, vel, k=5, radius_m=2.0)
    sample = crowd_pressure(0.0, density=4.0, positions=pos, velocities=vel,
                            k=5, radius_m=2.0)
    assert sample.pressure == pytest.approx(4.0 * var)


def test_pressure_is_zero_without_motion():
    pos = grid_positions()
    vel = np.zeros_like(pos)
    assert crowd_pressure(0.0, 5.0, pos, vel).pressure == pytest.approx(0.0)


def test_pressure_is_zero_in_an_empty_region():
    sample = crowd_pressure(0.0, 0.0, np.empty((0, 2)), np.empty((0, 2)))
    assert sample.pressure == 0.0
    assert sample.n_velocities == 0


# -- track velocities -------------------------------------------------------


def det_at_pixel(u, v, tid):
    return Detection(u - 10, v - 60, u + 10, v, conf=0.9, track_id=tid)


def test_first_frame_yields_no_velocity(ground):
    tv = TrackVelocities(ground, lag_s=0.3)
    pos, vel = tv.update(0.0, [det_at_pixel(640, 500, 1)])
    assert len(vel) == 0


def test_velocity_recovers_a_known_ground_speed(ground):
    """Move a track 0.5 m in 0.5 s; expect 1.0 m/s."""
    tv = TrackVelocities(ground, lag_s=1.0)
    start_px = ground.to_image([(1.0, 1.0)])[0]
    end_px = ground.to_image([(1.5, 1.0)])[0]

    tv.update(0.0, [det_at_pixel(*start_px, 1)])
    pos, vel = tv.update(0.5, [det_at_pixel(*end_px, 1)])

    assert len(vel) == 1
    assert np.linalg.norm(vel[0]) == pytest.approx(1.0, rel=1e-6)
    assert vel[0][0] == pytest.approx(1.0, rel=1e-6)   # motion is along +x
    assert vel[0][1] == pytest.approx(0.0, abs=1e-6)


def test_untracked_detections_are_ignored(ground):
    tv = TrackVelocities(ground, lag_s=1.0)
    no_id = Detection(630, 440, 650, 500, conf=0.9, track_id=None)
    tv.update(0.0, [no_id])
    pos, vel = tv.update(0.5, [no_id])
    assert len(pos) == 0 and len(vel) == 0


def test_implausible_speed_is_discarded_as_an_id_switch(ground):
    """An id reused across the frame would imply a 20 m/s pedestrian."""
    tv = TrackVelocities(ground, lag_s=1.0, max_speed_mps=6.0)
    a = ground.to_image([(0.1, 1.0)])[0]
    b = ground.to_image([(2.9, 1.0)])[0]
    tv.update(0.0, [det_at_pixel(*a, 1)])
    pos, vel = tv.update(0.1, [det_at_pixel(*b, 1)])
    assert len(vel) == 0


def test_stale_tracks_are_pruned(ground):
    tv = TrackVelocities(ground, lag_s=0.3)
    tv.update(0.0, [det_at_pixel(640, 500, 1)])
    tv.update(10.0, [det_at_pixel(640, 500, 2)])
    assert 1 not in tv._hist


def test_reset_clears_history(ground):
    tv = TrackVelocities(ground, lag_s=1.0)
    tv.update(0.0, [det_at_pixel(640, 500, 1)])
    tv.reset()
    pos, vel = tv.update(0.5, [det_at_pixel(600, 480, 1)])
    assert len(vel) == 0


# -- optical flow fallback --------------------------------------------------


def textured_frame(rng, h=360, w=640):
    """Random texture, so Farneback has something to lock onto."""
    small = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def shifted(frame, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def test_flow_yields_nothing_on_the_first_frame(ground):
    from cdw.turbulence import FarnebackFlow

    rng = np.random.default_rng(0)
    flow = FarnebackFlow(ground, step=32)
    pos, vel = flow.update(0.0, textured_frame(rng))
    assert len(pos) == 0 and len(vel) == 0


def test_flow_is_near_zero_for_identical_frames(ground):
    from cdw.turbulence import FarnebackFlow

    rng = np.random.default_rng(1)
    frame = textured_frame(rng)
    flow = FarnebackFlow(ground, step=32)
    flow.update(0.0, frame)
    pos, vel = flow.update(1 / 15.0, frame)
    assert len(vel) > 0
    assert np.abs(vel).max() < 0.05          # m/s


def test_flow_recovers_the_direction_of_a_known_shift(ground):
    """Shifting the image right must read as motion in world +x."""
    from cdw.turbulence import FarnebackFlow

    rng = np.random.default_rng(2)
    frame = textured_frame(rng)
    flow = FarnebackFlow(ground, step=32)
    flow.update(0.0, frame)
    pos, vel = flow.update(1 / 15.0, shifted(frame, 8, 0))

    assert len(vel) > 0
    assert np.median(vel[:, 0]) > 0.1        # m/s, rightwards
    assert abs(np.median(vel[:, 1])) < abs(np.median(vel[:, 0]))


def test_flow_speed_scales_with_the_shift(ground):
    """Twice the pixel displacement is twice the measured speed."""
    from cdw.turbulence import FarnebackFlow

    rng = np.random.default_rng(3)
    frame = textured_frame(rng)

    def median_speed(dx):
        flow = FarnebackFlow(ground, step=32)
        flow.update(0.0, frame)
        _, vel = flow.update(1 / 15.0, shifted(frame, dx, 0))
        return float(np.median(np.linalg.norm(vel, axis=1)))

    slow, fast = median_speed(4), median_speed(8)
    assert fast == pytest.approx(2 * slow, rel=0.35)


def test_mask_restricts_where_flow_is_sampled(ground):
    from cdw.turbulence import FarnebackFlow

    rng = np.random.default_rng(4)
    frame = textured_frame(rng)
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[100:200, 100:300] = 255

    flow = FarnebackFlow(ground, step=16)
    flow.update(0.0, frame)
    pos_masked, _ = flow.update(1 / 15.0, shifted(frame, 6, 0), mask)

    flow2 = FarnebackFlow(ground, step=16)
    flow2.update(0.0, frame)
    pos_all, _ = flow2.update(1 / 15.0, shifted(frame, 6, 0))

    assert 0 < len(pos_masked) < len(pos_all)


def test_flow_reset_clears_the_previous_frame(ground):
    from cdw.turbulence import FarnebackFlow

    rng = np.random.default_rng(5)
    frame = textured_frame(rng)
    flow = FarnebackFlow(ground, step=32)
    flow.update(0.0, frame)
    flow.reset()
    pos, vel = flow.update(1 / 15.0, frame)
    assert len(vel) == 0                     # treated as a first frame again
