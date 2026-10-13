"""Pipeline integration tests, model-free.

These run the real Pipeline object over detections we control exactly, so a
failure here is wiring (wrong reference point, wrong area, stages fed out of
order) rather than detector behaviour.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cdw.calibrate import GroundPlane
from cdw.detect import SyntheticDetector
from cdw.pipeline import Pipeline, run_video

IMAGE_POINTS = [[380.0, 640.0], [900.0, 640.0], [760.0, 360.0], [520.0, 360.0]]
WORLD_POINTS = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]]
REPO = Path(__file__).resolve().parents[1]


def make_cfg(gt_path, **overrides):
    cfg = {
        "_dir": str(REPO / "configs"),
        "name": "test",
        "calibration": {"image_points": IMAGE_POINTS, "world_points": WORLD_POINTS},
        "region": {"name": "corridor", "polygon": IMAGE_POINTS},
        "detector": {
            "synthetic": str(gt_path),
            "reference_point": "box_bottom",
            "track": True,
        },
        "density": {"window_s": 3.0},
        "pressure": {"source": "tracks", "lag_s": 0.3, "k": 5, "radius_m": 2.0},
        "thresholds": "configs/thresholds.yaml",
    }
    cfg.update(overrides)
    return cfg


def write_gt(path, frames):
    Path(path).write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return path


def walkers(n_frames, count, speed_mps, jitter_m=0.0, seed=0, fps=15.0):
    """Ground-truth boxes for `count` people crossing a 3 m x 2 m region.

    A steady drift at ``speed_mps`` plus, optionally, independent per-frame
    positional jitter of ``jitter_m``. The jitter is *not* accumulated, which
    models people shuffling in place: it produces large, conflicting
    frame-to-frame velocities without walking anyone out of the region.

    Callers must keep ``speed_mps * n_frames / fps`` short enough that nobody
    reaches the far wall - clipping at the boundary would zero the velocities
    and quietly invalidate any motion assertion.
    """
    ground = GroundPlane(IMAGE_POINTS, WORLD_POINTS)
    rng = np.random.default_rng(seed)
    base = np.column_stack([
        rng.uniform(0.10, 0.40, count),
        rng.uniform(0.30, 1.70, count),
    ])
    travel = speed_mps * n_frames / fps
    assert base[:, 0].max() + travel < 2.95, "walkers would hit the far wall"

    frames = []
    for _ in range(n_frames):
        pos = base + (rng.normal(0.0, jitter_m, base.shape) if jitter_m else 0.0)
        boxes = []
        for tid in range(count):
            u, v = ground.to_image([pos[tid]])[0]
            boxes.append({"xyxy": [u - 10, v - 60, u + 10, v], "conf": 1.0,
                          "track_id": tid + 1})
        frames.append(boxes)
        base = base + np.array([speed_mps / fps, 0.0])
    return frames


@pytest.fixture
def blank_frame():
    return np.zeros((720, 1280, 3), np.uint8)


def run(cfg, n_frames, blank_frame, fps=15.0):
    pipe = Pipeline(cfg)
    return pipe, [pipe.process(i, i / fps, blank_frame) for i in range(n_frames)]


# -- construction -----------------------------------------------------------


def test_summary_reports_area_and_calibration_error(tmp_path, blank_frame):
    gt = write_gt(tmp_path / "gt.json", walkers(5, 6, 1.0))
    pipe = Pipeline(make_cfg(gt))
    summary = pipe.summary()
    assert summary["region_area_m2"] == pytest.approx(6.0)
    assert summary["calibration_error"]["max_m"] < 1e-9
    assert summary["reference_point"] == "box_bottom"


# -- density ----------------------------------------------------------------


def test_known_count_gives_known_density(tmp_path, blank_frame):
    """12 people in 6 m^2 must read 2.0 people/m^2, not approximately."""
    gt = write_gt(tmp_path / "gt.json", walkers(45, 12, 0.0))
    _, records = run(make_cfg(gt), 45, blank_frame)
    assert records[-1].count_region == 12
    assert records[-1].raw_density == pytest.approx(2.0)
    assert records[-1].density == pytest.approx(2.0)


def test_detections_outside_the_region_are_excluded(tmp_path, blank_frame):
    frames = [[
        {"xyxy": [630, 440, 650, 500], "conf": 1.0, "track_id": 1},   # inside
        {"xyxy": [40, 40, 60, 100], "conf": 1.0, "track_id": 2},      # far outside
    ]] * 10
    gt = write_gt(tmp_path / "gt.json", frames)
    _, records = run(make_cfg(gt), 10, blank_frame)
    assert records[-1].count_total == 2
    assert records[-1].count_region == 1


# -- motion -----------------------------------------------------------------


def test_orderly_flow_recovers_its_speed_and_stays_calm(tmp_path, blank_frame):
    """Everyone at 1.2 m/s: mean speed should match, pressure should be ~0.

    8 people in 6 m^2 is 1.33 people/m^2, i.e. LOS C - below the escalation
    band, so an orderly crowd at this density should stay NORMAL.
    """
    gt = write_gt(tmp_path / "gt.json", walkers(30, 8, 1.2))
    _, records = run(make_cfg(gt), 30, blank_frame)
    final = records[-1]
    assert final.n_velocities == 8
    assert final.mean_speed == pytest.approx(1.2, rel=0.05)
    assert final.pressure < 1e-6
    assert final.level == "NORMAL"


def test_disorderly_flow_produces_more_pressure_than_orderly(tmp_path, blank_frame):
    calm = write_gt(tmp_path / "calm.json", walkers(60, 14, 0.3, jitter_m=0.0, seed=1))
    rough = write_gt(tmp_path / "rough.json", walkers(60, 14, 0.3, jitter_m=0.08, seed=1))
    _, calm_rec = run(make_cfg(calm), 60, blank_frame)
    _, rough_rec = run(make_cfg(rough), 60, blank_frame)
    assert max(r.pressure for r in rough_rec) > max(r.pressure for r in calm_rec) * 10


# -- warning ----------------------------------------------------------------


def test_sparse_calm_crowd_never_warns(tmp_path, blank_frame):
    gt = write_gt(tmp_path / "gt.json", walkers(150, 4, 0.2, jitter_m=0.02))
    _, records = run(make_cfg(gt), 150, blank_frame)
    assert {r.level for r in records} == {"NORMAL"}


def test_reset_restores_initial_state(tmp_path, blank_frame):
    gt = write_gt(tmp_path / "gt.json", walkers(30, 6, 0.5))
    pipe = Pipeline(make_cfg(gt))
    for i in range(30):
        pipe.process(i, i / 15.0, blank_frame)
    pipe.reset()
    first = pipe.process(0, 0.0, blank_frame)
    assert first.density == pytest.approx(1.0)
    assert first.pressure_slope == 0.0
    assert first.level == "NORMAL"


# -- reference point --------------------------------------------------------


def test_reference_point_changes_which_detections_count(tmp_path, blank_frame):
    """A box straddling the far edge: feet inside, head above the region."""
    frames = [[{"xyxy": [630, 300, 650, 500], "conf": 1.0, "track_id": 1}]] * 5
    gt = write_gt(tmp_path / "gt.json", frames)

    cfg_feet = make_cfg(gt)
    _, feet = run(cfg_feet, 5, blank_frame)

    cfg_head = make_cfg(gt)
    cfg_head["detector"] = dict(cfg_head["detector"], reference_point="box_top")
    _, head = run(cfg_head, 5, blank_frame)

    assert feet[-1].count_region == 1
    assert head[-1].count_region == 0


# -- full video path --------------------------------------------------------


SYNTH_VIDEO = REPO / "data/raw/synthetic_bottleneck.mp4"
SYNTH_GT = REPO / "data/annotations/synthetic_bottleneck_gt.json"


@pytest.mark.skipif(
    not (SYNTH_VIDEO.exists() and SYNTH_GT.exists()),
    reason="run scripts/make_synthetic_clip.py first",
)
def test_scripted_jam_trips_the_warning():
    """The scripted scenario must escalate NORMAL -> WATCH -> WARNING.

    This is the end-to-end acceptance test for stages 3-6: a clip whose density
    climbs from 1.3 to 4.3 people/m^2 while flow breaks down should produce
    exactly one sustained warning episode, and it should start after the
    density signal does.
    """
    import yaml

    cfg = yaml.safe_load((REPO / "configs/synthetic.yaml").read_text(encoding="utf-8"))
    cfg["_dir"] = str(REPO / "configs")
    records, summary = run_video(cfg, SYNTH_VIDEO)

    assert summary["region_area_m2"] == pytest.approx(6.0)
    levels = [r.level for r in records]
    assert levels[0] == "NORMAL"
    assert "WATCH" in levels
    assert "WARNING" in levels
    assert levels.index("WATCH") < levels.index("WARNING")

    peak = max(r.density for r in records)
    assert 4.0 < peak < 4.6                      # 26 people in 6 m^2

    warn_t = records[levels.index("WARNING")].t
    assert 8.0 < warn_t < 16.0                   # inside the scripted ramp


@pytest.mark.skipif(not SYNTH_GT.exists(), reason="run scripts/make_synthetic_clip.py first")
def test_synthetic_detector_is_exhausted_cleanly():
    det = SyntheticDetector(SYNTH_GT)
    n = len(det)
    for _ in range(n):
        det.detect()
    assert det.detect() == []                    # past the end, not an error
    det.reset()
    assert len(det.detect()) > 0
