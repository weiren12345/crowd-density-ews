"""Detection tests.

The Detection geometry tests need no model. The HeadDetector tests do, and skip
cleanly when the weights are not present - so a fresh clone still runs green
without downloading anything.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cdw.detect import REFERENCE_POINTS, Detection, SyntheticDetector

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "yolov8n.pt"
ASSET = REPO / ".venv/Lib/site-packages/ultralytics/assets/bus.jpg"


# -- Detection geometry -----------------------------------------------------


@pytest.fixture
def det():
    return Detection(100.0, 200.0, 140.0, 320.0, conf=0.9, track_id=7)


def test_centre(det):
    assert det.centre == (120.0, 260.0)


def test_box_bottom_is_the_feet(det):
    assert det.box_bottom == (120.0, 320.0)


def test_box_top_is_the_head(det):
    assert det.box_top == (120.0, 200.0)


@pytest.mark.parametrize("mode", REFERENCE_POINTS)
def test_every_reference_point_mode_resolves(det, mode):
    x, y = det.reference_point(mode)
    assert det.x1 <= x <= det.x2
    assert det.y1 <= y <= det.y2


def test_unknown_reference_point_is_rejected(det):
    with pytest.raises(ValueError, match="reference point mode"):
        det.reference_point("elbow")


def test_as_dict_roundtrips_through_json(det):
    d = json.loads(json.dumps(det.as_dict()))
    assert d["xyxy"] == [100.0, 200.0, 140.0, 320.0]
    assert d["track_id"] == 7


# -- SyntheticDetector ------------------------------------------------------


def test_synthetic_detector_replays_in_order(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"frames": [
        [{"xyxy": [0, 0, 10, 10], "conf": 1.0, "track_id": 1}],
        [],
        [{"xyxy": [5, 5, 15, 15], "conf": 0.8, "track_id": 1},
         {"xyxy": [20, 20, 30, 30], "conf": 0.7, "track_id": 2}],
    ]}), encoding="utf-8")

    det = SyntheticDetector(path)
    assert len(det) == 3
    assert len(det.detect()) == 1
    assert len(det.detect()) == 0
    third = det.detect()
    assert [d.track_id for d in third] == [1, 2]
    assert det.detect() == []            # exhausted, not an error


def test_synthetic_detector_track_is_the_same_stream(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"frames": [[{"xyxy": [0, 0, 10, 10]}]]}), encoding="utf-8")
    det = SyntheticDetector(path)
    assert len(det.track()) == 1         # track and detect are one replay


# -- HeadDetector (needs weights) -------------------------------------------

needs_model = pytest.mark.skipif(
    not (WEIGHTS.exists() and ASSET.exists()),
    reason="yolov8n.pt or the ultralytics sample asset is not present",
)


@pytest.fixture(scope="module")
def sample_image():
    import cv2
    return cv2.imread(str(ASSET))


@needs_model
def test_detector_finds_people_and_filters_classes(sample_image):
    from cdw.detect import HeadDetector

    det = HeadDetector(weights=str(WEIGHTS), conf=0.25, imgsz=640, keep_classes=(0,))
    results = det.detect(sample_image)
    assert len(results) >= 3                       # the sample has several people
    assert all(d.cls == 0 for d in results)        # class filter honoured
    assert all(0.25 <= d.conf <= 1.0 for d in results)
    assert all(d.x2 > d.x1 and d.y2 > d.y1 for d in results)


@needs_model
def test_raising_confidence_cannot_add_detections(sample_image):
    from cdw.detect import HeadDetector

    low = HeadDetector(weights=str(WEIGHTS), conf=0.10, imgsz=640, keep_classes=(0,))
    high = HeadDetector(weights=str(WEIGHTS), conf=0.70, imgsz=640, keep_classes=(0,))
    assert len(high.detect(sample_image)) <= len(low.detect(sample_image))


@needs_model
def test_detect_assigns_no_track_ids(sample_image):
    from cdw.detect import HeadDetector

    det = HeadDetector(weights=str(WEIGHTS), conf=0.25, imgsz=640, keep_classes=(0,))
    assert all(d.track_id is None for d in det.detect(sample_image))


@needs_model
def test_tracking_assigns_and_holds_ids(sample_image):
    """Pan the scene and check ids survive the motion."""
    import cv2
    from cdw.detect import HeadDetector
    from collections import Counter

    det = HeadDetector(weights=str(WEIGHTS), conf=0.25, imgsz=640, keep_classes=(0,))
    h, w = sample_image.shape[:2]
    seen = Counter()
    for i in range(12):
        M = np.float32([[1, 0, -i * 6], [0, 1, 0]])
        frame = cv2.warpAffine(sample_image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        for d in det.track(frame):
            if d.track_id is not None:
                seen[d.track_id] += 1

    assert seen, "tracking produced no ids at all"
    # At least one person should be held for the whole pan; a tracker that
    # re-ids every frame would show all counts equal to 1.
    assert max(seen.values()) >= 10


@needs_model
def test_reset_restarts_numbering(sample_image):
    from cdw.detect import HeadDetector

    det = HeadDetector(weights=str(WEIGHTS), conf=0.25, imgsz=640, keep_classes=(0,))
    for _ in range(3):
        det.track(sample_image)
    det.reset()
    ids = [d.track_id for d in det.track(sample_image) if d.track_id is not None]
    assert ids and min(ids) == 1
