"""Detection and tracking — the only learned component in the system.

Everything downstream (density, pressure, warning) is deterministic Python
operating on the output of this module. That separation is deliberate: it keeps
the warning logic auditable and lets the non-ML stack be tested without a model
(see SyntheticDetector).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Detection", "HeadDetector", "SyntheticDetector", "REFERENCE_POINTS"]

#: How to reduce a box to the single pixel we project onto the ground plane.
#: - "box_bottom": bottom-centre. Geometrically correct for full-body boxes,
#:   because the feet are on the ground plane the homography describes.
#: - "centre": box centre. Correct for a head-detection model's own output in
#:   the sense that it is the head, but the head is ~1.7 m off the plane, so it
#:   projects long (away from the camera). See docs/limitations.md.
#: - "box_top": top-centre of a full-body box, i.e. a crude head proxy.
REFERENCE_POINTS = ("box_bottom", "centre", "box_top")


@dataclass(slots=True)
class Detection:
    """One detected person or head in one frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    track_id: int | None = None
    cls: int = 0

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def box_bottom(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def box_top(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y1)

    def reference_point(self, mode: str = "box_bottom") -> tuple[float, float]:
        if mode not in REFERENCE_POINTS:
            raise ValueError(f"reference point mode must be one of {REFERENCE_POINTS}")
        return getattr(self, mode)

    def as_dict(self) -> dict:
        return {
            "xyxy": [self.x1, self.y1, self.x2, self.y2],
            "conf": self.conf,
            "track_id": self.track_id,
            "cls": self.cls,
        }


class HeadDetector:
    """Ultralytics YOLO wrapper, usable with pretrained or fine-tuned weights.

    Two modes matter for this project:

    * Bootstrap: ``weights="yolov8n.pt"``, ``keep_classes=(0,)`` — COCO
      person boxes. Full bodies, so ``reference_point="box_bottom"`` puts the
      projected point on the ground where it belongs. Good enough to prove the
      pipeline end to end before any training happens.
    * Trained: ``weights="runs/.../best.pt"``, single class. Heads only, so the
      reference point becomes the head and the height bias applies.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        imgsz: int = 1280,
        device: str | None = None,
        keep_classes: tuple[int, ...] | None = None,
        iou: float = 0.5,
        tracker: str = "bytetrack.yaml",
        max_det: int = 3000,
    ):
        from ultralytics import YOLO  # imported lazily: heavy, and optional for tests

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.keep_classes = keep_classes
        self.iou = iou
        self.tracker = tracker
        self.weights = weights
        # Ultralytics defaults this to 300. A dense crowd frame can hold far
        # more people than that, and the cap is silent: it would look exactly
        # like recall collapsing, which is the one thing this project measures.
        self.max_det = max_det

    def _common_kwargs(self) -> dict:
        kw = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device is not None:
            kw["device"] = self.device
        if self.keep_classes is not None:
            kw["classes"] = list(self.keep_classes)
        return kw

    def detect(self, frame) -> list[Detection]:
        """Stateless per-frame detection. No track ids."""
        res = self.model.predict(frame, **self._common_kwargs())[0]
        return _from_ultralytics(res)

    def track(self, frame) -> list[Detection]:
        """Detection plus ByteTrack association, so detections carry ids.

        Stateful: frames must be fed in order, and a new video needs a fresh
        detector (or ``reset()``) or tracks will be carried across the cut.
        """
        res = self.model.track(
            frame, persist=True, tracker=self.tracker, **self._common_kwargs()
        )[0]
        return _from_ultralytics(res)

    def reset(self) -> None:
        """Drop tracker state between clips."""
        for attr in ("trackers", "predictor"):
            if getattr(self.model, attr, None) is not None:
                try:
                    setattr(self.model, attr, None)
                except AttributeError:
                    pass


def _from_ultralytics(res) -> list[Detection]:
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), int)
    ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
    out = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        out.append(
            Detection(
                x1, y1, x2, y2,
                conf=float(conf[i]),
                track_id=int(ids[i]) if ids is not None else None,
                cls=int(cls[i]),
            )
        )
    return out


class SyntheticDetector:
    """Replays known positions as if they had been detected.

    Exists so the density / pressure / warning stack can be validated against
    ground truth it cannot get wrong, independently of detector recall. Takes a
    JSON file of ``{"frames": [[{"xyxy": [...], "conf": c, "track_id": i}, ...], ...]}``
    as produced by scripts/make_synthetic_clip.py.
    """

    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.frames: list[list[dict]] = data["frames"]
        self.meta: dict = data.get("meta", {})
        self._i = 0

    def __len__(self) -> int:
        return len(self.frames)

    def _emit(self, index: int) -> list[Detection]:
        if index >= len(self.frames):
            return []
        return [
            Detection(*d["xyxy"], conf=d.get("conf", 1.0), track_id=d.get("track_id"))
            for d in self.frames[index]
        ]

    def detect(self, frame=None) -> list[Detection]:
        out = self._emit(self._i)
        self._i += 1
        return out

    track = detect

    def reset(self) -> None:
        self._i = 0
