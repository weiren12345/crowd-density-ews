"""End-to-end pipeline: video in, per-frame density / pressure / warning out.

Wires the stages together and nothing more. All of the judgement lives in the
stage modules and the config files; this file is plumbing, so that swapping the
detector or the thresholds does not mean editing logic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibrate import GroundPlane, HeightCorrection
from .density import DensityEstimator, Region
from .detect import HeadDetector, SyntheticDetector
from .turbulence import FarnebackFlow, TrackVelocities, crowd_pressure
from .warning import WarningRule

__all__ = ["Pipeline", "FrameRecord", "load_config", "run_video"]

CSV_FIELDS = [
    "frame", "t", "count_total", "count_region", "raw_density", "density",
    "los", "level", "pressure", "pressure_slope", "velocity_variance",
    "mean_speed", "n_velocities", "pressure_source", "density_signal",
    "pressure_signal", "held_for_s", "reason",
]


@dataclass(slots=True)
class FrameRecord:
    frame: int
    t: float
    count_total: int
    count_region: int
    raw_density: float
    density: float
    los: str
    level: str
    pressure: float
    pressure_slope: float
    velocity_variance: float
    mean_speed: float
    n_velocities: int
    pressure_source: str
    density_signal: bool
    pressure_signal: bool
    held_for_s: float
    reason: str
    detections: list = field(default_factory=list, repr=False)

    def csv_row(self) -> dict:
        d = asdict(self)
        d.pop("detections", None)
        return d


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_dir"] = str(Path(path).resolve().parent)
    return cfg


def _resolve(cfg: dict, value: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(value)
    if p.is_absolute():
        return p
    root = Path(cfg.get("_dir", ".")).parent
    candidate = root / p
    return candidate if candidate.exists() else Path(value)


class Pipeline:
    """Holds the configured stages and processes frames in order."""

    def __init__(self, cfg: dict, detector=None):
        self.cfg = cfg

        cal = cfg["calibration"]
        if "file" in cal:
            self.ground = GroundPlane.load(_resolve(cfg, cal["file"]))
        else:
            self.ground = GroundPlane(cal["image_points"], cal["world_points"])

        # Only meaningful when the reference point is NOT on the ground plane,
        # i.e. once the head detector replaces full-body boxes.
        hc_cfg = cal.get("height_correction")
        self.height_correction = HeightCorrection.from_dict(hc_cfg) if hc_cfg else None

        self.region = Region(
            cfg["region"]["polygon"],
            self.ground,
            name=cfg["region"].get("name", "region"),
            height_correction=self.height_correction,
        )

        det_cfg = cfg.get("detector", {})
        self.reference_point = det_cfg.get("reference_point", "box_bottom")
        self.use_tracking = bool(det_cfg.get("track", True))
        self.detector = detector if detector is not None else self._build_detector(det_cfg)

        self.density = DensityEstimator(
            self.region, window_s=float(cfg.get("density", {}).get("window_s", 3.0))
        )

        p_cfg = cfg.get("pressure", {})
        self.pressure_source = p_cfg.get("source", "tracks")
        self.k = int(p_cfg.get("k", 5))
        self.radius_m = p_cfg.get("radius_m", 2.0)
        self.tracks = TrackVelocities(
            self.ground,
            lag_s=float(p_cfg.get("lag_s", 0.3)),
            max_speed_mps=float(p_cfg.get("max_speed_mps", 6.0)),
            height_correction=self.height_correction,
        )
        self.flow = (
            FarnebackFlow(
                self.ground,
                step=int(p_cfg.get("flow_step", 16)),
                height_correction=self.height_correction,
            )
            if self.pressure_source in ("flow", "both")
            else None
        )

        thresholds = cfg.get("thresholds", "configs/thresholds.yaml")
        self.rule = WarningRule.from_file(_resolve(cfg, thresholds))

        self._mask: np.ndarray | None = None

    def _build_detector(self, det_cfg: dict):
        if "synthetic" in det_cfg:
            return SyntheticDetector(_resolve(self.cfg, det_cfg["synthetic"]))
        keep = det_cfg.get("keep_classes")
        return HeadDetector(
            weights=det_cfg.get("weights", "yolov8n.pt"),
            conf=float(det_cfg.get("conf", 0.25)),
            imgsz=int(det_cfg.get("imgsz", 1280)),
            device=det_cfg.get("device"),
            keep_classes=tuple(keep) if keep else None,
            iou=float(det_cfg.get("iou", 0.5)),
            tracker=det_cfg.get("tracker", "bytetrack.yaml"),
        )

    # -- per frame ----------------------------------------------------------

    def process(self, frame_idx: int, t: float, frame) -> FrameRecord:
        dets = (
            self.detector.track(frame) if self.use_tracking else self.detector.detect(frame)
        )
        inside = self.region.filter(dets, self.reference_point)
        dsample = self.density.update(t, len(inside))

        positions, velocities, source = self._velocity_field(t, frame, inside)
        psample = crowd_pressure(
            t, dsample.density, positions, velocities,
            k=self.k, radius_m=self.radius_m, source=source,
        )
        wsample = self.rule.update(t, dsample.density, psample.pressure)

        return FrameRecord(
            frame=frame_idx,
            t=round(t, 4),
            count_total=len(dets),
            count_region=len(inside),
            raw_density=round(dsample.raw_density, 4),
            density=round(dsample.density, 4),
            los=wsample.los,
            level=wsample.level.value,
            pressure=round(psample.pressure, 6),
            pressure_slope=round(wsample.pressure_slope, 6),
            velocity_variance=round(psample.velocity_variance, 6),
            mean_speed=round(psample.mean_speed, 4),
            n_velocities=psample.n_velocities,
            pressure_source=psample.source,
            density_signal=wsample.density_signal,
            pressure_signal=wsample.pressure_signal,
            held_for_s=round(wsample.held_for_s, 2),
            reason=wsample.reason,
            detections=inside,
        )

    def _velocity_field(self, t: float, frame, inside):
        """Velocities from tracks, from flow, or tracks with a flow fallback."""
        if self.pressure_source == "flow":
            return (*self._flow_field(t, frame), "flow")

        pos, vel = self.tracks.update(t, inside, self.reference_point)
        if self.pressure_source == "tracks":
            return pos, vel, "tracks"

        # "both": prefer tracks, fall back when tracking has thinned out.
        min_frac = float(self.cfg.get("pressure", {}).get("fallback_below_frac", 0.5))
        enough = len(inside) == 0 or len(vel) >= max(2, min_frac * len(inside))
        if enough:
            return pos, vel, "tracks"
        return (*self._flow_field(t, frame), "flow")

    def _flow_field(self, t: float, frame):
        if self.flow is None:
            self.flow = FarnebackFlow(self.ground, height_correction=self.height_correction)
        if self._mask is None or self._mask.shape[:2] != frame.shape[:2]:
            self._mask = self.region.mask(frame.shape)
        return self.flow.update(t, frame, self._mask)

    def reset(self) -> None:
        self.density.reset()
        self.tracks.reset()
        self.rule.reset()
        if self.flow is not None:
            self.flow.reset()
        if hasattr(self.detector, "reset"):
            self.detector.reset()

    def summary(self) -> dict:
        return {
            "region": self.region.name,
            "region_area_m2": round(self.region.area_m2, 3),
            "calibration_error": self.ground.reprojection_error(),
            "reference_point": self.reference_point,
            "pressure_source": self.pressure_source,
            "density_window_s": self.density.window_s,
            "height_correction": (
                self.height_correction.to_dict() if self.height_correction else None
            ),
        }


def run_video(
    cfg: dict,
    video: str | Path,
    out_csv: str | Path | None = None,
    out_video: str | Path | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    detector=None,
    progress=None,
):
    """Run the pipeline over a video file. Returns (records, summary).

    ``stride`` skips frames to trade temporal resolution for speed; timestamps
    stay in real seconds, so density smoothing and pressure slopes remain
    correct, just coarser.
    """
    pipe = Pipeline(cfg, detector=detector)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = float(cfg.get("video", {}).get("fps_fallback", 25.0))

    writer = None
    records: list[FrameRecord] = []
    idx = 0
    kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rec = pipe.process(idx, idx / fps, frame)
                records.append(rec)
                if out_video is not None:
                    from .viz import draw_overlay

                    canvas = draw_overlay(frame.copy(), pipe, rec, records)
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            str(out_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps / stride,
                            (w, h),
                        )
                    writer.write(canvas)
                kept += 1
                if progress is not None:
                    progress(rec)
                if max_frames is not None and kept >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = pipe.summary()
    summary.update({"frames_processed": kept, "fps": fps, "stride": stride, "video": str(video)})

    if out_csv is not None:
        write_csv(records, out_csv)
        Path(str(out_csv)).with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return records, summary


def write_csv(records: list[FrameRecord], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(r.csv_row())
