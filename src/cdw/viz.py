"""Overlay rendering, for demos and for eyeballing whether the numbers are sane.

Visual sanity checks catch calibration mistakes that no unit test will: if the
region outline does not sit on the ground where you measured it, the area is
wrong and every density figure downstream is wrong with it.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["draw_overlay", "LEVEL_COLOURS"]

#: BGR, deliberately not red/green only - the demo should stay legible in a
#: greyscale printout of the report.
LEVEL_COLOURS = {
    "NORMAL": (120, 170, 120),
    "WATCH": (40, 180, 230),
    "WARNING": (40, 40, 220),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, pipe, record, history=None):
    """Draw region, detections and a HUD onto ``frame`` (modified in place)."""
    colour = LEVEL_COLOURS.get(record.level, (200, 200, 200))

    _draw_region(frame, pipe.region.polygon, colour)
    for det in record.detections:
        _draw_detection(frame, det, colour)
    _draw_hud(frame, pipe, record, colour)
    if history:
        _draw_pressure_strip(frame, history)
    return frame


def _draw_region(frame, polygon, colour):
    pts = polygon.astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, dst=frame)
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)


def _draw_detection(frame, det, colour):
    p1 = (int(det.x1), int(det.y1))
    p2 = (int(det.x2), int(det.y2))
    cv2.rectangle(frame, p1, p2, colour, 1)
    rx, ry = det.reference_point("box_bottom")
    cv2.circle(frame, (int(rx), int(ry)), 2, colour, -1)
    if det.track_id is not None:
        cv2.putText(frame, str(det.track_id), (p1[0], max(10, p1[1] - 3)),
                    _FONT, 0.35, colour, 1, cv2.LINE_AA)


def _draw_hud(frame, pipe, record, colour):
    h, w = frame.shape[:2]
    lines = [
        f"{record.level}   LOS {record.los}",
        f"density  {record.density:5.2f} people/m2   (raw {record.raw_density:5.2f})",
        f"pressure {record.pressure:7.4f}   slope {record.pressure_slope:+7.4f}/s",
        f"in region {record.count_region:3d} / {record.count_total:3d}   "
        f"area {pipe.region.area_m2:.1f} m2",
        f"velocities {record.n_velocities:3d} from {record.pressure_source}",
        record.reason,
    ]
    pad, lh = 10, 20
    box_h = pad * 2 + lh * len(lines)
    box_w = 460
    panel = frame[0:box_h, 0:box_w].copy()
    cv2.rectangle(panel, (0, 0), (box_w, box_h), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.65, frame[0:box_h, 0:box_w], 0.35, 0,
                    dst=frame[0:box_h, 0:box_w])
    for i, text in enumerate(lines):
        c = colour if i == 0 else (235, 235, 235)
        scale = 0.62 if i == 0 else 0.46
        cv2.putText(frame, text, (pad, pad + lh * (i + 1) - 6),
                    _FONT, scale, c, 1, cv2.LINE_AA)

    if record.level == "WARNING":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, 6)

    cv2.putText(
        frame,
        "detects dangerous conditions; does not predict a crush",
        (pad, h - 10), _FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
    )


def _draw_pressure_strip(frame, history, width=460, height=60, margin=10):
    """Small trailing plot of pressure, so a rise is visible rather than inferred."""
    h, w = frame.shape[:2]
    series = [r.pressure for r in history[-240:]]
    if len(series) < 2:
        return
    x0, y0 = margin, h - height - 28
    x1, y1 = x0 + width, y0 + height
    if x1 >= w or y0 <= 0:
        return

    panel = frame[y0:y1, x0:x1].copy()
    cv2.rectangle(panel, (0, 0), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.6, frame[y0:y1, x0:x1], 0.4, 0, dst=frame[y0:y1, x0:x1])

    lo, hi = min(series), max(series)
    span = (hi - lo) or 1.0
    pts = [
        (
            int(x0 + i * (width - 1) / (len(series) - 1)),
            int(y1 - 2 - (v - lo) / span * (height - 4)),
        )
        for i, v in enumerate(series)
    ]
    cv2.polylines(frame, [np.array(pts, np.int32)], False, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"pressure  {lo:.3f}-{hi:.3f}", (x0 + 4, y0 + 14),
                _FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
