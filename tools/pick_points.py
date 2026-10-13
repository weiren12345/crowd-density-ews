"""Click calibration points or a region polygon on a frame.

Calibration mode
----------------
Click the 4 corners of something on the ground whose real dimensions you have
measured - a rectangle of tape, four paving stones, a line-marked box. Click
them in order, going around the shape (not diagonally). Then give the real
width and height in metres.

    python tools/pick_points.py --mode calibration \
        --video data/raw/site1.mp4 --frame 0 \
        --width 4.0 --height 3.0 \
        --out configs/calibration_site1.json

The tool prints the reprojection error. With exactly 4 points it will be
essentially zero - that means the fit is exact, NOT that the calibration is
accurate. Accuracy comes from your tape measure. To actually check it, click
more than 4 points covering the area and watch the residuals.

Region mode
-----------
Click the polygon you want density measured inside, then press ENTER.

    python tools/pick_points.py --mode region \
        --video data/raw/site1.mp4 --frame 0 \
        --calibration configs/calibration_site1.json

It prints the polygon and its real area, ready to paste into a camera config.

Keys: left click add, u undo, r reset, ENTER accept, q/ESC quit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdw.calibrate import GroundPlane  # noqa: E402

WINDOW = "pick points - click, u undo, r reset, ENTER accept, q quit"


def load_frame(video: str, index: int):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video}")
    return frame


def pick(frame, want: int | None, labels=None):
    """Collect clicked points. ``want`` fixes the count, None means any number."""
    points: list[tuple[float, float]] = []
    base = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and (want is None or len(points) < want):
            points.append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1600, base.shape[1]), min(900, base.shape[0]))
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = base.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 255), -1)
            tag = labels[i] if labels and i < len(labels) else str(i + 1)
            cv2.putText(canvas, tag, (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points, np.int32)],
                          isClosed=len(points) > 2, color=(0, 255, 255), thickness=2)

        need = f"{len(points)}/{want}" if want else f"{len(points)}"
        hint = labels[len(points)] if labels and len(points) < len(labels) else ""
        cv2.putText(canvas, f"points {need}   {hint}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 10):
            if want is not None and len(points) != want:
                continue
            if want is None and len(points) < 3:
                continue
            cv2.destroyAllWindows()
            return points


def do_calibration(args) -> int:
    frame = load_frame(args.video, args.frame)

    if args.world:
        world = [tuple(float(v) for v in p.split(",")) for p in args.world.split(";")]
        labels = [f"world {w}" for w in world]
    else:
        world = [(0.0, 0.0), (args.width, 0.0), (args.width, args.height), (0.0, args.height)]
        labels = [
            "1: origin corner",
            f"2: {args.width} m along the width",
            f"3: far corner ({args.width} x {args.height} m)",
            f"4: {args.height} m along the height",
        ]

    pts = pick(frame, want=len(world), labels=labels)
    if pts is None:
        print("cancelled")
        return 1

    ground = GroundPlane(pts, world)
    err = ground.reprojection_error()

    payload = ground.to_dict()
    payload["note"] = (
        f"Clicked on frame {args.frame} of {Path(args.video).name}. "
        "Record in your logbook what was physically measured."
    )
    payload.pop("H", None)   # refit from points on load, so edits cannot go stale
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"reprojection error: rms {err['rms_m']*100:.2f} cm, max {err['max_m']*100:.2f} cm")
    if len(world) == 4:
        print("  (4 points determine the homography exactly, so this is ~0 by")
        print("   construction. It does not measure accuracy. Use --world with")
        print("   more than 4 measured points if you want a real residual.)")
    return 0


def do_region(args) -> int:
    frame = load_frame(args.video, args.frame)
    pts = pick(frame, want=None)
    if pts is None:
        print("cancelled")
        return 1

    poly = [[round(x), round(y)] for x, y in pts]
    print("\nregion:")
    print(f"  name: {args.name}")
    print(f"  polygon: {json.dumps(poly)}")

    if args.calibration:
        ground = GroundPlane.load(args.calibration)
        area = ground.polygon_area_m2(pts)
        print(f"\nreal area: {area:.2f} m2")
        print(f"at 4 people/m2 that is {area*4:.0f} people in frame - if that")
        print("looks implausible for this view, the calibration is wrong.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("calibration", "region"), required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--width", type=float, default=4.0, help="calibration: metres")
    ap.add_argument("--height", type=float, default=3.0, help="calibration: metres")
    ap.add_argument("--world", help='calibration: explicit "x,y;x,y;..." in metres')
    ap.add_argument("--out", default="configs/calibration.json")
    ap.add_argument("--calibration", help="region: calibration file, to report area")
    ap.add_argument("--name", default="bottleneck", help="region: name")
    args = ap.parse_args()

    return do_calibration(args) if args.mode == "calibration" else do_region(args)


if __name__ == "__main__":
    raise SystemExit(main())
