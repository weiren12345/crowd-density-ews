"""Blind hand-counting tool for the recall study.

The headline limitation of this project is that detector recall collapses in
exactly the density regime that matters. Measuring that needs ground truth that
the detector cannot have influenced, so this tool is deliberately blind:

* it never draws model output,
* it presents frames in a shuffled order seeded per annotator, so you cannot
  anneal to the scenario as it unfolds,
* it does not show you your own previous count for a frame,
* it records how long you spent, so rushed frames can be identified later.

Annotate every head you can see inside the region outline, including partially
occluded ones you are confident about. Skip (press s) rather than guess if a
frame is genuinely unreadable - a skipped frame is data, a guessed one is
contamination.

    python tools/count_heads.py --manifest data/frames/site1/manifest.json \
        --annotator alice

A second annotator runs the same command with a different --annotator on a
subset, and scripts/eval_recall.py reports the agreement between them.

Keys: left click add head, right click / u remove nearest, s skip frame,
      ENTER accept and advance, b go back, q save and quit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

WINDOW = "count heads - click each head, ENTER next, u undo, s skip, q quit"


def seeded_order(items, annotator: str):
    """Deterministic per-annotator shuffle, so a session can be resumed."""
    seed = int(hashlib.sha256(annotator.encode("utf-8")).hexdigest()[:8], 16)
    order = list(items)
    random.Random(seed).shuffle(order)
    return order


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["frames"], data.get("region_polygon"), data.get("meta", {})


def annotate_one(image, polygon, index, total, existing=None):
    """Return (points, skipped) or None if the annotator quit."""
    points: list[tuple[float, float]] = list(existing or [])
    started = time.time()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            d = [(px - x) ** 2 + (py - y) ** 2 for px, py in points]
            points.pop(int(np.argmin(d)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1600, image.shape[1]), min(900, image.shape[0]))
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = image.copy()
        if polygon:
            cv2.polylines(canvas, [np.array(polygon, np.int32)], True, (0, 255, 255), 2)
        for x, y in points:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.circle(canvas, (int(x), int(y)), 9, (0, 0, 255), 1)

        # Count is shown because the annotator needs to track their own place,
        # not as feedback against any model output.
        cv2.rectangle(canvas, (0, 0), (430, 40), (20, 20, 20), -1)
        cv2.putText(canvas, f"frame {index+1}/{total}    heads: {len(points)}",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return None
        if key == ord("u") and points:
            points.pop()
        elif key == ord("s"):
            return {"points": [], "skipped": True, "seconds": time.time() - started}
        elif key == ord("b"):
            return {"back": True}
        elif key in (13, 10):
            return {
                "points": points,
                "skipped": False,
                "seconds": time.time() - started,
            }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--out", default=None, help="default: alongside the manifest")
    ap.add_argument("--limit", type=int, default=None,
                    help="annotate only the first N of the shuffled order")
    args = ap.parse_args()

    frames, polygon, meta = load_manifest(args.manifest)
    out_path = Path(args.out) if args.out else (
        Path(args.manifest).parent / f"annotations_{args.annotator}.json"
    )

    done = {}
    if out_path.exists():
        done = json.loads(out_path.read_text(encoding="utf-8")).get("frames", {})
        print(f"resuming: {len(done)} frames already annotated")

    order = seeded_order(frames, args.annotator)
    if args.limit:
        order = order[: args.limit]

    i = 0
    try:
        while i < len(order):
            entry = order[i]
            key = str(entry["frame"])
            if key in done and not args.limit:
                i += 1
                continue

            path = Path(args.manifest).parent / entry["file"]
            image = cv2.imread(str(path))
            if image is None:
                print(f"missing image, skipping: {path}")
                i += 1
                continue

            result = annotate_one(image, polygon, i, len(order))
            if result is None:
                break
            if result.get("back"):
                i = max(0, i - 1)
                continue

            done[key] = {
                "file": entry["file"],
                "count": len(result["points"]),
                "points": [[round(x, 1), round(y, 1)] for x, y in result["points"]],
                "skipped": result["skipped"],
                "seconds": round(result["seconds"], 1),
            }
            i += 1
    finally:
        cv2.destroyAllWindows()
        out_path.write_text(
            json.dumps(
                {
                    "annotator": args.annotator,
                    "manifest": str(args.manifest),
                    "blind": True,
                    "frames": done,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        counted = [v for v in done.values() if not v["skipped"]]
        print(f"\nwrote {out_path}")
        print(f"{len(counted)} frames counted, {len(done) - len(counted)} skipped")
        if counted:
            median = sorted(v["seconds"] for v in counted)[len(counted) // 2]
            print(f"median {median:.0f}s per frame, "
                  f"{sum(v['count'] for v in counted)} heads total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
