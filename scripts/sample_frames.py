"""Draw a density-stratified sample of frames for the recall study.

A uniform random sample of a clip is mostly quiet frames, and tells you almost
nothing about behaviour at 5 people/m^2 because it contains three such frames.
Stratifying by the detector's own density estimate fixes the sample size per
band - at the cost of the strata being defined by the thing under test, which
is a bias to state in the writeup rather than one to pretend away. The strata
are only a sampling device; the x-axis of the final curve is the *hand* count.

    python scripts/sample_frames.py --config configs/site1.yaml \
        --video data/raw/site1.mp4 --out data/frames/site1 --per-bin 20

Writes to --out:
    frame_000123.png      the sampled frames
    manifest.json         what tools/count_heads.py reads (no model output)
    detections.json       low-threshold detections, for scripts/eval_recall.py

Detections are stored at a deliberately low confidence so that the threshold
sweep can be done offline without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdw.calibrate import GroundPlane  # noqa: E402
from cdw.density import Region  # noqa: E402
from cdw.detect import HeadDetector  # noqa: E402
from cdw.pipeline import _resolve, load_config  # noqa: E402

DEFAULT_EDGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 99.0]


def build_region(cfg):
    cal = cfg["calibration"]
    ground = (
        GroundPlane.load(_resolve(cfg, cal["file"]))
        if "file" in cal
        else GroundPlane(cal["image_points"], cal["world_points"])
    )
    return Region(cfg["region"]["polygon"], ground, cfg["region"].get("name", "region"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=20)
    ap.add_argument("--stride", type=int, default=5,
                    help="scan every Nth frame; also enforces spacing between samples")
    ap.add_argument("--scan-conf", type=float, default=0.05,
                    help="low threshold used for both scanning and stored detections")
    ap.add_argument("--edges", default=",".join(str(e) for e in DEFAULT_EDGES),
                    help="density bin edges in people/m^2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    edges = [float(e) for e in args.edges.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    region = build_region(cfg)
    det_cfg = cfg.get("detector", {})
    reference_point = det_cfg.get("reference_point", "box_bottom")
    keep = det_cfg.get("keep_classes")

    detector = HeadDetector(
        weights=det_cfg.get("weights", "yolov8n.pt"),
        conf=args.scan_conf,
        imgsz=int(det_cfg.get("imgsz", 1280)),
        device=det_cfg.get("device"),
        keep_classes=tuple(keep) if keep else None,
    )

    # Pass 1: scan, recording the detector density estimate per candidate frame.
    print("pass 1: scanning")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    candidates = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            dets = detector.detect(frame)
            inside = region.filter(dets, reference_point)
            # At scan confidence; this is a sampling statistic only.
            candidates.append((idx, len(inside) / region.area_m2))
            if len(candidates) % 100 == 0:
                print(f"  {idx} frames", end="\r")
        idx += 1
    cap.release()
    print(f"  scanned {idx} frames, {len(candidates)} candidates")

    # Stratify and sample.
    bins = defaultdict(list)
    for frame_idx, density in candidates:
        b = int(np.digitize(density, edges) - 1)
        bins[max(0, min(b, len(edges) - 2))].append(frame_idx)

    rng = np.random.default_rng(args.seed)
    chosen = []
    for b in sorted(bins):
        pool = bins[b]
        take = min(args.per_bin, len(pool))
        picked = sorted(rng.choice(pool, size=take, replace=False).tolist())
        chosen.extend(picked)
        print(f"  bin {edges[b]:.1f}-{edges[b+1]:.1f} p/m2: "
              f"{len(pool)} available, {take} sampled")
    chosen = sorted(set(chosen))

    # Pass 2: write out the chosen frames and their detections.
    print(f"pass 2: writing {len(chosen)} frames")
    cap = cv2.VideoCapture(args.video)
    manifest, detections = [], {}
    wanted = set(chosen)
    idx = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            name = f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_dir / name), frame)
            manifest.append({"frame": idx, "file": name, "t": round(idx / fps, 3)})
            dets = region.filter(detector.detect(frame), reference_point)
            detections[str(idx)] = [d.as_dict() for d in dets]
            wanted.discard(idx)
        idx += 1
    cap.release()

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "meta": {
                    "video": str(args.video),
                    "fps": fps,
                    "region_area_m2": round(region.area_m2, 4),
                    "stratified_by": "detector density estimate at scan confidence",
                    "scan_conf": args.scan_conf,
                    "edges": edges,
                    "seed": args.seed,
                },
                "region_polygon": region.polygon.tolist(),
                "frames": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "detections.json").write_text(
        json.dumps(
            {
                "meta": {
                    "weights": detector.weights,
                    "scan_conf": args.scan_conf,
                    "imgsz": detector.imgsz,
                    "reference_point": reference_point,
                    "region_area_m2": round(region.area_m2, 4),
                },
                "frames": detections,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {out_dir}")
    print(f"next: python tools/count_heads.py --manifest {out_dir/'manifest.json'} "
          f"--annotator <name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
