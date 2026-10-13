"""Validate the pipeline against a dataset that ships its own calibration.

Everything so far has been proven on data we generated ourselves. This closes
the loop: real video, somebody else's homography, and annotations in world
metres, so the pipeline's output can be checked against ground truth it had no
hand in producing.

Trajectory datasets (ETH, UCY and relatives) suit this because their labels are
already in metres - that is why they ship an H matrix at all.

Three things get checked, in increasing order of how much they tell you:

1. **Scale.** Project the image corners and report the implied scene size. A
   reversed or wrongly-scaled matrix still runs and still produces numbers;
   this is what catches it.
2. **Density.** True count inside the region comes from the annotations, so
   true density is count / measured area. Compare against what the detector
   produces. The gap IS the detector's contribution to density error, measured
   on real video.
3. **Velocity.** Differencing annotated trajectories gives true speeds. Compare
   against TrackVelocities. Until now that code has only ever been checked
   against motion we scripted.

    python scripts/validate_calibrated.py \
        --video data/raw/seq.mp4 --homography H.txt \
        --annotations obsmat.txt --out runs/validate_eth

Annotation format: whitespace-separated, one row per person per frame. Default
column order is ``frame id x y`` with x/y in world metres; override with
--columns if the dataset differs (ETH obsmat, for instance, interleaves
velocities and uses x/z/y).

If density tracks ground truth here, the geometry is right and any later
disagreement on your own footage is the detector or the calibration - not the
arithmetic. That is a much stronger claim than "the tests pass".
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
from cdw.density import DensityEstimator, Region  # noqa: E402
from cdw.detect import HeadDetector  # noqa: E402
from cdw.turbulence import TrackVelocities  # noqa: E402


def load_annotations(path, columns: str, scale: float, frame_scale: float):
    """-> {frame_index: {ped_id: (x_m, y_m)}}"""
    idx = {name: i for i, name in enumerate(columns.split(","))}
    for required in ("frame", "id", "x", "y"):
        if required not in idx:
            raise ValueError(f"--columns must name {required}; got {columns}")

    out: dict[int, dict[int, tuple[float, float]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) <= max(idx.values()):
            continue
        try:
            frame = int(round(float(f[idx["frame"]]) * frame_scale))
            ped = int(float(f[idx["id"]]))
            x = float(f[idx["x"]]) * scale
            y = float(f[idx["y"]]) * scale
        except ValueError:
            continue
        out.setdefault(frame, {})[ped] = (x, y)
    return out


def region_from_annotations(ground: GroundPlane, ann, image_size, margin=0.9):
    """Fall back to a region covering where people actually are.

    Better than using the whole frame: the far field of a ground-plane
    homography is where one pixel covers metres, and including it would make
    the area enormous and the density meaningless.
    """
    pts = np.array([p for frame in ann.values() for p in frame.values()])
    if len(pts) < 4:
        raise ValueError("too few annotated positions to infer a region")
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    lo = (lo - [cx, cy]) * margin + [cx, cy]
    hi = (hi - [cx, cy]) * margin + [cx, cy]
    world_poly = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]

    image_poly = ground.to_image(world_poly)
    w, h = image_size
    if not np.all(np.isfinite(image_poly)):
        raise ValueError("inferred region does not project back into the image")
    clipped = np.clip(image_poly, [0, 0], [w - 1, h - 1])
    return clipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--homography", required=True, help="3x3 matrix, whitespace separated")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction", default="image_to_world",
                    choices=("image_to_world", "world_to_image"))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on the homography output, into metres")
    ap.add_argument("--ann-scale", type=float, default=1.0,
                    help="multiplier on annotation x/y, into metres")
    ap.add_argument("--columns", default="frame,id,x,y")
    ap.add_argument("--frame-scale", type=float, default=1.0,
                    help="annotation frame number -> video frame index "
                         "(datasets often annotate every Nth frame)")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--keep-classes", default="0",
                    help="comma separated, or 'none' for a single-class head model")
    ap.add_argument("--reference-point", default="box_bottom")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--region", default=None, help="JSON image polygon; inferred if absent")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. scale check ----------------------------------------------------
    H = np.loadtxt(args.homography)
    ground = GroundPlane.from_matrix(H, direction=args.direction, scale=args.scale)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scene = ground.describe_scene((w, h))
    print(f"video {w}x{h} at {fps:g} fps")
    print(f"implied scene extent: {scene.get('extent_x_m', float('nan')):.1f} m x "
          f"{scene.get('extent_y_m', float('nan')):.1f} m  -> {scene['reason']}")
    if not scene["ok"]:
        print("\nSTOPPING. The calibration does not describe a plausible scene.")
        print("Try --direction world_to_image, or --scale, before reading anything")
        print("else from this run: every density below would be wrong.")
        cap.release()
        return 1

    ann = load_annotations(args.annotations, args.columns, args.ann_scale, args.frame_scale)
    print(f"{len(ann)} annotated frames, "
          f"{len({p for f in ann.values() for p in f})} distinct people")

    # -- region ------------------------------------------------------------
    if args.region:
        image_poly = np.array(json.loads(args.region), dtype=float)
    else:
        image_poly = region_from_annotations(ground, ann, (w, h))
        print("region inferred from annotated positions (5th-95th percentile)")

    region = Region(image_poly, ground, name="validation")
    print(f"region area: {region.area_m2:.2f} m2")

    # -- detector ----------------------------------------------------------
    keep = None if args.keep_classes.lower() == "none" else tuple(
        int(c) for c in args.keep_classes.split(",")
    )
    det = HeadDetector(weights=args.weights, conf=args.conf, imgsz=args.imgsz,
                       device=args.device, keep_classes=keep)
    density = DensityEstimator(region, window_s=1.0)
    tracks = TrackVelocities(ground, lag_s=0.4)

    rows = []
    idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in ann:
            t = idx / fps

            true_pts = np.array(list(ann[idx].values()))
            true_in = [p for p in true_pts if region.contains(ground.to_image([p])[0])]
            true_count = len(true_in)

            dets = det.track(frame)
            inside = region.filter(dets, args.reference_point)
            sample = density.update(t, len(inside))
            _, vel = tracks.update(t, inside, args.reference_point)

            rows.append({
                "frame": idx,
                "t": round(t, 3),
                "true_count": true_count,
                "det_count": len(inside),
                "true_density": round(true_count / region.area_m2, 4),
                "det_density": round(sample.raw_density, 4),
                "det_speed": round(float(np.linalg.norm(vel, axis=1).mean()), 3) if len(vel) else None,
            })
            processed += 1
            if processed % 25 == 0:
                print(f"  {processed} frames", end="\r")
            if args.max_frames and processed >= args.max_frames:
                break
        idx += 1
    cap.release()

    if not rows:
        print("\nno annotated frame matched a video frame index.")
        print("The annotation frame numbering probably differs from the video's;")
        print("try --frame-scale (e.g. 0.1 if annotations are every 10th frame).")
        return 1

    # -- true speeds from the annotations ---------------------------------
    true_speeds = []
    frames_sorted = sorted(ann)
    for a, b in zip(frames_sorted, frames_sorted[1:]):
        dt = (b - a) / fps
        if dt <= 0 or dt > 1.0:
            continue
        shared = set(ann[a]) & set(ann[b])
        for p in shared:
            d = np.linalg.norm(np.array(ann[b][p]) - np.array(ann[a][p]))
            if d / dt < 6.0:
                true_speeds.append(d / dt)

    tc = np.array([r["true_count"] for r in rows], float)
    dc = np.array([r["det_count"] for r in rows], float)
    td = np.array([r["true_density"] for r in rows], float)
    dd = np.array([r["det_density"] for r in rows], float)
    speeds = np.array([r["det_speed"] for r in rows if r["det_speed"] is not None], float)

    summary = {
        "video": args.video,
        "scene": scene,
        "region_area_m2": round(region.area_m2, 3),
        "frames": len(rows),
        "count": {
            "mean_true": float(tc.mean()),
            "mean_detected": float(dc.mean()),
            "pooled_ratio": float(dc.sum() / tc.sum()) if tc.sum() else None,
            "correlation": (
                float(np.corrcoef(tc, dc)[0, 1])
                if len(tc) > 2 and tc.std() > 0 and dc.std() > 0
                else None
            ),
        },
        "density": {
            "mean_true": float(td.mean()),
            "mean_detected": float(dd.mean()),
            "mean_abs_error": float(np.abs(dd - td).mean()),
            "bias": float((dd - td).mean()),
        },
        "speed": {
            "true_mean_mps": float(np.mean(true_speeds)) if true_speeds else None,
            "detected_mean_mps": float(speeds.mean()) if len(speeds) else None,
            "n_true_samples": len(true_speeds),
        },
    }

    with open(out_dir / "validation.csv", "w", encoding="utf-8", newline="") as fh:
        import csv
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    (out_dir / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    c, d, s = summary["count"], summary["density"], summary["speed"]
    print(f"\n\nframes compared   {len(rows)}")
    ratio = "n/a" if c["pooled_ratio"] is None else f"{c['pooled_ratio']:.3f}"
    print(f"count             true {c['mean_true']:.1f}, "
          f"detected {c['mean_detected']:.1f}, ratio {ratio}")
    if c["correlation"] is not None:
        print(f"correlation       {c['correlation']:.3f}   "
              "(does detected count track the real one?)")
    else:
        print("correlation       undefined (one series never varies)")
    print(f"density           true {d['mean_true']:.3f}, detected {d['mean_detected']:.3f} "
          f"people/m2")
    print(f"                  mean abs error {d['mean_abs_error']:.3f}, "
          f"bias {d['bias']:+.3f}")
    if s["true_mean_mps"] is not None:
        print(f"speed             true {s['true_mean_mps']:.2f} m/s from "
              f"{s['n_true_samples']} annotation pairs")
        if s["detected_mean_mps"] is not None:
            print(f"                  pipeline {s['detected_mean_mps']:.2f} m/s")
    print(f"\nwrote {out_dir}")
    print("\nA negative density bias is the expected signature of missed detections.")
    print("A speed mismatch points at the homography or the frame rate, not the detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
