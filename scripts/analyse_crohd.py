"""Run the measurement stack over CroHD and check it against ground truth.

Two things happen here, and the second is the point:

1. **Measure.** Per frame: how many people the detector finds, how fast they
   are moving, and how disordered that motion is.
2. **Validate.** CroHD ships ground-truth tracks, so the same quantities can be
   computed from the annotations. Until now ``TrackVelocities`` has only ever
   been checked against motion we scripted ourselves; this is the first time it
   meets real pedestrians.

No calibration is used. Positions stay in pixels, so speeds are px/s. Two
consequences worth being explicit about:

* Comparing pipeline speed against ground-truth speed *within* a sequence is
  valid - both are in the same pixels.
* Comparing *across* sequences is not, because a pixel covers a different
  amount of ground in each. For that, speeds are also reported in
  **head-widths per second**: a head is roughly constant in the real world, so
  dividing by the median head width approximates a metric scale without
  needing a homography. That is the normalisation the fundamental diagram
  needs in the next step.

Tracking needs consecutive frames - ByteTrack associates between neighbours,
so feeding every Nth frame would destroy the identities the velocities depend
on. Frames are therefore processed in order and subsampling happens only in
what gets recorded.

    python scripts/analyse_crohd.py --root C:/path/to/HT21 \\
        --weights runs/detect/heads_crohd/weights/best.pt --out runs/crohd
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdw.calibrate import GroundPlane  # noqa: E402
from cdw.turbulence import local_velocity_variance  # noqa: E402


def load_gt(gt_path: Path):
    """frame -> {track_id: (cx, cy, w, h)} in pixels."""
    out: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        f = line.strip().split(",")
        if len(f) < 6:
            continue
        if len(f) >= 7 and float(f[6]) == 0:
            continue
        frame, tid = int(float(f[0])), int(float(f[1]))
        x, y, w, h = (float(v) for v in f[2:6])
        out[frame][tid] = (x + w / 2, y + h / 2, w, h)
    return out


class PixelVelocities:
    """Per-track velocity in px/s, differenced over a lag.

    Deliberately mirrors cdw.turbulence.TrackVelocities, but in pixels so the
    same computation can be run over ground-truth tracks for comparison.
    """

    def __init__(self, lag_s: float = 0.4, max_speed_px: float = 400.0):
        self.lag_s = lag_s
        self.max_speed_px = max_speed_px
        self._hist: dict[int, deque] = defaultdict(deque)

    def update(self, t: float, items: dict[int, tuple]):
        positions, velocities = [], []
        for tid, (cx, cy) in items.items():
            h = self._hist[tid]
            h.append((t, np.array([cx, cy])))
            while len(h) > 1 and t - h[0][0] > self.lag_s:
                h.popleft()
            if len(h) >= 2:
                t0, p0 = h[0]
                dt = t - t0
                if dt > 1e-6:
                    v = (np.array([cx, cy]) - p0) / dt
                    if np.linalg.norm(v) <= self.max_speed_px:
                        positions.append([cx, cy])
                        velocities.append(v)
        for tid in [k for k, v in self._hist.items() if v and t - v[-1][0] > 2.0]:
            del self._hist[tid]
        if not positions:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.array(positions), np.array(velocities)


def analyse(seq_dir: Path, detector, fps: float, max_frames: int, record_every: int,
            lag_s: float):
    import cv2

    gt = load_gt(seq_dir / "gt" / "gt.txt")
    frames = sorted(gt)[:max_frames]

    det_vel = PixelVelocities(lag_s=lag_s)
    gt_vel = PixelVelocities(lag_s=lag_s)
    detector.reset()

    rows = []
    for i, fr in enumerate(frames):
        img = cv2.imread(str(seq_dir / "img1" / f"{fr:06d}.jpg"))
        if img is None:
            continue
        t = fr / fps

        dets = detector.track(img)
        det_items = {
            d.track_id: d.centre for d in dets if d.track_id is not None
        }
        gt_items = {tid: (b[0], b[1]) for tid, b in gt[fr].items()}

        dpos, dvel = det_vel.update(t, det_items)
        gpos, gvel = gt_vel.update(t, gt_items)

        if i % record_every:
            continue

        head_w = float(np.median([b[2] for b in gt[fr].values()])) if gt[fr] else np.nan
        radius = 5 * head_w if np.isfinite(head_w) else 100.0

        def stats(pos, vel):
            if len(vel) == 0:
                return 0, np.nan, np.nan
            speed = float(np.linalg.norm(vel, axis=1).mean())
            var = local_velocity_variance(pos, vel, k=5, radius_m=radius)
            return len(vel), speed, var

        n_d, sp_d, var_d = stats(dpos, dvel)
        n_g, sp_g, var_g = stats(gpos, gvel)

        rows.append({
            "sequence": seq_dir.name,
            "frame": fr,
            "t": round(t, 3),
            "true_count": len(gt[fr]),
            "det_count": len(dets),
            "median_head_w_px": round(head_w, 2),
            "true_speed_px_s": round(sp_g, 2) if np.isfinite(sp_g) else "",
            "det_speed_px_s": round(sp_d, 2) if np.isfinite(sp_d) else "",
            "true_speed_hw_s": round(sp_g / head_w, 4) if np.isfinite(sp_g) else "",
            "det_speed_hw_s": round(sp_d / head_w, 4) if np.isfinite(sp_d) else "",
            "true_var": round(var_g, 2) if np.isfinite(var_g) else "",
            "det_var": round(var_d, 2) if np.isfinite(var_d) else "",
            "n_true_tracks": n_g,
            "n_det_tracks": n_d,
        })
        if len(rows) % 20 == 0:
            print(f"  {seq_dir.name}: {len(rows)} recorded", end="\r")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="HT21 dir containing train/")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--lag-s", type=float, default=0.4)
    ap.add_argument("--sequences", default=None, help="comma separated, default all")
    args = ap.parse_args()

    from cdw.detect import HeadDetector

    train = Path(args.root) / "train"
    seqs = sorted(d for d in train.iterdir() if (d / "gt" / "gt.txt").exists())
    if args.sequences:
        want = {s.strip() for s in args.sequences.split(",")}
        seqs = [s for s in seqs if s.name in want]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = HeadDetector(weights=args.weights, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, max_det=args.max_det)

    all_rows = []
    for seq in seqs:
        print(f"\n{seq.name}")
        all_rows += analyse(seq, detector, 25.0, args.max_frames,
                            args.record_every, args.lag_s)

    with open(out_dir / "crohd_frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    # -- validation summary, per sequence --------------------------------
    print(f"\n{'seq':>10} {'frames':>7} {'count ratio':>12} {'speed true':>11} "
          f"{'speed det':>10} {'speed err':>10} {'corr':>6}")
    summary = {}
    for seq in sorted({r["sequence"] for r in all_rows}):
        rs = [r for r in all_rows if r["sequence"] == seq]
        tc = np.array([r["true_count"] for r in rs], float)
        dc = np.array([r["det_count"] for r in rs], float)
        ok = [r for r in rs if r["true_speed_px_s"] != "" and r["det_speed_px_s"] != ""]
        ts = np.array([r["true_speed_px_s"] for r in ok], float)
        ds = np.array([r["det_speed_px_s"] for r in ok], float)
        corr = (float(np.corrcoef(ts, ds)[0, 1])
                if len(ts) > 2 and ts.std() > 0 and ds.std() > 0 else float("nan"))
        rel = float(np.abs(ds - ts).mean() / ts.mean()) if len(ts) and ts.mean() else float("nan")
        summary[seq] = {
            "frames": len(rs),
            "count_ratio": float(dc.sum() / tc.sum()),
            "mean_true_count": float(tc.mean()),
            "mean_true_speed_px_s": float(ts.mean()) if len(ts) else None,
            "mean_det_speed_px_s": float(ds.mean()) if len(ds) else None,
            "speed_rel_error": rel,
            "speed_correlation": corr,
        }
        print(f"{seq:>10} {len(rs):7d} {dc.sum()/tc.sum():12.3f} "
              f"{ts.mean():11.1f} {ds.mean():10.1f} {rel*100:9.0f}% {corr:6.3f}")

    (out_dir / "crohd_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(f"\nwrote {out_dir}")
    print("\nSpeeds are px/s and comparable only WITHIN a sequence.")
    print("Use the head-width columns when comparing across scenes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
