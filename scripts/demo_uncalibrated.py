"""Run the whole system on real crowd video, without calibration.

Every component has been validated separately. This runs them together, end to
end - the demonstration the project has been missing.

Note on the footage: HT21-01, 02 and 03 are detector TRAINING scenes; only
HT21-04 was held out. Running this on 02 or 03 therefore shows the measurement
and escalation logic working, not detector generalisation - that is evidenced
on ShanghaiTech, which the model never saw. Use --sequence HT21-04 for a run
where the detector is also on unseen data.

Uncalibrated, so the units differ from the calibrated pipeline:

    occupancy        people inside the region, counted (validated to ~98%)
    speed            head-widths per second, which is scale-free
    net rate         people per second, from the occupancy trend
    disorder         variance of local velocity, in head-widths^2/s^2

**The threshold is scene-relative, and that is deliberate.** The Fruin bands
need calibration we do not have and citations we have not verified, so instead
the system learns this scene's own baseline from an initial quiet period and
flags departures from it: occupancy above its own high percentile AND rising.
Both signals come from the footage itself. Nothing is borrowed, and nothing is
invented.

What that buys, and what it costs: the warning means "unusual for this camera",
not "dangerous in absolute terms". It needs no survey and no literature, and it
cannot tell you a crowd is near a crush. It is an anomaly signal, and the
overlay says so.

    python scripts/demo_uncalibrated.py --root C:/path/to/HT21 \\
        --weights runs/detect/heads_crohd/weights/best.pt --out runs/demo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdw.flow import FlowBalance  # noqa: E402
from cdw.turbulence import local_velocity_variance  # noqa: E402

LEVEL_COLOUR = {"NORMAL": (120, 170, 120), "WATCH": (40, 180, 230),
                "ELEVATED": (40, 40, 220)}


class SceneBaseline:
    """Learns what ordinary looks like for one camera, then flags departures.

    Deliberately not a model of danger. It is a model of this scene's normal,
    which is the only thing uncalibrated footage can actually support.
    """

    def __init__(self, learn_s: float = 20.0, percentile: float = 85.0):
        self.learn_s = learn_s
        self.percentile = percentile
        self._occ: list[float] = []
        self.threshold: float | None = None

    def update(self, t: float, occupancy: float) -> bool:
        """Returns True once the baseline is established."""
        if self.threshold is None:
            self._occ.append(occupancy)
            if t >= self.learn_s and len(self._occ) > 10:
                self.threshold = float(np.percentile(self._occ, self.percentile))
            return self.threshold is not None
        return True

    def is_high(self, occupancy: float) -> bool:
        return self.threshold is not None and occupancy > self.threshold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="HT21 dir; or use --video")
    ap.add_argument("--video", default=None, help="a video file instead of --root")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequence", default="HT21-04")
    ap.add_argument("--max-frames", type=int, default=997)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--region-frac", type=float, default=0.6)
    ap.add_argument("--learn-s", type=float, default=15.0)
    ap.add_argument("--rate-threshold", type=float, default=0.3,
                    help="people/s above which occupancy counts as rising")
    ap.add_argument("--dwell-s", type=float, default=2.0)
    ap.add_argument(
        "--exit-frac", type=float, default=0.90,
        help="hysteresis: having entered WATCH above the baseline, occupancy "
             "must fall below baseline*exit_frac to leave it. Without a "
             "deadband the level flaps every few frames whenever occupancy "
             "sits on the threshold, which is unreadable and useless.",
    )
    ap.add_argument("--clear-s", type=float, default=3.0,
                    help="how long the quieter condition must hold before "
                         "de-escalating")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    import cv2
    from cdw.detect import HeadDetector

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"cannot open {args.video}")
            return 1
        frames_src = []
        while len(frames_src) < args.max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            frames_src.append(fr)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        label = Path(args.video).stem
    elif args.root:
        seq = Path(args.root) / "train" / args.sequence
        paths = sorted((seq / "img1").glob("*.jpg"))[: args.max_frames]
        if not paths:
            print(f"no frames under {seq}")
            return 1
        frames_src = paths
        src_fps = 25.0
        label = args.sequence
    else:
        print("give either --root or --video")
        return 1
    probe = (frames_src[0] if args.video else cv2.imread(str(frames_src[0])))
    h, w = probe.shape[:2]
    rw, rh = w * args.region_frac, h * args.region_frac
    x0, y0 = (w - rw) / 2, (h - rh) / 2
    region = (x0, y0, x0 + rw, y0 + rh)
    poly = np.array([[x0, y0], [x0 + rw, y0], [x0 + rw, y0 + rh], [x0, y0 + rh]], np.int32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    det = HeadDetector(weights=args.weights, conf=args.conf, imgsz=args.imgsz,
                       device=args.device, max_det=1000)
    det.reset()
    flow = FlowBalance(window_s=30.0)
    baseline = SceneBaseline(learn_s=args.learn_s)

    hist: dict[int, deque] = {}
    writer = None
    rows = []
    trace: deque[tuple[float, float]] = deque(maxlen=300)
    both_since = None
    clear_since = None
    level = "NORMAL"
    fps = src_fps

    def in_region(p):
        return region[0] <= p[0] <= region[2] and region[1] <= p[1] <= region[3]

    for i, item in enumerate(frames_src):
        img = item if args.video else cv2.imread(str(item))
        if img is None:
            continue
        t = i / fps

        dets = det.track(img)
        inside = [d for d in dets if in_region(d.centre)]
        ids_inside = {d.track_id for d in inside if d.track_id is not None}
        ids_all = {d.track_id for d in dets if d.track_id is not None}

        # velocities, in head-widths per second so the scale cancels
        pos, vel, hw = [], [], []
        for d in inside:
            if d.track_id is None:
                continue
            width = max(1.0, d.x2 - d.x1)
            q = hist.setdefault(d.track_id, deque())
            q.append((t, np.array(d.centre), width))
            while len(q) > 1 and t - q[0][0] > 0.4:
                q.popleft()
            if len(q) >= 2:
                t0, p0, _ = q[0]
                dt = t - t0
                if dt > 1e-6:
                    v = (np.array(d.centre) - p0) / dt / width
                    if np.linalg.norm(v) < 20:
                        pos.append(np.array(d.centre) / width)
                        vel.append(v)
                        hw.append(width)
        for tid in [k for k, q in hist.items() if q and t - q[-1][0] > 2.0]:
            del hist[tid]

        occ = len(inside)
        fs = flow.update(t, ids_inside, ids_all, occupancy=occ)
        speed = float(np.linalg.norm(vel, axis=1).mean()) if vel else 0.0
        disorder = local_velocity_variance(pos, vel, k=5, radius_m=5.0) if len(vel) > 2 else 0.0

        ready = baseline.update(t, occ)
        rising = fs.net_rate > args.rate_threshold

        # Hysteresis: a higher bar to enter a state than to stay in it. Without
        # it the level oscillates whenever occupancy sits on the threshold.
        if level in ("WATCH", "ELEVATED"):
            high = baseline.is_high(occ * (1.0 / args.exit_frac))
        else:
            high = baseline.is_high(occ)

        both = ready and high and rising
        both_since = (both_since or t) if both else None
        held = (t - both_since) if both_since is not None else 0.0

        quiet = not (high and rising)
        clear_since = (clear_since or t) if quiet else None
        cleared = clear_since is not None and (t - clear_since) >= args.clear_s

        if not ready:
            level = "LEARNING"
        elif level == "ELEVATED" and not cleared:
            pass                                  # hold until the clear time
        elif both and held >= args.dwell_s:
            level = "ELEVATED"
        elif high:
            level = "WATCH"
        elif level == "WATCH" and not cleared:
            pass                                  # hold WATCH through a dip
        else:
            level = "NORMAL"

        trace.append((t, occ))
        rows.append({
            "frame": i + 1, "t": round(t, 3), "occupancy": occ,
            "speed_hw_s": round(speed, 3), "disorder": round(disorder, 4),
            "net_rate": round(fs.net_rate, 3),
            "inflow": round(fs.inflow_rate, 3), "outflow": round(fs.outflow_rate, 3),
            "baseline": round(baseline.threshold, 1) if baseline.threshold else "",
            "level": level,
        })

        if not args.no_video:
            colour = LEVEL_COLOUR.get(level, (200, 200, 200))
            cv2.polylines(img, [poly], True, colour, 3)
            for d in inside:
                cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                              colour, 2)
            panel = img[0:235, 0:720].copy()
            cv2.rectangle(panel, (0, 0), (720, 235), (18, 18, 18), -1)
            cv2.addWeighted(panel, 0.72, img[0:235, 0:720], 0.28, 0,
                            dst=img[0:235, 0:720])
            lines = [
                (f"{level}", 1.25, colour),
                (f"occupancy   {occ:4d}   (scene baseline "
                 f"{baseline.threshold:.0f})" if baseline.threshold
                 else f"occupancy   {occ:4d}   (learning baseline)", 0.78, (235,)*3),
                (f"net rate    {fs.net_rate:+.2f} people/s", 0.78, (235,)*3),
                (f"in {fs.inflow_rate:.2f} / out {fs.outflow_rate:.2f} per s", 0.78, (235,)*3),
                (f"speed       {speed:.2f} head-widths/s", 0.78, (235,)*3),
                (f"disorder    {disorder:.3f}", 0.78, (235,)*3),
            ]
            y = 44
            for text, scale, col in lines:
                cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                            col, 2, cv2.LINE_AA)
                y += 34 if scale > 1 else 30
            if len(trace) > 2:
                ts = [p[1] for p in trace]
                lo, hi = min(ts), max(ts)
                span = (hi - lo) or 1.0
                pts = [(int(18 + j * 680 / (len(ts) - 1)),
                        int(228 - (v - lo) / span * 44)) for j, v in enumerate(ts)]
                cv2.polylines(img, [np.array(pts, np.int32)], False, colour, 2, cv2.LINE_AA)
            cv2.putText(img, "uncalibrated: scene-relative anomaly, not absolute density",
                        (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),
                        2, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(img)
            if i in (int(args.max_frames * 0.25), int(args.max_frames * 0.6),
                     int(args.max_frames * 0.9)):
                cv2.imwrite(str(out_dir / f"still_{i:04d}.jpg"), img)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames_src)}  {level}", end="\r")

    if writer is not None:
        writer.release()

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    occ = np.array([r["occupancy"] for r in rows], float)
    levels = [r["level"] for r in rows]
    summary = {
        "sequence": label, "frames": len(rows),
        "duration_s": round(len(rows) / fps, 1),
        "region_frac": args.region_frac,
        "scene_baseline_occupancy": baseline.threshold,
        "occupancy": {"mean": float(occ.mean()), "min": float(occ.min()),
                      "max": float(occ.max())},
        "levels": {lv: levels.count(lv) for lv in set(levels)},
        "mode": "uncalibrated, scene-relative threshold",
        "caveat": "flags departure from this camera's learned normal; it is "
                  "not an absolute danger threshold and does not detect a crush",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{label}: {len(rows)} frames, {len(rows)/fps:.0f}s")
    print(f"occupancy mean {occ.mean():.0f}, range {occ.min():.0f}-{occ.max():.0f}")
    print(f"scene baseline (85th pct of first {args.learn_s:.0f}s): "
          f"{baseline.threshold:.0f}")
    for lv, n in sorted(summary["levels"].items(), key=lambda kv: -kv[1]):
        print(f"  {lv:9s} {n:4d} frames ({n/len(rows)*100:.0f}%)")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
