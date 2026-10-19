"""Validate flow balance on CroHD, and forecast occupancy.

Flow balance is the one forward-looking part of the system, so it needs a
check that does not depend on having an incident to predict. Conservation
provides one: people do not appear or vanish, so

    occupancy(t) - occupancy(0)  ==  entries - exits

If the counted crossings reproduce the measured occupancy change, the crossing
logic is sound. Where they drift apart, the gap is caused by track churn -
identities lost and reacquired in a dense crowd - and the size of that drift is
the honest limit on any projection built from it.

Run on ground-truth tracks first: that establishes what the method can do with
perfect identities. Then through the pipeline, which shows what tracking
quality actually costs.

    python scripts/flow_forecast.py --root C:/path/to/HT21 \\
        --weights runs/detect/heads_crohd/weights/best.pt --out runs/flow
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdw.flow import FlowBalance  # noqa: E402


def load_gt(gt_path: Path):
    out: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        f = line.strip().split(",")
        if len(f) < 6:
            continue
        if len(f) >= 7 and float(f[6]) == 0:
            continue
        frame, tid = int(float(f[0])), int(float(f[1]))
        x, y, w, h = (float(v) for v in f[2:6])
        out[frame][tid] = (x + w / 2, y + h / 2)
    return out


def centred_region(width: int, height: int, frac: float = 0.6):
    """A box inside the frame, so crossings happen in view.

    Using the frame edge as the boundary would conflate people walking out of
    the region with people walking out of shot.
    """
    w, h = width * frac, height * frac
    x0, y0 = (width - w) / 2, (height - h) / 2
    return (x0, y0, x0 + w, y0 + h)


def inside(region, pt) -> bool:
    x0, y0, x1, y1 = region
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def run_source(frames, region, fps, window_s, label):
    """frames: {frame_no: {track_id: (cx, cy)}} -> per-frame rows."""
    fb = FlowBalance(window_s=window_s)
    rows = []
    cum_entries = cum_exits = 0
    first_occ = None
    for fr in sorted(frames):
        t = fr / fps
        people = frames[fr]
        ins = {tid for tid, p in people.items() if inside(region, p)}
        s = fb.update(t, ins, set(people))
        cum_entries += s.entries
        cum_exits += s.exits
        if first_occ is None:
            first_occ = s.occupancy
        rows.append({
            "source": label, "frame": fr, "t": round(t, 3),
            "occupancy": s.occupancy,
            "inflow_rate": round(s.inflow_rate, 3),
            "outflow_rate": round(s.outflow_rate, 3),
            "net_rate": round(s.net_rate, 3),
            "crossing_net_rate": round(s.crossing_net_rate, 3),
            "discrepancy": round(s.crossing_discrepancy, 3),
            "proj_30s": round(s.projected_occupancy(30.0), 1),
        })
    return rows, first_occ


def conservation_check(frames, region, fps):
    """Cumulative crossings vs measured occupancy change, over the whole clip."""
    fb = FlowBalance(window_s=1e9)        # never forget, we want totals
    occ = []
    for fr in sorted(frames):
        people = frames[fr]
        ins = {tid for tid, p in people.items() if inside(region, p)}
        s = fb.update(fr / fps, ins, set(people))
        occ.append(s.occupancy)
    entries = sum(1 for _, d in fb._events if d > 0)
    exits = sum(1 for _, d in fb._events if d < 0)
    measured = occ[-1] - occ[0]
    implied = entries - exits
    return {
        "entries": entries, "exits": exits,
        "implied_change": implied,
        "measured_change": measured,
        "drift": implied - measured,
        "mean_occupancy": float(np.mean(occ)),
        "drift_per_100_crossings": (
            (implied - measured) / max(1, entries + exits) * 100
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--sequence", default="HT21-04")
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--region-frac", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seq = Path(args.root) / "train" / args.sequence
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    probe = cv2.imread(str(next((seq / "img1").glob("*.jpg"))))
    h, w = probe.shape[:2]
    region = centred_region(w, h, args.region_frac)
    print(f"{args.sequence}: {w}x{h}, region {args.region_frac:.0%} of frame")

    gt = load_gt(seq / "gt" / "gt.txt")
    gt = {k: gt[k] for k in sorted(gt)[: args.max_frames]}

    results = {"sequence": args.sequence, "region_frac": args.region_frac}
    all_rows = []

    print("\nground truth:")
    rows, _ = run_source(gt, region, 25.0, args.window_s, "ground_truth")
    all_rows += rows
    cons = conservation_check(gt, region, 25.0)
    results["ground_truth"] = cons
    print(f"  entries {cons['entries']}, exits {cons['exits']}")
    print(f"  occupancy change: measured {cons['measured_change']:+d}, "
          f"implied by crossings {cons['implied_change']:+d}, "
          f"drift {cons['drift']:+d}")
    print(f"  drift per 100 crossings: {cons['drift_per_100_crossings']:+.1f}")

    if args.weights:
        from cdw.detect import HeadDetector
        det = HeadDetector(weights=args.weights, conf=0.25, imgsz=args.imgsz,
                           device=args.device, max_det=1000)
        det.reset()
        pipe_frames = {}
        n = min(args.max_frames, len(list((seq / "img1").glob("*.jpg"))))
        print("\npipeline:")
        for fr in range(1, n + 1):
            img = cv2.imread(str(seq / "img1" / f"{fr:06d}.jpg"))
            if img is None:
                continue
            people = {d.track_id: d.centre for d in det.track(img)
                      if d.track_id is not None}
            if people:
                pipe_frames[fr] = people
            if fr % 200 == 0:
                print(f"  {fr}/{n}", end="\r")
        rows, _ = run_source(pipe_frames, region, 25.0, args.window_s, "pipeline")
        all_rows += rows
        cons_p = conservation_check(pipe_frames, region, 25.0)
        results["pipeline"] = cons_p
        print(f"  entries {cons_p['entries']}, exits {cons_p['exits']}")
        print(f"  occupancy change: measured {cons_p['measured_change']:+d}, "
              f"implied {cons_p['implied_change']:+d}, drift {cons_p['drift']:+d}")
        print(f"  drift per 100 crossings: {cons_p['drift_per_100_crossings']:+.1f}")

    with open(out_dir / "flow_frames.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        wtr.writeheader()
        wtr.writerows(all_rows)
    (out_dir / "flow_summary.json").write_text(json.dumps(results, indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {out_dir}")
    print("\nDrift is the honest limit on any projection: it is the share of")
    print("crossings that track churn adds or loses. A projection is only as")
    print("trustworthy as the crossing count behind it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
