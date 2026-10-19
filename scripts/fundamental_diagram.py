"""Measure the pedestrian fundamental diagram, and locate the critical density.

The problem this solves: a warning system needs a threshold, and we have no
crush footage to calibrate one against. The fundamental diagram is the way out.

Pedestrian flow obeys a well-established relationship - as density rises, speed
falls, and flow (density x speed) rises to a maximum and then collapses. The
density at peak flow is the **critical density**: below it the crowd is stable,
above it flow breaks down and disturbances amplify. That point can be located
from entirely ordinary footage. No incident required.

Two choices make this measurable here:

**Per-person, not per-frame.** Density is a local quantity. One number per
frame averages a packed corner together with an empty one and destroys the
relationship. Measuring each person's own local crowding and their own speed
turns a few hundred frames into tens of thousands of samples spanning a wide
density range.

**Head-widths, not pixels.** A head is roughly constant in the real world, so
dividing by the median head width in each frame removes the per-scene pixel
scale without needing a homography. Optionally converted to approximate metres
with an assumed head width - clearly an assumption, and reported as one.

    python scripts/fundamental_diagram.py --root C:/path/to/HT21 --out runs/fd

By default this measures the diagram from GROUND TRUTH tracks: that establishes
the physics independently of our detector. Pass --weights to additionally
measure it through the pipeline, which tests whether the system can recover the
same relationship it would have to rely on in deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

K_NEIGHBOURS = 6


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
        out[frame][tid] = (x + w / 2, y + h / 2, w)
    return out


def per_person_samples(frames, fps, lag_s, max_speed_px):
    """Yield (local_spacing_headwidths, speed_headwidths_per_s) per person.

    Local spacing is the mean distance to the K nearest neighbours, which is
    the standard way to estimate local density from point positions.
    """
    hist: dict[int, deque] = defaultdict(deque)
    out_spacing, out_speed = [], []

    for frame_no in sorted(frames):
        people = frames[frame_no]
        if len(people) <= K_NEIGHBOURS:
            continue
        t = frame_no / fps
        ids = list(people)
        pos = np.array([[people[i][0], people[i][1]] for i in ids], float)
        # Each person is normalised by THEIR OWN head width, not the frame
        # median. Under perspective a distant person has a small head and small
        # pixel spacing; a frame-wide scale factor mis-scales both and
        # manufactures spurious high-density, high-speed samples at the back of
        # the scene. Per-person scaling is what makes this depth-invariant.
        head_w_each = np.array([people[i][2] for i in ids], float)

        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        idx = np.argpartition(d, K_NEIGHBOURS - 1, axis=1)[:, :K_NEIGHBOURS]
        spacing_px = np.array([d[i, idx[i]].mean() for i in range(len(ids))])

        # Smooth the scale estimate over each neighbourhood. A single box width
        # is noisy, and because density scales as 1/w^2 while speed scales as
        # 1/w, that noise drives both UP together - a shared denominator that
        # manufactures a positive density-speed correlation out of nothing.
        # Neighbours sit at similar depth, so their median preserves the
        # perspective variation while cancelling most of the per-box noise.
        head_w_each = np.array([
            np.median(np.concatenate(([head_w_each[i]], head_w_each[idx[i]])))
            for i in range(len(ids))
        ])

        for i, tid in enumerate(ids):
            hw = head_w_each[i]
            if not np.isfinite(hw) or hw <= 1:
                continue
            h = hist[tid]
            h.append((t, pos[i]))
            while len(h) > 1 and t - h[0][0] > lag_s:
                h.popleft()
            if len(h) < 2:
                continue
            t0, p0 = h[0]
            dt = t - t0
            if dt <= 1e-6:
                continue
            speed_px = float(np.linalg.norm(pos[i] - p0) / dt)
            if speed_px > max_speed_px:
                continue
            out_spacing.append(spacing_px[i] / hw)
            out_speed.append(speed_px / hw)

        for tid in [k for k, v in hist.items() if v and t - v[-1][0] > 2.0]:
            del hist[tid]

    return np.array(out_spacing), np.array(out_speed)


def build_diagram(spacing_hw, speed_hw, head_width_m, n_bins=18):
    """Bin by density and return the diagram plus the critical point."""
    # Density from mean neighbour spacing. For points on a plane the area per
    # person scales with spacing^2; the constant is absorbed into the units,
    # which is why this is reported as a proxy rather than an absolute.
    # For points on a regular lattice of pitch a, the mean distance to the 6
    # nearest is 1.138a, giving rho = 1/a^2 ~ 1.295/d^2. Real crowds are not a
    # lattice, so this is a proxy: the SHAPE of the curve is what matters, not
    # the absolute constant.
    density_hw = 1.295 / np.maximum(spacing_hw, 1e-6) ** 2        # per head-width^2
    density_m2 = density_hw / (head_width_m ** 2)
    speed_ms = speed_hw * head_width_m

    keep = np.isfinite(density_m2) & np.isfinite(speed_ms) & (density_m2 < 20)
    density_m2, speed_ms = density_m2[keep], speed_ms[keep]

    lo, hi = np.percentile(density_m2, [1, 99])
    edges = np.linspace(lo, hi, n_bins + 1)
    rows = []
    for a, b in zip(edges, edges[1:]):
        m = (density_m2 >= a) & (density_m2 < b)
        if m.sum() < 30:
            continue
        d_mid = float(density_m2[m].mean())
        v = float(speed_ms[m].mean())
        rows.append({
            "density_m2": d_mid,
            "speed_ms": v,
            "speed_sd": float(speed_ms[m].std()),
            "flow": d_mid * v,
            "n": int(m.sum()),
        })
    crit = max(rows, key=lambda r: r["flow"]) if rows else None
    return rows, crit, len(density_m2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None,
                    help="also measure through the pipeline, not just ground truth")
    ap.add_argument("--head-width-m", type=float, default=0.18,
                    help="assumed real head width, used only to express the "
                         "result in metres. An assumption, not a measurement.")
    ap.add_argument("--lag-s", type=float, default=0.4)
    ap.add_argument("--max-frames", type=int, default=3500)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = Path(args.root) / "train"
    seqs = sorted(d for d in train.iterdir() if (d / "gt" / "gt.txt").exists())

    results = {}

    # -- ground truth ------------------------------------------------------
    all_sp, all_sv = [], []
    print("ground truth:")
    for seq in seqs:
        gt = load_gt(seq / "gt" / "gt.txt")
        gt = {k: gt[k] for k in sorted(gt)[: args.max_frames]}
        sp, sv = per_person_samples(gt, 25.0, args.lag_s, 400.0)
        print(f"  {seq.name}: {len(sp):,} person-samples")
        all_sp.append(sp)
        all_sv.append(sv)
    sp = np.concatenate(all_sp)
    sv = np.concatenate(all_sv)
    rows, crit, n = build_diagram(sp, sv, args.head_width_m)
    results["ground_truth"] = {"diagram": rows, "critical": crit, "n_samples": n}

    print(f"\nfundamental diagram from {n:,} person-observations")
    print(f"  {'density':>9} {'speed':>8} {'flow':>8} {'n':>8}")
    for r in rows:
        print(f"  {r['density_m2']:9.2f} {r['speed_ms']:8.3f} {r['flow']:8.3f} {r['n']:8d}")
    if crit:
        print(f"\ncritical density (peak flow): {crit['density_m2']:.2f} people/m^2"
              f"  at speed {crit['speed_ms']:.2f} m/s")

    # -- through the pipeline ---------------------------------------------
    if args.weights:
        import cv2
        from cdw.detect import HeadDetector

        det = HeadDetector(weights=args.weights, conf=0.25, imgsz=args.imgsz,
                           device=args.device, max_det=1000)
        all_sp, all_sv = [], []
        print("\npipeline:")
        for seq in seqs:
            det.reset()
            frames: dict[int, dict] = {}
            n_fr = min(args.max_frames, len(list((seq / "img1").glob("*.jpg"))))
            for fr in range(1, n_fr + 1):
                img = cv2.imread(str(seq / "img1" / f"{fr:06d}.jpg"))
                if img is None:
                    continue
                people = {}
                for d in det.track(img):
                    if d.track_id is not None:
                        cx, cy = d.centre
                        people[d.track_id] = (cx, cy, d.x2 - d.x1)
                if people:
                    frames[fr] = people
                if fr % 200 == 0:
                    print(f"  {seq.name}: {fr}/{n_fr}", end="\r")
            s, v = per_person_samples(frames, 25.0, args.lag_s, 400.0)
            print(f"  {seq.name}: {len(s):,} person-samples")
            all_sp.append(s)
            all_sv.append(v)
        sp2 = np.concatenate(all_sp)
        sv2 = np.concatenate(all_sv)
        rows2, crit2, n2 = build_diagram(sp2, sv2, args.head_width_m)
        results["pipeline"] = {"diagram": rows2, "critical": crit2, "n_samples": n2}
        if crit2:
            print(f"\npipeline critical density: {crit2['density_m2']:.2f} people/m^2")

    results["assumptions"] = {
        "head_width_m": args.head_width_m,
        "note": "Metric values depend on the assumed head width. The shape of "
                "the curve and the location of peak flow in density units do "
                "not; only the axis labels scale.",
    }
    (out_dir / "fundamental_diagram.json").write_text(json.dumps(results, indent=2),
                                                      encoding="utf-8")
    print(f"\nwrote {out_dir/'fundamental_diagram.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
