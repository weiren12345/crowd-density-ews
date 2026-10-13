"""Generate a synthetic bottleneck clip with exact ground truth.

Purpose: prove the density / pressure / warning stack end to end without a
model, without footage, and without ethics approval. Because the positions are
known exactly, any wrong number here is a bug in our arithmetic rather than a
detector failure - which is the only way to tell those two apart later.

Writes:
    data/raw/synthetic_bottleneck.mp4
    data/annotations/synthetic_bottleneck_gt.json

The scripted scenario: a steady stream that jams. Count climbs from 8 to 26 in
a 6 m^2 region (1.3 -> 4.3 people/m^2), mean flow speed collapses from 1.2 to
0.15 m/s, and velocity noise grows - the combination that should drive crowd
pressure up and trip the two-signal rule.

Usage:
    python scripts/make_synthetic_clip.py
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

# Geometry shared with configs/calibration_synthetic.json - keep in step.
IMAGE_SIZE = (1280, 720)
IMAGE_POINTS = [(380.0, 640.0), (900.0, 640.0), (760.0, 360.0), (520.0, 360.0)]
WORLD_POINTS = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]  # metres, 6 m^2

FPS = 15.0
DURATION_S = 24.0

SHOULDER_M = 0.45   # drawn body width
STATURE_M = 1.70    # drawn body height


def ramp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """Linear ramp from v0 to v1 between t0 and t1, clamped outside."""
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def target_count(t: float) -> int:
    return int(round(ramp(t, 6.0, 16.0, 8, 26)))


def noise_sigma(t: float) -> float:
    """Velocity noise in m/s. Rising noise is the turbulence signal."""
    return ramp(t, 8.0, 18.0, 0.05, 0.90)


def flow_speed(t: float) -> float:
    """Mean forward speed in m/s, collapsing as the jam forms."""
    return ramp(t, 6.0, 16.0, 1.20, 0.15)


class Agent:
    __slots__ = ("tid", "pos")

    def __init__(self, tid: int, pos: np.ndarray):
        self.tid = tid
        self.pos = pos


def simulate(rng: np.random.Generator):
    """Yield (t, agents) per frame."""
    n_frames = int(FPS * DURATION_S)
    agents: list[Agent] = []
    next_id = 1

    for i in range(n_frames):
        t = i / FPS
        want = target_count(t)

        while len(agents) < want:
            agents.append(
                Agent(next_id, np.array([rng.uniform(0.1, 0.5), rng.uniform(0.15, 1.85)]))
            )
            next_id += 1
        while len(agents) > want:
            agents.pop(rng.integers(len(agents)))

        sigma = noise_sigma(t)
        v_flow = flow_speed(t)
        dt = 1.0 / FPS
        for a in agents:
            v = np.array([v_flow, 0.0]) + rng.normal(0.0, sigma, size=2)
            a.pos = a.pos + v * dt
            # Reflect at the region edges so the population stays put and the
            # count follows the schedule rather than the geometry.
            for axis, hi in ((0, 3.0), (1, 2.0)):
                if a.pos[axis] < 0.05:
                    a.pos[axis] = 0.05 + (0.05 - a.pos[axis])
                elif a.pos[axis] > hi - 0.05:
                    a.pos[axis] = (hi - 0.05) - (a.pos[axis] - (hi - 0.05))
                a.pos[axis] = float(np.clip(a.pos[axis], 0.05, hi - 0.05))

        yield t, agents


def boxes_for(agents, ground: GroundPlane):
    """Full-body boxes whose bottom-centre is the agent's ground position."""
    out = []
    for a in agents:
        u, v = ground.to_image([a.pos])[0]
        mpp = ground.metres_per_pixel_at((u, v))
        if not np.isfinite(mpp) or mpp <= 1e-9:
            continue
        w_px = SHOULDER_M / mpp
        h_px = STATURE_M / mpp
        out.append(
            {
                "xyxy": [
                    float(u - w_px / 2), float(v - h_px),
                    float(u + w_px / 2), float(v),
                ],
                "conf": 1.0,
                "track_id": int(a.tid),
            }
        )
    return out


def render(frame, boxes, ground: GroundPlane):
    """Paint a floor, the region outline and crude bodies, for flow texture."""
    frame[:] = 55
    cv2.rectangle(frame, (0, 0), (IMAGE_SIZE[0], 300), (80, 72, 68), -1)

    # Floor grid at 0.5 m, which also makes a wrong homography obvious by eye.
    for x in np.arange(0.0, 3.01, 0.5):
        p = ground.to_image([(x, 0.0), (x, 2.0)]).astype(int)
        cv2.line(frame, tuple(p[0]), tuple(p[1]), (90, 90, 90), 1)
    for y in np.arange(0.0, 2.01, 0.5):
        p = ground.to_image([(0.0, y), (3.0, y)]).astype(int)
        cv2.line(frame, tuple(p[0]), tuple(p[1]), (90, 90, 90), 1)

    for b in sorted(boxes, key=lambda d: d["xyxy"][3]):
        x1, y1, x2, y2 = b["xyxy"]
        cx = int((x1 + x2) / 2)
        wpx = max(3, int(x2 - x1))
        shade = 150 + (b["track_id"] * 37) % 80
        cv2.rectangle(frame, (int(x1), int(y1 + wpx * 0.8)), (int(x2), int(y2)),
                      (shade - 40, shade - 20, shade), -1)
        cv2.circle(frame, (cx, int(y1 + wpx * 0.45)), max(2, int(wpx * 0.45)),
                   (shade - 70, shade - 50, shade - 20), -1)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--video", default=str(ROOT / "data/raw/synthetic_bottleneck.mp4"))
    ap.add_argument(
        "--gt", default=str(ROOT / "data/annotations/synthetic_bottleneck_gt.json")
    )
    args = ap.parse_args()

    ground = GroundPlane(IMAGE_POINTS, WORLD_POINTS)
    rng = np.random.default_rng(args.seed)

    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gt).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        args.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, IMAGE_SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {args.video}")

    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    all_frames = []
    trajectory_rows = []        # frame_id ped_id x y, in world metres
    try:
        for i, (t, agents) in enumerate(simulate(rng)):
            boxes = boxes_for(agents, ground)
            all_frames.append(boxes)
            # Mirrors the layout trajectory datasets ship, so the validation
            # script can be exercised on known data before real data arrives.
            for a in agents:
                trajectory_rows.append((i, a.tid, float(a.pos[0]), float(a.pos[1])))
            writer.write(render(frame, boxes, ground))
    finally:
        writer.release()

    # A borrowed-calibration fixture: the homography on its own, as a dataset
    # would distribute it, with no correspondences attached.
    h_path = Path(args.video).with_name("synthetic_H.txt")
    np.savetxt(h_path, ground.H, fmt="%.12g")

    traj_path = Path(args.video).with_name("synthetic_annotations.txt")
    traj_path.write_text(
        "\n".join(f"{f} {p} {x:.4f} {y:.4f}" for f, p, x, y in trajectory_rows),
        encoding="utf-8",
    )

    Path(args.gt).write_text(
        json.dumps(
            {
                "meta": {
                    "fps": FPS,
                    "image_size": list(IMAGE_SIZE),
                    "region_area_m2": ground.polygon_area_m2(IMAGE_POINTS),
                    "seed": args.seed,
                    "note": "exact ground truth; positions are known, not detected",
                },
                "frames": all_frames,
            }
        ),
        encoding="utf-8",
    )

    counts = [len(f) for f in all_frames]
    area = ground.polygon_area_m2(IMAGE_POINTS)
    print(f"wrote {args.video}")
    print(f"wrote {args.gt}")
    print(f"wrote {h_path}  (homography alone, as a dataset ships it)")
    print(f"wrote {traj_path}  ({len(trajectory_rows)} world-coordinate rows)")
    print(
        f"{len(all_frames)} frames at {FPS:g} fps, region {area:.2f} m2, "
        f"count {min(counts)}-{max(counts)} "
        f"({min(counts)/area:.2f}-{max(counts)/area:.2f} people/m2)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
