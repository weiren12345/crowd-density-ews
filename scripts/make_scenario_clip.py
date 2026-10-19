"""Generate a SYNTHETIC calibrated scenario: a bottleneck filling up.

Purpose, stated plainly so it cannot be mistaken: this shows what the system
WOULD report given calibrated footage of a deteriorating crowd. It is not
evidence about real crowds. Every person, distance and movement here was
fabricated by this script.

It exists because the real footage available to this project (CroHD) contains
only ordinary functioning crowds, so the warning logic correctly stays quiet
and never demonstrates its escalation. A simulated scenario with a known
ground plane closes that gap honestly: the geometry is declared rather than
surveyed, and the clip is watermarked on every frame.

The camera geometry mimics the real demonstration - elevated, looking down
across a paved area - so the two sit side by side sensibly.

The scenario:

    0-10 s    light flow, people crossing freely
    10-22 s   inflow rises
    22 s on   the exit constricts, so outflow collapses while inflow continues

That is the condition the system is built to notice: not merely a dense crowd,
but one that is filling faster than it empties, with movement becoming slower
and more disordered as it does. Speed is coupled to local density, so the
slowdown emerges from the crowding rather than being scripted directly.

Writes the clip, its ground truth, its calibration and a ready config:

    data/raw/scenario_bottleneck.mp4
    data/annotations/scenario_bottleneck_gt.json
    configs/calibration_scenario.json
    configs/scenario.yaml

Then:
    python scripts/run_pipeline.py --config configs/scenario.yaml \\
        --video data/raw/scenario_bottleneck.mp4 --out runs/scenario
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

IMAGE_SIZE = (1280, 720)
FPS = 25.0
DURATION_S = 56.0

# --- Declared geometry -----------------------------------------------------
# FABRICATED. On real footage these come from a tape measure; here they are
# simply asserted, which is the whole point of the exercise.
REGION_W_M = 6.0
REGION_D_M = 5.0                      # 30 m^2
IMAGE_POINTS = [(430.0, 660.0), (1010.0, 660.0), (835.0, 330.0), (605.0, 330.0)]
WORLD_POINTS = [(0.0, 0.0), (REGION_W_M, 0.0), (REGION_W_M, REGION_D_M), (0.0, REGION_D_M)]

SHOULDER_M = 0.45
STATURE_M = 1.70
HEAD_M = 0.20

FREE_SPEED = 1.30                     # m/s unimpeded
JAM_DENSITY = 6.5                     # people/m^2 at which movement stops


def inflow_rate(t: float) -> float:
    """People entering per second."""
    if t < 6.0:
        return 2.2
    if t < 16.0:
        return 2.2 + (7.5 - 2.2) * (t - 6.0) / 10.0
    return 7.5


def exit_capacity(t: float) -> float:
    """Fraction of free speed available in the exit strip.

    Drops at 22 s: the constriction that turns a busy crowd into a filling one.
    """
    if t < 14.0:
        return 1.0
    return max(0.06, 1.0 - 0.94 * min(1.0, (t - 14.0) / 6.0))


class Agent:
    __slots__ = ("tid", "pos", "vel", "pace")

    def __init__(self, tid, pos, pace=1.0):
        self.tid = tid
        self.pos = pos
        self.vel = np.array([0.0, -FREE_SPEED])
        # Individual walking pace. Without this everyone moves identically for
        # a given density, local velocity variance stays near zero, and the
        # crowd-pressure signal never rises - which is not how real jams behave.
        self.pace = pace


def local_density(positions, i, radius=1.0):
    if len(positions) < 2:
        return 0.0
    d = np.linalg.norm(positions - positions[i], axis=1)
    return float((d < radius).sum() - 1) / (np.pi * radius ** 2)


def simulate(rng):
    n_frames = int(FPS * DURATION_S)
    agents: list[Agent] = []
    next_id = 1
    carry = 0.0
    dt = 1.0 / FPS

    for _ in range(26):                  # pre-populate: ~0.9 people/m^2
        agents.append(Agent(next_id,
                            np.array([rng.uniform(0.2, REGION_W_M - 0.2),
                                      rng.uniform(0.2, REGION_D_M - 0.2)]),
                            pace=float(np.clip(rng.normal(1.0, 0.18), 0.5, 1.5))))
        next_id += 1

    for i in range(n_frames):
        t = i / FPS

        carry += inflow_rate(t) * dt
        while carry >= 1.0:
            carry -= 1.0
            agents.append(Agent(next_id,
                                np.array([rng.uniform(0.25, REGION_W_M - 0.25),
                                          REGION_D_M + rng.uniform(0.0, 0.4)]),
                                pace=float(np.clip(rng.normal(1.0, 0.18), 0.5, 1.5))))
            next_id += 1

        if agents:
            pos = np.array([a.pos for a in agents])
            cap = exit_capacity(t)
            for j, a in enumerate(agents):
                rho = local_density(pos, j)
                # Speed falls with local density: the fundamental diagram,
                # imposed here rather than measured.
                speed = FREE_SPEED * a.pace * max(0.03, 1.0 - rho / JAM_DENSITY)
                if a.pos[1] < 0.9:
                    speed *= cap
                # Disorder grows with crowding: people shuffle sideways when
                # they cannot move forward, and the queue goes stop-and-go.
                jitter = 0.75 * min(1.0, rho / 3.5)
                if rho > 2.0 and rng.random() < 0.18:
                    speed *= rng.uniform(0.0, 0.35)      # momentary stall
                a.vel = np.array([rng.normal(0.0, jitter), -speed])
                a.pos = a.pos + a.vel * dt
                a.pos[0] = float(np.clip(a.pos[0], 0.05, REGION_W_M - 0.05))

        agents = [a for a in agents if a.pos[1] > -0.35]
        yield t, agents


def boxes_for(agents, ground):
    out = []
    for a in agents:
        if not (0.0 <= a.pos[1] <= REGION_D_M + 0.5):
            continue
        u, v = ground.to_image([a.pos])[0]
        mpp = ground.metres_per_pixel_at((u, v))
        if not np.isfinite(mpp) or mpp <= 1e-9:
            continue
        head_px = HEAD_M / mpp
        out.append({
            "xyxy": [float(u - head_px / 2), float(v - STATURE_M / mpp - head_px / 2),
                     float(u + head_px / 2), float(v - STATURE_M / mpp + head_px / 2)],
            "conf": 1.0, "track_id": int(a.tid),
            "ground": [float(a.pos[0]), float(a.pos[1])],
        })
    return out


def render(frame, agents, ground, t):
    frame[:] = (62, 64, 66)
    # far surface
    cv2.rectangle(frame, (0, 0), (IMAGE_SIZE[0], 300), (78, 80, 84), -1)

    # paving, on the declared ground plane
    for x in np.arange(-3.0, REGION_W_M + 3.01, 1.0):
        p = ground.to_image([(x, -1.0), (x, REGION_D_M + 2.0)])
        cv2.line(frame, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (86, 88, 92), 1)
    for y in np.arange(-1.0, REGION_D_M + 2.01, 1.0):
        p = ground.to_image([(-3.0, y), (REGION_W_M + 3.0, y)])
        cv2.line(frame, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (86, 88, 92), 1)

    # the constriction, drawn so the scenario reads visually
    for side in (-0.9, REGION_W_M + 0.9):
        p = ground.to_image([(side, 0.0), (side, 1.2)])
        cv2.line(frame, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (120, 120, 130), 6)

    for a in sorted(agents, key=lambda q: q.pos[1], reverse=True):
        if not (-0.4 <= a.pos[1] <= REGION_D_M + 0.6):
            continue
        u, v = ground.to_image([a.pos])[0]
        mpp = ground.metres_per_pixel_at((u, v))
        if not np.isfinite(mpp) or mpp <= 1e-9:
            continue
        sh = max(2, int(SHOULDER_M / mpp / 2))
        head_r = max(1, int(HEAD_M / mpp / 2))
        top = int(v - STATURE_M / mpp)
        shade = 90 + (a.tid * 53) % 90
        cv2.ellipse(frame, (int(u), int(v)), (sh, max(1, sh // 2)), 0, 0, 360,
                    (44, 46, 48), -1)                                    # shadow
        cv2.ellipse(frame, (int(u), (top + int(v)) // 2),
                    (sh, max(2, int((v - top) / 2))), 0, 0, 360,
                    (shade - 25, shade - 10, shade), -1)                 # body
        cv2.circle(frame, (int(u), top), head_r, (shade - 45, shade - 25, shade + 15), -1)
        cv2.circle(frame, (int(u), top), head_r, (35, 35, 40), 1)

    cv2.putText(frame, "SYNTHETIC SCENARIO - fabricated geometry and movement",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "not evidence about real crowds",
                (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 200, 255), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--video", default=str(ROOT / "data/raw/scenario_bottleneck.mp4"))
    ap.add_argument("--gt", default=str(ROOT / "data/annotations/scenario_bottleneck_gt.json"))
    args = ap.parse_args()

    ground = GroundPlane(IMAGE_POINTS, WORLD_POINTS)
    area = ground.polygon_area_m2(IMAGE_POINTS)
    rng = np.random.default_rng(args.seed)

    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gt).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, IMAGE_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {args.video}")

    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    all_frames, counts = [], []
    try:
        for t, agents in simulate(rng):
            boxes = boxes_for(agents, ground)
            all_frames.append([{k: b[k] for k in ("xyxy", "conf", "track_id")}
                               for b in boxes])
            counts.append(sum(1 for a in agents if 0 <= a.pos[1] <= REGION_D_M))
            writer.write(render(frame, agents, ground, t))
    finally:
        writer.release()

    Path(args.gt).write_text(json.dumps({
        "meta": {"fps": FPS, "image_size": list(IMAGE_SIZE),
                 "region_area_m2": area, "seed": args.seed,
                 "synthetic": True,
                 "note": "FABRICATED scenario. Geometry declared, not surveyed."},
        "frames": all_frames}), encoding="utf-8")

    cal = {"note": "FABRICATED calibration for a synthetic scenario. On real "
                   "footage these points come from measured ground features.",
           "image_points": [list(p) for p in IMAGE_POINTS],
           "world_points": [list(p) for p in WORLD_POINTS]}
    (ROOT / "configs/calibration_scenario.json").write_text(json.dumps(cal, indent=2),
                                                            encoding="utf-8")

    (ROOT / "configs/scenario.yaml").write_text(f"""# SYNTHETIC calibrated scenario.
#
# Shows what the system would report given calibrated footage of a bottleneck
# that fills faster than it empties. Geometry and movement are FABRICATED;
# this is a demonstration of behaviour, not evidence about real crowds.
#
# Detection is replayed from ground truth, so this exercises calibration,
# density, crowd pressure and the warning rule without detector error.

name: scenario_bottleneck

calibration:
  file: configs/calibration_scenario.json

region:
  name: bottleneck
  polygon: {[[int(x), int(y)] for x, y in IMAGE_POINTS]}

detector:
  synthetic: data/annotations/scenario_bottleneck_gt.json
  reference_point: centre
  track: true

density:
  window_s: 3.0

pressure:
  source: tracks
  lag_s: 0.4
  k: 5
  radius_m: 2.0

thresholds: configs/thresholds.yaml
""", encoding="utf-8")

    counts = np.array(counts)
    print(f"wrote {args.video}")
    print(f"wrote {args.gt}")
    print("wrote configs/calibration_scenario.json, configs/scenario.yaml")
    print(f"\n{len(all_frames)} frames at {FPS:g} fps ({len(all_frames)/FPS:.0f}s)")
    print(f"region {area:.1f} m2")
    print(f"occupancy {counts.min()}-{counts.max()} "
          f"({counts.min()/area:.2f}-{counts.max()/area:.2f} people/m2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
