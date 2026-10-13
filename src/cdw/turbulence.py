"""Crowd pressure: density x variance of local velocity.

Following the crowd-pressure measure of Helbing, Johansson and Al-Abideen,
which rose sharply before the flow breakdowns they analysed in video of the
2006 Mina/Jamarat crowd disaster. Pressure is reported in
(people/m^2)(m/s)^2. Verify the definition against the paper before quoting it
in the proposal; see docs/references.md.

Two independent sources of the velocity field:

* ``TrackVelocities`` - per-person velocities from ByteTrack ids projected to
  the ground plane. Preferred, but id switching degrades it exactly where it
  matters most, in dense crowds.
* ``FarnebackFlow`` - dense optical flow, no identities needed. Survives
  density better, but measures apparent motion of texture rather than of
  people, so it cannot distinguish one person moving fast from a group
  shuffling.

Running both and reporting where they disagree is itself a result worth having
in the writeup.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from .calibrate import GroundPlane

__all__ = [
    "TrackVelocities",
    "FarnebackFlow",
    "local_velocity_variance",
    "crowd_pressure",
    "PressureSample",
]


@dataclass(slots=True)
class PressureSample:
    t: float
    pressure: float           # (people/m^2)(m/s)^2
    velocity_variance: float  # (m/s)^2
    mean_speed: float         # m/s
    n_velocities: int
    source: str               # "tracks" or "flow"


class TrackVelocities:
    """Per-track ground-plane velocity by finite difference over a short lag.

    A single-frame difference is dominated by box jitter, so we difference
    against the oldest sample still inside ``lag_s``.
    """

    def __init__(
        self,
        ground: GroundPlane,
        lag_s: float = 0.3,
        max_speed_mps: float = 6.0,
        height_correction=None,
    ):
        self.ground = ground
        self.lag_s = lag_s
        self.max_speed_mps = max_speed_mps
        # Must match the correction the region uses, or speeds are measured on
        # the dilated head plane and come out systematically too fast.
        self.height_correction = height_correction
        self._hist: dict[int, deque[tuple[float, np.ndarray]]] = defaultdict(deque)

    def update(self, t: float, detections, reference_point: str = "box_bottom"):
        """Feed one frame. Returns (positions, velocities) in metres and m/s.

        Only detections carrying a track id contribute. Velocities above
        ``max_speed_mps`` are dropped as association errors - a pedestrian does
        not cross 6 m/s, so such a value is an id switch, not a sprint.
        """
        tracked = [d for d in detections if d.track_id is not None]
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray | None] = []

        if tracked:
            pts = np.array([d.reference_point(reference_point) for d in tracked])
            world = self.ground.to_world(pts)
            if self.height_correction is not None:
                world = self.height_correction.apply(world)
            for d, w in zip(tracked, world):
                h = self._hist[d.track_id]
                h.append((t, w))
                while len(h) > 1 and t - h[0][0] > self.lag_s:
                    h.popleft()
                positions.append(w)
                v: np.ndarray | None = None
                if len(h) >= 2:
                    t0, w0 = h[0]
                    dt = t - t0
                    if dt > 1e-6:
                        cand = (w - w0) / dt
                        if np.linalg.norm(cand) <= self.max_speed_mps:
                            v = cand
                velocities.append(v)

        self._prune(t)
        keep = [(p, v) for p, v in zip(positions, velocities) if v is not None]
        if not keep:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.array([p for p, _ in keep]), np.array([v for _, v in keep])

    def _prune(self, t: float, stale_s: float = 2.0) -> None:
        for tid in [k for k, v in self._hist.items() if v and t - v[-1][0] > stale_s]:
            del self._hist[tid]

    def reset(self) -> None:
        self._hist.clear()


class FarnebackFlow:
    """Dense optical flow inside a region, converted to m/s on the ground plane.

    The pixel->metre scale varies across the frame under perspective, so each
    sampled flow vector is converted using the homography's local Jacobian at
    that pixel rather than one global scale factor.
    """

    def __init__(self, ground: GroundPlane, step: int = 16, height_correction=None):
        self.ground = ground
        self.step = step
        self.height_correction = height_correction
        self._prev_gray: np.ndarray | None = None
        self._prev_t: float | None = None
        self._jac_cache: dict[tuple[int, int], np.ndarray] = {}

    def update(self, t: float, frame, mask: np.ndarray | None = None):
        """Feed one frame. Returns (positions_m, velocities_mps)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._prev_gray is None or self._prev_t is None:
            self._prev_gray, self._prev_t = gray, t
            return np.empty((0, 2)), np.empty((0, 2))

        dt = t - self._prev_t
        if dt <= 1e-6:
            return np.empty((0, 2)), np.empty((0, 2))

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        self._prev_gray, self._prev_t = gray, t

        h, w = gray.shape[:2]
        ys = np.arange(self.step // 2, h, self.step)
        xs = np.arange(self.step // 2, w, self.step)
        positions, velocities = [], []
        for y in ys:
            for x in xs:
                if mask is not None and mask[y, x] == 0:
                    continue
                du, dv = flow[y, x]
                try:
                    jac = self._jacobian(int(x), int(y))
                    pos = self.ground.to_world([(x, y)])[0]
                except ValueError:
                    continue
                if self.height_correction is not None:
                    # Flow is measured on whatever surface the texture sits on;
                    # in a crowd that is mostly heads, so apply the same
                    # correction to both the position and the displacement.
                    pos = self.height_correction.apply([pos])[0]
                    jac = jac * self.height_correction.scale
                velocities.append((jac @ np.array([du, dv])) / dt)
                positions.append(pos)

        if not positions:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.array(positions), np.array(velocities)

    def _jacobian(self, x: int, y: int) -> np.ndarray:
        key = (x, y)
        if key not in self._jac_cache:
            self._jac_cache[key] = self.ground.jacobian_at((x, y))
        return self._jac_cache[key]

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_t = None


def local_velocity_variance(
    positions, velocities, k: int = 5, radius_m: float | None = None
) -> float:
    """Mean over people of the velocity variance in their neighbourhood.

    "Local" matters: a crowd all walking one way at 1.2 m/s has high global
    speed but near-zero local variance, and is not dangerous. A crowd with
    people pushed in conflicting directions has low mean speed and high local
    variance, which is the condition of interest.

    Neighbourhood is the ``k`` nearest others, optionally capped at
    ``radius_m``. Falls back to global variance when there are too few points
    to form neighbourhoods.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
    n = len(pos)
    if n == 0:
        return 0.0
    if n < 3:
        return float(vel.var(axis=0).sum()) if n > 1 else 0.0

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]

    per_person = []
    for i in range(n):
        neigh = idx[i]
        if radius_m is not None:
            neigh = neigh[d[i, neigh] <= radius_m]
        if len(neigh) == 0:
            continue
        group = np.vstack([vel[i], vel[neigh]])
        per_person.append(group.var(axis=0).sum())
    return float(np.mean(per_person)) if per_person else 0.0


def crowd_pressure(
    t: float,
    density: float,
    positions,
    velocities,
    k: int = 5,
    radius_m: float | None = 2.0,
    source: str = "tracks",
) -> PressureSample:
    """Pressure = density x local velocity variance."""
    var = local_velocity_variance(positions, velocities, k=k, radius_m=radius_m)
    vel = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
    mean_speed = float(np.linalg.norm(vel, axis=1).mean()) if len(vel) else 0.0
    return PressureSample(
        t=t,
        pressure=float(density * var),
        velocity_variance=var,
        mean_speed=mean_speed,
        n_velocities=len(vel),
        source=source,
    )
