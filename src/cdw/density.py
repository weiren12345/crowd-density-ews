"""Density: counts inside a measured region, smoothed over time.

Plain arithmetic on detector output. The two things that make the number mean
anything are the region's real area (from the homography) and the smoothing
window, which suppresses per-frame detector flicker without hiding real change.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .calibrate import GroundPlane, HeightCorrection, polygon_area

__all__ = ["Region", "DensityEstimator", "DensitySample"]


class Region:
    """A polygon drawn in image coordinates with a known real-world area.

    Without a height correction, containment is tested in image space. A
    projective map takes a polygon to a polygon and preserves which side of
    each edge a point falls on, provided the region does not straddle the
    horizon — which a ground-plane region in front of the camera never does.

    With a height correction configured, the reference point is no longer on
    the ground plane, so the test has to happen after the correction: project
    to world, contract toward the nadir, then test against the world polygon.
    """

    def __init__(
        self,
        image_polygon,
        ground: GroundPlane,
        name: str = "region",
        height_correction: HeightCorrection | None = None,
    ):
        self.polygon = np.asarray(image_polygon, dtype=np.float64).reshape(-1, 2)
        if len(self.polygon) < 3:
            raise ValueError("a region needs at least 3 points")
        self.ground = ground
        self.name = name
        self.height_correction = height_correction
        # The region outline is drawn on the ground itself, so its area needs
        # no correction - only the people projected into it do.
        self.world_polygon = ground.to_world(self.polygon)
        self.area_m2 = polygon_area(self.world_polygon)
        if self.area_m2 <= 0:
            raise ValueError("region projects to zero area; check the calibration")
        self._cv_poly = self.polygon.astype(np.float32)
        self._cv_world_poly = self.world_polygon.astype(np.float32)

    def contains(self, pixel) -> bool:
        if self.height_correction is None:
            x, y = float(pixel[0]), float(pixel[1])
            return cv2.pointPolygonTest(self._cv_poly, (x, y), False) >= 0
        wx, wy = self.to_ground(pixel)
        return cv2.pointPolygonTest(self._cv_world_poly, (float(wx), float(wy)), False) >= 0

    def to_ground(self, pixel) -> np.ndarray:
        """One pixel -> its corrected ground-plane position in metres."""
        world = self.ground.to_world([pixel])
        if self.height_correction is not None:
            world = self.height_correction.apply(world)
        return world[0]

    def mask(self, shape) -> np.ndarray:
        """Binary mask of the region, for optical-flow sampling."""
        h, w = shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [self.polygon.astype(np.int32)], 255)
        return m

    def filter(self, detections, reference_point: str = "box_bottom") -> list:
        """Detections whose ground reference point falls inside the region."""
        return [d for d in detections if self.contains(d.reference_point(reference_point))]

    def world_positions(self, detections, reference_point: str = "box_bottom") -> np.ndarray:
        """(N,2) corrected ground-plane positions in metres for the detections."""
        if not detections:
            return np.empty((0, 2))
        pts = np.array([d.reference_point(reference_point) for d in detections])
        world = self.ground.to_world(pts)
        if self.height_correction is not None:
            world = self.height_correction.apply(world)
        return world


@dataclass(slots=True)
class DensitySample:
    t: float
    count: int
    raw_density: float       # people/m^2 this frame
    density: float           # people/m^2 smoothed over the window
    window_frames: int       # how many frames the smoothed value averages


class DensityEstimator:
    """Count -> people/m^2, averaged over a trailing time window.

    The window is in *seconds*, not frames, so the behaviour does not change
    when the clip's frame rate does.
    """

    def __init__(self, region: Region, window_s: float = 3.0):
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.region = region
        self.window_s = window_s
        self._buf: deque[tuple[float, float]] = deque()

    def update(self, t: float, count: int) -> DensitySample:
        raw = count / self.region.area_m2
        self._buf.append((t, raw))
        cutoff = t - self.window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        vals = [v for _, v in self._buf]
        return DensitySample(
            t=t,
            count=count,
            raw_density=raw,
            density=float(np.mean(vals)),
            window_frames=len(vals),
        )

    def reset(self) -> None:
        self._buf.clear()

    @property
    def area_m2(self) -> float:
        return self.region.area_m2
