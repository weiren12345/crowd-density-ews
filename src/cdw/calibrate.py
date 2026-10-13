"""Ground-plane calibration: image pixels <-> real-world metres.

A homography is valid for a single flat plane. We fit it to the *ground*, so
every point we push through it is implicitly assumed to lie on the ground. A
detected head does not, which introduces a systematic bias documented in
docs/limitations.md. One calibration per camera position.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

__all__ = ["GroundPlane", "HeightCorrection", "polygon_area"]


class HeightCorrection:
    """Undo the long-projection error from pushing head points through a
    ground-plane homography.

    A head is about 1.7 m above the plane the homography describes, so the ray
    from the camera through the head meets the ground *beyond* the person. For
    a camera at height ``H`` whose nadir (the ground point directly below it)
    is ``N``, a head at true horizontal distance ``R`` from ``N`` projects to

        R' = R * H / (H - h)

    i.e. a pure radial dilation about the nadir. Inverting it is one multiply:

        R = R' * (H - h) / H

    The error is not small. At H = 6 m and h = 1.7 m, someone 5 m from the
    nadir lands 2.0 m too far away - comparable to the width of the region
    being measured. Mitigating it costs two measurements: the camera height,
    and where the nadir falls in world coordinates.

    This is exact only for people of height ``object_height_m`` standing
    upright on the plane; real stature varies by roughly +/- 0.1 m, which
    leaves a residual of order 0.1 m at the distances above. That residual is
    worth stating in the writeup, and is still far better than ignoring the
    effect.
    """

    def __init__(self, nadir_xy, camera_height_m: float, object_height_m: float = 1.7):
        if camera_height_m <= object_height_m:
            raise ValueError(
                "camera_height_m must exceed object_height_m; a camera at or "
                "below head height cannot see the ground plane behind people"
            )
        self.nadir = np.asarray(nadir_xy, dtype=np.float64).reshape(2)
        self.camera_height_m = float(camera_height_m)
        self.object_height_m = float(object_height_m)
        self.scale = (self.camera_height_m - self.object_height_m) / self.camera_height_m

    def apply(self, world_points) -> np.ndarray:
        """Contract head-plane projections back onto the ground plane."""
        pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return pts
        return self.nadir + (pts - self.nadir) * self.scale

    def to_dict(self) -> dict:
        return {
            "nadir": self.nadir.tolist(),
            "camera_height_m": self.camera_height_m,
            "object_height_m": self.object_height_m,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeightCorrection":
        return cls(
            d["nadir"],
            float(d["camera_height_m"]),
            float(d.get("object_height_m", 1.7)),
        )


def polygon_area(points: np.ndarray) -> float:
    """Unsigned area of a simple polygon via the shoelace formula."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


class GroundPlane:
    """Maps image pixels to metres on the ground plane.

    Built from >=4 correspondences between image points and world points whose
    real-world spacing is known (a paving grid, a measured rectangle of tape).
    """

    def __init__(self, image_points=None, world_points=None, H=None):
        if image_points is None and world_points is None and H is None:
            raise ValueError("give either correspondences or a homography matrix")

        fitted = H is None
        if image_points is None and world_points is None:
            # Supplied matrix with no correspondences, e.g. a dataset's own
            # calibration. Nothing to measure residuals against.
            self.image_points = np.empty((0, 2))
            self.world_points = np.empty((0, 2))
        else:
            self.image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
            self.world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
            if len(self.image_points) != len(self.world_points):
                raise ValueError("image_points and world_points must be the same length")
            if fitted and len(self.image_points) < 4:
                raise ValueError("need at least 4 correspondences to fit a homography")

        if fitted:
            H, _ = cv2.findHomography(self.image_points, self.world_points, method=0)
            if H is None:
                raise ValueError(
                    "homography fit failed; are the 4 points collinear or duplicated?"
                )

        self.H = np.asarray(H, dtype=np.float64)
        if self.H.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got {self.H.shape}")
        if not np.all(np.isfinite(self.H)):
            raise ValueError("homography contains non-finite values")
        if abs(np.linalg.det(self.H)) < 1e-12:
            raise ValueError("homography is singular and cannot be inverted")
        self.H_inv = np.linalg.inv(self.H)
        self.fitted_from_points = fitted

    @classmethod
    def from_matrix(cls, H, direction: str = "image_to_world", scale: float = 1.0):
        """Build from a homography somebody else measured.

        Trajectory datasets (ETH, UCY and similar) ship an ``H.txt`` because
        their annotations are already in world metres. Reusing it means testing
        this pipeline rather than our own tape measure.

        Two things vary between datasets and are not detectable automatically:

        ``direction``
            Whether their matrix maps image->world (ours) or world->image. Get
            it backwards and everything still *runs*, producing densities wrong
            by orders of magnitude. ``describe_scene()`` is the check.
        ``scale``
            Multiplier into metres, for a matrix expressed in other units
            (e.g. 0.01 if it yields centimetres).

        Because there are no correspondences, ``reprojection_error()`` reports
        nothing: residuals require points to be residual *from*. Validate with
        ``describe_scene()`` instead.
        """
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got {H.shape}")
        if direction == "world_to_image":
            H = np.linalg.inv(H)
        elif direction != "image_to_world":
            raise ValueError('direction must be "image_to_world" or "world_to_image"')
        if scale != 1.0:
            H = H.copy()
            H[:2, :] *= scale
        return cls(H=H)

    def describe_scene(self, image_size) -> dict:
        """Project the image corners and report the implied real-world extent.

        The sanity check for a borrowed homography. A pedestrian scene should
        come out tens of metres across; if this says 0.4 m or 40 km, the
        matrix direction or its units are wrong, and every density downstream
        would have been silently wrong too.
        """
        w, h = int(image_size[0]), int(image_size[1])
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        try:
            world = self.to_world(corners)
        except ValueError as exc:
            return {"ok": False, "reason": f"corner projects to the horizon: {exc}"}

        extent_x = float(world[:, 0].max() - world[:, 0].min())
        extent_y = float(world[:, 1].max() - world[:, 1].min())
        area = polygon_area(world)
        plausible = (0.5 <= extent_x <= 1000.0) and (0.5 <= extent_y <= 1000.0)
        return {
            "ok": bool(plausible),
            "extent_x_m": extent_x,
            "extent_y_m": extent_y,
            "frame_area_m2": float(area),
            "corners_world": world.tolist(),
            "reason": (
                "plausible for a pedestrian scene"
                if plausible
                else "implausible extent - check the matrix direction and units"
            ),
        }

    # -- projection ---------------------------------------------------------

    def to_world(self, pixels) -> np.ndarray:
        """Pixels (N,2) -> ground-plane metres (N,2)."""
        return _apply(self.H, pixels)

    def to_image(self, metres) -> np.ndarray:
        """Ground-plane metres (N,2) -> pixels (N,2)."""
        return _apply(self.H_inv, metres)

    def jacobian_at(self, pixel) -> np.ndarray:
        """2x2 matrix d(world)/d(pixel) at one pixel, in metres per pixel.

        Used to convert a pixel-space displacement (e.g. an optical-flow
        vector) into metres. The scale is not constant across the frame under
        perspective, so this must be evaluated locally rather than once.
        """
        px = np.asarray(pixel, dtype=np.float64).reshape(2)
        u, v = px
        h = self.H
        d = h[2, 0] * u + h[2, 1] * v + h[2, 2]
        if abs(d) < 1e-12:
            raise ValueError(f"pixel {px} projects to the horizon; scale undefined")
        nx = h[0, 0] * u + h[0, 1] * v + h[0, 2]
        ny = h[1, 0] * u + h[1, 1] * v + h[1, 2]
        return np.array(
            [
                [(h[0, 0] * d - nx * h[2, 0]) / d**2, (h[0, 1] * d - nx * h[2, 1]) / d**2],
                [(h[1, 0] * d - ny * h[2, 0]) / d**2, (h[1, 1] * d - ny * h[2, 1]) / d**2],
            ]
        )

    def metres_per_pixel_at(self, pixel) -> float:
        """Isotropic approximation of local scale: sqrt of the area ratio."""
        return float(np.sqrt(abs(np.linalg.det(self.jacobian_at(pixel)))))

    # -- quality ------------------------------------------------------------

    def reprojection_error(self) -> dict:
        """Residuals of the fit itself, in metres.

        Report these in the writeup: they bound how precise any density figure
        can be. A max residual comparable to a person's own footprint means the
        calibration, not the detector, is the limiting factor.

        Returns ``fitted: False`` and no residuals for a borrowed homography -
        there are no correspondences to measure against, and reporting a zero
        here would imply a precision nobody established.
        """
        if len(self.image_points) == 0:
            return {
                "fitted": False,
                "mean_m": None,
                "max_m": None,
                "rms_m": None,
                "per_point_m": [],
                "note": "homography supplied directly; no correspondences to residual from",
            }
        pred = self.to_world(self.image_points)
        err = np.linalg.norm(pred - self.world_points, axis=1)
        return {
            "fitted": True,
            "mean_m": float(err.mean()),
            "max_m": float(err.max()),
            "rms_m": float(np.sqrt((err**2).mean())),
            "per_point_m": [float(e) for e in err],
        }

    def polygon_area_m2(self, image_polygon) -> float:
        """Real-world area of a polygon drawn in image coordinates."""
        return polygon_area(self.to_world(image_polygon))

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "image_points": self.image_points.tolist(),
            "world_points": self.world_points.tolist(),
            "H": self.H.tolist(),
            "fitted_from_points": self.fitted_from_points,
            "reprojection_error": self.reprojection_error(),
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "GroundPlane":
        pts = d.get("image_points") or None
        world = d.get("world_points") or None
        if pts is None or world is None:
            if not d.get("H"):
                raise ValueError("calibration has neither correspondences nor a matrix")
            return cls(H=d["H"])
        return cls(pts, world, H=d.get("H"))

    @classmethod
    def load(cls, path) -> "GroundPlane":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_rectangle(cls, image_points, width_m: float, height_m: float) -> "GroundPlane":
        """Convenience fit from 4 image points known to be a rectangle.

        Points must be given clockwise or counter-clockwise starting at the
        corner that becomes the world origin.
        """
        world = [(0.0, 0.0), (width_m, 0.0), (width_m, height_m), (0.0, height_m)]
        return cls(image_points, world)


def _apply(M: np.ndarray, points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    out = hom @ M.T
    w = out[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("a point projects to the horizon; check the calibration")
    return out[:, :2] / w
