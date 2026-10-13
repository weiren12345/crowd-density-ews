"""Calibration tests.

The point of these is that an area error is invisible downstream: a region
measured 20% too large makes every density 20% too low, and nothing in the
pipeline will complain. So the area has to be pinned against geometry we can
compute by hand.
"""

import numpy as np
import pytest

from cdw.calibrate import GroundPlane, polygon_area

# A trapezoid in the image that is a 3 m x 2 m rectangle on the ground.
IMAGE_POINTS = [(380.0, 640.0), (900.0, 640.0), (760.0, 360.0), (520.0, 360.0)]
WORLD_POINTS = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]


@pytest.fixture
def ground():
    return GroundPlane(IMAGE_POINTS, WORLD_POINTS)


def test_shoelace_area_of_unit_square():
    assert polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)]) == pytest.approx(1.0)


def test_shoelace_is_orientation_independent():
    cw = [(0, 0), (0, 1), (1, 1), (1, 0)]
    ccw = list(reversed(cw))
    assert polygon_area(cw) == pytest.approx(polygon_area(ccw))


def test_fit_is_exact_for_four_points(ground):
    err = ground.reprojection_error()
    # Four correspondences determine a homography exactly, so residuals are
    # numerical noise. Anything larger means the points were mis-ordered.
    assert err["max_m"] < 1e-9


def test_corners_map_to_their_world_points(ground):
    got = ground.to_world(IMAGE_POINTS)
    assert np.allclose(got, WORLD_POINTS, atol=1e-9)


def test_region_area_matches_hand_computed_area(ground):
    assert ground.polygon_area_m2(IMAGE_POINTS) == pytest.approx(6.0, abs=1e-9)


def test_half_the_region_is_half_the_area(ground):
    """Midpoints of the long edges split the rectangle into two 3 m^2 halves."""
    near_mid = np.mean([IMAGE_POINTS[0], IMAGE_POINTS[1]], axis=0)
    far_mid = np.mean([IMAGE_POINTS[2], IMAGE_POINTS[3]], axis=0)
    half = [IMAGE_POINTS[0], near_mid, far_mid, IMAGE_POINTS[3]]
    assert ground.polygon_area_m2(half) == pytest.approx(3.0, abs=1e-6)


def test_roundtrip_image_world_image(ground):
    probes = [(500.0, 600.0), (700.0, 500.0), (640.0, 420.0)]
    back = ground.to_image(ground.to_world(probes))
    assert np.allclose(back, probes, atol=1e-6)


def test_scale_is_larger_further_from_the_camera(ground):
    """Under perspective, one pixel covers more ground further away."""
    near = ground.metres_per_pixel_at((640, 630))
    far = ground.metres_per_pixel_at((640, 370))
    assert far > near * 1.5


def test_jacobian_matches_finite_difference(ground):
    """The analytic Jacobian is what converts flow vectors to m/s."""
    px = np.array([620.0, 520.0])
    analytic = ground.jacobian_at(px)
    h = 1e-4
    numeric = np.empty((2, 2))
    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        fwd = ground.to_world([px + step])[0]
        bwd = ground.to_world([px - step])[0]
        numeric[:, j] = (fwd - bwd) / (2 * h)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-9)


def test_save_and_load_roundtrip(ground, tmp_path):
    path = tmp_path / "cal.json"
    ground.save(path)
    loaded = GroundPlane.load(path)
    assert np.allclose(loaded.to_world(IMAGE_POINTS), ground.to_world(IMAGE_POINTS))
    assert loaded.polygon_area_m2(IMAGE_POINTS) == pytest.approx(6.0, abs=1e-9)


def test_from_rectangle_matches_explicit_world_points():
    gp = GroundPlane.from_rectangle(IMAGE_POINTS, width_m=3.0, height_m=2.0)
    assert gp.polygon_area_m2(IMAGE_POINTS) == pytest.approx(6.0, abs=1e-9)


def test_too_few_points_is_rejected():
    with pytest.raises(ValueError, match="at least 4"):
        GroundPlane(IMAGE_POINTS[:3], WORLD_POINTS[:3])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        GroundPlane(IMAGE_POINTS, WORLD_POINTS[:3])


def test_collinear_points_are_rejected():
    collinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    with pytest.raises(ValueError):
        GroundPlane(collinear, WORLD_POINTS)


# -- head-height correction -------------------------------------------------


def head_pixel(ground, person_xy, nadir, camera_h, head_h):
    """Where a person's head appears, given the camera geometry.

    The ray from the camera through the head meets the ground at a point
    dilated radially about the nadir by H/(H-h), so that ground point is what
    the head's pixel corresponds to.
    """
    p = np.asarray(person_xy, float)
    n = np.asarray(nadir, float)
    long_point = n + (p - n) * (camera_h / (camera_h - head_h))
    return ground.to_image([long_point])[0]


def test_correction_recovers_the_true_position(ground):
    from cdw.calibrate import HeightCorrection

    nadir, camera_h, head_h = (1.5, -4.0), 6.0, 1.7
    hc = HeightCorrection(nadir, camera_h, head_h)

    for person in [(0.5, 0.5), (1.5, 1.0), (2.5, 1.8)]:
        px = head_pixel(ground, person, nadir, camera_h, head_h)
        uncorrected = ground.to_world([px])[0]
        corrected = hc.apply([uncorrected])[0]
        assert np.allclose(corrected, person, atol=1e-9)


def test_uncorrected_head_projection_is_badly_wrong(ground):
    """Quantifies the error the correction exists to remove."""
    nadir, camera_h, head_h = (1.5, -4.0), 6.0, 1.7
    person = (2.5, 1.8)
    px = head_pixel(ground, person, nadir, camera_h, head_h)
    uncorrected = ground.to_world([px])[0]
    error = np.linalg.norm(uncorrected - np.array(person))
    assert error > 1.5          # metres - larger than a person is wide


def test_correction_is_identity_for_a_point_at_the_nadir():
    from cdw.calibrate import HeightCorrection

    hc = HeightCorrection((1.5, -4.0), 6.0, 1.7)
    assert np.allclose(hc.apply([(1.5, -4.0)])[0], (1.5, -4.0))


def test_zero_height_object_needs_no_correction():
    from cdw.calibrate import HeightCorrection

    hc = HeightCorrection((1.5, -4.0), 6.0, object_height_m=0.0)
    assert hc.scale == pytest.approx(1.0)
    assert np.allclose(hc.apply([(2.0, 3.0)])[0], (2.0, 3.0))


def test_scale_shrinks_as_the_camera_gets_lower():
    from cdw.calibrate import HeightCorrection

    high = HeightCorrection((0, 0), 12.0, 1.7).scale
    low = HeightCorrection((0, 0), 4.0, 1.7).scale
    assert high > low           # a lower camera means a worse dilation


def test_camera_below_head_height_is_rejected():
    from cdw.calibrate import HeightCorrection

    with pytest.raises(ValueError, match="must exceed"):
        HeightCorrection((0, 0), 1.5, 1.7)


def test_correction_roundtrips_through_a_dict():
    from cdw.calibrate import HeightCorrection

    hc = HeightCorrection((1.5, -4.0), 6.0, 1.7)
    back = HeightCorrection.from_dict(hc.to_dict())
    assert back.scale == pytest.approx(hc.scale)
    assert np.allclose(back.nadir, hc.nadir)


def test_empty_input_is_safe():
    from cdw.calibrate import HeightCorrection

    hc = HeightCorrection((0, 0), 6.0, 1.7)
    assert hc.apply(np.empty((0, 2))).shape == (0, 2)


# -- borrowed homographies --------------------------------------------------


def test_from_matrix_reproduces_the_fitted_plane(ground):
    borrowed = GroundPlane.from_matrix(ground.H)
    probes = [(500.0, 600.0), (700.0, 500.0), (640.0, 420.0)]
    assert np.allclose(borrowed.to_world(probes), ground.to_world(probes))


def test_from_matrix_inverts_a_world_to_image_matrix(ground):
    borrowed = GroundPlane.from_matrix(ground.H_inv, direction="world_to_image")
    probes = [(500.0, 600.0), (700.0, 500.0)]
    assert np.allclose(borrowed.to_world(probes), ground.to_world(probes))


def test_from_matrix_applies_a_unit_scale(ground):
    """A matrix yielding centimetres becomes metres with scale=0.01."""
    H_cm = ground.H.copy()
    H_cm[:2, :] *= 100.0
    borrowed = GroundPlane.from_matrix(H_cm, scale=0.01)
    assert np.allclose(borrowed.to_world(IMAGE_POINTS), ground.to_world(IMAGE_POINTS))


def test_borrowed_calibration_reports_no_residuals(ground):
    err = GroundPlane.from_matrix(ground.H).reprojection_error()
    assert err["fitted"] is False
    assert err["rms_m"] is None       # must not claim a precision nobody measured


def test_fitted_calibration_still_reports_residuals(ground):
    assert ground.reprojection_error()["fitted"] is True


def test_describe_scene_accepts_a_plausible_calibration(ground):
    info = GroundPlane.from_matrix(ground.H).describe_scene((1280, 720))
    assert info["ok"] is True
    assert 0.5 < info["extent_x_m"] < 1000


def test_describe_scene_rejects_a_reversed_matrix(ground):
    """The failure this check exists to catch.

    Using a world->image matrix as if it were image->world still runs, and
    silently produces densities wrong by orders of magnitude. The implied
    scene extent is what gives it away.
    """
    reversed_ = GroundPlane.from_matrix(ground.H_inv)   # deliberately not flipped
    info = reversed_.describe_scene((1280, 720))
    assert info["ok"] is False
    assert "implausible" in info["reason"]


def test_bad_direction_is_rejected(ground):
    with pytest.raises(ValueError, match="direction must be"):
        GroundPlane.from_matrix(ground.H, direction="sideways")


def test_non_3x3_matrix_is_rejected():
    with pytest.raises(ValueError, match="3x3"):
        GroundPlane.from_matrix(np.eye(4))


def test_singular_matrix_is_rejected():
    with pytest.raises(ValueError, match="singular"):
        GroundPlane(H=np.zeros((3, 3)))


def test_non_finite_matrix_is_rejected():
    H = np.eye(3)
    H[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        GroundPlane(H=H)


def test_empty_construction_is_rejected():
    with pytest.raises(ValueError, match="either correspondences or a homography"):
        GroundPlane()


def test_borrowed_calibration_roundtrips_through_a_dict(ground):
    borrowed = GroundPlane.from_matrix(ground.H)
    back = GroundPlane.from_dict(borrowed.to_dict())
    assert np.allclose(back.to_world(IMAGE_POINTS), borrowed.to_world(IMAGE_POINTS))
    assert back.reprojection_error()["fitted"] is False


def test_borrowed_calibration_still_computes_area(ground):
    """Area is what density divides by, so it must work without correspondences."""
    borrowed = GroundPlane.from_matrix(ground.H)
    assert borrowed.polygon_area_m2(IMAGE_POINTS) == pytest.approx(6.0, abs=1e-9)
