import numpy as np

from vservo.camera.camera import CameraModel
from vservo.controller.jacobian import compute_image_jacobian
from vservo.controller.velocity import compute_control_velocity, update_pose
from vservo.utils.constants import CONVERGENCE_THRESHOLD, DT


def test_compute_control_velocity_zero_error():
    cam = CameraModel()
    pts_world = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    current_pixels, pts_camera = cam.project_points(pts_world)
    jacobian = compute_image_jacobian(cam, pts_camera)

    velocity, err_norm = compute_control_velocity(
        current_pixels.reshape(-1),
        current_pixels.reshape(-1),
        jacobian,
        gain=0.5,
    )

    np.testing.assert_allclose(err_norm, 0.0, atol=1e-12)
    np.testing.assert_allclose(velocity, np.zeros((6, 1)), atol=1e-12)


def test_compute_control_velocity_direction():
    cam = CameraModel()
    pts_world = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    current_pixels, pts_camera = cam.project_points(pts_world)
    jacobian = compute_image_jacobian(cam, pts_camera)

    desired_pixels = current_pixels.copy()
    desired_pixels[0, 0] += 10.0  # move target to the right

    velocity, err_norm = compute_control_velocity(
        current_pixels.reshape(-1),
        desired_pixels.reshape(-1),
        jacobian,
        gain=0.5,
    )

    assert err_norm > CONVERGENCE_THRESHOLD
    vx, _, _, _, wy, _ = velocity.flatten()

    assert vx < 0.0
    assert wy < 0.0


def test_update_pose_translation_only():
    cam = CameraModel()
    cam.set_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    velocity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_position, new_angles = update_pose(cam, velocity)

    expected_position = np.array([DT, 0.0, 0.0])
    np.testing.assert_allclose(new_position, expected_position, atol=1e-9)
    np.testing.assert_allclose(new_angles, np.zeros(3), atol=1e-9)


def test_update_pose_rotation_only():
    cam = CameraModel()
    cam.set_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    omega_z = np.pi  # rad/s
    velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, omega_z])
    _, new_angles = update_pose(cam, velocity)

    expected_yaw = omega_z * DT
    np.testing.assert_allclose(new_angles[2], expected_yaw, atol=1e-9)


def test_update_pose_respects_world_orientation():
    cam = CameraModel()
    cam.set_pose(0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2)  # 90 deg yaw

    velocity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    new_position, _ = update_pose(cam, velocity)

    expected_position = np.array([0.0, DT, 0.0])
    np.testing.assert_allclose(new_position, expected_position, atol=1e-9)
