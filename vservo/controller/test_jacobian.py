import numpy as np

from vservo.camera.camera import CameraModel
from vservo.controller.jacobian import compute_image_jacobian


def test_jacobian_shape():
    cam = CameraModel()
    points_camera = np.array(
        [
            [0.1, -0.2, 1.0],
            [0.3, 0.4, 2.0],
            [-0.5, 0.2, 1.5],
            [0.0, -0.1, 0.8],
        ],
        dtype=np.float64,
    )

    jacobian = compute_image_jacobian(cam, points_camera)
    assert jacobian.shape == (8, 6)


def test_jacobian_center_point_values():
    cam = CameraModel()
    point_on_axis = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    jacobian = compute_image_jacobian(cam, point_on_axis)

    expected_u_row = np.array([-cam.focal_x, 0.0, 0.0, 0.0, -cam.focal_x, 0.0])
    expected_v_row = np.array([0.0, -cam.focal_y, 0.0, cam.focal_y, 0.0, 0.0])

    np.testing.assert_allclose(jacobian[0], expected_u_row, atol=1e-9)
    np.testing.assert_allclose(jacobian[1], expected_v_row, atol=1e-9)


def test_jacobian_depth_scaling():
    cam = CameraModel()
    point_z1 = np.array([[0.1, -0.05, 1.0]], dtype=np.float64)
    point_z2 = np.array([[0.1, -0.05, 2.0]], dtype=np.float64)

    jacobian_z1 = compute_image_jacobian(cam, point_z1)
    jacobian_z2 = compute_image_jacobian(cam, point_z2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.abs(jacobian_z2) / np.abs(jacobian_z1)

    non_zero_mask = np.abs(jacobian_z1) > 1e-9

    # Entries dominated by depth terms (everything except yaw on u-row and omega_x on v-row)
    depth_sensitive_mask = np.ones_like(non_zero_mask, dtype=bool)
    depth_sensitive_mask[0::2, 4] = False  # yaw column on u-row
    depth_sensitive_mask[1::2, 3] = False  # omega_x column on v-row

    depth_mask = non_zero_mask & depth_sensitive_mask
    depth_ratios = ratios[depth_mask]

    assert depth_ratios.size > 0
    assert np.all(depth_ratios < 0.6)

    # Remaining entries should not grow with depth
    residual_mask = non_zero_mask & ~depth_sensitive_mask
    residual_ratios = ratios[residual_mask]
    assert residual_ratios.size > 0
    assert np.all(residual_ratios <= 1.0)
