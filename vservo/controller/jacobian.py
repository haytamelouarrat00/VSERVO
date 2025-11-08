import numpy as np
from numpy.typing import NDArray

# Standard interaction matrix for point features
from vservo.camera.camera import CameraModel


def compute_image_jacobian(
    camera_model: CameraModel, pts_camera: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the image Jacobian (interaction matrix) for multiple points.
    **Columns 1-3: Translational velocities**
        - Column 1 (vx): How moving camera right affects pixels
        - Column 2 (vy): How moving camera down affects pixels
        - Column 3 (vz): How moving camera forward affects pixels

        **Columns 4-6: Rotational velocities**
        - Column 4 (ωx): How rotating around X-axis (pitch) affects pixels
        - Column 5 (ωy): How rotating around Y-axis (yaw) affects pixels
        - Column 6 (ωz): How rotating around Z-axis (roll) affects pixels

    Args:
        pts_camera: Nx3 array of 3D points in camera frame [X, Y, Z]

    Returns:
        (2N)×6 Jacobian matrix relating camera velocity to image feature velocity
    """
    pts = np.asarray(pts_camera, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)

    if pts.shape[1] != 3:
        raise ValueError("pts_camera must have shape (..., 3).")

    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    if np.any(np.isclose(Z, 0.0)):
        raise ValueError("Point depth Z must be non-zero.")

    fx = float(camera_model.focal_x)
    fy = float(camera_model.focal_y)

    Z_inv = 1.0 / Z
    Z_inv_sq = Z_inv**2
    X_over_Z = X * Z_inv
    Y_over_Z = Y * Z_inv

    n_points = pts.shape[0]
    L = np.zeros((2 * n_points, 6), dtype=np.float64)

    # Rows associated with u (even indices)
    L[0::2, 0] = -fx * Z_inv
    L[0::2, 1] = 0.0
    L[0::2, 2] = fx * X_over_Z * Z_inv
    L[0::2, 3] = fx * X_over_Z * Y_over_Z
    L[0::2, 4] = -fx * (1.0 + X_over_Z**2)
    L[0::2, 5] = fx * Y_over_Z

    # Rows associated with v (odd indices)
    L[1::2, 0] = 0.0
    L[1::2, 1] = -fy * Z_inv
    L[1::2, 2] = fy * Y_over_Z * Z_inv
    L[1::2, 3] = fy * (1.0 + Y_over_Z**2)
    L[1::2, 4] = -fy * X_over_Z * Y_over_Z
    L[1::2, 5] = -fy * X_over_Z

    return L
