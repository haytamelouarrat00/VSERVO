import numpy as np
from numpy.typing import NDArray

from vservo.utils.constants import LAMBDA_GAIN, CONVERGENCE_THRESHOLD, DT, EPSILON


def compute_control_velocity(
    current_features: NDArray[np.float64],
    desired_features: NDArray[np.float64],
    jacobian: NDArray[np.float64],
    gain: float = LAMBDA_GAIN,
) -> tuple[NDArray[np.float64], float]:
    """
    Compute camera velocity using IBVS control law.

    Args:
        current_features: 2N×1 vector of current pixel positions [u1, v1, u2, v2, ...]
        desired_features: 2N×1 vector of desired pixel positions [u1*, v1*, u2*, v2*, ...]
        jacobian: 2N×6 image Jacobian matrix
        gain: Control gain λ (typically 0.1 to 1.0)

    Returns:
        6×1 velocity vector [vx, vy, vz, ωx, ωy, ωz] in camera frame
    """
    assert current_features.shape == desired_features.shape
    s = current_features.reshape(-1, 1)  # 4x2 → 8x1
    s_star = desired_features.reshape(-1, 1)  # 4x2 → 8x1
    err = s - s_star
    err_norm = np.linalg.norm(err)
    if err_norm < CONVERGENCE_THRESHOLD:
        return np.zeros((6, 1)), err_norm
    L_plus = np.linalg.pinv(jacobian)
    v = -gain * L_plus @ err
    return v, err_norm


def skew(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Create skew-symmetric matrix from 3-vector."""
    ax, ay, az = a
    return np.array(
        [
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_euler(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Rz*Ry*Rx rotation matrix to Euler angles (rX, rY, rZ)."""
    if abs(R[2, 0]) < 1.0 - EPSILON:
        rY = np.arcsin(-R[2, 0])
        rX = np.arctan2(R[2, 1], R[2, 2])
        rZ = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch at +-90 degrees
        rY = np.pi / 2 * np.sign(-R[2, 0])
        rX = np.arctan2(-R[0, 1], R[1, 1])
        rZ = 0.0
    return np.array([rX, rY, rZ], dtype=np.float64)


