import numpy as np
from numpy.typing import NDArray
from vservo.utils.constants import Fx, Fy, Cu, Cv
from vservo.utils.constants import DT, EPSILON
from vservo.controller.velocity import skew

class CameraModel:
    def __init__(self) -> None:
        # Camera intrinsic parameters: how the camera sees
        self.focal_x: float = Fx
        self.focal_y: float = Fy
        self.centroid_x: float = Cu
        self.centroid_y: float = Cv

        # Camera extrinsic parameters: where the camera is in 3D space
        self.X: float = 0.0
        self.Y: float = 0.0
        self.Z: float = 0.0
        self.rX: float = 0.0
        self.rY: float = 0.0
        self.rZ: float = 0.0

        self.R: NDArray[np.float64] = np.zeros((3, 3))
        self.T: NDArray[np.float64] = np.vstack((self.X, self.Y, self.Z))

    def set_pose(
        self, X: float, Y: float, Z: float, rX: float, rY: float, rZ: float
    ) -> None:
        """Update camera pose parameters."""
        self.X = X
        self.Y = Y
        self.Z = Z
        self.rX = rX
        self.rY = rY
        self.rZ = rZ

    def get_rotation(self):
        return np.array([self.rX, self.rY, self.rZ])

    def homogenous_transformation_matrix(self) -> NDArray[np.float64]:
        """Compute 4x4 homogeneous transformation matrix (world → camera)."""
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.rX), -np.sin(self.rX)],
                [0, np.sin(self.rX), np.cos(self.rX)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(self.rY), 0, np.sin(self.rY)],
                [0, 1, 0],
                [-np.sin(self.rY), 0, np.cos(self.rY)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(self.rZ), -np.sin(self.rZ), 0],
                [np.sin(self.rZ), np.cos(self.rZ), 0],
                [0, 0, 1],
            ]
        )

        self.R = Rz @ Ry @ Rx
        self.T = np.array([[self.X], [self.Y], [self.Z]])

        camera_pose = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1]))
        assert camera_pose.shape == (4, 4)
        return camera_pose

    def get_rotation_matrix(self) -> NDArray[np.float64]:
        """
        Current 3x3 rotation matrix from camera frame to world frame.

        Returns:
            3x3 rotation matrix
        """
        # Recompute to ensure the matrix reflects the latest pose parameters.
        self.homogenous_transformation_matrix()
        return self.R.copy()

    def get_position(self) -> NDArray[np.float64]:
        """Return current camera position in world coordinates."""
        return np.array([self.X, self.Y, self.Z], dtype=np.float64)

    def get_euler_angles(self) -> NDArray[np.float64]:
        """Return current roll, pitch, yaw (rX, rY, rZ)."""
        return np.array([self.rX, self.rY, self.rZ], dtype=np.float64)

    def world_2_camera(self, T_camera: NDArray[np.float64], pts_3d_world: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform 3D points from world coordinates to camera coordinates.

        Args:
            T_camera: 4x4 pose of the camera (camera → world)
            pts_3d_world: (..., 3) array of 3D points in world frame

        Returns:
            (..., 3) array of 3D points expressed in camera frame
        """
        T_camera = np.asarray(T_camera, dtype=np.float64)
        if T_camera.shape != (4, 4):
            raise ValueError("Camera pose matrix must be 4x4.")

        points = np.asarray(pts_3d_world, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("Input points must have shape (..., 3).")

        was_1d = points.ndim == 1
        points_reshaped = points.reshape(-1, 3)

        # Homogeneous coordinates
        ones = np.ones((points_reshaped.shape[0], 1), dtype=np.float64)
        points_h = np.hstack((points_reshaped, ones))

        # Invert pose (camera → world) to get world → camera transform
        T_world_to_camera = np.linalg.inv(T_camera)

        points_camera_h = (T_world_to_camera @ points_h.T).T
        points_camera = points_camera_h[:, :3]

        if was_1d:
            return points_camera[0]
        return points_camera

    def project(self, pts_3d_camera: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Project 3D points (in camera frame) to 2D image coordinates.

        Args:
            pts_3d_camera: (..., 3) array of 3D points in camera frame

        Returns:
            (..., 2) array of 2D pixel coordinates [u, v]
        """
        points = np.asarray(pts_3d_camera, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("Input points must have shape (..., 3).")

        was_1d = points.ndim == 1
        points_reshaped = points.reshape(-1, 3)

        X = points_reshaped[:, 0]
        Y = points_reshaped[:, 1]
        Z = points_reshaped[:, 2]

        if np.any(np.isclose(Z, 0.0)):
            raise ValueError("Points must have non-zero Z value in camera frame.")

        u = self.focal_x * (X / Z) + self.centroid_x
        v = self.focal_y * (Y / Z) + self.centroid_y

        pixels = np.column_stack((u, v))
        if was_1d:
            return pixels[0]
        return pixels

    def backproject(self, pts2d: np.ndarray, depth=1.0) -> np.ndarray:
        u, v = pts2d.T
        X = (u - self.centroid_x) * depth / self.focal_x
        Y = (v - self.centroid_y) * depth / self.focal_y
        Z = np.full_like(X, depth)
        return np.column_stack((X, Y, Z))

    def project_points(self, pts_3d_world: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Complete projection pipeline: world 3D → camera 3D → image 2D.

        Args:
            pts_3d_world: Nx3 array of 3D points in world coordinates

        Returns:
            tuple of:
                - pts_2d_image: Nx2 array of pixel coordinates
                - pts_3d_camera: Nx3 array of points in camera frame (needed for Jacobian!)
        """
        T_camera = self.homogenous_transformation_matrix()
        pts_3d_camera = self.world_2_camera(T_camera, pts_3d_world)
        pts_2d_image = self.project(pts_3d_camera)
        return pts_2d_image, pts_3d_camera

    def get_pose(self):
        return np.array([
            self.X, self.Y, self.Z, self.rX, self.rY, self.rZ
        ])

    def generate_random_pose(self, target_points, position_range, rotation_range,
                             margin=50, max_attempts=100):

        x_range, y_range, z_range = position_range

        # Compute center of target points (to stay roughly aimed at them)
        target_center = np.mean(target_points, axis=0)

        for attempt in range(max_attempts):
            # --- Generate random position ---
            # Sample around the target center
            X = target_center[0] + np.random.uniform(-x_range, x_range)
            Y = target_center[1] + np.random.uniform(-y_range, y_range)

            # Z should be negative (camera behind points, looking toward +Z)
            # Sample a distance, then make it negative
            Z = target_center[2] - np.random.uniform(0.5, z_range)

            # --- Generate random rotation ---
            # Small random rotations around each axis
            rX = np.random.uniform(-rotation_range, rotation_range)
            rY = np.random.uniform(-rotation_range, rotation_range)
            rZ = np.random.uniform(-rotation_range, rotation_range)

            # --- Check if this pose is valid ---
            if self.is_pose_valid(X, Y, Z, rX, rY, rZ, target_points,
                             self.focal_x, self.centroid_y*2, self.centroid_x*2, margin):
                return X, Y, Z, rX, rY, rZ

        # Failed to find valid pose
        print(f"WARNING: Could not find valid pose after {max_attempts} attempts")
        return None

    def is_pose_valid(self, X, Y, Z, rX, rY, rZ, target_points,
                      focal_length, image_width, image_height, margin):

        # --- Build transformation matrix for this pose ---
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rX), -np.sin(rX)],
            [0, np.sin(rX), np.cos(rX)]
        ])
        Ry = np.array([
            [np.cos(rY), 0, np.sin(rY)],
            [0, 1, 0],
            [-np.sin(rY), 0, np.cos(rY)]
        ])
        Rz = np.array([
            [np.cos(rZ), -np.sin(rZ), 0],
            [np.sin(rZ), np.cos(rZ), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx
        T_translation = np.array([[X], [Y], [Z]])

        # Build 4x4 homogeneous transformation
        T_cam = np.eye(4)
        T_cam[:3, :3] = R
        T_cam[:3, 3:4] = T_translation

        # Inverse to transform world → camera
        T_world_to_cam = np.linalg.inv(T_cam)

        # --- Transform points to camera frame ---
        n_points = len(target_points)
        points_homogeneous = np.hstack([target_points, np.ones((n_points, 1))])
        points_camera_h = (T_world_to_cam @ points_homogeneous.T).T
        points_camera = points_camera_h[:, :3]

        # --- Check 1: All points in front of camera ---
        Z_values = points_camera[:, 2]
        if np.any(Z_values <= 0.1):  # At least 10cm in front
            return False

        # --- Check 2: Project to image and check bounds ---
        X_cam = points_camera[:, 0]
        Y_cam = points_camera[:, 1]
        Z_cam = points_camera[:, 2]

        # Assume square pixels (focal_x = focal_y = focal_length)
        cu = image_width / 2.0
        cv = image_height / 2.0

        u = focal_length * (X_cam / Z_cam) + cu
        v = focal_length * (Y_cam / Z_cam) + cv

        # Check all pixels within bounds (with margin)
        if np.any(u < margin) or np.any(u > image_width - margin):
            return False
        if np.any(v < margin) or np.any(v > image_height - margin):
            return False

        # All checks passed!
        return True

    def check_features_visible(self, pixels, points_camera):
        """
        Check if all features are visible in image.

        Returns:
            (bool, str): (all_visible, error_message)
        """

        image_width = 2 * self.centroid_y
        image_height = 2 * self.centroid_x
        # Check behind camera
        if np.any(points_camera[:, 2] <= 0):
            return False, "Point(s) behind camera (Z <= 0)"

        # Check image bounds
        if np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= image_width):
            return False, "Point(s) outside horizontal FOV"

        if np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= image_height):
            return False, "Point(s) outside vertical FOV"

        return True, ""

    def rotation_matrix_to_euler(self, R: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def update_pose(
            self, velocity: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Integrate camera pose given camera-frame twist.

        Args:
            camera_model: Camera to update.
            velocity: 6-vector [vx, vy, vz, wx, wy, wz] in camera frame.

        Returns:
            Tuple of (new_position, new_euler_angles).
        """
        twist = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if twist.size != 6:
            raise ValueError("Velocity must contain 6 elements [vx, vy, vz, wx, wy, wz].")

        R_wc = self.get_rotation_matrix()
        v_linear_cam = twist[:3]
        omega_cam = twist[3:]

        # Linear velocity expressed in world frame.
        v_linear_world = R_wc @ v_linear_cam
        new_position = self.get_position() + v_linear_world * DT

        omega_norm = np.linalg.norm(omega_cam)
        if omega_norm < EPSILON:
            delta_R = np.eye(3) + skew(omega_cam) * DT
        else:
            axis = omega_cam / omega_norm
            theta = omega_norm * DT
            K = skew(axis)
            delta_R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

        R_new = R_wc @ delta_R
        angles = self.rotation_matrix_to_euler(R_new)
        self.set_pose(
            float(new_position[0]),
            float(new_position[1]),
            float(new_position[2]),
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
        )
        return new_position, angles
