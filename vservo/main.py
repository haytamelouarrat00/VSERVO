import numpy as np

from vservo.camera.camera import CameraModel
from vservo.controller.jacobian import compute_image_jacobian
from vservo.controller.velocity import compute_control_velocity
from vservo.utils.constants import LAMBDA_GAIN

from vservo.utils.viz import plot_velocity_history

PTS_WORLD = np.array(
    [
        [-0.1, -0.1, 0.5],  # Top-left
        [0.1, -0.1, 0.5],  # Top-right
        [0.1, 0.1, 0.5],  # Bottom-right
        [-0.1, 0.1, 0.5],  # Bottom-left
    ],
    dtype=np.float64,
)

MAX_ITERATIONS = 300
ERROR_THRESHOLD = 5e-3 # pixels
PROGRESS_PRINT_INTERVAL = 10


def _format_pose(cam: CameraModel) -> str:
    return (
        f"Position=({cam.X:.5f}, {cam.Y:.5f}, {cam.Z:.5f}) | "
        f"Rotation=({cam.rX:.5f}, {cam.rY:.5f}, {cam.rZ:.5f})"
    )


def run_ibvs_simulation():
    cam = CameraModel()

    # Desired configuration at identity pose
    cam.set_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    desired_pixels, _ = cam.project_points(PTS_WORLD)
    print("Desired configuration (identity pose):")
    print(desired_pixels)
    print()

    # Sample a valid starting pose
    random_pose = cam.generate_random_pose(
        PTS_WORLD, position_range=(0.3, 0.3, 0.8), rotation_range=0.3
    )
    if random_pose is None:
        raise RuntimeError("Failed to generate valid random pose.")

    if isinstance(random_pose, dict):
        cam.set_pose(**random_pose)
    else:
        cam.set_pose(*random_pose)
    print("Initial (random) pose:")
    print(f"  {_format_pose(cam)}")

    initial_pixels, _ = cam.project_points(PTS_WORLD)
    initial_error = np.linalg.norm(initial_pixels.flatten() - desired_pixels.flatten())
    print(f"Initial error: {initial_error:.5f} pixels")
    print("=" * 60)

    # Storage
    error_history: list[float] = []
    position_history: list[np.ndarray] = []
    feature_history: list[np.ndarray] = []
    velocity_history: list[np.ndarray] = []
    rotation_history: list[np.ndarray] = []

    converged = False
    iteration = 0
    desired_features = desired_pixels.reshape(-1, 1)

    # Main IBVS loop
    while not converged and iteration < MAX_ITERATIONS:
        current_pixels, points_camera = cam.project_points(PTS_WORLD)
        feature_history.append(current_pixels.copy())
        position_history.append(cam.get_position().copy())
        rotation_history.append(cam.get_euler_angles().copy())

        jacobian = compute_image_jacobian(cam, points_camera)
        current_features = current_pixels.reshape(-1, 1)

        velocity, error_norm = compute_control_velocity(
            current_features, desired_features, jacobian, gain=LAMBDA_GAIN
        )

        velocity_history.append(velocity.flatten().copy())
        error_history.append(float(error_norm))

        if iteration % PROGRESS_PRINT_INTERVAL == 0:
            print(
                f"Iteration {iteration:3d}: Error = {error_norm:6.5f} px | "
                f"{_format_pose(cam)}"
            )

        if error_norm < ERROR_THRESHOLD:
            converged = True
            print(f"\n✓ Converged at iteration {iteration}!")
            print(f"  Final error: {error_norm:.3f} pixels")
            break

        cam.update_pose(velocity)
        iteration += 1

    final_pixels, _ = cam.project_points(PTS_WORLD)
    final_error = error_history[-1] if error_history else initial_error

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if converged:
        print("✓ SUCCESS: Visual servoing converged!")
    else:
        print(f"✗ Max iterations reached. Final error: {final_error:.5f} px")

    print(f"\nTotal iterations: {iteration}")
    print(f"Error reduction: {initial_error:.5f} → {final_error:.5f} pixels")
    print("\nFinal camera pose:")
    print(f"  {_format_pose(cam)}")
    print("\nFinal pixel positions:")
    print(final_pixels)
    print("\nDesired pixel positions:")
    print(desired_pixels)
    print("\nPixel errors (final - desired):")
    print(final_pixels - desired_pixels)
    print(f"\nMax pixel error: {np.max(np.abs(final_pixels - desired_pixels)):.5f} pixels")

    return {
        "error_history": np.array(error_history),
        "position_history": np.array(position_history),
        "feature_history": feature_history,
        "velocity_history": np.array(velocity_history),
        "rotation_history": np.array(rotation_history),
        "desired_pixels": desired_pixels,
        "points_world": PTS_WORLD,
        "converged": converged,
        "iterations": iteration,
    }


def main():
    results = run_ibvs_simulation()
    return results


# python
if __name__ == "__main__":
    results = run_ibvs_simulation()
    err_history = results["error_history"]
    position_history = results["position_history"]
    feature_history = results["feature_history"]
    velocity_history = results["velocity_history"]
    desired_pixels = results["desired_pixels"]
    points_world = results["points_world"]
    plot_velocity_history(velocity_history)
