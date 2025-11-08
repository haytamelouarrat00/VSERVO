import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from vservo.main import run_ibvs_simulation

COLORS = ["tab:red", "tab:green", "tab:blue", "tab:orange"]


def _world_axis_limits(points: np.ndarray, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    mins = points.min(axis=0) - margin
    maxs = points.max(axis=0) + margin
    return mins, maxs


def _image_axis_limits(features: list[np.ndarray], desired: np.ndarray, pad: float = 40.0):
    stack = np.vstack(features + [desired])
    mins = stack.min(axis=0) - pad
    maxs = stack.max(axis=0) + pad
    return mins, maxs


def run_minimal_viewer(
    results: dict | None = None,
    interval_ms: int = 120,
    save_path: str | None = None,
    show: bool = True,
    show_feature_traces: bool = False,
) -> dict:
    """
    Run the IBVS simulation and animate camera pose + image observations.

    Args:
        results: Optional precomputed simulation results.
        interval_ms: Delay (ms) between frames / iterations.
        save_path: Optional GIF path; if provided, each frame corresponds to an iteration.
        show: Whether to display the live Matplotlib window.
        show_feature_traces: If True, plot each pixel's trajectory up to the current frame.

    Returns:
        Simulation results dictionary.
    """
    if results is None:
        results = run_ibvs_simulation()

    positions = np.asarray(results["position_history"])
    rotations = np.asarray(results["rotation_history"])
    feature_history = [np.asarray(f) for f in results["feature_history"]]
    desired_pixels = np.asarray(results["desired_pixels"])
    world_points = np.asarray(results["points_world"])
    error_history = np.asarray(results["error_history"])
    velocity_history = np.asarray(results["velocity_history"])

    n_frames = len(feature_history)
    if n_frames == 0:
        raise RuntimeError("Simulation did not produce any iterations to visualize.")

    world_cloud = np.vstack((world_points, positions))
    world_min, world_max = _world_axis_limits(world_cloud, margin=0.1)
    image_min, image_max = _image_axis_limits(feature_history, desired_pixels, pad=40.0)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[0.9, 0.8], hspace=0.35, wspace=0.3)
    ax_pose = fig.add_subplot(gs[0, 0], projection="3d")
    ax_img = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[1, 1])
    info_text = fig.text(
        0.02,
        0.97,
        "",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, boxstyle="round"),
    )

    def _style_pose_axis():
        ax_pose.set_xlim(world_min[0], world_max[0])
        ax_pose.set_ylim(world_min[1], world_max[1])
        ax_pose.set_zlim(world_min[2], world_max[2])
        ax_pose.set_xlabel("X (m)")
        ax_pose.set_ylabel("Y (m)")
        ax_pose.set_zlabel("Z (m)")
        ax_pose.set_title("Camera Pose")

    def _style_image_axis():
        ax_img.set_xlim(image_min[0], image_max[0])
        ax_img.set_ylim(image_max[1], image_min[1])  # Image coordinates (v downwards)
        ax_img.set_xlabel("u (pixels)")
        ax_img.set_ylabel("v (pixels)")
        ax_img.set_title("Image Plane")
        ax_img.grid(True, alpha=0.2)

    def _style_error_axis():
        ax_error.set_xlabel("Iteration")
        ax_error.set_ylabel("Error (px)")
        ax_error.set_title("Error Convergence")
        ax_error.grid(True, alpha=0.3)

    def _style_velocity_axis():
        ax_vel.set_xlabel("Iteration")
        ax_vel.set_ylabel("Velocity")
        ax_vel.set_title("Velocity Commands")
        ax_vel.grid(True, alpha=0.3)

    def _draw_frame(frame: int):
        ax_pose.cla()
        ax_img.cla()
        ax_error.cla()
        ax_vel.cla()
        _style_pose_axis()
        _style_image_axis()
        _style_error_axis()
        _style_velocity_axis()

        traj = positions[: frame + 1]
        ax_pose.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", linewidth=2)
        ax_pose.scatter(
            traj[0, 0], traj[0, 1], traj[0, 2], color="red", s=60, label="Start", depthshade=False
        )
        ax_pose.scatter(
            traj[-1, 0],
            traj[-1, 1],
            traj[-1, 2],
            color="green",
            s=80,
            marker="*",
            label="Current",
            depthshade=False,
        )
        ax_pose.scatter(
            world_points[:, 0],
            world_points[:, 1],
            world_points[:, 2],
            color="orange",
            marker="^",
            s=40,
            alpha=0.6,
            label="Targets",
        )
        ax_pose.legend(loc="upper right", fontsize=8)

        current_pixels = feature_history[frame]
        for idx in range(current_pixels.shape[0]):
            if show_feature_traces and frame > 0:
                trace = np.array([fh[idx] for fh in feature_history[: frame + 1]])
                ax_img.plot(
                    trace[:, 0],
                    trace[:, 1],
                    color=COLORS[idx % len(COLORS)],
                    linewidth=1.2,
                    alpha=0.7,
                )
            ax_img.scatter(
                current_pixels[idx, 0],
                current_pixels[idx, 1],
                color=COLORS[idx % len(COLORS)],
                s=60,
                edgecolors="black",
            )
            ax_img.scatter(
                desired_pixels[idx, 0],
                desired_pixels[idx, 1],
                marker="x",
                color=COLORS[idx % len(COLORS)],
                s=80,
                linewidths=2,
            )

        err_y = error_history[: frame + 1]
        ax_error.plot(np.arange(1, frame + 2), err_y, color="tab:blue", linewidth=2)
        ax_error.axhline(y=2.0, color="tab:red", linestyle="--", linewidth=1, alpha=0.6)

        if velocity_history.size > 0:
            vel_seq = velocity_history[: frame + 1]
            iters = np.arange(1, vel_seq.shape[0] + 1)
            ax_vel.plot(iters, vel_seq[:, 0], color="tab:red", linewidth=1, label="vx")
            ax_vel.plot(iters, vel_seq[:, 1], color="tab:green", linewidth=1, label="vy")
            ax_vel.plot(iters, vel_seq[:, 2], color="tab:blue", linewidth=1, label="vz")
            ax_vel.plot(iters, vel_seq[:, 3], color="tab:pink", linewidth=1, linestyle="--", label="ωx")
            ax_vel.plot(iters, vel_seq[:, 4], color="tab:olive", linewidth=1, linestyle="--", label="ωy")
            ax_vel.plot(iters, vel_seq[:, 5], color="tab:purple", linewidth=1, linestyle="--", label="ωz")
            ax_vel.legend(fontsize=8, ncol=2, loc="upper right")

        info_text.set_text(
            "Iteration: {}/{}\n"
            "Position: [{:.3f}, {:.3f}, {:.3f}]\n"
            "Rotation: [{:.3f}, {:.3f}, {:.3f}]\n"
            "Error: {:.2f} px".format(
                frame + 1,
                n_frames,
                positions[frame, 0],
                positions[frame, 1],
                positions[frame, 2],
                rotations[frame, 0],
                rotations[frame, 1],
                rotations[frame, 2],
                error_history[frame],
            )
        )

    anim = FuncAnimation(fig, _draw_frame, frames=n_frames, interval=interval_ms, repeat=False)

    if save_path:
        fps = 1000.0 / interval_ms if interval_ms > 0 else 15.0
        print(f"Saving animation to '{save_path}' ({fps:.1f} fps)…")
        anim.save(save_path, writer="pillow", fps=fps)
        print("Saved animation.")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


if __name__ == "__main__":
    run_minimal_viewer(show_feature_traces=True, save_path="ibvs.gif")
