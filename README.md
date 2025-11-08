# VSERVO_1: Visual Servoing Playground

> Minimal tools for simulating Image-Based Visual Servoing (IBVS), analyzing convergence, and visualizing how a pinhole camera “moves” to align observed features with their desired pixel positions.

## TL;DR

- **Simulation first:** `run_ibvs_simulation()` (see `vservo/main.py`) sets up a pinhole camera, generates a valid random pose, and runs a classic IBVS loop with an image Jacobian controller.
- **Live visualization:** `python -m vservo.minimal_viewer` animates the 3D camera pose, per-feature pixel motion, error decay, and velocity commands. Optionally export the animation to a GIF (see below).
- **Composable code:** camera, Jacobian, velocity control, visualization, and utility modules stay decoupled so you can reuse them in notebooks or other robotics stacks.

## Features

- Deterministic camera model (`vservo/camera/camera.py`) with projection/back-projection, pose integration, and point visibility checks.
- Image Jacobian calculations for IBVS (`vservo/controller/jacobian.py`) and control law utilities (`vservo/controller/velocity.py`).
- Random but valid camera pose generation + convergence loop that tracks error, pose, velocities, and per-point pixel history.
- Minimal viewer (`vservo/minimal_viewer.py`) for side-by-side live visualization of pose, image-plane features, error curve, and velocity magnitudes.
- Ready-to-plot helpers (`vservo/utils/viz.py`) for comprehensive or simple error plots if you want static reporting instead of live animation.

## What We Built (and How)

1. **Core IBVS loop (`run_ibvs_simulation`)** – Initializes a pinhole camera, samples a visibility-safe random pose, and iteratively:
   - Projects the known 3D target points.
   - Builds the image Jacobian from camera-frame coordinates.
   - Applies the standard IBVS velocity command (`compute_control_velocity`).
   - Integrates pose updates directly in the camera model (`CameraModel.update_pose`).
   - Logs error, pose, velocity, and pixel history for downstream tools.
2. **Pose-aware camera utilities** – Added Jacobian-friendly projection/back-projection helpers, skew-matrix utilities, and random-pose validation that keeps all features in view.
3. **Visualization stack** – Two complementary paths:
   - Static plots (`utils/viz.py`) for reports/CI.
   - A “minimal viewer” animation that shows 3D pose evolution, per-feature pixel motion (with optional traces), real-time error decay, and both linear/angular velocity magnitudes. The viewer can also export GIFs, one frame per IBVS iteration.
4. **Project scaffolding** – A publishable README, placeholders for hero/GIF assets, and a modular package layout so components can be reused in notebooks or robotics codebases.

## Example Results

Sample console output (random pose will differ):

```
Desired configuration (identity pose):
[[ 80. 160.]
 [400. 160.]
 [400. 480.]
 [ 80. 480.]]

Initial (random) pose:
  Position=(0.16884, 0.06207, -0.12103) | Rotation=(0.16887, -0.25306, -0.13282)
Initial error: 137.19327 pixels
============================================================
Iteration   0: Error = 137.19327 px | Position=(0.16884, 0.06207, -0.12103) | Rotation=(0.16887, -0.25306, -0.13282)
Iteration  10: Error = 82.47206 px | Position=(0.09043, 0.02830, -0.07374) | Rotation=(0.08217, -0.15399, -0.06590)
Iteration  50: Error = 10.63277 px | Position=(0.00876, 0.00170, -0.00946) | Rotation=(0.00653, -0.01736, -0.00685)
Iteration 100: Error = 0.81814 px | Position=(0.00064, 0.00012, -0.00072) | Rotation=(0.00047, -0.00129, -0.00052)
Iteration 150: Error = 0.06295 px | Position=(0.00005, 0.00001, -0.00006) | Rotation=(0.00004, -0.00010, -0.00004)
Iteration 200: Error = 0.00484 px | Position=(0.00000, 0.00000, -0.00000) | Rotation=(0.00000, -0.00001, -0.00000)

✓ Converged at iteration 200!
  Final error: 0.005 pixels

============================================================
FINAL RESULTS
============================================================
✓ SUCCESS: Visual servoing converged!

Total iterations: 200
Error reduction: 137.19327 → 0.00484 pixels

Final camera pose:
  Position=(0.00000, 0.00000, -0.00000) | Rotation=(0.00000, -0.00001, -0.00000)

Final pixel positions:
[[ 80.00226461 160.00233877]
 [399.9993484  160.00282387]
 [399.99855002 480.00057318]
 [ 80.001114   479.99910929]]

Desired pixel positions:
[[ 80. 160.]
 [400. 160.]
 [400. 480.]
 [ 80. 480.]]

Pixel errors (final - desired):
[[ 0.00226461  0.00233877]
 [-0.0006516   0.00282387]
 [-0.00144998  0.00057318]
 [ 0.001114   -0.00089071]]

Max pixel error: 0.00282 pixels
```

The minimal viewer turns that run into an animation showing how the camera translates/rotates to re-center the four feature points. Drop the actual GIF into `docs/media/ibvs-placeholder.gif` once generated.

## Getting Started

```bash
git clone https://github.com/your-org/vservo.git
cd vservo
python -m vservo.main
```

This prints a textual summary of the IBVS run (initial pose, error evolution, final pose, pixel errors). To watch the simulation evolve:

```bash
python -m vservo.minimal_viewer --show_feature_traces
```

> **Note:** CLI flags are minimal; tweak them by editing `vservo/minimal_viewer.py` or importing `run_minimal_viewer()` from Python. Example:

```python
from vservo.minimal_viewer import run_minimal_viewer

results = run_minimal_viewer(
    interval_ms=80,
    save_path="docs/media/ibvs_run.gif",  # omit to skip saving
    show_feature_traces=True,
    show=True,
)
```

## Project Layout

```
vservo/
├── camera/                # Camera model and tests
├── controller/            # Jacobian + velocity control utilities
├── utils/                 # Viz helpers, constants, misc utilities
├── main.py                # Entry point for running a simulation
└── minimal_viewer.py      # Live animator / GIF exporter
```

Add your own targets, controller tweaks, or plotting backends inside these packages. `run_ibvs_simulation()` returns a dictionary with keys like `error_history`, `position_history`, `velocity_history`, etc., so it’s easy to extend.

## Saving Visualizations

The viewer can export a GIF where each frame corresponds to one IBVS iteration:

```python
run_minimal_viewer(save_path="docs/media/ibvs.gif", show=False)
```

![GIF Placeholder](/media/ibvs.gif)

Prefer static plots? Use `vservo/utils/viz.py` functions, e.g.:

```python
from vservo.utils.viz import plot_error_convergence

results = run_ibvs_simulation()
plot_error_convergence(results["error_history"], results["converged"], save_path="docs/media/error.png")
```

## Roadmap

- [ ] Plug in custom target trajectories / camera intrinsics
- [ ] Add CLI flags for viewer options (interval, traces, export path, etc.)
- [ ] Extend to 6-DOF robot arm or quadrotor kinematics
- [ ] Benchmark controllers (adaptive gains, H∞, etc.)

## Contributing

1. Fork the repo & create a branch (`git checkout -b feature/your-idea`)
2. Make your changes and add tests where possible
3. Run linters/tests (coming soon!)
4. Submit a PR with a short description and screenshots/GIFs if relevant

Issues and discussions are welcome—IBVS edge cases, dataset support, or alternative controllers are all fair game.

---

Made with ☕ and NumPy. Drop a ⭐ if this helped you debug IBVS!
