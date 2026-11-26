"""Track a moving and rotating 3-D ellipsoid with a spatial MEM filter.

Run from an installed development checkout with
``python examples/basic/mem_ekf_star_3d_tracking.py``. The example uses NumPy
and generates uniformly distributed measurements inside an ellipsoid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pyrecest.filters import MEMEKFStar3DTracker, MEMQKF3DTracker, MEMRBPF3DTracker

TRACKERS = {
    "ekf-star": MEMEKFStar3DTracker,
    "qkf": MEMQKF3DTracker,
    "rbpf": MEMRBPF3DTracker,
}


def sample_unit_ball(rng: np.random.Generator, count: int) -> np.ndarray:
    """Return columns uniformly distributed in the 3-D unit ball."""
    directions = rng.normal(size=(3, count))
    directions /= np.linalg.norm(directions, axis=0)
    return directions * rng.uniform(size=count) ** (1.0 / 3.0)


def orientation_error_degrees(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Compare labelled ellipsoid axes, allowing equivalent axis sign flips."""
    sign_choices = ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])
    angles = [
        Rotation.from_matrix(reference.T @ estimated @ np.diag(signs)).magnitude()
        for signs in sign_choices
    ]
    return float(np.rad2deg(min(angles)))


def run_example(
    seed: int = 7, n_steps: int = 60, n_measurements: int = 50, tracker_name="ekf-star"
):
    """Return the final tracker and simulation histories for a seeded scenario."""
    if n_steps <= 0 or n_measurements <= 0:
        raise ValueError("n_steps and n_measurements must be positive")
    rng = np.random.default_rng(seed)
    dt = 0.2
    transition = np.eye(6)
    transition[:3, 3:] = dt * np.eye(3)
    true_state = np.array([0.0, 0.0, 0.0, 0.6, -0.25, 0.15])
    true_axes = np.array([4.0, 2.5, 1.4])
    true_orientation = Rotation.from_euler(
        "xyz", [28.0, -33.0, 41.0], degrees=True
    ).as_matrix()
    rotation_increment = np.array([0.012, -0.009, 0.014])
    increment_matrix = Rotation.from_rotvec(rotation_increment).as_matrix()
    measurement_noise_cov = 0.08**2 * np.eye(3)

    # Keep particle sampling independent of the common simulated observations.
    particle_options = {"rng": seed} if tracker_name == "rbpf" else {}
    tracker = TRACKERS[tracker_name](
        kinematic_state=true_state + np.array([0.4, -0.3, 0.25, 0.04, -0.03, 0.02]),
        covariance=np.diag([0.5**2] * 3 + [0.15**2] * 3),
        orientation=true_orientation
        @ Rotation.from_rotvec([0.1, -0.08, 0.09]).as_matrix(),
        semi_axes=np.array([3.5, 2.15, 1.25]),
        shape_covariance=np.diag([0.15**2] * 3 + [0.6**2, 0.4**2, 0.25**2]),
        **particle_options,
    )
    true_extent = true_orientation @ np.diag(true_axes**2) @ true_orientation.T
    initial_relative_extent_error = np.linalg.norm(
        tracker.extent - true_extent
    ) / np.linalg.norm(true_extent)
    initial_orientation_error = orientation_error_degrees(
        tracker.orientation, true_orientation
    )
    history = {
        "time": [0.0],
        "center_error": [np.linalg.norm(tracker.kinematic_state[:3] - true_state[:3])],
        "orientation_error_degrees": [initial_orientation_error],
        "relative_extent_error": [initial_relative_extent_error],
        "estimated_axes": [np.array(tracker.semi_axes)],
        "true_axes": true_axes,
        "initial_relative_extent_error": float(initial_relative_extent_error),
        "initial_orientation_error_degrees": initial_orientation_error,
    }

    for step in range(n_steps):
        true_state = transition @ true_state
        true_orientation = true_orientation @ increment_matrix
        tracker.predict_linear(
            transition,
            sys_noise=np.diag([1e-5] * 3 + [1e-6] * 3),
            shape_sys_noise=np.diag([1e-6] * 3 + [1e-7] * 3),
            rotation_increment=rotation_increment,
        )
        measurements = (
            true_state[:3, None]
            + true_orientation
            @ np.diag(true_axes)
            @ sample_unit_ball(rng, n_measurements)
            + rng.multivariate_normal(
                np.zeros(3), measurement_noise_cov, size=n_measurements
            ).T
        )
        # A 3 x N array contains one point in each column. The default C_h=I/5
        # matches sample_unit_ball; additive sensor covariance stays separate.
        tracker.update(measurements, meas_noise_cov=measurement_noise_cov)
        true_extent = true_orientation @ np.diag(true_axes**2) @ true_orientation.T
        history["time"].append((step + 1) * dt)
        history["center_error"].append(
            np.linalg.norm(tracker.kinematic_state[:3] - true_state[:3])
        )
        history["orientation_error_degrees"].append(
            orientation_error_degrees(tracker.orientation, true_orientation)
        )
        history["relative_extent_error"].append(
            np.linalg.norm(tracker.extent - true_extent) / np.linalg.norm(true_extent)
        )
        history["estimated_axes"].append(np.array(tracker.semi_axes))

    history["true_state"] = true_state
    history["true_orientation"] = true_orientation
    history["last_measurements"] = measurements
    return tracker, history


def save_plot(tracker, history, output: Path) -> None:
    """Save the final 3-D ellipsoid and estimation-error histories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 8), layout="constrained")
    ax = figure.add_subplot(2, 2, 1, projection="3d")
    measurements = history["last_measurements"]
    ax.scatter(*measurements, s=6, color="0.5", alpha=0.5, label="Last scan")
    azimuth, polar = np.meshgrid(
        np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 20)
    )
    sphere = np.array(
        [
            np.cos(azimuth) * np.sin(polar),
            np.sin(azimuth) * np.sin(polar),
            np.cos(polar),
        ]
    )
    for orientation, axes, center, color, label in (
        (
            history["true_orientation"],
            history["true_axes"],
            history["true_state"][:3],
            "tab:blue",
            "True",
        ),
        (
            tracker.orientation,
            tracker.semi_axes,
            tracker.kinematic_state[:3],
            "tab:orange",
            "Estimated",
        ),
    ):
        surface = (
            orientation @ np.diag(axes) @ sphere.reshape(3, -1) + center[:, None]
        ).reshape(sphere.shape)
        ax.plot_wireframe(
            *surface, rstride=2, cstride=2, color=color, alpha=0.65, label=label
        )
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Final ellipsoid")
    center = history["true_state"][:3]
    radius = 1.15 * max(np.max(history["true_axes"]), np.max(tracker.semi_axes))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize="small")

    for panel, key, title, ylabel in (
        (2, "center_error", "Center estimate", "Position error"),
        (3, "relative_extent_error", "Ellipsoid extent", "Relative Frobenius error"),
        (4, "orientation_error_degrees", "Orientation estimate", "Error (degrees)"),
    ):
        ax = figure.add_subplot(2, 2, panel)
        ax.plot(history["time"], history[key])
        ax.set(xlabel="Time", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tracker", choices=TRACKERS, default="ekf-star")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a saved figure, such as tracking-3d.png",
    )
    args = parser.parse_args()
    tracker, history = run_example(seed=args.seed, tracker_name=args.tracker)
    print("Scans: 60; measurements per scan: 50")
    print(f"Center error: {history['center_error'][-1]:.4f}")
    print(
        f"Velocity error: {np.linalg.norm(tracker.kinematic_state[3:] - history['true_state'][3:]):.4f}"
    )
    print("True semi-axes:", history["true_axes"])
    print("Estimated semi-axes:", np.round(tracker.semi_axes, 4))
    print(
        f"Orientation error: {history['initial_orientation_error_degrees']:.3f} -> {history['orientation_error_degrees'][-1]:.3f} degrees"
    )
    print(
        f"Relative extent error: {history['initial_relative_extent_error']:.3f} -> {history['relative_extent_error'][-1]:.3f}"
    )
    if args.output is not None:
        save_plot(tracker, history, args.output)
        print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
