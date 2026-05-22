"""Filter an SO(3)^K time series with masks and confidence weights."""

from __future__ import annotations

import numpy as np
from pyrecest.backend import array, cos, pi, sin, stack, to_numpy
from pyrecest.filters import run_so3_product_sequence_filter


def z_quaternion(angle):
    """Return a scalar-last quaternion for a yaw rotation."""
    return array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])


def x_quaternion(angle):
    """Return a scalar-last quaternion for a roll rotation."""
    return array([sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)])


def identity_transition(particles, time_index, rng):
    """Simple dynamics callback used by the example.

    Real applications can replace this with learned, kinematic, or inertial
    dynamics. The callback receives all particles shaped (N, K, 4), the time
    index being predicted, and a NumPy random generator.
    """
    del time_index, rng
    return particles


def main() -> None:
    n_steps = 12
    measurements = stack(
        [
            stack(
                [
                    z_quaternion(0.04 * time_index),
                    x_quaternion(0.02 * time_index + 0.1 * np.sin(time_index / 3.0)),
                    z_quaternion(pi / 8.0),
                ],
                axis=0,
            )
            for time_index in range(n_steps)
        ],
        axis=0,
    )

    mask = np.ones((n_steps, 3), dtype=float)
    mask[3:6, 1] = 0.0
    mask[8, 0] = 0.0

    confidence = np.ones((n_steps, 3), dtype=float)
    confidence[:, 2] = 0.6
    confidence[6:, 1] = 0.8
    confidence *= mask

    result = run_so3_product_sequence_filter(
        measurements,
        mask,
        transition_callback=identity_transition,
        noise_std=0.15,
        num_particles=200,
        resample_threshold=0.5,
        partition="singleton",
        confidence=confidence,
        proposal_gain=0.1,
        rng=np.random.default_rng(0),
    )

    print("estimate shape:", result.estimates.shape)
    print("mean ESS:", float(np.mean(to_numpy(result.effective_sample_size))))
    print("resampled frames:", np.flatnonzero(to_numpy(result.resampled)).tolist())


if __name__ == "__main__":
    main()
