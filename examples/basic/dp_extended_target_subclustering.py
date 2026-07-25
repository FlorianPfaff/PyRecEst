"""Dirichlet-process subclustering for one extended-target measurement subset.

The RFS tracker would decide that this subset belongs to one extended target; the
DP helper then proposes target-internal measurement subclusters, such as two
scattering centers or two contour fragments, without fixing that number in
advance.
"""

import numpy as np
from pyrecest.experimental.dp_measurement_subclusters import (
    DPMeasurementSubclusterConfig,
    fit_dp_measurement_subclusters,
)


def run_example():
    """Return a DP partition for a compact two-scatterer measurement subset."""
    target_centered_measurements = np.array(
        [
            [-2.0, -0.1],
            [-1.9, 0.1],
            [-2.1, 0.0],
            [2.0, 0.1],
            [2.1, -0.1],
            [1.9, 0.0],
        ],
        dtype=float,
    )
    config = DPMeasurementSubclusterConfig(
        concentration=0.8,
        measurement_variance=0.03,
        prior_variance=25.0,
    )
    return fit_dp_measurement_subclusters(target_centered_measurements, config)


def main():
    """Print a compact partition summary."""
    result = run_example()
    print("assignments:", result.assignments.tolist())
    print("log predictive likelihood:", f"{result.log_predictive_likelihood:.3f}")
    for index, atom in enumerate(result.atoms):
        mean = ", ".join(f"{entry:.3f}" for entry in atom.posterior_mean)
        print(f"subcluster {index}: count={atom.count}, posterior_mean=[{mean}]")


if __name__ == "__main__":
    main()
