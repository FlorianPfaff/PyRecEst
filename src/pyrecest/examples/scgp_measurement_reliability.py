"""Reusable SCGP reliability-weighted measurement example.

The scenario contains two plausible contour returns and one outlier. It compares
an unweighted update, a reliability-weighted update, and a masked update. The
module is intentionally importable so downstream packages can reuse the same
example instead of keeping their own copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrecest.backend import array, eye, linalg
from pyrecest.filters import FullSCGPTracker


@dataclass(frozen=True)
class SCGPMeasurementReliabilityExampleResult:
    """Outputs of the reliability-weighted SCGP example."""

    measurements: Any
    measurement_weights: Any
    active_measurement_mask: Any
    prior_kinematic_estimate: Any
    unweighted_kinematic_estimate: Any
    weighted_kinematic_estimate: Any
    masked_kinematic_estimate: Any
    unweighted_active_measurement_indices: tuple[int, ...]
    weighted_active_measurement_indices: tuple[int, ...]
    masked_active_measurement_indices: tuple[int, ...]
    unweighted_quadratic_form: float | None
    weighted_quadratic_form: float | None
    masked_quadratic_form: float | None

    @property
    def weighted_position_shift(self) -> float:
        """Return the position shift induced by the weighted update."""
        return float(
            linalg.norm(
                self.weighted_kinematic_estimate[:2] - self.prior_kinematic_estimate[:2]
            )
        )

    @property
    def unweighted_position_shift(self) -> float:
        """Return the position shift induced by the unweighted update."""
        return float(
            linalg.norm(
                self.unweighted_kinematic_estimate[:2]
                - self.prior_kinematic_estimate[:2]
            )
        )

    @property
    def masked_position_shift(self) -> float:
        """Return the position shift induced by the masked update."""
        return float(
            linalg.norm(
                self.masked_kinematic_estimate[:2] - self.prior_kinematic_estimate[:2]
            )
        )


def make_scgp_reliability_example_tracker() -> FullSCGPTracker:
    """Return a small SCGP tracker used by the reliability example."""
    n_base_points = 8
    reference_extent = array([1.0] * n_base_points)
    return FullSCGPTracker(
        n_base_points,
        kinematic_state=array([0.0, 0.0, 0.0, 1.0, 0.0]),
        kinematic_covariance=1e-4 * eye(5),
        shape_state=reference_extent,
        shape_covariance=0.05 * eye(n_base_points),
        measurement_noise=0.02 * eye(2),
        radial_noise_variance=0.01,
        extent_forgetting_rate=0.2,
        reference_extent=reference_extent,
    )


def scgp_reliability_example_measurements():
    """Return two plausible contour measurements and one outlier."""
    return array(
        [
            [1.4, 0.1],
            [0.2, 1.2],
            [3.0, 3.0],
        ]
    )


def run_scgp_measurement_reliability_example():
    """Run unweighted, weighted, and masked SCGP updates for one batch.

    The unweighted tracker treats all three measurements equally. The weighted
    tracker retains the second contour return with lower reliability and skips
    the outlier by assigning weight zero. The masked tracker explicitly disables
    the outlier while leaving the two plausible contour measurements active.
    """
    measurements = scgp_reliability_example_measurements()
    measurement_weights = array([1.0, 0.25, 0.0])
    active_measurement_mask = array([True, True, False])

    unweighted_tracker = make_scgp_reliability_example_tracker()
    weighted_tracker = make_scgp_reliability_example_tracker()
    masked_tracker = make_scgp_reliability_example_tracker()
    prior_kinematic_estimate = array(weighted_tracker.get_point_estimate_kinematics())

    unweighted_tracker.update(measurements)
    weighted_tracker.update(
        measurements,
        measurement_weights=measurement_weights,
    )
    masked_tracker.update(
        measurements,
        active_measurement_mask=active_measurement_mask,
    )

    return SCGPMeasurementReliabilityExampleResult(
        measurements=measurements,
        measurement_weights=measurement_weights,
        active_measurement_mask=active_measurement_mask,
        prior_kinematic_estimate=prior_kinematic_estimate,
        unweighted_kinematic_estimate=array(
            unweighted_tracker.get_point_estimate_kinematics()
        ),
        weighted_kinematic_estimate=array(
            weighted_tracker.get_point_estimate_kinematics()
        ),
        masked_kinematic_estimate=array(masked_tracker.get_point_estimate_kinematics()),
        unweighted_active_measurement_indices=tuple(
            unweighted_tracker.last_active_measurement_indices or ()
        ),
        weighted_active_measurement_indices=tuple(
            weighted_tracker.last_active_measurement_indices or ()
        ),
        masked_active_measurement_indices=tuple(
            masked_tracker.last_active_measurement_indices or ()
        ),
        unweighted_quadratic_form=unweighted_tracker.last_quadratic_form,
        weighted_quadratic_form=weighted_tracker.last_quadratic_form,
        masked_quadratic_form=masked_tracker.last_quadratic_form,
    )


def main() -> None:
    """Print a compact summary of the reliability example."""
    result = run_scgp_measurement_reliability_example()
    print("measurements:", result.measurements)
    print("measurement weights:", result.measurement_weights)
    print("active measurement mask:", result.active_measurement_mask)
    print("unweighted active indices:", result.unweighted_active_measurement_indices)
    print("weighted active indices:", result.weighted_active_measurement_indices)
    print("masked active indices:", result.masked_active_measurement_indices)
    print("unweighted kinematic estimate:", result.unweighted_kinematic_estimate)
    print("weighted kinematic estimate:", result.weighted_kinematic_estimate)
    print("masked kinematic estimate:", result.masked_kinematic_estimate)
    print("unweighted position shift:", result.unweighted_position_shift)
    print("weighted position shift:", result.weighted_position_shift)
    print("masked position shift:", result.masked_position_shift)


if __name__ == "__main__":
    main()
