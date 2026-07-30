import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
import pytest
from pyrecest.experimental.dvs.trackers import DVSFullSCGPTracker
from pyrecest.experimental.dvs.vectorized_flow import (
    tracker_signed_normal_flows_vectorized,
)


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="DVS vectorized-flow tests currently use numpy.testing assertions",
)
@pytest.mark.parametrize("velocity", ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0]))
def test_vectorized_flow_matches_scalar_tracker(velocity):
    tracker = DVSFullSCGPTracker(
        16,
        kinematic_state=np.asarray([50.0, 50.0, 0.1]),
        kinematic_covariance=np.eye(3),
        shape_state=np.full(16, 10.0),
        shape_covariance=np.eye(16),
        velocities=False,
        measurement_noise=np.eye(2),
    )
    measurements = np.asarray(
        [[60.0, 50.0], [40.0, 50.0], [50.0, 56.0], [50.0, 44.0]],
        dtype=float,
    )
    vectorized = tracker_signed_normal_flows_vectorized(tracker, measurements, velocity)
    scalar = np.asarray(
        [tracker.signed_normal_flow_for_measurement(m, velocity) for m in measurements],
        dtype=float,
    )
    assert np.allclose(vectorized, scalar, atol=1e-8)


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="DVS vectorized-flow tests currently use numpy.testing assertions",
)
def test_vectorized_flow_preserves_extreme_finite_velocity_direction():
    tracker = DVSFullSCGPTracker(
        16,
        kinematic_state=np.asarray([50.0, 50.0, 0.1]),
        kinematic_covariance=np.eye(3),
        shape_state=np.full(16, 10.0),
        shape_covariance=np.eye(16),
        velocities=False,
        measurement_noise=np.eye(2),
    )
    measurements = np.asarray(
        [[60.0, 50.0], [40.0, 50.0], [50.0, 56.0], [50.0, 44.0]],
        dtype=float,
    )

    reference = tracker_signed_normal_flows_vectorized(
        tracker, measurements, [1.0, 1.0]
    )
    extreme = tracker_signed_normal_flows_vectorized(
        tracker, measurements, [1e308, 1e308]
    )

    assert np.isfinite(extreme).all()
    np.testing.assert_allclose(extreme, reference, atol=1e-8)
