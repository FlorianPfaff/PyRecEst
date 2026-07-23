"""Regression tests for UKF-M measurement-vector shape validation."""

import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, zeros
from pyrecest.filters.ukf_on_manifolds import UKFOnManifolds


def _make_filter(observation_function):
    def propagation(state, omega, noise, dt):  # pylint: disable=unused-argument
        return state + noise

    return UKFOnManifolds(
        f=propagation,
        h=observation_function,
        phi=lambda state, tangent: state + tangent,
        phi_inv=lambda reference, state: state - reference,
        Q=eye(2) * 0.1,
        R=eye(2) * 0.5,
        alpha=1.0e-3,
        state0=zeros(2),
        P0=eye(2),
    )


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="UKFOnManifolds.update is not supported on JAX",
)
class TestUKFOnManifoldsMeasurementShapes(unittest.TestCase):
    """Reject values that flatten or broadcast to the measurement dimension."""

    def test_row_measurement_is_rejected(self):
        ukf = _make_filter(lambda state: state)

        with self.assertRaisesRegex(ValueError, "measurement must have shape"):
            ukf.update(array([[1.0, -1.0]]))

    def test_column_measurement_is_rejected(self):
        ukf = _make_filter(lambda state: state)

        with self.assertRaisesRegex(ValueError, "measurement must have shape"):
            ukf.update(array([[1.0], [-1.0]]))

    def test_scalar_observation_output_is_rejected(self):
        ukf = _make_filter(lambda state: 0.0)

        with self.assertRaisesRegex(
            ValueError, "observation function output must have shape"
        ):
            ukf.update(array([1.0, -1.0]))

    def test_row_observation_output_is_rejected(self):
        ukf = _make_filter(lambda state: array([[state[0], state[1]]]))

        with self.assertRaisesRegex(
            ValueError, "observation function output must have shape"
        ):
            ukf.update(array([1.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
