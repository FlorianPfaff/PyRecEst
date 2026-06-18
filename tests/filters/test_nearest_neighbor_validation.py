import unittest
from unittest.mock import patch

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Nearest-neighbor validation tests require the numpy backend fixtures.",
)
class NearestNeighborValidationTest(unittest.TestCase):
    def _tracker(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = [KalmanFilter(GaussianDistribution(zeros(4), eye(4)))]
        return tracker

    def test_duplicate_filter_handles_raise_value_error(self):
        shared_filter = KalmanFilter(GaussianDistribution(zeros(4), eye(4)))
        tracker = GlobalNearestNeighbor()

        with self.assertRaisesRegex(ValueError, "same handle"):
            tracker.filter_state = [shared_filter, shared_filter]

    def test_predict_linear_rejects_nonzero_mean_gaussian_system_noise(self):
        tracker = self._tracker()
        nonzero_mean_noise = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]), eye(4), check_validity=False
        )

        with self.assertRaisesRegex(ValueError, "zero mean"):
            tracker.predict_linear(eye(4), nonzero_mean_noise)

    def test_update_linear_unsupported_backend_raises_not_implemented(self):
        tracker = self._tracker()

        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "numpy backend"):
                tracker.update_linear(array([[0.0], [0.0]]), eye(4)[:2, :], eye(2))

    def test_update_linear_dimension_mismatch_raises_value_error(self):
        tracker = self._tracker()

        with self.assertRaisesRegex(ValueError, "measurement matrix"):
            tracker.update_linear(array([[0.0]]), eye(4)[:2, :], eye(2))


if __name__ == "__main__":
    unittest.main()
