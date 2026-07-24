import unittest

import numpy as np
import pyrecest.backend
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Global nearest-neighbor association is not supported on this backend",
)
class GlobalNearestNeighborInputDimensionTest(unittest.TestCase):
    @staticmethod
    def _tracker():
        tracker = GlobalNearestNeighbor(
            association_param={"distance_metric_pos": "Euclidean"}
        )
        tracker.filter_state = [
            KalmanFilter(GaussianDistribution(np.zeros(2), np.eye(2)))
        ]
        return tracker

    def test_rejects_one_dimensional_measurements(self):
        tracker = self._tracker()

        with self.assertRaisesRegex(
            ValueError, r"measurements must have shape \(dim_meas, n_meas\)"
        ):
            tracker.find_association(np.zeros(2), np.eye(2), np.eye(2))

    def test_rejects_measurement_matrix_row_mismatch(self):
        tracker = self._tracker()

        with self.assertRaisesRegex(ValueError, "measurement matrix"):
            tracker.find_association(
                np.zeros((2, 1)),
                np.zeros((1, 2)),
                np.eye(2),
            )

    def test_rejects_covariances_with_wrong_measurement_dimension(self):
        tracker = self._tracker()
        measurements = np.zeros((2, 1))

        invalid_covariances = (
            np.eye(3),
            np.zeros((3, 3, 1)),
        )
        for cov_mats_meas in invalid_covariances:
            with self.subTest(shape=cov_mats_meas.shape):
                with self.assertRaisesRegex(
                    ValueError, "cov_mats_meas must have shape"
                ):
                    tracker.find_association(
                        measurements,
                        np.eye(2),
                        cov_mats_meas,
                    )


if __name__ == "__main__":
    unittest.main()
