import unittest

from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.smoothers import RauchTungStriebelSmoother


class RauchTungStriebelMeasurementMatrixShapeTest(unittest.TestCase):
    def test_rejects_measurement_matrix_with_wrong_row_count(self):
        smoother = RauchTungStriebelSmoother()

        with self.assertRaisesRegex(
            ValueError,
            r"measurement_matrices must contain matrices with shape \(1, 2\)",
        ):
            smoother.filter(
                initial_state=GaussianDistribution(zeros(2), eye(2)),
                measurements=array([1.0, 2.0]),
                measurement_matrices=eye(2),
                meas_noise_covariances=array([1.0, 1.0]),
                system_matrices=eye(2),
                sys_noise_covariances=zeros((2, 2)),
            )


if __name__ == "__main__":
    unittest.main()
