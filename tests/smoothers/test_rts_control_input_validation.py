import unittest

from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.smoothers import RauchTungStriebelSmoother
from pyrecest.smoothers.abstract_smoother import AbstractSmoother


class RauchTungStriebelControlInputValidationTest(unittest.TestCase):
    def test_rejects_broadcastable_control_vector_with_wrong_dimension(self):
        smoother = RauchTungStriebelSmoother()

        with self.assertRaisesRegex(
            ValueError, r"sys_inputs must contain vectors with shape \(2,\)"
        ):
            smoother.filter(
                initial_state=GaussianDistribution(zeros(2), eye(2)),
                measurements=[zeros(2), zeros(2)],
                measurement_matrices=zeros((2, 2)),
                meas_noise_covariances=eye(2),
                system_matrices=eye(2),
                sys_noise_covariances=zeros((2, 2)),
                sys_inputs=array([1.0]),
            )

    def test_accepts_all_omitted_per_step_control_inputs(self):
        normalized = AbstractSmoother._normalize_vector_sequence(
            [None, None],
            length=2,
            name="sys_inputs",
            vector_dim=2,
        )

        self.assertEqual(normalized, [None, None])


if __name__ == "__main__":
    unittest.main()
