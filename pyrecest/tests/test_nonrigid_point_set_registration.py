import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, asarray, concatenate, eye, linalg, ones, zeros
from pyrecest.utils.nonrigid_point_set_registration import (
    ThinPlateSplineTransform,
    estimate_thin_plate_spline,
    joint_tps_registration_assignment,
)


class TestThinPlateSplineEstimation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_thin_plate_spline_recovers_known_warp(self):
        source = array(
            [
                [0.0, 0.0],
                [1.0, 0.2],
                [0.2, 1.1],
                [1.3, 1.4],
                [2.1, 0.1],
                [2.0, 1.8],
            ]
        )
        polynomial = concatenate([ones((source.shape[0], 1)), source], axis=1)
        raw_weights = array(
            [
                [0.06, -0.02],
                [-0.03, 0.01],
                [0.01, 0.03],
                [-0.02, -0.01],
                [0.00, 0.02],
                [-0.02, -0.03],
            ]
        )
        projector = eye(source.shape[0]) - polynomial @ linalg.pinv(polynomial)
        tps_weights = projector @ raw_weights

        true_transform = ThinPlateSplineTransform(
            control_points=source,
            weights=array(tps_weights),
            affine_coefficients=array(
                [
                    [1.5, -0.7],
                    [1.02, 0.05],
                    [-0.04, 0.98],
                ]
            ),
        )
        target = true_transform.apply(source)

        estimated = estimate_thin_plate_spline(source, target, regularization=0.0)

        query = array([[0.5, 0.7], [1.7, 0.8], [1.2, 1.9]])
        npt.assert_allclose(estimated.apply(source), target, atol=1e-9)
        npt.assert_allclose(
            estimated.apply(query), true_transform.apply(query), atol=1e-8
        )


class TestJointThinPlateSplineRegistrationAssignment(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_tps_registration_assignment_recovers_permuted_matches(self):
        reference = array(
            [
                [0.0, 0.0],
                [1.0, 0.1],
                [0.2, 1.1],
                [1.3, 1.5],
                [2.2, -0.1],
                [2.4, 1.2],
                [3.0, 0.5],
            ]
        )
        true_transform = ThinPlateSplineTransform(
            control_points=reference,
            weights=array(
                [
                    [0.04, -0.01],
                    [-0.03, 0.02],
                    [0.01, 0.03],
                    [-0.02, -0.02],
                    [0.00, 0.02],
                    [0.01, -0.03],
                    [-0.01, 0.00],
                ]
            ),
            affine_coefficients=array(
                [
                    [2.2, -1.0],
                    [1.01, 0.04],
                    [-0.03, 0.99],
                ]
            ),
        )
        transformed = true_transform.apply(reference)
        permutation = array([4, 1, 6, 0, 5, 2, 3])
        moving = transformed[permutation]

        result = joint_tps_registration_assignment(
            reference,
            moving,
            max_cost=1.5,
            regularization=1e-8,
            tolerance=1e-10,
        )

        expected_assignment = zeros(reference.shape[0], dtype=int)
        expected_assignment[permutation] = array(list(range(reference.shape[0])))

        npt.assert_array_equal(result.assignment, expected_assignment)
        npt.assert_allclose(
            result.transform.apply(reference),
            transformed,
            atol=1e-6,
        )
        self.assertTrue(result.converged)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_tps_registration_assignment_handles_missing_points_and_outliers(
        self,
    ):
        reference = array(
            [
                [0.0, 0.0],
                [0.8, 0.2],
                [0.1, 1.0],
                [1.1, 1.2],
                [1.9, -0.1],
                [2.0, 1.5],
            ]
        )
        true_transform = ThinPlateSplineTransform(
            control_points=reference,
            weights=array(
                [
                    [0.02, 0.00],
                    [-0.01, 0.01],
                    [0.00, 0.02],
                    [-0.02, -0.01],
                    [0.01, 0.00],
                    [0.00, -0.02],
                ]
            ),
            affine_coefficients=array(
                [
                    [1.2, -0.9],
                    [1.0, 0.03],
                    [-0.01, 1.0],
                ]
            ),
        )
        observed_indices = array([1, 3, 5])
        moving = true_transform.apply(reference[observed_indices])
        moving = array(
            [
                moving[0],
                moving[1],
                moving[2],
                [20.0, 20.0],
            ]
        )

        result = joint_tps_registration_assignment(
            reference,
            moving,
            initial_transform=ThinPlateSplineTransform.from_translation(
                array([1.1, -0.8])
            ),
            max_cost=0.5,
            regularization=1e-8,
        )

        assignment = asarray(result.assignment)
        self.assertEqual(int((assignment >= 0).sum()), 3)
        self.assertTrue(bool((assignment[[0, 2, 4]] == -1).all()))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_tps_registration_assignment_supports_custom_cost_function(self):
        reference = array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        moving = array([[0.3, -0.1], [1.3, -0.1], [2.3, -0.1]])
        preference = array(
            [
                [0.0, 10.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 0.0],
            ]
        )

        def cost_function(transformed_reference, moving_points):
            del transformed_reference, moving_points
            return preference

        result = joint_tps_registration_assignment(
            reference,
            moving,
            initial_transform=ThinPlateSplineTransform.from_translation(
                array([0.3, -0.1])
            ),
            cost_function=cost_function,
            max_cost=1.0,
            regularization=1e-8,
        )

        npt.assert_array_equal(result.assignment, array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
