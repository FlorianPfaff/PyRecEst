import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, cos, deg2rad, eye, sin, vstack, zeros
from pyrecest.utils.point_set_registration import (
    AffineTransform,
    estimate_transform,
    joint_registration_assignment,
    solve_gated_assignment,
)


class TestEstimateTransform(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_affine_transform_recovers_ground_truth(self):
        source = array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.5, 2.0]],
        )
        true_matrix = array([[1.1, 0.2], [-0.15, 0.95]])
        true_offset = array([3.0, -1.5])
        target = (true_matrix @ source.T).T + true_offset

        estimated = estimate_transform(source, target, model="affine")

        npt.assert_allclose(estimated.matrix, true_matrix, atol=1e-10)
        npt.assert_allclose(estimated.offset, true_offset, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_rigid_transform_recovers_rotation_and_translation(self):
        source = array(
            [[0.0, 0.0], [1.0, 0.2], [0.4, 1.1], [1.3, 1.6], [2.1, -0.3]],
        )
        angle = deg2rad(array(25.0))
        true_rotation = array(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]],
        )
        true_offset = array([-0.8, 2.4])
        target = (true_rotation @ source.T).T + true_offset

        estimated = estimate_transform(source, target, model="rigid")

        npt.assert_allclose(estimated.matrix, true_rotation, atol=1e-10)
        npt.assert_allclose(estimated.offset, true_offset, atol=1e-10)


class TestGatedAssignment(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_solve_gated_assignment_leaves_rows_unmatched(self):
        cost_matrix = array([[0.1, 10.0], [10.0, 10.0]])
        assignment = solve_gated_assignment(cost_matrix, max_cost=1.0)
        npt.assert_array_equal(assignment, array([0, -1]))


class TestJointRegistrationAssignment(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_registration_assignment_recovers_permuted_affine_matches(self):
        reference = array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 0.5],
                [1.5, 1.6],
                [2.3, 1.9],
            ],
        )
        true_matrix = array([[1.03, 0.08], [-0.04, 0.97]])
        true_offset = array([4.0, -2.0])
        transformed = (true_matrix @ reference.T).T + true_offset
        permutation = array([3, 5, 1, 4, 0, 2])
        moving = transformed[permutation]

        result = joint_registration_assignment(
            reference,
            moving,
            model="affine",
            max_cost=5.0,
            tolerance=1e-12,
        )

        expected_assignment = zeros(reference.shape[0], dtype=int)
        expected_assignment[permutation] = array(list(range(reference.shape[0])))
        npt.assert_array_equal(result.assignment, expected_assignment)
        npt.assert_allclose(result.transform.matrix, true_matrix, atol=1e-10)
        npt.assert_allclose(result.transform.offset, true_offset, atol=1e-10)
        self.assertTrue(result.converged)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_registration_assignment_handles_missing_points_and_outliers(self):
        reference = array(
            [[0.0, 0.0], [1.0, 0.0], [0.3, 0.9], [1.1, 1.2], [2.0, -0.2]],
        )
        shift = array([6.0, -3.0])
        moving = reference[[1, 3, 4]] + shift
        moving = vstack([moving, array([[50.0, 50.0]])])

        result = joint_registration_assignment(
            reference,
            moving,
            model="translation",
            initial_transform=AffineTransform(eye(2), shift + array([0.1, -0.1])),
            max_cost=0.35,
        )

        self.assertEqual(int((result.assignment >= 0).sum()), 3)
        self.assertTrue(bool((result.assignment[[0, 2]] == -1).all()))
        npt.assert_allclose(result.transform.offset, shift, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_registration_assignment_supports_custom_cost_function(self):
        reference = array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        moving = reference + array([0.5, -0.2])
        preference = array(
            [
                [0.0, 10.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 0.0],
            ],
        )

        def cost_function(transformed_reference, moving_points):
            del transformed_reference, moving_points
            return preference

        result = joint_registration_assignment(
            reference,
            moving,
            model="translation",
            cost_function=cost_function,
            max_cost=1.0,
        )

        npt.assert_array_equal(result.assignment, array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
