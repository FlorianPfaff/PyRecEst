import unittest

import pyrecest.backend
from pyrecest.backend import abs as backend_abs
from pyrecest.backend import (
    allclose,
    arange,
    argmax,
    array,
    array_equal,
    diagonal,
)
from pyrecest.backend import max as backend_max
from pyrecest.backend import (
    mean,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    transpose,
)
from pyrecest.distributions import SdHalfCondSdHalfGridDistribution
from pyrecest.distributions._so3_helpers import exp_map_identity, quaternion_multiply
from pyrecest.filters import (
    HyperhemisphericalGridFilter,
    quaternion_grid_transition_density,
    so3_right_multiplication_grid_transition,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Not supported on JAX backend",
)
class TestSO3GridTransition(unittest.TestCase):
    def setUp(self):
        self.filter_ = HyperhemisphericalGridFilter(32, 3)
        self.grid = self.filter_.filter_state.get_grid()
        self.manifold_size = self.filter_.filter_state.get_manifold_size()

    def test_returns_normalized_conditional_density(self):
        transition = so3_right_multiplication_grid_transition(
            self.grid,
            array([0.0, 0.0, 0.0]),
            24.0,
        )

        self.assertIsInstance(transition, SdHalfCondSdHalfGridDistribution)
        self.assertTrue(allclose(transition.get_grid(), self.grid, atol=1e-12))
        column_integrals = mean(transition.grid_values, axis=0) * self.manifold_size
        self.assertTrue(allclose(column_integrals, 1.0, atol=1e-10))

    def test_identity_increment_peaks_on_current_grid_cell(self):
        transition = so3_right_multiplication_grid_transition(
            self.grid,
            array([0.0, 0.0, 0.0]),
            80.0,
        )

        column_maxima = backend_max(transition.grid_values, axis=0)
        self.assertTrue(
            allclose(diagonal(transition.grid_values), column_maxima, atol=1e-12)
        )

    def test_nonzero_tangent_increment_peaks_at_rotated_grid_cell(self):
        tangent_increment = array([0.7, 0.2, -0.1])
        transition = so3_right_multiplication_grid_transition(
            self.grid,
            tangent_increment,
            40.0,
        )

        delta_quaternion = exp_map_identity(tangent_increment)[0]
        targets = quaternion_multiply(self.grid, delta_quaternion)
        expected_indices = argmax(backend_abs(self.grid @ transpose(targets)), axis=0)
        actual_indices = argmax(transition.grid_values, axis=0)

        self.assertTrue(array_equal(actual_indices, expected_indices))
        self.assertGreater(
            int(backend_sum(actual_indices != arange(self.grid.shape[0]))),
            0,
        )

    def test_accepts_tangent_and_quaternion_increments(self):
        tangent_increment = array([0.1, -0.2, 0.3])
        delta_quaternion = exp_map_identity(tangent_increment)[0]

        transition_from_tangent = so3_right_multiplication_grid_transition(
            self.grid,
            tangent_increment,
            18.0,
        )
        transition_from_quaternion = so3_right_multiplication_grid_transition(
            self.grid,
            delta_quaternion,
            18.0,
        )
        transition_from_alias = quaternion_grid_transition_density(
            self.grid,
            tangent_increment,
            18.0,
        )

        self.assertTrue(
            allclose(
                transition_from_tangent.grid_values,
                transition_from_quaternion.grid_values,
                atol=1e-12,
            )
        )
        self.assertTrue(
            allclose(
                transition_from_tangent.grid_values,
                transition_from_alias.grid_values,
                atol=1e-12,
            )
        )

    def test_antipodal_increment_and_grid_representatives_are_invariant(self):
        delta_quaternion = exp_map_identity(array([0.2, -0.1, 0.3]))[0]
        transition = so3_right_multiplication_grid_transition(
            self.grid,
            delta_quaternion,
            20.0,
        )
        transition_from_antipodal_increment = so3_right_multiplication_grid_transition(
            self.grid,
            -delta_quaternion,
            20.0,
        )
        transition_from_antipodal_grid = so3_right_multiplication_grid_transition(
            -self.grid,
            delta_quaternion,
            20.0,
        )

        self.assertTrue(
            allclose(
                transition.grid_values,
                transition_from_antipodal_increment.grid_values,
                atol=1e-12,
            )
        )
        self.assertTrue(
            allclose(
                transition.grid_values,
                transition_from_antipodal_grid.grid_values,
                atol=1e-12,
            )
        )
        self.assertTrue(
            allclose(transition_from_antipodal_grid.get_grid(), self.grid, atol=1e-12)
        )

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            so3_right_multiplication_grid_transition(
                self.grid,
                array([0.0, 0.0, 0.0]),
                0.0,
            )
        with self.assertRaises(ValueError):
            so3_right_multiplication_grid_transition(
                self.grid[:, :3],
                array([0.0, 0.0, 0.0]),
                1.0,
            )
        with self.assertRaises(ValueError):
            so3_right_multiplication_grid_transition(
                self.grid,
                array([0.0, 0.0]),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
