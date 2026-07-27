import unittest

import numpy as np
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import (
    estimate_transform,
    solve_gated_assignment,
)


class TestPointSetRegistrationMixedScalarValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_booleans_hidden_by_mixed_coercion(self):
        target = array([[0.0, 0.0], [1.0, 0.0]])
        invalid_sources = (
            [[0.0, False], [1.0, 0.0]],
            [[0.0, np.bool_(False)], [1.0, 0.0]],
        )

        for source in invalid_sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    estimate_transform(source, target, model="translation")

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_assignment_rejects_booleans_hidden_by_mixed_coercion(self):
        invalid_cost_matrices = (
            [[0.0, True]],
            [[0.0, np.bool_(True)]],
        )

        for cost_matrix in invalid_cost_matrices:
            with self.subTest(cost_matrix=cost_matrix):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    solve_gated_assignment(cost_matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_real_mixed_numeric_inputs_remain_supported(self):
        source = [[0, 0.0], [1, 0.0]]
        target = array([[2.0, 3.0], [3.0, 3.0]])

        transform = estimate_transform(source, target, model="translation")
        assignment = solve_gated_assignment([[0, 1.5], [2.5, 0]])

        np.testing.assert_allclose(transform.offset, [2.0, 3.0])
        np.testing.assert_array_equal(assignment, [0, 1])


if __name__ == "__main__":
    unittest.main()
