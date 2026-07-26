import unittest

import numpy as np
from pyrecest.filters.discrete_state import (
    mode_transition_matrix,
    sticky_mode_transition_matrix,
)


class TestStickyModeTransitionCountValidation(unittest.TestCase):
    def test_rejects_non_integer_mode_counts(self):
        invalid_counts = (
            True,
            np.bool_(True),
            2.0,
            2.5,
            np.array([2]),
            np.timedelta64(2, "ns"),
        )

        for n_modes in invalid_counts:
            with self.subTest(n_modes=n_modes):
                with self.assertRaisesRegex(
                    ValueError, "n_modes must be a positive integer"
                ):
                    sticky_mode_transition_matrix(n_modes, stickiness=0.8)

    def test_alias_uses_the_same_validation(self):
        with self.assertRaisesRegex(ValueError, "n_modes must be a positive integer"):
            mode_transition_matrix(False, stickiness=0.8)

    def test_accepts_numpy_integer_scalar(self):
        matrix = sticky_mode_transition_matrix(np.int64(2), stickiness=0.75)

        np.testing.assert_allclose(
            matrix,
            np.array(
                [
                    [0.75, 0.25],
                    [0.25, 0.75],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
