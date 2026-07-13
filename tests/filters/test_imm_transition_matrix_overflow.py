import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


class TestIMMTransitionMatrixOverflow(unittest.TestCase):
    def test_normalizes_finite_rows_without_overflow(self):
        max_float = np.finfo(float).max
        transition_matrix = np.array(
            [
                [max_float, max_float / 2.0],
                [1.0, 1.0],
            ]
        )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            with self.assertWarnsRegex(UserWarning, "Renormalizing rows"):
                normalized = (
                    InteractingMultipleModelFilter._prepare_transition_matrix(
                        transition_matrix, 2
                    )
                )

        npt.assert_allclose(
            normalized,
            np.array(
                [
                    [2.0 / 3.0, 1.0 / 3.0],
                    [0.5, 0.5],
                ]
            ),
        )
        npt.assert_allclose(normalized.sum(axis=1), np.ones(2))


if __name__ == "__main__":
    unittest.main()
