import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, diag
from pyrecest.filters import EKFSplineTracker, MEMSOEKFTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Finite-difference tracker validation tests use the NumPy backend.",
)
class TestFiniteDifferenceStepValidation(unittest.TestCase):
    @staticmethod
    def _make_mem_soekf(finite_difference_step):
        return MEMSOEKFTracker(
            array([0.0, 0.0, 1.0, -1.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 2.0, 1.0]),
            diag(array([0.01, 0.1, 0.2])),
            measurement_matrix=array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ),
            finite_difference_step=finite_difference_step,
        )

    def test_mem_soekf_rejects_nonfinite_step(self):
        for invalid_step in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(step=invalid_step), self.assertRaisesRegex(
                ValueError,
                "finite_difference_step",
            ):
                self._make_mem_soekf(invalid_step)

    def test_ekf_spline_rejects_nonfinite_step(self):
        for invalid_step in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(step=invalid_step), self.assertRaisesRegex(
                ValueError,
                "finite_difference_step",
            ):
                EKFSplineTracker(finite_difference_step=invalid_step)


if __name__ == "__main__":
    unittest.main()
