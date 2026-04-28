import unittest

import numpy.testing as npt
from pyrecest.backend import array, pi
from pyrecest.distributions import CircularDiracDistribution, VonMisesDistribution
from pyrecest.smoothers import SlidingWindowManifoldMeanSmoother


class SlidingWindowManifoldMeanSmootherTest(unittest.TestCase):
    _CIRCULAR_ATOL = 1e-6

    def test_centered_window_smooths_linear_sequence(self):
        smoother = SlidingWindowManifoldMeanSmoother(window_size=3)

        smoothed_values = smoother.smooth(
            [array([0.0]), array([2.0]), array([4.0]), array([6.0])]
        )

        self.assertEqual(len(smoothed_values), 4)
        npt.assert_allclose(smoothed_values[0], array([1.0]))
        npt.assert_allclose(smoothed_values[1], array([2.0]))
        npt.assert_allclose(smoothed_values[2], array([4.0]))
        npt.assert_allclose(smoothed_values[3], array([5.0]))

    def test_trailing_window_smooths_only_past_values(self):
        smoother = SlidingWindowManifoldMeanSmoother(
            window_size=2,
            alignment="trailing",
        )

        smoothed_values = smoother.smooth([array([0.0]), array([2.0]), array([4.0])])

        npt.assert_allclose(smoothed_values[0], array([0.0]))
        npt.assert_allclose(smoothed_values[1], array([1.0]))
        npt.assert_allclose(smoothed_values[2], array([3.0]))

    def test_explicit_circular_dirac_factory_handles_wraparound(self):
        smoother = SlidingWindowManifoldMeanSmoother(
            window_size=3,
            dirac_distribution_factory=CircularDiracDistribution,
        )

        smoothed_values = smoother.smooth(
            [array([2.0 * pi - 0.1]), array([0.0]), array([0.1])]
        )

        smoothed_angle = float(smoothed_values[1][0])
        self.assertLess(
            min(abs(smoothed_angle), abs(smoothed_angle - 2.0 * pi)),
            self._CIRCULAR_ATOL,
        )

    def test_distribution_inputs_infer_circular_manifold(self):
        smoother = SlidingWindowManifoldMeanSmoother(window_size=3)
        states = [
            VonMisesDistribution(2.0 * pi - 0.1, 4.0),
            VonMisesDistribution(0.0, 4.0),
            VonMisesDistribution(0.1, 4.0),
        ]

        smoothed_values = smoother.smooth(states)

        smoothed_angle = float(smoothed_values[1][0])
        self.assertLess(
            min(abs(smoothed_angle), abs(smoothed_angle - 2.0 * pi)),
            self._CIRCULAR_ATOL,
        )

    def test_window_weights_are_applied_to_edges(self):
        smoother = SlidingWindowManifoldMeanSmoother(
            window_size=3,
            window_weights=array([1.0, 2.0, 1.0]),
        )

        smoothed_values = smoother.smooth([array([0.0]), array([2.0]), array([4.0])])

        npt.assert_allclose(smoothed_values[0], array([2.0 / 3.0]))
        npt.assert_allclose(smoothed_values[1], array([2.0]))
        npt.assert_allclose(smoothed_values[2], array([10.0 / 3.0]))


if __name__ == "__main__":
    unittest.main()
