import copy
import unittest

import numpy as np
from parameterized import parameterized
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.kernel_sme_filter import KernelSMEFilter
from scipy.linalg import block_diag


class TestKernelSMEFilter(unittest.TestCase):
    measurement_matrix_2DCV = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    gaussians_2DCV = [
        GaussianDistribution(np.zeros(4), np.diag(np.arange(1, 5))),
        GaussianDistribution(np.arange(1, 5), 2 * np.eye(4)),
        GaussianDistribution(-np.arange(1, 5), np.diag(np.arange(4, 0, -1))),
    ]

    measurement_matrix_3DCV = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
    )
    gaussians_3DCV = [
        GaussianDistribution(np.zeros(6), np.diag(np.arange(1, 7))),
        GaussianDistribution(np.arange(1, 7), 2 * np.eye(6)),
        GaussianDistribution(-np.arange(1, 7), np.diag(np.arange(6, 0, -1))),
    ]

    measurement_matrix_3DCA = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )
    gaussians_3DCA = [
        GaussianDistribution(np.zeros(9), np.diag(np.arange(1, 10))),
        GaussianDistribution(np.arange(1, 10), 2 * np.eye(9)),
        GaussianDistribution(-np.arange(1, 10), np.diag(np.arange(9, 0, -1))),
    ]

    scenario_names = ["2DCV", "3DCV", "3DCA"]

    def setUp(self):
        # Set up your test case variables here
        self.initial_priors = [
            GaussianDistribution(np.array([0, 0]), np.array([[1, 0], [0, 1]])),
            GaussianDistribution(np.array([5, 5]), np.array([[1, 0], [0, 1]])),
        ]
        self.system_matrix = np.array([[1, 0.1], [0, 1]])
        self.sys_noise_cov = np.array([[0.01, 0], [0, 0.01]])
        self.measurement_noise_cov = np.array([[0.1, 0], [0, 0.1]])
        self.measurements = np.array([[1, 1.1], [5.5, 4.9]])
        self.perfect_meas_ordered_2DCV = self.measurement_matrix_2DCV @ np.column_stack(
            [g.mu for g in self.gaussians_2DCV]
        )

    @parameterized.expand([(gaussians_2DCV,), (gaussians_3DCV,), (gaussians_3DCA,)])
    def testSetState(self, gaussians):
        tracker = KernelSMEFilter()
        tracker.filter_state = gaussians

    @parameterized.expand(
        [
            (gaussians_2DCV, (4, 3), (12,)),
            (gaussians_3DCV, (6, 3), (18,)),
            (gaussians_3DCA, (9, 3), (27,)),
        ]
    )
    def testGetState(
        self, gaussians, expected_point_estimate_shape, expected_filter_state_shape
    ):
        tracker = KernelSMEFilter(gaussians)
        self.assertEqual(tracker.filter_state.mu.shape, expected_filter_state_shape)
        self.assertEqual(
            tracker.filter_state.C.shape,
            (expected_filter_state_shape[0], expected_filter_state_shape[0]),
        )

        self.assertEqual(
            tracker.get_point_estimate().shape, expected_point_estimate_shape
        )
        self.assertEqual(
            tracker.get_point_estimate(True).shape, expected_filter_state_shape
        )

    def testPredictLinearAllSameMatsNoInputs2DCV(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_2DCV
        np.testing.assert_allclose(
            tracker.filter_state.mu,
            np.hstack(
                (
                    self.gaussians_2DCV[0].mu,
                    self.gaussians_2DCV[1].mu,
                    self.gaussians_2DCV[2].mu,
                )
            ),
        )
        tracker.predict_linear(
            np.block([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
            np.eye(4),
        )
        np.testing.assert_allclose(
            tracker.get_point_estimate(),
            np.column_stack(
                (np.zeros((4,)), np.array([3, 2, 7, 4]), -np.array([3, 2, 7, 4]))
            ),
        )
        np.testing.assert_allclose(
            tracker.C,
            block_diag(
                np.array([[3, 2], [2, 2]]),
                np.array([[7, 4], [4, 4]]),
                np.array([[4, 2], [2, 2]]),
                np.array([[4, 2], [2, 2]]),
                np.array([[7, 3], [3, 3]]),
                np.array([[3, 1], [1, 1]]),
            )
            + np.eye(12),
        )

    def testPredictLinearAllSameMatsNoInputs3DCV(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_3DCV
        tracker.predict_linear(
            np.array(
                [
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            np.eye(6),
        )
        np.testing.assert_allclose(
            tracker.get_point_estimate(),
            np.column_stack(
                (
                    np.zeros((6, 1)),
                    np.array([3, 2, 7, 4, 11, 6]),
                    -np.array([3, 2, 7, 4, 11, 6]),
                )
            ),
        )
        # jscpd:ignore-start
        np.testing.assert_allclose(
            tracker.C,
            block_diag(
                np.array([[3, 2], [2, 2]]),
                np.array([[7, 4], [4, 4]]),
                np.array([[11, 6], [6, 6]]),
                np.array([[4, 2], [2, 2]]),
                np.array([[4, 2], [2, 2]]),
                np.array([[4, 2], [2, 2]]),
                np.array([[11, 5], [5, 5]]),
                np.array([[7, 3], [3, 3]]),
                np.array([[3, 1], [1, 1]]),
            )
            + np.eye(18),
        )
        # jscpd:ignore-end

    @parameterized.expand(
        [
            (gaussians_2DCV, measurement_matrix_2DCV),
            (gaussians_3DCV, measurement_matrix_3DCV),
            (gaussians_3DCA, measurement_matrix_3DCA),
        ]
    )
    def test_update_independent_of_order(self, gaussians, measurement_matrix):
        tracker_ordered = KernelSMEFilter(gaussians)
        tracker_unordered = copy.deepcopy(tracker_ordered)

        perfect_meas_ordered = measurement_matrix @ np.column_stack(
            [g.mu for g in gaussians]
        )

        tracker_ordered.update_linear(
            perfect_meas_ordered,
            measurement_matrix,
            np.eye(perfect_meas_ordered.shape[0]),
        )
        tracker_unordered.update_linear(
            perfect_meas_ordered[:, ::-1],
            measurement_matrix,
            np.eye(perfect_meas_ordered.shape[0]),
        )

        # Check if the state estimates are close enough
        np.testing.assert_allclose(
            tracker_unordered.get_point_estimate(), tracker_ordered.get_point_estimate()
        )
        np.testing.assert_allclose(
            tracker_unordered.filter_state.C, tracker_ordered.filter_state.C
        )

    def test_update_with_clutter(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_2DCV

        measurementsUnorderedWithClutter = np.column_stack(
            (
                self.perfect_meas_ordered_2DCV[:, [1, 2]],
                np.array([2, 2]),
                self.perfect_meas_ordered_2DCV[:, 0].reshape(-1, 1),
            )
        )
        tracker.update_linear(
            measurementsUnorderedWithClutter, self.measurement_matrix_2DCV, np.eye(2)
        )

        priorMeans = np.column_stack([g.mu for g in self.gaussians_2DCV])
        posteriorMeansFirstUpdate = tracker.get_point_estimate()
        self.assertTrue(
            np.sum(np.linalg.norm(priorMeans[[0, 2], :] - np.array([[2], [2]]), axis=0))
            >= 10
        )
        self.assertTrue(
            np.sum(
                np.linalg.norm(
                    posteriorMeansFirstUpdate[[0, 2], :] - np.array([[2], [2]]), axis=0
                )
            )
            <= 8
        )

        measurementsUnorderedWithClutter = np.column_stack(
            (
                self.perfect_meas_ordered_2DCV[:, [1, 2]],
                -np.array([2, 2]),
                self.perfect_meas_ordered_2DCV[:, 0].reshape(-1, 1),
            )
        )
        for _ in range(15):
            tracker.update_linear(
                measurementsUnorderedWithClutter,
                self.measurement_matrix_2DCV,
                np.eye(2),
            )

        posteriorMeansAfterAllUpdates = tracker.get_point_estimate()
        self.assertTrue(
            np.sum(
                np.linalg.norm(
                    posteriorMeansAfterAllUpdates[[0, 2], :] + np.array([[2], [2]]),
                    axis=0,
                )
            )
            <= 8
        )
        self.assertTrue(
            np.sum(
                np.linalg.norm(
                    posteriorMeansFirstUpdate[[0, 2], :] + np.array([[2], [2]]), axis=0
                )
            )
            >= 10
        )
        self.assertTrue(
            np.sum(np.linalg.norm(priorMeans[[0, 2], :] - np.array([[2], [2]]), axis=0))
            >= 10
        )

    def test_gating(self):
        trackerWithGating = KernelSMEFilter()
        trackerNoGating = KernelSMEFilter()
        trackerWithGating.filter_state = self.gaussians_2DCV
        trackerNoGating.filter_state = self.gaussians_2DCV

        measurementsUnorderedWithClutter = np.column_stack(
            (
                self.perfect_meas_ordered_2DCV[:, [1, 2]],
                np.array([2, 2]),
                self.perfect_meas_ordered_2DCV[:, 0],
            )
        )
        # jscpd:ignore-start
        trackerWithGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            np.eye(2),
            0,
            np.zeros(2),
            1,
            True,
        )
        trackerNoGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            np.eye(2),
            0,
            np.zeros(2),
            1,
            False,
        )
        np.testing.assert_equal(trackerWithGating.x, trackerNoGating.x)
        np.testing.assert_equal(trackerWithGating.C, trackerNoGating.C)

        trackerNoGatingBackup = copy.deepcopy(trackerNoGating)
        measurementsUnorderedWithClutter = np.column_stack(
            (self.perfect_meas_ordered_2DCV[:, [1, 2, 0]], np.array([10, 10]))
        )
        trackerWithGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            np.eye(2),
            0,
            np.zeros(2),
            1,
            True,
        )
        trackerNoGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            np.eye(2),
            0,
            np.zeros(2),
            1,
            False,
        )
        self.assertFalse(np.allclose(trackerWithGating.x, trackerNoGating.x))
        self.assertFalse(np.allclose(trackerWithGating.C, trackerNoGating.C))

        trackerNoGating = trackerNoGatingBackup
        trackerNoGating.update_linear(
            self.perfect_meas_ordered_2DCV[:, [1, 2, 0]],
            self.measurement_matrix_2DCV,
            np.eye(2),
            0,
            np.zeros(2),
            1,
            False,
        )
        np.testing.assert_equal(trackerWithGating.x, trackerNoGating.x)
        np.testing.assert_equal(trackerWithGating.C, trackerNoGating.C)
        # jscpd:ignore-end

    def test_logging(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_2DCV
        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 1))
        self.assertEqual(np.size(tracker.posterior_estimates_over_time), 0)

        allGaussians = self.gaussians_2DCV

        measurements = self.perfect_meas_ordered_2DCV
        tracker.update_linear(measurements, self.measurement_matrix_2DCV, np.eye(2))

        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 1))
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 1))

        tracker.predict_linear(
            block_diag(np.array([[1, 1], [0, 1]]), np.array([[1, 1], [0, 1]])),
            np.eye(4),
            None,
        )

        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 2))
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 1))

        allGaussians = self.gaussians_2DCV
        perfectMeasOrdered = np.dot(
            self.measurement_matrix_2DCV, np.column_stack([g.mu for g in allGaussians])
        )
        measurements = perfectMeasOrdered
        tracker.update_linear(measurements, self.measurement_matrix_2DCV, np.eye(2))

        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 2))
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 2))
