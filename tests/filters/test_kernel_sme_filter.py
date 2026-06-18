import copy
import math
import unittest
from unittest.mock import patch

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized
from pyrecest.backend import (
    arange,
    array,
    column_stack,
    diag,
    eye,
    hstack,
    linalg,
    zeros,
)
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.kernel_sme_filter import KernelSMEFilter


class TestKernelSMEFilter(unittest.TestCase):
    measurement_matrix_2DCV = array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    gaussians_2DCV = [
        GaussianDistribution(zeros(4), diag(arange(1.0, 5.0))),
        GaussianDistribution(arange(1.0, 5.0), 2 * eye(4)),
        GaussianDistribution(-arange(1.0, 5.0), diag(arange(4.0, 0.0, -1.0))),
    ]

    measurement_matrix_3DCV = array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    gaussians_3DCV = [
        GaussianDistribution(zeros(6), diag(arange(1.0, 7.0))),
        GaussianDistribution(arange(1.0, 7.0), 2 * eye(6)),
        GaussianDistribution(-arange(1.0, 7.0), diag(arange(6.0, 0.0, -1.0))),
    ]

    measurement_matrix_3DCA = array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    gaussians_3DCA = [
        GaussianDistribution(zeros(9), diag(arange(1.0, 10.0))),
        GaussianDistribution(arange(1.0, 10.0), 2 * eye(9)),
        GaussianDistribution(-arange(1.0, 10.0), diag(arange(9.0, 0.0, -1.0))),
    ]

    scenario_names = ["2DCV", "3DCV", "3DCA"]

    def setUp(self):
        # Set up your test case variables here
        self.initial_priors = [
            GaussianDistribution(array([0.0, 0.0]), array([[1.0, 0.0], [0.0, 1.0]])),
            GaussianDistribution(array([5.0, 5.0]), array([[1.0, 0.0], [0.0, 1.0]])),
        ]
        self.system_matrix = array([[1.0, 0.1], [0.0, 1.0]])
        self.sys_noise_cov = array([[0.01, 0.0], [0.0, 0.01]])
        self.measurement_noise_cov = array([[0.1, 0.0], [0.0, 0.1]])
        self.measurements = array([[1.0, 1.1], [5.5, 4.9]])
        self.perfect_meas_ordered_2DCV = self.measurement_matrix_2DCV @ column_stack(
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
        npt.assert_allclose(
            tracker.filter_state.mu,
            hstack(
                (
                    self.gaussians_2DCV[0].mu,
                    self.gaussians_2DCV[1].mu,
                    self.gaussians_2DCV[2].mu,
                )
            ),
        )
        tracker.predict_linear(
            array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            eye(4),
        )
        npt.assert_allclose(
            tracker.get_point_estimate(),
            column_stack(
                (zeros((4,)), array([3.0, 2.0, 7.0, 4.0]), -array([3.0, 2.0, 7.0, 4.0]))
            ),
        )
        npt.assert_allclose(
            tracker.C,
            linalg.block_diag(
                array([[3.0, 2], [2, 2]]),
                array([[7, 4], [4, 4]]),
                array([[4, 2], [2, 2]]),
                array([[4, 2], [2, 2]]),
                array([[7, 3], [3, 3]]),
                array([[3, 1], [1, 1]]),
            )
            + eye(12),
        )

    def test_predict_linear_rejects_inputs(self):
        tracker = KernelSMEFilter(self.initial_priors)

        with self.assertRaisesRegex(NotImplementedError, "Inputs"):
            tracker.predict_linear(
                self.system_matrix,
                self.sys_noise_cov,
                inputs=array([1.0, 0.0]),
            )

    def testPredictLinearAllSameMatsNoInputs3DCV(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_3DCV
        tracker.predict_linear(
            array(
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            eye(6),
        )
        npt.assert_allclose(
            tracker.get_point_estimate(),
            column_stack(
                (
                    zeros((6, 1)),
                    array([3.0, 2.0, 7.0, 4.0, 11.0, 6.0]),
                    -array([3.0, 2.0, 7.0, 4.0, 11.0, 6.0]),
                )
            ),
        )
        # jscpd:ignore-start
        npt.assert_allclose(
            tracker.C,
            linalg.block_diag(
                array([[3, 2], [2, 2]]),
                array([[7, 4], [4, 4]]),
                array([[11, 6], [6, 6]]),
                array([[4, 2], [2, 2]]),
                array([[4, 2], [2, 2]]),
                array([[4, 2], [2, 2]]),
                array([[11, 5], [5, 5]]),
                array([[7, 3], [3, 3]]),
                array([[3, 1], [1, 1]]),
            )
            + eye(18),
        )
        # jscpd:ignore-end

    @parameterized.expand(
        [
            (gaussians_2DCV, measurement_matrix_2DCV),
            (gaussians_3DCV, measurement_matrix_3DCV),
            (gaussians_3DCA, measurement_matrix_3DCA),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_independent_of_order(self, gaussians, measurement_matrix):
        tracker_ordered = KernelSMEFilter(gaussians)
        tracker_unordered = copy.deepcopy(tracker_ordered)

        perfect_meas_ordered = measurement_matrix @ column_stack(
            [g.mu for g in gaussians]
        )

        tracker_ordered.update_linear(
            perfect_meas_ordered,
            measurement_matrix,
            eye(perfect_meas_ordered.shape[0]),
        )
        tracker_unordered.update_linear(
            perfect_meas_ordered[:, ::-1],
            measurement_matrix,
            eye(perfect_meas_ordered.shape[0]),
        )

        # Check if the state estimates are close enough
        npt.assert_allclose(
            tracker_unordered.get_point_estimate(), tracker_ordered.get_point_estimate()
        )
        npt.assert_allclose(
            tracker_unordered.filter_state.C, tracker_ordered.filter_state.C
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_with_clutter(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_2DCV

        measurementsUnorderedWithClutter = column_stack(
            (
                self.perfect_meas_ordered_2DCV[:, [1, 2]],
                array([2, 2]),
                self.perfect_meas_ordered_2DCV[:, 0].reshape(-1, 1),
            )
        )
        tracker.update_linear(
            measurementsUnorderedWithClutter, self.measurement_matrix_2DCV, eye(2)
        )

        priorMeans = column_stack([g.mu for g in self.gaussians_2DCV])
        posteriorMeansFirstUpdate = tracker.get_point_estimate()
        self.assertTrue(
            sum(linalg.norm(priorMeans[[0, 2], :] - array([[2], [2]]), axis=0)) >= 10
        )
        self.assertTrue(
            sum(
                linalg.norm(
                    posteriorMeansFirstUpdate[[0, 2], :] - array([[2], [2]]), axis=0
                )
            )
            <= 8
        )

        measurementsUnorderedWithClutter = column_stack(
            (
                self.perfect_meas_ordered_2DCV[:, [1, 2]],
                -array([2, 2]),
                self.perfect_meas_ordered_2DCV[:, 0].reshape(-1, 1),
            )
        )
        for _ in range(15):
            tracker.update_linear(
                measurementsUnorderedWithClutter,
                self.measurement_matrix_2DCV,
                eye(2),
            )

        posteriorMeansAfterAllUpdates = tracker.get_point_estimate()
        self.assertTrue(
            sum(
                linalg.norm(
                    posteriorMeansAfterAllUpdates[[0, 2], :] + array([[2], [2]]),
                    axis=0,
                )
            )
            <= 8
        )
        self.assertTrue(
            sum(
                linalg.norm(
                    posteriorMeansFirstUpdate[[0, 2], :] + array([[2], [2]]), axis=0
                )
            )
            >= 10
        )
        self.assertTrue(
            sum(linalg.norm(priorMeans[[0, 2], :] - array([[2], [2]]), axis=0)) >= 10
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_gating(self):
        trackerWithGating = KernelSMEFilter()
        trackerNoGating = KernelSMEFilter()
        trackerWithGating.filter_state = self.gaussians_2DCV
        trackerNoGating.filter_state = self.gaussians_2DCV

        measurementsUnorderedWithClutter = column_stack(
            (
                self.perfect_meas_ordered_2DCV[:, [1, 2]],
                array([2, 2]),
                self.perfect_meas_ordered_2DCV[:, 0],
            )
        )
        # jscpd:ignore-start
        trackerWithGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            eye(2),
            0,
            zeros(2),
            1,
            True,
        )
        trackerNoGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            eye(2),
            0,
            zeros(2),
            1,
            False,
        )
        npt.assert_equal(trackerWithGating.x, trackerNoGating.x)
        npt.assert_equal(trackerWithGating.C, trackerNoGating.C)

        trackerNoGatingBackup = copy.deepcopy(trackerNoGating)
        measurementsUnorderedWithClutter = column_stack(
            (self.perfect_meas_ordered_2DCV[:, [1, 2, 0]], array([10, 10]))
        )
        trackerWithGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            eye(2),
            0,
            zeros(2),
            1,
            True,
        )
        trackerNoGating.update_linear(
            measurementsUnorderedWithClutter,
            self.measurement_matrix_2DCV,
            eye(2),
            0,
            zeros(2),
            1,
            False,
        )
        npt.assert_allclose(trackerWithGating.x, trackerNoGating.x, rtol=0.001)
        npt.assert_allclose(trackerWithGating.C, trackerNoGating.C, rtol=0.001)

        trackerNoGating = trackerNoGatingBackup
        trackerNoGating.update_linear(
            self.perfect_meas_ordered_2DCV[:, [1, 2, 0]],
            self.measurement_matrix_2DCV,
            eye(2),
            0,
            zeros(2),
            1,
            False,
        )
        npt.assert_equal(trackerWithGating.x, trackerNoGating.x)
        npt.assert_equal(trackerWithGating.C, trackerNoGating.C)
        # jscpd:ignore-end

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_gating_rejecting_all_measurements_keeps_prior(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_2DCV
        prior_x = copy.deepcopy(tracker.x)
        prior_c = copy.deepcopy(tracker.C)

        far_measurements = array([[100.0, -100.0], [100.0, -100.0]])

        tracker.update_linear(
            far_measurements,
            self.measurement_matrix_2DCV,
            eye(2),
            0,
            zeros(2),
            1,
            True,
            gating_threshold=1.0,
        )

        npt.assert_allclose(tracker.x, prior_x)
        npt.assert_allclose(tracker.C, prior_c)
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 1))

    def test_update_rejects_gating_threshold_without_gating(self):
        tracker = KernelSMEFilter(self.initial_priors)

        with self.assertRaisesRegex(ValueError, "gating"):
            tracker.update_linear(
                self.measurements,
                eye(2),
                self.measurement_noise_cov,
                enable_gating=False,
                gating_threshold=1.0,
            )

    def test_gen_test_points_rejects_pytorch_backend(self):
        with patch.object(pyrecest.backend, "__backend_name__", "pytorch"):
            with self.assertRaisesRegex(NotImplementedError, "backend"):
                KernelSMEFilter.gen_test_points(array([[0.0]]), 1.0)

    def test_calc_pseudo_meas_rejects_jax_backend(self):
        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "backend"):
                KernelSMEFilter.calc_pseudo_meas(
                    array([[0.0]]), array([[0.0]]), 1.0
                )

    def test_calc_moments_rejects_bad_lambda_multimeas_length(self):
        with self.assertRaisesRegex(ValueError, "lambdaMultimeas"):
            KernelSMEFilter.calc_moments(
                array([0.0, 0.0, 1.0, 1.0]),
                eye(4),
                eye(2),
                eye(2),
                array([[0.0], [0.0]]),
                1.0,
                2,
                lambdaMultimeas=array([1.0, 2.0, 3.0]),
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_logging(self):
        tracker = KernelSMEFilter()
        tracker.filter_state = self.gaussians_2DCV
        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 1))
        self.assertEqual(math.prod(tracker.posterior_estimates_over_time.shape), 0)

        allGaussians = self.gaussians_2DCV

        measurements = self.perfect_meas_ordered_2DCV
        tracker.update_linear(measurements, self.measurement_matrix_2DCV, eye(2))

        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 1))
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 1))

        tracker.predict_linear(
            linalg.block_diag(
                array([[1.0, 1.0], [0.0, 1.0]]), array([[1.0, 1.0], [0.0, 1.0]])
            ),
            eye(4),
            None,
        )

        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 2))
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 1))

        allGaussians = self.gaussians_2DCV
        # measurement matrix is meas_dim x state_dim. Hence, we need to stack the state dimension first (unlike our normal convention)
        perfectMeasOrdered = self.measurement_matrix_2DCV @ column_stack(
            [g.mu for g in allGaussians]
        )
        measurements = perfectMeasOrdered
        tracker.update_linear(measurements, self.measurement_matrix_2DCV, eye(2))

        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 2))
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 2))
