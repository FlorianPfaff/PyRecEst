import copy
import unittest
from unittest.mock import patch

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import argsort, array, column_stack, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.gaussian_mixture_phd_filter import (
    GaussianMixturePHDFilter,
    GaussianMixturePHDState,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Currently only supported for the numpy backend",
)
class TestGaussianMixturePHDFilter(unittest.TestCase):
    def test_constructor_rejects_unsupported_backend(self):
        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "numpy backend"):
                GaussianMixturePHDFilter(
                    log_prior_estimates=False,
                    log_posterior_estimates=False,
                )

    def test_filter_state_roundtrip(self):
        component = GaussianDistribution(array([0.0, 0.0]), eye(2))
        tracker = GaussianMixturePHDFilter(
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = GaussianMixturePHDState([component], array([0.75]))

        self.assertEqual(len(tracker.filter_state.dists), 1)
        npt.assert_allclose(tracker.filter_state.w, array([0.75]))
        npt.assert_allclose(
            tracker.filter_state.dists[0].mu,
            array([0.0, 0.0]),
        )

    def test_predict_linear_scales_weights_and_adds_births(self):
        tracker = GaussianMixturePHDFilter(
            initial_components=[
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
            ],
            initial_weights=array([0.8]),
            birth_components=[
                GaussianDistribution(array([5.0, 5.0]), 2.0 * eye(2)),
            ],
            birth_weights=array([0.2]),
            survival_probability=0.9,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )

        tracker.predict_linear(eye(2), 0.1 * eye(2))
        state = tracker.filter_state

        self.assertEqual(len(state.dists), 2)
        npt.assert_allclose(state.w, array([0.72, 0.2]))
        npt.assert_allclose(state.dists[0].mu, array([0.0, 0.0]))
        npt.assert_allclose(state.dists[0].C, 1.1 * eye(2))
        npt.assert_allclose(state.dists[1].mu, array([5.0, 5.0]))
        npt.assert_allclose(state.dists[1].C, 2.0 * eye(2))

    def test_predict_linear_rejects_nonzero_mean_gaussian_system_noise(self):
        tracker = GaussianMixturePHDFilter(
            initial_components=[
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
            ],
            initial_weights=array([0.8]),
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )

        with self.assertRaisesRegex(ValueError, "zero mean"):
            tracker.predict_linear(
                eye(2),
                GaussianDistribution(array([1.0, 0.0]), 0.1 * eye(2)),
            )

    def test_update_linear_extracts_reasonable_point_estimate(self):
        tracker = GaussianMixturePHDFilter(
            initial_components=[
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
            ],
            initial_weights=array([0.9]),
            detection_probability=0.95,
            clutter_intensity=1e-4,
            extraction_threshold=0.3,
            merging_threshold=0.01,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )

        tracker.update_linear(
            array([[0.2], [-0.1]]),
            eye(2),
            0.1 * eye(2),
        )

        expected_mean = array([0.18181818, -0.09090909])
        npt.assert_allclose(
            tracker.get_point_estimate().reshape((-1,)),
            expected_mean,
            atol=5e-3,
        )
        self.assertEqual(tracker.get_number_of_targets(), 1)
        self.assertGreater(tracker.get_expected_number_of_targets(), 0.9)

    def test_update_is_independent_of_measurement_order(self):
        tracker_a = GaussianMixturePHDFilter(
            initial_components=[
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([10.0, 0.0]), eye(2)),
            ],
            initial_weights=array([0.9, 0.9]),
            detection_probability=0.95,
            clutter_intensity=1e-4,
            extraction_threshold=0.3,
            merging_threshold=0.5,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker_b = copy.deepcopy(tracker_a)

        measurements = column_stack(
            (
                array([0.1, 0.0]),
                array([10.1, -0.1]),
            )
        )

        tracker_a.update_linear(measurements, eye(2), 0.1 * eye(2))
        tracker_b.update_linear(measurements[:, ::-1], eye(2), 0.1 * eye(2))

        estimates_a = tracker_a.get_point_estimate()
        estimates_b = tracker_b.get_point_estimate()

        sort_indices_a = argsort(estimates_a[0])
        sort_indices_b = argsort(estimates_b[0])

        npt.assert_allclose(
            estimates_a[:, sort_indices_a],
            estimates_b[:, sort_indices_b],
            atol=1e-6,
        )

    def test_merge_combines_nearby_components(self):
        tracker = GaussianMixturePHDFilter(
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = GaussianMixturePHDState(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([0.1, 0.0]), eye(2)),
                GaussianDistribution(array([5.0, 5.0]), eye(2)),
            ],
            array([0.4, 0.3, 0.2]),
        )

        tracker.merge()

        self.assertEqual(len(tracker.filter_state.dists), 2)
        self.assertAlmostEqual(
            tracker.get_expected_number_of_targets(),
            0.9,
            places=9,
        )
