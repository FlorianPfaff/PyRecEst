import unittest

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.gaussian_mixture_phd_filter import GaussianMixturePHDFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Currently only supported for the numpy backend",
)
class TestGaussianMixturePHDFilterDimensions(unittest.TestCase):
    def test_constructor_rejects_mismatched_birth_dimension(self):
        with self.assertRaisesRegex(
            ValueError, "Birth components must have dimension 2"
        ):
            GaussianMixturePHDFilter(
                initial_components=[GaussianDistribution(array([0.0, 0.0]), eye(2))],
                initial_weights=array([0.8]),
                birth_components=[
                    GaussianDistribution(array([5.0, 5.0, 0.0]), eye(3)),
                ],
                birth_weights=array([0.2]),
                log_prior_estimates=False,
                log_posterior_estimates=False,
            )

    def test_set_birth_model_rejects_mismatched_dimension(self):
        tracker = GaussianMixturePHDFilter(
            initial_components=[GaussianDistribution(array([0.0, 0.0]), eye(2))],
            initial_weights=array([0.8]),
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )

        with self.assertRaisesRegex(
            ValueError, "Birth components must have dimension 2"
        ):
            tracker.set_birth_model(
                [GaussianDistribution(array([5.0, 5.0, 0.0]), eye(3))],
                array([0.2]),
            )

    def test_predict_linear_rejects_mismatched_temporary_birth_dimension(self):
        tracker = GaussianMixturePHDFilter(
            initial_components=[GaussianDistribution(array([0.0, 0.0]), eye(2))],
            initial_weights=array([0.8]),
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )

        with self.assertRaisesRegex(
            ValueError, "Birth components must have dimension 2"
        ):
            tracker.predict_linear(
                eye(2),
                0.1 * eye(2),
                birth_components=[
                    GaussianDistribution(array([5.0, 5.0, 0.0]), eye(3)),
                ],
                birth_weights=array([0.2]),
            )
