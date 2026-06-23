import unittest

import pyrecest.backend
from pyrecest.backend import array, ones, zeros
from pyrecest.distributions.hypersphere_subset.bayesian_complex_watson_mixture_model import (
    BayesianComplexWatsonMixtureModel,
)


def _observations():
    return array([[1.0 + 0.0j], [0.0 + 0.0j]])


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Bayesian complex Watson fitting is not supported on JAX.",
)
class BayesianComplexWatsonMixtureValidationTest(unittest.TestCase):
    def test_estimate_posterior_requires_parameter_schema(self):
        params = BayesianComplexWatsonMixtureModel.parameters_default(2, 1)
        del params["prior"]["alpha"]

        with self.assertRaisesRegex(ValueError, "prior.alpha"):
            BayesianComplexWatsonMixtureModel.estimate_posterior(
                _observations(), params
            )

    def test_estimate_posterior_rejects_non_hermitian_initial_B(self):
        params = BayesianComplexWatsonMixtureModel.parameters_default(2, 1)
        params["initial"]["B"] = ones((2, 2, 1), dtype=complex) * (1.0 + 1.0j)

        with self.assertRaisesRegex(ValueError, "Hermitian"):
            BayesianComplexWatsonMixtureModel.estimate_posterior(
                _observations(), params
            )

    def test_estimate_posterior_rejects_saliencies_length_mismatch(self):
        params = BayesianComplexWatsonMixtureModel.parameters_default(2, 1)
        params["prior"]["saliencies"] = array([1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "saliencies"):
            BayesianComplexWatsonMixtureModel.estimate_posterior(
                _observations(), params
            )

    def test_fit_default_rejects_large_feature_dimension(self):
        observations = zeros((100, 1), dtype=complex)

        with self.assertRaisesRegex(ValueError, "D < 100"):
            BayesianComplexWatsonMixtureModel.fit_default(observations, 1)

    def test_fit_default_rejects_non_matrix_observations(self):
        observations = zeros((3,), dtype=complex)

        with self.assertRaisesRegex(ValueError, "shape"):
            BayesianComplexWatsonMixtureModel.fit_default(observations, 1)


if __name__ == "__main__":
    unittest.main()
