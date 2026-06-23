import unittest

import pyrecest.backend
from pyrecest.backend import zeros
from pyrecest.distributions.hypersphere_subset.bayesian_complex_watson_mixture_model import (
    BayesianComplexWatsonMixtureModel,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Bayesian complex Watson fitting is not supported on JAX.",
)
class BayesianComplexWatsonDefaultValidationTest(unittest.TestCase):
    def test_fit_default_rejects_feature_dimension_limit(self):
        Z = zeros((100, 1), dtype=complex)

        with self.assertRaisesRegex(ValueError, "D < 100"):
            BayesianComplexWatsonMixtureModel.fit_default(Z, K=1)


if __name__ == "__main__":
    unittest.main()
