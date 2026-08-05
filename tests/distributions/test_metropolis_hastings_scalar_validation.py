import unittest

import numpy as np
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
    _validate_integer_sample_parameter,
)


class VectorLogDensityDistribution(AbstractManifoldSpecificDistribution):
    def __init__(self):
        super().__init__(dim=1)

    @property
    def input_dim(self):
        return 1

    def get_manifold_size(self):
        return 1.0

    def pdf(self, xs):
        return array(1.0)

    def ln_pdf(self, xs):
        return array([0.0, -1.0])

    def mean(self):
        return array([0.0])


class MetropolisHastingsScalarValidationTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="This regression test targets the non-JAX MH implementation.",
    )
    def test_sample_metropolis_hastings_rejects_vector_log_density(self):
        distribution = VectorLogDensityDistribution()

        def proposal(x):
            return x

        with self.assertRaisesRegex(ValueError, "scalar"):
            distribution.sample_metropolis_hastings(
                n=1,
                burn_in=0,
                skipping=1,
                proposal=proposal,
                start_point=array([0.0]),
            )

    def test_sample_metropolis_hastings_rejects_masked_chain_controls(self):
        distribution = VectorLogDensityDistribution()
        controls = {
            "n": np.ma.array(1, mask=True),
            "burn_in": np.ma.array(0, mask=True),
            "skipping": np.ma.array(1, mask=True),
        }

        for parameter, masked_value in controls.items():
            kwargs = {"n": 1, "burn_in": 0, "skipping": 1}
            kwargs[parameter] = masked_value
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, "integer"):
                    distribution.sample_metropolis_hastings(**kwargs)

    def test_unmasked_scalar_wrapper_remains_valid(self):
        self.assertEqual(
            _validate_integer_sample_parameter(
                np.ma.array(3, mask=False), "n", minimum=1
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
