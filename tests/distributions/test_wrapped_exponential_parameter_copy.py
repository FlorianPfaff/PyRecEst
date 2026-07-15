import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.circle.wrapped_exponential_distribution import (
    WrappedExponentialDistribution,
)


class WrappedExponentialParameterCopyTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="JAX arrays are immutable and cannot expose caller-side aliasing.",
    )
    def test_constructor_owns_mutable_rate_storage(self):
        rate = array(2.0)
        distribution = WrappedExponentialDistribution(rate)
        evaluation_points = array([0.0, 0.5, 1.0])
        expected_pdf = to_numpy(distribution.pdf(evaluation_points)).copy()

        rate[...] = 4.0

        npt.assert_allclose(to_numpy(rate), 4.0)
        npt.assert_allclose(to_numpy(distribution.lambda_), 2.0)
        npt.assert_allclose(
            to_numpy(distribution.pdf(evaluation_points)),
            expected_pdf,
        )


if __name__ == "__main__":
    unittest.main()
