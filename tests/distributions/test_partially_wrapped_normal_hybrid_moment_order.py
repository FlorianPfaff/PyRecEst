import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class PartiallyWrappedNormalHybridMomentOrderTest(unittest.TestCase):
    def test_groups_all_cosines_before_all_sines(self):
        mu = array([0.2, 1.1, 3.0])
        covariance = array(
            [
                [0.4, 0.0, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        distribution = PartiallyWrappedNormalDistribution(
            mu,
            covariance,
            bound_dim=2,
        )

        expected = np.array(
            [
                np.cos(0.2) * np.exp(-0.4 / 2.0),
                np.cos(1.1) * np.exp(-0.8 / 2.0),
                np.sin(0.2) * np.exp(-0.4 / 2.0),
                np.sin(1.1) * np.exp(-0.8 / 2.0),
                3.0,
            ]
        )

        npt.assert_allclose(distribution.hybrid_moment(), expected)


if __name__ == "__main__":
    unittest.main()
