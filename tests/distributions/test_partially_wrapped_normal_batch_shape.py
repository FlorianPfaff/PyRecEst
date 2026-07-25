import unittest

import numpy.testing as npt
from pyrecest.backend import array, reshape
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class PartiallyWrappedNormalBatchShapeTest(unittest.TestCase):
    def test_pdf_preserves_nested_batch_shape(self):
        dist = PartiallyWrappedNormalDistribution(
            array([5.0, 1.0]),
            array([[2.0, 1.0], [1.0, 1.0]]),
            bound_dim=1,
        )
        xs = array(
            [
                [[0.5, 1.0], [1.5, 0.5], [2.0, 2.0]],
                [[0.25, 1.25], [1.25, 0.75], [2.25, 1.75]],
            ]
        )

        values = dist.pdf(xs)
        flat_values = reshape(dist.pdf(reshape(xs, (-1, 2))), (2, 3))

        self.assertEqual(values.shape, (2, 3))
        npt.assert_allclose(values, flat_values, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
