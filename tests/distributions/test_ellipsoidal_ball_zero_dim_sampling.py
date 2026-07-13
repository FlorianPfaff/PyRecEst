import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, zeros
from pyrecest.distributions import EllipsoidalBallUniformDistribution


class TestEllipsoidalBallZeroDimSampling(unittest.TestCase):
    def test_sample_returns_one_empty_coordinate_row_per_draw(self):
        dist = EllipsoidalBallUniformDistribution(array([]), zeros((0, 0)))

        samples = dist.sample(3)

        self.assertEqual(samples.shape, (3, 0))


if __name__ == "__main__":
    unittest.main()
