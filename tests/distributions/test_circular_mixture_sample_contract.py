import unittest

from pyrecest.backend import array, zeros
from pyrecest.distributions import CircularMixture
from pyrecest.distributions.circle.abstract_circular_distribution import (
    AbstractCircularDistribution,
)


class _MalformedCircularDistribution(AbstractCircularDistribution):
    def __init__(self):
        super().__init__()

    def pdf(self, xs):
        return zeros(array(xs).shape)

    def sample(self, n):
        del n
        return array([0.1, 0.2])


class TestCircularMixtureSampleContract(unittest.TestCase):
    def test_rejects_component_returning_multiple_values_for_one_sample(self):
        mixture = CircularMixture(
            [_MalformedCircularDistribution()],
            array([1.0]),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"component sample output must have shape \(1,\), got \(2,\)",
        ):
            mixture.sample(1)


if __name__ == "__main__":
    unittest.main()
