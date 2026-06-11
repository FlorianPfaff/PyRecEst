import unittest

from pyrecest.backend import array
from pyrecest.distributions.hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_mixture import ToroidalMixture


class ToroidalMixtureTest(unittest.TestCase):
    def test_constructor_rejects_invalid_distribution_list(self):
        invalid_cases = (
            ([], array([]), "at least one"),
            (
                [CustomHypertoroidalDistribution(lambda xs: xs[:, 0], 2)],
                array([1.0]),
                "toroidal distributions",
            ),
        )

        for dists, weights, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ToroidalMixture(dists, weights)


if __name__ == "__main__":
    unittest.main()
