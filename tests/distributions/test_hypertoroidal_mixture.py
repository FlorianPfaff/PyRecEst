import unittest

from pyrecest.distributions.hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import (
    HypertoroidalMixture,
)
from pyrecest.distributions.nonperiodic.custom_linear_distribution import (
    CustomLinearDistribution,
)


class HypertoroidalMixtureTest(unittest.TestCase):
    def test_constructor_rejects_nonhypertoroidal_components(self):
        linear_dist = CustomLinearDistribution(lambda xs: xs[:, 0], 1)

        with self.assertRaisesRegex(TypeError, "hypertoroidal"):
            HypertoroidalMixture([linear_dist])

    def test_to_circular_mixture_rejects_multidimensional_mixture(self):
        dist = CustomHypertoroidalDistribution(lambda xs: xs[:, 0], 2)
        mixture = HypertoroidalMixture([dist])

        with self.assertRaisesRegex(ValueError, "dim == 1"):
            mixture.to_circular_mixture()

    def test_to_toroidal_mixture_rejects_wrong_dimension(self):
        dist = CustomHypertoroidalDistribution(lambda xs: xs, 1)
        mixture = HypertoroidalMixture([dist])

        with self.assertRaisesRegex(ValueError, "dim == 2"):
            mixture.to_toroidal_mixture()


if __name__ == "__main__":
    unittest.main()
