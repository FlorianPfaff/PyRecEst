import unittest

from pyrecest.backend import __backend_name__ as backend_name
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.custom_hypercylindrical_distribution import (
    CustomHypercylindricalDistribution,
)
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_distribution import (
    HypercylindricalStateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


class _HypercylindricalSubdivisionSubclass(
    HypercylindricalStateSpaceSubdivisionDistribution
):
    pass


@unittest.skipIf(backend_name != "numpy", reason="Factory uses SciPy integration")
class HypercylindricalSubdivisionFactorySubclassTest(unittest.TestCase):
    def setUp(self):
        circular = VonMisesDistribution(array(0.0), array(1.5))
        linear = GaussianDistribution(array([0.5]), array([[0.4]]))

        def density(x):
            return circular.pdf(x[:, 0]) * linear.pdf(x[:, 1:])

        self.density = density
        self.source = CustomHypercylindricalDistribution(density, 1, 1)

    def test_inherited_from_function_preserves_subclass(self):
        converted = _HypercylindricalSubdivisionSubclass.from_function(
            self.density, 5, 1, 1
        )

        self.assertIs(type(converted), _HypercylindricalSubdivisionSubclass)

    def test_conversion_gateway_preserves_requested_subclass(self):
        converted = convert_distribution(
            self.source,
            _HypercylindricalSubdivisionSubclass,
            no_of_grid_points=5,
        )

        self.assertIs(type(converted), _HypercylindricalSubdivisionSubclass)


if __name__ == "__main__":
    unittest.main()
