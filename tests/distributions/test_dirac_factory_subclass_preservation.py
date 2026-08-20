import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.circle.circular_dirac_distribution import (
    CircularDiracDistribution,
)
from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypertorus.hypertoroidal_dirac_distribution import (
    HypertoroidalDiracDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.distributions.se2_dirac_distribution import SE2DiracDistribution
from pyrecest.distributions.se3_dirac_distribution import SE3DiracDistribution


class _LinearDiracSubclass(LinearDiracDistribution):
    pass


class _CircularDiracSubclass(CircularDiracDistribution):
    pass


class _CircularGridSubclass(CircularGridDistribution):
    pass


class _HypertoroidalDiracSubclass(HypertoroidalDiracDistribution):
    pass


class _HypertoroidalGridSubclass(HypertoroidalGridDistribution):
    pass


class _SE2DiracSubclass(SE2DiracDistribution):
    pass


class _SE3DiracSubclass(SE3DiracDistribution):
    pass


class DiracFactorySubclassPreservationTest(unittest.TestCase):
    def test_linear_conversion_factory_preserves_requested_subclass(self):
        source = LinearDiracDistribution(array([0.0, 1.0]))

        converted = convert_distribution(
            source, _LinearDiracSubclass, n_particles=2
        )

        self.assertIsInstance(converted, _LinearDiracSubclass)

    def test_circular_conversion_factory_preserves_requested_subclass(self):
        source = CircularDiracDistribution(array([0.0, 1.0]))

        converted = convert_distribution(
            source, _CircularDiracSubclass, n_particles=2
        )

        self.assertIsInstance(converted, _CircularDiracSubclass)

    def test_circular_grid_conversion_factory_preserves_requested_subclass(self):
        source = VonMisesDistribution(0.3, 2.0)

        converted = convert_distribution(
            source, _CircularGridSubclass, no_of_gridpoints=9
        )

        self.assertIsInstance(converted, _CircularGridSubclass)

    def test_hypertoroidal_conversion_factory_preserves_requested_subclass(self):
        source = HypertoroidalDiracDistribution(
            array([[0.0, 0.5], [1.0, 1.5]])
        )

        converted = convert_distribution(
            source, _HypertoroidalDiracSubclass, n_particles=2
        )

        self.assertIsInstance(converted, _HypertoroidalDiracSubclass)

    def test_hypertoroidal_grid_factory_preserves_requested_subclass(self):
        source = HypertoroidalUniformDistribution(2)

        converted = convert_distribution(
            source, _HypertoroidalGridSubclass, n_grid_points=(3, 3)
        )

        self.assertIsInstance(converted, _HypertoroidalGridSubclass)

    def test_se2_conversion_factory_preserves_requested_subclass(self):
        source = SE2DiracDistribution(array([[0.0, 1.0, 2.0]]))

        converted = convert_distribution(source, _SE2DiracSubclass, n_particles=2)

        self.assertIsInstance(converted, _SE2DiracSubclass)

    def test_se3_conversion_factory_preserves_requested_subclass(self):
        source = SE3DiracDistribution(
            array([[1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])
        )

        converted = convert_distribution(source, _SE3DiracSubclass, n_particles=2)

        self.assertIsInstance(converted, _SE3DiracSubclass)


if __name__ == "__main__":
    unittest.main()
