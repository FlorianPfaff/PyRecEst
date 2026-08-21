import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)


class _LinearBoxParticleSubclass(LinearBoxParticleDistribution):
    pass


class LinearBoxParticleFactorySubclassPreservationTest(unittest.TestCase):
    def test_conversion_factory_preserves_requested_subclass(self):
        source = LinearDiracDistribution(array([[1.0, 2.0]]))

        converted = convert_distribution(
            source,
            _LinearBoxParticleSubclass,
            n_particles=3,
            box_half_width=0.25,
        )

        self.assertIsInstance(converted, _LinearBoxParticleSubclass)


if __name__ == "__main__":
    unittest.main()
