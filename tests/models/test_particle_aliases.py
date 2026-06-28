import unittest

from pyrecest.models.likelihood import DensityTransitionModel
from pyrecest.models.particle import (
    DensityTransitionModel as ParticleDensityTransitionModel,
)


class TestParticleAliases(unittest.TestCase):
    def test_density_transition_alias_matches_canonical_model(self):
        self.assertIs(ParticleDensityTransitionModel, DensityTransitionModel)


if __name__ == "__main__":
    unittest.main()
