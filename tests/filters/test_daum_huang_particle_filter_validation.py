import unittest

import numpy as np
from pyrecest.filters import EDHParticleFlowFilter


class DaumHuangParticleFlowValidationTest(unittest.TestCase):
    def test_constructor_rejects_nonfinite_jitter(self):
        for invalid_jitter in (np.nan, np.inf, -np.inf):
            with self.subTest(jitter=invalid_jitter):
                with self.assertRaisesRegex(ValueError, "jitter"):
                    EDHParticleFlowFilter(n_particles=2, dim=1, jitter=invalid_jitter)


if __name__ == "__main__":
    unittest.main()
