import unittest

from pyrecest.diagnostics import ParticleFilterResult


class ParticleFilterResultTest(unittest.TestCase):
    def test_mapping_compatibility_and_aliases(self):
        result = ParticleFilterResult(
            estimates=[1, 2],
            effective_sample_size=[3.0, 4.0],
            resampled=[False, True],
            particle_spread=[0.1, 0.2],
        )

        self.assertEqual(result["effective_sample_size"], [3.0, 4.0])
        self.assertEqual(result.ess_history, [3.0, 4.0])
        self.assertEqual(result.resampling_flags, [False, True])
        result["source"] = "test"
        self.assertEqual(result.metadata["source"], "test")


if __name__ == "__main__":
    unittest.main()
