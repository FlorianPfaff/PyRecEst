import unittest

from pyrecest.backend import array
from pyrecest.diagnostics import ParticleFilterResult


class _OverflowFloat:
    def __float__(self):
        raise OverflowError("simulated overflow")


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

    def test_summary_statistics_flatten_backend_arrays(self):
        result = ParticleFilterResult(
            estimates=[],
            effective_sample_size=array([3.0, 4.0, 5.0]),
            resampled=array([False, True, True]),
            particle_spread=array([0.3, 0.2, 0.1]),
            block_effective_sample_size=array([[3.0, 2.0], [4.0, 1.0]]),
        )

        summary = result.summary_statistics()

        self.assertEqual(result.resampling_count, 2)
        self.assertAlmostEqual(result.resampling_fraction, 2.0 / 3.0)
        self.assertAlmostEqual(summary["mean_effective_sample_size"], 4.0)
        self.assertAlmostEqual(summary["min_effective_sample_size"], 3.0)
        self.assertAlmostEqual(summary["final_effective_sample_size"], 5.0)
        self.assertAlmostEqual(summary["mean_particle_spread"], 0.2)
        self.assertAlmostEqual(summary["final_particle_spread"], 0.1)
        self.assertAlmostEqual(summary["mean_block_effective_sample_size"], 2.5)
        self.assertAlmostEqual(summary["min_block_effective_sample_size"], 1.0)

    def test_summary_statistics_ignore_malformed_values(self):
        result = ParticleFilterResult(
            estimates=[],
            effective_sample_size=["bad", [2.0, 4.0]],
            resampled="not-a-bool-sequence",
            particle_spread=["bad", 0.5],
        )

        summary = result.summary_statistics()

        self.assertEqual(result.resampling_count, 0)
        self.assertEqual(result.resampling_fraction, 0.0)
        self.assertAlmostEqual(summary["mean_effective_sample_size"], 3.0)
        self.assertAlmostEqual(summary["final_effective_sample_size"], 4.0)
        self.assertAlmostEqual(summary["mean_particle_spread"], 0.5)

    def test_summary_statistics_ignore_overflowing_numeric_values(self):
        result = ParticleFilterResult(
            estimates=[],
            effective_sample_size=[_OverflowFloat(), [2.0, 4.0]],
            resampled=[],
            particle_spread=[_OverflowFloat(), 0.5],
        )

        summary = result.summary_statistics()

        self.assertAlmostEqual(summary["mean_effective_sample_size"], 3.0)
        self.assertAlmostEqual(summary["final_effective_sample_size"], 4.0)
        self.assertAlmostEqual(summary["mean_particle_spread"], 0.5)


if __name__ == "__main__":
    unittest.main()
