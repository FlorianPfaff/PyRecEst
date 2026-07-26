import unittest

from pyrecest.filters import (
    FixedLagBuffer,
    MeasurementTimeBuffer,
    OutOfSequenceParticleUpdater,
)


class _DummyParticleFilter:
    def __init__(self):
        self.filter_state = {"updates": 0}


class OutOfSequenceMaxLagValidationTest(unittest.TestCase):
    def test_rejects_nonfinite_max_lag_across_public_entry_points(self):
        constructors = {
            "fixed_lag_buffer": lambda max_lag: FixedLagBuffer(max_lag=max_lag),
            "measurement_time_buffer": lambda max_lag: MeasurementTimeBuffer(
                max_lag=max_lag
            ),
            "particle_updater": lambda max_lag: OutOfSequenceParticleUpdater(
                _DummyParticleFilter(), max_lag=max_lag
            ),
        }

        for max_lag in (float("nan"), float("inf"), -float("inf")):
            for name, constructor in constructors.items():
                with self.subTest(max_lag=max_lag, constructor=name):
                    with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                        constructor(max_lag)


if __name__ == "__main__":
    unittest.main()
