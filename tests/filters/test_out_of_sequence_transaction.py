import unittest

from pyrecest.filters import OutOfSequenceParticleUpdater


class _DiagnosticsOnlyParticleFilter:
    def __init__(self):
        self.filter_state = {"updates": 0}

    def update_nonlinear_using_likelihood(self, likelihood, measurement=None):
        self.filter_state = {"updates": self.filter_state["updates"] + 1}
        return likelihood(measurement, None)


class OutOfSequenceTransactionTest(unittest.TestCase):
    @staticmethod
    def _accepted(_measurement, _particles):
        return {"accepted": True}

    def test_failed_event_is_rolled_back_and_does_not_poison_replay(self):
        updater = OutOfSequenceParticleUpdater(
            _DiagnosticsOnlyParticleFilter(), initial_time=0.0
        )

        def fail(_measurement, _particles):
            raise RuntimeError("failed update")

        with self.assertRaisesRegex(RuntimeError, "failed update"):
            updater.update_nonlinear_using_likelihood(1.0, fail)

        self.assertEqual(updater.event_count, 0)
        self.assertEqual(updater.current_time, 0.0)
        self.assertEqual(updater.filter_state, {"updates": 0})

        updater.update_nonlinear_using_likelihood(1.0, self._accepted)
        result = updater.update_nonlinear_using_likelihood(0.5, self._accepted)

        self.assertTrue(result.out_of_sequence)
        self.assertEqual(updater.event_count, 2)
        self.assertEqual(updater.filter_state, {"updates": 2})

    def test_invalid_diagnostics_roll_back_successful_filter_mutation(self):
        updater = OutOfSequenceParticleUpdater(
            _DiagnosticsOnlyParticleFilter(), initial_time=0.0
        )

        with self.assertRaisesRegex(ValueError, "accepted diagnostic"):
            updater.update_nonlinear_using_likelihood(
                1.0,
                lambda _measurement, _particles: {"accepted": "maybe"},
            )

        self.assertEqual(updater.event_count, 0)
        self.assertEqual(updater.current_time, 0.0)
        self.assertEqual(updater.filter_state, {"updates": 0})


if __name__ == "__main__":
    unittest.main()
