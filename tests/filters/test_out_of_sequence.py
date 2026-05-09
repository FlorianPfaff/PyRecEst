import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.filters import (
    EuclideanParticleFilter,
    FixedLagBuffer,
    KalmanFilter,
    MeasurementTimeBuffer,
    OutOfSequenceKalmanUpdater,
    OutOfSequenceParticleUpdater,
    retrodict_linear_gaussian,
)


class OutOfSequenceMeasurementTest(unittest.TestCase):
    def test_fixed_lag_buffer_orders_and_trims(self):
        buffer = FixedLagBuffer(max_lag=1.0)
        buffer.append(2.0, "b")
        buffer.append(1.0, "a")
        buffer.append(3.0, "c")

        self.assertEqual([item.value for item in buffer.items], ["b", "c"])
        self.assertTrue(buffer.is_out_of_sequence(2.5))
        self.assertTrue(buffer.is_within_lag(2.0))
        self.assertFalse(buffer.is_within_lag(1.5))
        self.assertEqual(buffer.latest_at_or_before(2.5).value, "b")

    def test_measurement_time_buffer_reports_oosm(self):
        buffer = MeasurementTimeBuffer(max_lag=2.0)
        record = buffer.add(5.0, array([1.0]), source="sensor-a")

        self.assertEqual(record.time, 5.0)
        self.assertEqual(record.metadata["source"], "sensor-a")
        self.assertTrue(buffer.is_out_of_sequence(4.0))
        self.assertTrue(buffer.is_within_lag(3.0))
        self.assertFalse(buffer.is_within_lag(2.9))

    def test_retrodict_linear_gaussian_inverts_square_transition(self):
        previous_mean, previous_covariance = retrodict_linear_gaussian(
            mean=array([4.0]),
            covariance=array([[8.0]]),
            system_matrix=array([[2.0]]),
            sys_input=array([1.0]),
        )

        self.assertTrue(allclose(previous_mean, array([1.5])))
        self.assertTrue(allclose(previous_covariance, array([[2.0]])))

    def test_kalman_oosm_replay_matches_chronological_processing(self):
        f_mat = array([[1.0]])
        q_mat = array([[0.25]])
        h_mat = array([[1.0]])
        r_mat = array([[1.0]])
        z1 = array([1.0])
        z2 = array([3.0])

        chronological = KalmanFilter((array([0.0]), array([[1.0]])))
        chronological.predict_linear(f_mat, q_mat)
        chronological.update_linear(z1, h_mat, r_mat)
        chronological.predict_linear(f_mat, q_mat)
        chronological.update_linear(z2, h_mat, r_mat)

        delayed = KalmanFilter((array([0.0]), array([[1.0]])))
        updater = OutOfSequenceKalmanUpdater(delayed, initial_time=0.0, max_lag=2.0)
        updater.predict_linear(1.0, f_mat, q_mat)
        updater.predict_linear(2.0, f_mat, q_mat)
        updater.update_linear(2.0, z2, h_mat, r_mat)
        result = updater.update_linear(
            1.0,
            z1,
            h_mat,
            r_mat,
            return_diagnostics=True,
        )

        self.assertTrue(result.out_of_sequence)
        self.assertGreater(result.replayed_event_count, 0)
        self.assertIsNotNone(result.diagnostics)
        self.assertTrue(
            allclose(delayed.get_point_estimate(), chronological.get_point_estimate())
        )
        self.assertTrue(allclose(delayed.filter_state.C, chronological.filter_state.C))

    def test_kalman_oosm_rejects_measurements_outside_lag(self):
        updater = OutOfSequenceKalmanUpdater(
            KalmanFilter((array([0.0]), array([[1.0]]))),
            initial_time=0.0,
            max_lag=0.5,
        )
        updater.predict_linear(1.0, array([[1.0]]), array([[0.1]]))
        updater.predict_linear(2.0, array([[1.0]]), array([[0.1]]))

        with self.assertRaises(ValueError):
            updater.update_linear(
                1.0,
                array([1.0]),
                array([[1.0]]),
                array([[1.0]]),
            )

    def test_particle_oosm_replay_matches_deterministic_chronological_processing(self):
        def shift(particles):
            return particles + 1.0

        def likelihood(measurement, particles):
            residual = particles[:, 0] - measurement[0]
            return 1.0 / (1.0 + residual * residual)

        chronological = self._particle_filter()
        chronological.predict_nonlinear(shift, function_is_vectorized=True)
        chronological.update_nonlinear_using_likelihood(
            likelihood,
            measurement=array([1.0]),
        )
        chronological.predict_nonlinear(shift, function_is_vectorized=True)
        chronological.update_nonlinear_using_likelihood(
            likelihood,
            measurement=array([3.0]),
        )

        delayed = self._particle_filter()
        updater = OutOfSequenceParticleUpdater(delayed, initial_time=0.0, max_lag=2.0)
        updater.predict_nonlinear(1.0, shift)
        updater.predict_nonlinear(2.0, shift)
        updater.update_nonlinear_using_likelihood(
            2.0,
            likelihood,
            measurement=array([3.0]),
        )
        result = updater.update_nonlinear_using_likelihood(
            1.0,
            likelihood,
            measurement=array([1.0]),
        )

        self.assertTrue(result.out_of_sequence)
        self.assertTrue(allclose(delayed.filter_state.d, chronological.filter_state.d))
        self.assertTrue(allclose(delayed.filter_state.w, chronological.filter_state.w))

    @staticmethod
    def _particle_filter():
        particle_filter = EuclideanParticleFilter(3, 1)
        particle_filter.set_resampling_criterion(lambda _: False)
        particle_filter.filter_state = LinearDiracDistribution(
            array([[0.0], [1.0], [2.0]]),
            array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        )
        return particle_filter


if __name__ == "__main__":
    unittest.main()
