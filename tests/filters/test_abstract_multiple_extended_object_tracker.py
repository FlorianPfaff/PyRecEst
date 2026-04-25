# pylint: disable=duplicate-code,too-many-arguments,too-many-positional-arguments
import unittest

import numpy.testing as npt
from pyrecest.backend import array, diag, zeros
from pyrecest.filters import (
    AbstractMultipleExtendedObjectTracker,
    ExtendedObjectAssociationResult,
    ExtendedObjectEstimate,
    MultipleExtendedObjectStepResult,
)


class _DummyMultipleExtendedObjectTracker(AbstractMultipleExtendedObjectTracker):
    def __init__(self):
        super().__init__(
            log_prior_extents=True,
            log_posterior_extents=True,
            log_prior_measurement_rates=True,
            log_posterior_measurement_rates=True,
            log_cardinality=True,
            log_associations=True,
        )
        self.predict_calls = []
        self.update_calls = []
        self.objects = [
            ExtendedObjectEstimate(
                label="a",
                kinematics=array([1.0, 2.0]),
                extent=diag(array([4.0, 1.0])),
                existence_probability=0.8,
                measurement_rate=3.0,
                status="confirmed",
            ),
            ExtendedObjectEstimate(
                label="b",
                kinematics=array([5.0, 6.0]),
                extent=diag(array([9.0, 4.0])),
                existence_probability=0.6,
                measurement_rate=1.5,
                status="tentative",
            ),
        ]

    def predict(  # pylint: disable=too-many-positional-arguments
        self,
        dt=None,
        dynamic_model=None,
        process_noise=None,
        survival_probability=None,
        birth_model=None,
        **kwargs,
    ):
        self.predict_calls.append(
            {
                "dt": dt,
                "dynamic_model": dynamic_model,
                "process_noise": process_noise,
                "survival_probability": survival_probability,
                "birth_model": birth_model,
                "kwargs": kwargs,
            }
        )

    def update(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        measurements,
        measurement_model=None,
        meas_noise_cov=None,
        detection_probability=None,
        clutter_model=None,
        measurement_partitions=None,
        sensor_state=None,
        **kwargs,
    ):
        self.update_calls.append(
            {
                "measurements": measurements,
                "measurement_model": measurement_model,
                "meas_noise_cov": meas_noise_cov,
                "detection_probability": detection_probability,
                "clutter_model": clutter_model,
                "measurement_partitions": measurement_partitions,
                "sensor_state": sensor_state,
                "kwargs": kwargs,
            }
        )
        association = ExtendedObjectAssociationResult(
            object_to_measurement_indices={"a": [0, 1]},
            clutter_indices=[2],
            selected_partition=[[0, 1], [2]],
            log_likelihood=-1.25,
        )
        return MultipleExtendedObjectStepResult(
            estimates=self.get_object_estimates(),
            association=association,
            expected_number_of_objects=self.get_expected_number_of_targets(),
        )

    def get_object_estimates(
        self,
        extraction_threshold=None,
        max_objects=None,
        confirmed_only=None,
    ):
        estimates = list(self.objects)
        if confirmed_only:
            estimates = [
                estimate for estimate in estimates if estimate.status == "confirmed"
            ]
        if extraction_threshold is not None:
            estimates = [
                estimate
                for estimate in estimates
                if estimate.existence_probability is None
                or estimate.existence_probability >= extraction_threshold
            ]
        if max_objects is not None:
            estimates = estimates[:max_objects]
        return estimates

    def get_contour_points(self, n, labels=None, scaling_factor=1.0, **kwargs):
        del kwargs
        selected_labels = self.get_track_labels() if labels is None else labels
        return {label: scaling_factor * zeros((n, 2)) for label in selected_labels}


class AbstractMultipleExtendedObjectTrackerTest(unittest.TestCase):
    def test_vectorized_and_structured_estimates_are_available(self):
        tracker = _DummyMultipleExtendedObjectTracker()

        point_estimate = tracker.get_point_estimate()
        self.assertEqual(point_estimate.shape, (6, 2))
        npt.assert_array_equal(
            point_estimate[:, 0], array([1.0, 2.0, 4.0, 0.0, 0.0, 1.0])
        )
        npt.assert_array_equal(
            tracker.get_point_estimate(flatten_vector=True), point_estimate.flatten()
        )
        npt.assert_array_equal(
            tracker.get_point_estimate(include_extent=False),
            array([[1.0, 5.0], [2.0, 6.0]]),
        )

        self.assertEqual(tracker.get_number_of_targets(), 2)
        self.assertEqual(tracker.get_track_labels(), ["a", "b"])
        self.assertEqual(tracker.get_track_labels(confirmed_only=True), ["a"])
        self.assertEqual(tracker.get_measurement_rate_estimates(), [3.0, 1.5])
        self.assertAlmostEqual(tracker.get_expected_number_of_targets(), 1.4)

    def test_step_returns_associations_and_records_histories(self):
        tracker = _DummyMultipleExtendedObjectTracker()
        measurements = array([[1.0, 1.2, 10.0], [2.0, 2.1, 10.0]])

        result = tracker.step(
            measurements,
            predict_kwargs={"dt": 0.5},
            update_kwargs={"measurement_partitions": [[0, 1], [2]]},
        )

        self.assertEqual(tracker.predict_calls[0]["dt"], 0.5)
        self.assertEqual(
            tracker.update_calls[0]["measurement_partitions"], [[0, 1], [2]]
        )
        self.assertEqual(
            result.association.object_to_measurement_indices, {"a": [0, 1]}
        )
        self.assertEqual(result.association.clutter_indices, [2])
        self.assertIs(tracker.latest_step_result, result)

        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 1))
        self.assertEqual(tracker.posterior_estimates_over_time.shape, (12, 1))
        self.assertEqual(tracker.prior_extents_over_time.shape, (8, 1))
        self.assertEqual(tracker.posterior_extents_over_time.shape, (8, 1))
        npt.assert_array_equal(
            tracker.prior_measurement_rates_over_time, array([[3.0], [1.5]])
        )
        self.assertEqual(tracker.cardinality_over_time, [1.4])
        self.assertEqual(tracker.associations_over_time[0].clutter_indices, [2])

    def test_default_contour_contract_uses_labels(self):
        tracker = _DummyMultipleExtendedObjectTracker()

        contours = tracker.get_contour_points(5, labels=["a"], scaling_factor=2.0)

        self.assertEqual(set(contours.keys()), {"a"})
        self.assertEqual(contours["a"].shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
