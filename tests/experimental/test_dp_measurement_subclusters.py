import numpy as np
import pytest
from pyrecest.experimental.dp_measurement_subclusters import (
    DPMeasurementSubclusterConfig,
    active_subcluster_responsibilities,
    dp_measurement_subset_log_likelihood,
    fit_dp_measurement_subclusters,
    score_dp_measurement_partition,
)


def test_separated_measurements_open_two_subclusters():
    measurements = np.array(
        [
            [-5.0, 0.0],
            [-5.2, 0.1],
            [5.0, 0.0],
            [5.1, -0.1],
        ],
        dtype=float,
    )
    config = DPMeasurementSubclusterConfig(
        concentration=1.0,
        measurement_variance=0.05,
        prior_variance=100.0,
    )

    result = fit_dp_measurement_subclusters(measurements, config)

    assert result.num_subclusters == 2
    assert result.assignments.tolist() == [0, 0, 1, 1]
    means = result.posterior_means[np.argsort(result.posterior_means[:, 0])]
    np.testing.assert_allclose(means[:, 0], np.array([-5.1, 5.05]), atol=0.05)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(4))


def test_max_subclusters_forces_reuse():
    measurements = np.array([[-5.0, 0.0], [5.0, 0.0]], dtype=float)
    config = DPMeasurementSubclusterConfig(
        concentration=10.0,
        measurement_variance=0.01,
        prior_variance=100.0,
        max_subclusters=1,
    )

    result = fit_dp_measurement_subclusters(measurements, config)

    assert result.num_subclusters == 1
    assert result.assignments.tolist() == [0, 0]


def test_partition_score_prefers_separate_far_blobs():
    measurements = np.array(
        [
            [-5.0, 0.0],
            [-5.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
        ],
        dtype=float,
    )
    config = DPMeasurementSubclusterConfig(
        concentration=1.0,
        measurement_variance=0.02,
        prior_variance=100.0,
    )

    single_score = score_dp_measurement_partition(
        measurements, np.array([0, 0, 0, 0]), config
    )
    split_score = score_dp_measurement_partition(
        measurements, np.array([0, 0, 1, 1]), config
    )

    assert split_score > single_score


def test_subset_likelihood_matches_fit_result():
    measurements = np.array([[0.0], [0.2], [0.1]], dtype=float)
    config = DPMeasurementSubclusterConfig(concentration=0.5, measurement_variance=0.2)

    result = fit_dp_measurement_subclusters(measurements, config)

    assert dp_measurement_subset_log_likelihood(measurements, config) == pytest.approx(
        result.log_predictive_likelihood
    )


def test_empty_measurement_set_is_valid():
    result = fit_dp_measurement_subclusters(np.empty((0, 2)))

    assert result.num_subclusters == 0
    assert result.assignments.shape == (0,)
    assert result.responsibilities.shape == (0, 0)
    assert result.log_predictive_likelihood == 0.0


def test_active_responsibilities_can_be_recomputed():
    measurements = np.array([[-1.0], [-0.9], [1.0]], dtype=float)
    result = fit_dp_measurement_subclusters(
        measurements,
        DPMeasurementSubclusterConfig(
            concentration=1.0,
            measurement_variance=0.01,
            prior_variance=10.0,
        ),
    )

    responsibilities = active_subcluster_responsibilities(measurements, result.atoms)

    assert responsibilities.shape == result.responsibilities.shape
    np.testing.assert_allclose(
        responsibilities.sum(axis=1), np.ones(measurements.shape[0])
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("concentration", 0.0),
        ("concentration", np.nan),
        ("measurement_variance", -1.0),
        ("prior_variance", np.inf),
        ("max_subclusters", 0),
        ("max_subclusters", True),
    ],
)
def test_config_rejects_invalid_values(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        DPMeasurementSubclusterConfig(**{keyword: value})


def test_rejects_invalid_measurement_inputs():
    with pytest.raises(ValueError, match="shape"):
        fit_dp_measurement_subclusters(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        fit_dp_measurement_subclusters(np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError, match="prior_mean"):
        fit_dp_measurement_subclusters(
            np.array([[0.0, 1.0]]),
            DPMeasurementSubclusterConfig(prior_mean=np.array([0.0])),
        )


def test_partition_score_rejects_mismatched_assignments():
    with pytest.raises(ValueError, match="assignments"):
        score_dp_measurement_partition(np.zeros((2, 1)), np.array([0]))
