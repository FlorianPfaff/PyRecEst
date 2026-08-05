from __future__ import annotations

import numpy as np
import pytest

from pyrecest.tracking import (
    GaussianMixtureMeasurementFactor,
    balance_mixture_responsibilities,
    blend_mixture_responsibilities_with_uniform,
)


def test_isotropic_squared_responsibilities_match_direct_formula() -> None:
    means = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    sigmas = np.asarray([1.0, 2.0])
    log_priors = np.log(np.asarray([0.75, 0.25]))
    factor = (
        GaussianMixtureMeasurementFactor.from_isotropic_standard_deviations(
            means,
            sigmas,
            log_weights=log_priors,
            loss="squared",
        )
    )

    result = factor.evaluate(np.asarray([0.5, 0.0]))
    expected_log = log_priors - 0.5 * np.asarray(
        [0.5**2, (1.5 / 2.0) ** 2]
    )
    expected = np.exp(expected_log - np.max(expected_log))
    expected /= expected.sum()

    np.testing.assert_allclose(result.responsibilities, expected)
    np.testing.assert_allclose(result.residual_norms, [0.5, 1.5])
    np.testing.assert_allclose(
        result.mahalanobis_distances,
        [0.5, 0.75],
    )


def test_huber_cost_limits_outlier_penalty() -> None:
    factor = (
        GaussianMixtureMeasurementFactor.from_isotropic_standard_deviations(
            [[0.0], [10.0]],
            [1.0, 1.0],
            loss="huber",
            huber_delta=2.0,
        )
    )

    result = factor.evaluate([0.0])

    np.testing.assert_allclose(result.component_costs, [0.0, 18.0])
    assert result.responsibilities[0] > 0.999999


def test_full_covariance_and_observation_matrix_are_supported() -> None:
    factor = GaussianMixtureMeasurementFactor(
        component_means=[[1.0, 0.0], [0.0, 1.0]],
        component_covariances=[
            np.diag([4.0, 1.0]),
            np.diag([1.0, 9.0]),
        ],
        observation_matrix=[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    result = factor.evaluate([1.0, 7.0, 0.0])

    assert factor.state_dimension == 3
    np.testing.assert_allclose(
        result.mahalanobis_distances,
        [0.0, np.hypot(1.0, 1.0 / 3.0)],
    )


def test_moment_match_includes_noise_and_between_component_spread() -> None:
    factor = (
        GaussianMixtureMeasurementFactor.from_isotropic_standard_deviations(
            [[0.0, 0.0], [2.0, 0.0]],
            [1.0, 3.0],
        )
    )

    moment = factor.moment_match([0.25, 0.75])

    np.testing.assert_allclose(moment.mean, [1.5, 0.0])
    expected_covariance = np.diag([7.75, 7.0])
    np.testing.assert_allclose(moment.covariance, expected_covariance)
    assert moment.isotropic_variance == pytest.approx(7.375)


def test_responsibility_balancing_preserves_within_group_ratios() -> None:
    weights = np.asarray([0.6, 0.3, 0.1])
    balanced = balance_mixture_responsibilities(
        weights,
        ["raw", "raw", "translated"],
        1.0,
    )

    assert balanced[:2].sum() == pytest.approx(0.5)
    assert balanced[2] == pytest.approx(0.5)
    assert balanced[0] / balanced[1] == pytest.approx(2.0)


def test_uniform_blend_and_validation() -> None:
    blended = blend_mixture_responsibilities_with_uniform(
        [1.0, 0.0],
        0.2,
    )
    np.testing.assert_allclose(blended, [0.9, 0.1])

    with pytest.raises(ValueError, match="positive definite"):
        GaussianMixtureMeasurementFactor(
            [[0.0]],
            [[[0.0]]],
        )
    with pytest.raises(ValueError, match="masked"):
        GaussianMixtureMeasurementFactor.from_isotropic_standard_deviations(
            np.ma.array([[0.0]], mask=[[True]]),
            [1.0],
        )
