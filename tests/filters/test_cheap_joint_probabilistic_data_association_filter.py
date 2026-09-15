# pylint: disable=protected-access,no-name-in-module,no-member
"""Cheap JPDA reference cases, normalization, and Gaussian-update contracts."""

import numpy as np
import numpy.testing as npt
import pytest

import pyrecest.backend
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import (
    CJPDAF,
    JPDAF,
    CheapJointProbabilisticDataAssociationFilter,
    CheapJPDAF,
    KalmanFilter,
)

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Cheap JPDAF is explicitly NumPy-only.",
)


def _log_weights(weights):
    weights = np.asarray(weights, dtype=float)
    result = np.full_like(weights, -np.inf)
    np.log(weights, out=result, where=weights > 0)
    return result


def _marginals(weights, tracker_class=CheapJPDAF):
    log_weights = _log_weights(weights)
    eligible = [np.flatnonzero(np.isfinite(row)).tolist() for row in log_weights]
    tracker = tracker_class()
    # P_D=0.5 and kappa=1 make likelihoods equal detection-to-miss odds.
    return tracker._compute_association_probabilities(
        log_weights, eligible, 0.5, np.ones(log_weights.shape[1])
    )[0]


def _tracker(tracker_class=CheapJPDAF, means=(-1.0, 1.0), **parameters):
    return tracker_class(
        [
            KalmanFilter(GaussianDistribution(np.array([mean, 0.0]), np.eye(2)))
            for mean in means
        ],
        association_param={
            "detection_probability": 0.9,
            "clutter_intensity": 0.01,
            "gating_distance_threshold": 100.0,
            **parameters,
        },
    )


def test_public_aliases():
    assert CheapJPDAF is CJPDAF is CheapJointProbabilisticDataAssociationFilter
    assert issubclass(CheapJPDAF, JPDAF)


def test_hand_computed_competition_and_complementary_miss():
    weights = np.array([[2.0, 3.0], [5.0, 7.0]])
    beta = _marginals(weights)
    expected = np.array([[2 / 11, 3 / 13], [5 / 15, 7 / 16]])
    npt.assert_allclose(beta[:, 1:], expected)
    npt.assert_allclose(beta[:, 0], 1 - expected.sum(axis=1))
    # The uncorrected Fitzgerald miss weight does not normalize these rows.
    assert np.all(1 / (1 + weights.sum(axis=1)) < beta[:, 0])


@pytest.mark.parametrize(
    "weights",
    [
        [[2.0, 3.0, 0.0]],
        [[2.0], [5.0], [0.0]],
        [[2.0, 3.0, 0.0, 0.0], [0.0, 0.0, 5.0, 7.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ],
)
def test_agrees_with_exact_jpda_in_simple_cases(weights):
    npt.assert_allclose(_marginals(weights), _marginals(weights, JPDAF), atol=1e-14)


def test_general_case_is_an_approximation_not_exact_enumeration():
    weights = np.ones((2, 2))
    npt.assert_allclose(_marginals(weights), [[0.5, 0.25, 0.25]] * 2)
    npt.assert_allclose(_marginals(weights, JPDAF), [[3 / 7, 2 / 7, 2 / 7]] * 2)


def test_randomized_formula_rows_and_measurement_exclusivity():
    rng = np.random.default_rng(117)
    for _ in range(100):
        weights = rng.lognormal(0, 3, (11, 13))
        weights[rng.random(weights.shape) < 0.5] = 0.0
        beta = _marginals(weights)
        denominator = (
            1
            + weights.sum(axis=1, keepdims=True)
            + weights.sum(axis=0, keepdims=True)
            - weights
        )
        npt.assert_allclose(beta[:, 1:], weights / denominator, atol=1e-14)
        npt.assert_allclose(beta.sum(axis=1), 1.0, atol=1e-14)
        assert np.all(np.isfinite(beta))
        assert np.all(beta >= 0.0)
        assert np.all(beta <= 1.0 + 1e-14)
        assert np.all(beta[:, 1:].sum(axis=0) <= 1.0 + 1e-14)
        assert np.all(beta[:, 1:][weights == 0.0] == 0.0)


def test_permutation_equivariance_of_soft_marginals():
    weights = np.array([[2.0, 0.0, 3.0], [5.0, 7.0, 0.1], [0.0, 1.0, 4.0]])
    tracks, measurements = [2, 0, 1], [1, 2, 0]
    expected = _marginals(weights)[tracks][:, [0, 2, 3, 1]]
    npt.assert_allclose(_marginals(weights[tracks][:, measurements]), expected)


def test_log_space_does_not_underflow_disconnected_components():
    log_weights = np.array([[1000.0, -np.inf], [-np.inf, 0.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        beta = CheapJPDAF._cheap_marginals(log_weights)
    npt.assert_allclose(beta, [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])


def test_tiny_missed_detection_probability_is_preserved():
    beta = CheapJPDAF._cheap_marginals(np.array([[700.0]]))
    assert beta[0, 0] > 0.0
    npt.assert_allclose(beta[0, 0], np.exp(-700.0), rtol=1e-13, atol=0)


def test_extreme_competition_and_empty_rows_stay_normalized():
    log_weights = np.array([[1000.0, -1000.0], [1000.0, -np.inf], [-np.inf, -np.inf]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        beta = CheapJPDAF._cheap_marginals(log_weights)
    npt.assert_allclose(beta, [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])


def test_greedy_diagnostic_is_feasible_but_not_claimed_to_be_map():
    # Track-order greedy chooses (0, 1), while the joint MAP event is (1, 0).
    log_weights = _log_weights([[10.0, 9.0], [100.0, 1.1]])
    npt.assert_array_equal(CheapJPDAF._greedy_assignment(log_weights), [0, 1])
    tracker = _tracker(means=(0.0, 0.0))
    beta, diagnostic = tracker.find_association_probabilities(
        np.zeros((2, 1)), np.eye(2), np.eye(2)
    )
    npt.assert_array_equal(diagnostic, [0, -1])
    npt.assert_array_equal(tracker.latest_greedy_association, diagnostic)
    npt.assert_allclose(tracker.latest_association_probabilities, beta)
    assert tracker.latest_map_association is None
    npt.assert_array_equal(
        tracker.find_association(np.zeros((2, 1)), np.eye(2), np.eye(2)), diagnostic
    )


def test_exact_event_limit_is_not_used(monkeypatch):
    measurements = np.zeros((2, 12))
    exact = _tracker(JPDAF, means=(0.0,) * 12, max_enumerated_events=2)
    with pytest.raises(RuntimeError, match="max_enumerated_events"):
        exact.find_association_probabilities(measurements, np.eye(2), np.eye(2))

    def no_enumeration(*_args, **_kwargs):
        raise AssertionError("Cheap JPDAF must not call the exact event solver")

    monkeypatch.setattr(JPDAF, "_compute_association_probabilities", no_enumeration)
    cheap = _tracker(means=(0.0,) * 12, max_enumerated_events=1)
    cheap.update_linear(measurements, np.eye(2), np.eye(2))
    npt.assert_allclose(cheap.latest_association_probabilities.sum(axis=1), 1.0)
    assert np.isfinite(cheap.get_point_estimate()).all()


def test_no_measurements_leave_state_and_covariance_unchanged():
    tracker = _tracker()
    means = tracker.get_point_estimate().copy()
    covariances = [state.C.copy() for state in tracker.filter_state]
    tracker.update_linear(np.empty((2, 0)), np.eye(2), np.eye(2))
    npt.assert_allclose(tracker.latest_association_probabilities, [[1.0], [1.0]])
    npt.assert_array_equal(tracker.latest_greedy_association, [-1, -1])
    npt.assert_allclose(tracker.get_point_estimate(), means)
    for state, covariance in zip(tracker.filter_state, covariances):
        npt.assert_allclose(state.C, covariance)


def test_empty_bank_clears_cached_diagnostics():
    tracker = _tracker()
    tracker.find_association_probabilities(np.zeros((2, 1)), np.eye(2), np.eye(2))
    # Isolate association caches from the inherited empty-bank history logger.
    tracker.log_prior_estimates = False
    tracker.filter_state = []
    with pytest.warns(UserWarning, match="zero targets"):
        beta, diagnostic = tracker.find_association_probabilities(
            np.zeros((2, 1)), np.eye(2), np.eye(2)
        )
    assert beta.shape == (0, 1)
    assert diagnostic.shape == (0,)
    assert tracker.latest_association_probabilities.shape == (0, 1)
    assert tracker.latest_greedy_association.shape == (0,)
    assert tracker.latest_map_association is None
    assert tracker._latest_posterior_hypotheses == []


def test_all_measurements_outside_gate_leave_priors_unchanged():
    tracker = _tracker(gating_distance_threshold=0.1)
    before = tracker.get_point_estimate().copy()
    with pytest.warns(UserWarning, match="gating threshold"):
        tracker.update_linear(np.full((2, 3), 100.0), np.eye(2), np.eye(2))
    npt.assert_allclose(tracker.latest_association_probabilities, [[1, 0, 0, 0]] * 2)
    npt.assert_allclose(tracker.get_point_estimate(), before)
    for state in tracker.filter_state:
        npt.assert_allclose(state.C, np.eye(2))


def test_heterogeneous_covariances_and_vector_clutter_match_exact_single_track():
    parameters = {"clutter_intensity": np.array([0.01, 0.04])}
    cheap, exact = (
        _tracker(cls, means=(0.0,), **parameters) for cls in (CheapJPDAF, JPDAF)
    )
    measurements = np.array([[-0.5, 1.0], [0.1, -0.2]])
    covariances = np.stack([0.2 * np.eye(2), 2 * np.eye(2)], axis=2)
    for tracker in (cheap, exact):
        tracker.update_linear(measurements, np.eye(2), covariances)
    npt.assert_allclose(
        cheap.latest_association_probabilities, exact.latest_association_probabilities
    )
    npt.assert_allclose(cheap.get_point_estimate(), exact.get_point_estimate())
    npt.assert_allclose(cheap.filter_state[0].C, exact.filter_state[0].C)


def test_mixture_update_includes_between_hypothesis_covariance():
    tracker = _tracker(means=(0.0,))
    measurements = np.array([[-2.0, 2.0], [0.0, 0.0]])
    tracker.update_linear(measurements, np.eye(2), np.eye(2))
    beta = tracker.latest_association_probabilities[0]
    means = [np.zeros(2), np.array([-1.0, 0.0]), np.array([1.0, 0.0])]
    covariances = [np.eye(2), 0.5 * np.eye(2), 0.5 * np.eye(2)]
    mean = sum(weight * value for weight, value in zip(beta, means))
    covariance = sum(
        weight * (cov + np.outer(value - mean, value - mean))
        for weight, value, cov in zip(beta, means, covariances)
    )
    npt.assert_allclose(tracker.filter_state[0].mu, mean, atol=1e-14)
    npt.assert_allclose(tracker.filter_state[0].C, covariance)
    assert covariance[0, 0] > 1.0
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_prediction_uses_existing_filter_bank():
    tracker = _tracker()
    before = tracker.get_point_estimate().copy()
    tracker.predict_linear(np.eye(2), 0.1 * np.eye(2))
    npt.assert_allclose(tracker.get_point_estimate(), before)
    for state in tracker.filter_state:
        npt.assert_allclose(state.C, 1.1 * np.eye(2))


@pytest.mark.parametrize("backend", ["jax", "pytorch"])
def test_unsupported_backends_fail_explicitly(monkeypatch, backend):
    tracker = _tracker()
    monkeypatch.setattr(pyrecest.backend, "__backend_name__", backend)
    with pytest.raises(NotImplementedError, match="numpy backend"):
        tracker.update_linear(np.zeros((2, 1)), np.eye(2), np.eye(2))
    with pytest.raises(NotImplementedError, match="numpy backend"):
        tracker.find_association_probabilities(np.zeros((2, 1)), np.eye(2), np.eye(2))


@pytest.mark.parametrize("clutter", [0.0, -1.0, np.nan, np.inf, [0.01, np.nan]])
def test_invalid_clutter_rejected(clutter):
    tracker = _tracker(clutter_intensity=clutter)
    with pytest.raises(ValueError, match="clutter_intensity"):
        tracker.find_association_probabilities(np.zeros((2, 2)), np.eye(2), np.eye(2))


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_detection_probability_rejected(probability):
    tracker = _tracker(detection_probability=probability)
    with pytest.raises(ValueError, match="detection_probability"):
        tracker.find_association_probabilities(np.zeros((2, 1)), np.eye(2), np.eye(2))


def test_invalid_covariance_shape_and_pairwise_costs_rejected():
    tracker = _tracker()
    measurements = np.zeros((2, 1))
    with pytest.raises(ValueError, match="cov_mats_meas must have shape"):
        tracker.find_association_probabilities(measurements, np.eye(2), np.ones((1, 1)))
    with pytest.raises(NotImplementedError, match="pairwise_cost_matrix"):
        tracker.update_linear(
            measurements, np.eye(2), np.eye(2), pairwise_cost_matrix=np.zeros((2, 1))
        )
