import numpy as np
import pytest
from pyrecest.evidence import EvidenceComputationMode, resolve_evidence_computation_mode
from pyrecest.filters import sparse_second_order_grid_evidence


def test_evidence_computation_mode_resolves_boolean_and_string_aliases():
    full = resolve_evidence_computation_mode(return_smoothed=True)
    fast = resolve_evidence_computation_mode("evidence-only")

    assert full.mode == "full_smoothing"
    assert full.return_smoothed
    assert not full.evidence_only_requested
    assert fast.mode == "evidence_only"
    assert not fast.return_smoothed
    assert fast.evidence_only_requested
    assert fast.to_diagnostics()["evidence_computation_mode"] == "evidence_only"


def test_evidence_computation_mode_accepts_numpy_bool_flags():
    full = resolve_evidence_computation_mode(return_smoothed=np.bool_(True))
    fast = resolve_evidence_computation_mode(return_smoothed=np.bool_(False))

    assert full.mode == "full_smoothing"
    assert fast.mode == "evidence_only"


def test_evidence_computation_mode_rejects_invalid_boolean_strings():
    with pytest.raises(ValueError, match="return_smoothed"):
        resolve_evidence_computation_mode(return_smoothed="definitely")
    with pytest.raises(ValueError, match="return_smoothed"):
        resolve_evidence_computation_mode("full_smoothing", return_smoothed="false")


def test_evidence_computation_mode_rejects_inconsistent_flags():
    with pytest.raises(ValueError, match="evidence_only"):
        EvidenceComputationMode(mode="evidence_only", return_smoothed=True)
    with pytest.raises(ValueError, match="full_smoothing"):
        EvidenceComputationMode(mode="full_smoothing", return_smoothed=False)
    with pytest.raises(ValueError, match="unknown"):
        resolve_evidence_computation_mode("posterior-only")


def test_evidence_computation_mode_rejects_truthy_non_bool_return_smoothed():
    for return_smoothed in ("false", "true", 0, 1, np.array([False])):
        with pytest.raises(ValueError, match="return_smoothed must be a bool"):
            resolve_evidence_computation_mode(return_smoothed=return_smoothed)


def test_resolver_rejects_conflicting_explicit_mode_and_boolean_flag():
    with pytest.raises(ValueError, match="mode and return_smoothed"):
        resolve_evidence_computation_mode("evidence_only", return_smoothed=True)
    with pytest.raises(ValueError, match="mode and return_smoothed"):
        resolve_evidence_computation_mode(
            EvidenceComputationMode.full_smoothing(), return_smoothed=False
        )


def test_sparse_second_order_grid_accepts_evidence_only_mode_object():
    log_likelihood = np.log(
        np.array(
            [
                [0.7, 0.3],
                [0.4, 0.6],
                [0.5, 0.5],
            ],
            dtype=float,
        )
    )

    def initial_pair_initializer(scaled):
        prev = np.array([0, 0, 1, 1])
        curr = np.array([0, 1, 0, 1])
        prior = np.array([0.5, 0.5])
        values = prior[prev] * 0.5 * scaled[0, prev] * scaled[1, curr]
        return prev, curr, values, [2, 2]

    def transition_row_builder(prev: int, curr: int, transition_index: int):
        del prev, transition_index
        if curr == 0:
            return np.array([0, 1]), np.array([0.8, 0.2])
        return np.array([0, 1]), np.array([0.3, 0.7])

    full = sparse_second_order_grid_evidence(
        log_likelihood,
        initial_pair_initializer,
        transition_row_builder,
        evidence_mode=EvidenceComputationMode.full_smoothing(),
    )
    fast = sparse_second_order_grid_evidence(
        log_likelihood,
        initial_pair_initializer,
        transition_row_builder,
        evidence_mode=EvidenceComputationMode.evidence_only(),
    )

    assert fast.smoothed_log_probabilities is None
    assert full.smoothed_log_probabilities is not None
    assert full.log_marginal_likelihood == pytest.approx(
        fast.log_marginal_likelihood, abs=1e-12
    )
    np.testing.assert_allclose(np.exp(fast.terminal_log_probabilities).sum(), 1.0)
    assert fast.diagnostics["evidence_computation_mode"] == "evidence_only"
    assert fast.diagnostics["evidence_only"] == 1
    assert fast.diagnostics["backward_transition_rows"] == "skipped_evidence_only"
    assert fast.diagnostics["smoothed_posterior_returned"] == 0
    assert fast.diagnostics["terminal_posterior_returned"] == 1
