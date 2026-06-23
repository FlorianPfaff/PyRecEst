import math

import numpy as np
from pyrecest.filters.sparse_second_order_grid import sparse_second_order_grid_evidence


def test_sparse_second_order_grid_matches_dense_tiny_hmm_and_evidence_only():
    log_likelihood = np.log(
        np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.8, 0.1],
                [0.6, 0.3, 0.1],
            ],
            dtype=float,
        )
    )
    initial = np.array(
        [
            [0.6, 0.2, 0.1],
            [0.3, 0.6, 0.3],
            [0.1, 0.2, 0.6],
        ],
        dtype=float,
    )
    initial /= initial.sum(axis=0, keepdims=True)

    transition = {}
    for time_index in (1, 2):
        rows = np.zeros((3, 3, 3), dtype=float)
        for prev in range(3):
            for curr in range(3):
                weights = np.array([0.2, 0.5, 0.3], dtype=float)
                weights += 0.10 * (np.arange(3) == curr)
                weights += 0.05 * (np.arange(3) == prev)
                weights /= weights.sum()
                rows[:, prev, curr] = weights
        transition[time_index] = rows

    def init(scaled):
        prev = []
        curr = []
        values = []
        counts = []
        prior = np.ones(3) / 3.0
        for src in range(3):
            for dst in range(3):
                prev.append(src)
                curr.append(dst)
                values.append(
                    prior[src] * scaled[0, src] * initial[dst, src] * scaled[1, dst]
                )
            counts.append(3)
        return np.asarray(prev), np.asarray(curr), np.asarray(values), counts

    def row(prev, curr, transition_index):
        return np.arange(3), transition[transition_index][:, prev, curr]

    def cache_key(prev, curr, transition_index):
        # Deliberately do not include transition_index; both transition rows are equal
        # in this toy model, so the cache must be safe and log evidence must remain exact.
        return prev, curr

    result = sparse_second_order_grid_evidence(
        log_likelihood,
        init,
        row,
        transition_cache_key_builder=cache_key,
        return_smoothed=True,
    )
    expected = _dense_second_order_log_evidence(log_likelihood, initial, transition)

    assert abs(result.log_marginal_likelihood - expected) < 1e-12
    assert result.smoothed_log_probabilities is not None
    np.testing.assert_allclose(
        np.exp(result.smoothed_log_probabilities).sum(axis=1), 1.0
    )
    np.testing.assert_allclose(np.exp(result.terminal_log_probabilities).sum(), 1.0)
    assert result.diagnostics["transition_row_cache_hits"] > 0
    assert result.diagnostics["transition_row_cache_misses"] > 0

    evidence_only = sparse_second_order_grid_evidence(
        log_likelihood,
        init,
        row,
        transition_cache_key_builder=cache_key,
        return_smoothed=False,
    )
    assert evidence_only.smoothed_log_probabilities is None
    assert (
        evidence_only.diagnostics["backward_transition_rows"] == "skipped_evidence_only"
    )
    assert (
        abs(evidence_only.log_marginal_likelihood - result.log_marginal_likelihood)
        < 1e-12
    )
    np.testing.assert_allclose(
        np.exp(evidence_only.terminal_log_probabilities).sum(),
        1.0,
    )


def test_sparse_second_order_grid_validates_transition_rows():
    log_likelihood = np.zeros((3, 2), dtype=float)

    def init(scaled):
        return np.array([0]), np.array([0]), np.array([1.0]), [1]

    def bad_row(prev, curr, transition_index):
        return np.array([0, 1]), np.array([1.0])

    try:
        sparse_second_order_grid_evidence(log_likelihood, init, bad_row)
    except ValueError as exc:
        assert "matching shapes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("bad transition row should fail")


def test_sparse_second_order_grid_validates_initial_pair_weights_and_shape():
    log_likelihood = np.zeros((3, 2), dtype=float)

    def row(prev, curr, transition_index):
        del prev, curr, transition_index
        return np.array([0]), np.array([1.0])

    def negative_weight_init(scaled):
        del scaled
        return np.array([0, 1]), np.array([0, 1]), np.array([2.0, -1.0]), [1, 1]

    _assert_raises_value_error_containing(
        "initial pair weights",
        lambda: sparse_second_order_grid_evidence(
            log_likelihood, negative_weight_init, row
        ),
    )

    def nonfinite_weight_init(scaled):
        del scaled
        return np.array([0, 1]), np.array([0, 1]), np.array([1.0, np.nan]), [1, 1]

    _assert_raises_value_error_containing(
        "initial pair weights",
        lambda: sparse_second_order_grid_evidence(
            log_likelihood, nonfinite_weight_init, row
        ),
    )

    def nonvector_init(scaled):
        del scaled
        return (
            np.array([[0], [1]]),
            np.array([[0], [1]]),
            np.array([[0.5], [0.5]]),
            [1, 1],
        )

    _assert_raises_value_error_containing(
        "one-dimensional",
        lambda: sparse_second_order_grid_evidence(log_likelihood, nonvector_init, row),
    )


def _assert_raises_value_error_containing(expected_message, callback):
    try:
        callback()
    except ValueError as exc:
        assert expected_message in str(exc)
    else:  # pragma: no cover
        raise AssertionError(f"expected ValueError containing {expected_message!r}")


def _dense_second_order_log_evidence(log_likelihood, initial_transition, transition):
    likelihood = np.exp(log_likelihood)
    n_time, n_states = likelihood.shape
    pair = np.zeros((n_states, n_states), dtype=float)
    for prev in range(n_states):
        for curr in range(n_states):
            pair[prev, curr] = (
                (1.0 / n_states)
                * likelihood[0, prev]
                * initial_transition[curr, prev]
                * likelihood[1, curr]
            )
    scale = float(pair.sum())
    logp = math.log(scale)
    pair /= scale
    for time_index in range(2, n_time):
        next_pair = np.zeros_like(pair)
        for prev in range(n_states):
            for curr in range(n_states):
                mass = pair[prev, curr]
                if mass <= 0.0:
                    continue
                for dst in range(n_states):
                    next_pair[curr, dst] += (
                        mass
                        * transition[time_index - 1][dst, prev, curr]
                        * likelihood[time_index, dst]
                    )
        scale = float(next_pair.sum())
        logp += math.log(scale)
        pair = next_pair / scale
    return logp
