"""Regression tests for atomic IDKF prediction and update failures."""

import pytest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye
from pyrecest.filters.information_form_distributed_kalman_filter import IdkfNode


def _snapshot_idkf_state(node):
    return (
        node.global_information_state.copy(),
        {
            key: contribution.copy()
            for key, contribution in node.contribution_bank.items()
        },
        set(node.seen_contribution_ids),
    )


def _assert_idkf_state_unchanged(node, snapshot):
    expected_state, expected_bank, expected_seen_ids = snapshot
    actual_state = node.global_information_state

    assert allclose(actual_state.Y, expected_state.Y)
    if expected_state.transform_to_end is None:
        assert actual_state.transform_to_end is None
    else:
        assert allclose(actual_state.transform_to_end, expected_state.transform_to_end)
    assert actual_state.epoch == expected_state.epoch
    assert actual_state.operation_count == expected_state.operation_count
    assert actual_state.operation_hash == expected_state.operation_hash

    assert node.contribution_bank.keys() == expected_bank.keys()
    for key, expected_contribution in expected_bank.items():
        actual_contribution = node.contribution_bank[key]
        assert allclose(actual_contribution.y, expected_contribution.y)
        assert actual_contribution.epoch == expected_contribution.epoch
        assert (
            actual_contribution.operation_count == expected_contribution.operation_count
        )
        assert (
            actual_contribution.operation_hash == expected_contribution.operation_hash
        )
    assert node.seen_contribution_ids == expected_seen_ids


def test_failed_prediction_preserves_all_idkf_state():
    covariances = (eye(2), eye(2))
    node = IdkfNode.from_local_gaussian(
        1,
        (array([1.0, 2.0]), covariances[0]),
        covariances,
    )
    remote = IdkfNode.from_local_gaussian(
        2,
        (array([3.0, 4.0]), covariances[1]),
        covariances,
    )
    node.receive_contribution(remote.export_contribution())
    snapshot = _snapshot_idkf_state(node)

    with pytest.raises(ValueError, match="input_by_node"):
        node.predict_linear(
            array([[1.0, 0.0], [0.0, 2.0]]),
            eye(2),
            sys_input=array([0.5, 1.0]),
        )

    _assert_idkf_state_unchanged(node, snapshot)


def test_failed_update_preserves_all_idkf_state():
    node = IdkfNode.from_local_gaussian(
        1,
        (array([1.0]), eye(1)),
        (eye(1),),
        measurement_matrix=eye(1),
        meas_noise=eye(1),
    )
    snapshot = _snapshot_idkf_state(node)
    invalid_models = (
        (eye(1), eye(1)),
        {"H": eye(1)},
    )

    with pytest.raises(AttributeError, match="Measurement model mappings"):
        node.update_linear(array([2.0]), measurement_models=invalid_models)

    _assert_idkf_state_unchanged(node, snapshot)


def test_broadcast_process_covariance_is_rejected_without_mutation():
    node = IdkfNode.from_local_gaussian(
        1,
        (array([1.0, 2.0]), eye(2)),
        (eye(2),),
    )
    snapshot = _snapshot_idkf_state(node)

    with pytest.raises(ValueError, match="sys_noise_cov must have shape"):
        node.predict_linear(eye(2), array([0.4, 0.2]))

    _assert_idkf_state_unchanged(node, snapshot)
