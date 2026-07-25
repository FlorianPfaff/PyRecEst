import numpy as np
import pytest
from pyrecest.experimental.multisensor_ddp_association import (
    BIRTH_LABEL,
    CLUTTER_LABEL,
    SensorAssociationBlock,
    multisensor_ddp_association_update,
    predict_ddp_base_weights,
)

_TEMPORAL_PAYLOADS = (
    np.timedelta64(1, "ns"),
    np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
    np.array(np.timedelta64(1, "ns"), dtype=object),
)


def test_multisensor_update_shares_target_atoms_across_sensors():
    radar = SensorAssociationBlock(
        "radar",
        log_likelihoods=np.log([[0.90, 0.02], [0.03, 0.85]]),
        clutter_log_weights=np.log([0.01, 0.01]),
        birth_log_weights=np.log([0.02, 0.02]),
        concentration=4.0,
    )
    camera = SensorAssociationBlock(
        "camera",
        log_likelihoods=np.log([[0.82, 0.05], [0.04, 0.80]]),
        clutter_log_weights=np.log([0.02, 0.02]),
        birth_log_weights=np.log([0.03, 0.03]),
        concentration=4.0,
    )

    result = multisensor_ddp_association_update(
        [0.45, 0.45],
        [radar, camera],
        target_labels=("target-a", "target-b"),
        birth_weight=0.10,
        prior_strength=0.5,
    )

    assert result.assignment_labels == (
        "target-a",
        "target-b",
        BIRTH_LABEL,
        CLUTTER_LABEL,
    )
    assert result.posterior_for_sensor("radar").hard_assignments == (
        "target-a",
        "target-b",
    )
    assert result.posterior_for_sensor("camera").hard_assignments == (
        "target-a",
        "target-b",
    )
    assert result.posterior_for_sensor("radar").responsibilities[0, 0] > 0.90
    assert result.posterior_for_sensor("camera").responsibilities[1, 1] > 0.85
    assert np.isclose(
        result.updated_target_weights.sum() + result.updated_birth_weight, 1.0
    )
    assert result.expected_clutter_count < 0.2


def test_clutter_evidence_competes_with_existing_target_and_birth_atoms():
    block = SensorAssociationBlock(
        "lidar",
        log_likelihoods=np.log([[0.01, 0.01]]),
        clutter_log_weights=np.log([0.80]),
        birth_log_weights=np.log([0.02]),
        concentration=1.0,
    )

    result = multisensor_ddp_association_update(
        [0.45, 0.45], [block], birth_weight=0.10
    )

    posterior = result.posterior_for_sensor("lidar")
    assert posterior.hard_assignments == (CLUTTER_LABEL,)
    assert posterior.clutter_responsibilities[0] > 0.90
    assert result.expected_clutter_count > 0.90


def test_point_target_projection_allows_each_existing_target_once_per_sensor():
    block = SensorAssociationBlock(
        "radar",
        log_likelihoods=np.log([[0.90, 0.10], [0.85, 0.08], [0.05, 0.80]]),
        clutter_log_weights=np.log([0.01, 0.01, 0.01]),
        birth_log_weights=np.log([0.02, 0.02, 0.02]),
        concentration=5.0,
    )

    result = multisensor_ddp_association_update(
        [0.5, 0.5], [block], birth_weight=0.05, point_target=True
    )
    target_responsibilities = result.posterior_for_sensor(
        "radar"
    ).target_responsibilities

    assert np.all(target_responsibilities.sum(axis=0) <= 1.0)
    assert result.posterior_for_sensor("radar").hard_assignments.count(0) == 1
    assert result.posterior_for_sensor("radar").hard_assignments.count(1) == 1


def test_point_target_projection_rejects_infeasible_finite_assignments():
    block = SensorAssociationBlock(
        "radar",
        log_likelihoods=np.log([[0.90], [0.85]]),
        clutter_log_weights=[-np.inf, -np.inf],
        birth_log_weights=[-np.inf, -np.inf],
        concentration=1.0,
    )

    with pytest.raises(ValueError, match="no finite feasible association"):
        multisensor_ddp_association_update(
            [1.0],
            [block],
            birth_weight=0.0,
            point_target=True,
        )


def test_predict_ddp_base_weights_applies_survival_and_birth_mass():
    target_weights, birth_weight = predict_ddp_base_weights(
        [0.7, 0.3],
        survival_probabilities=[0.5, 1.0],
        birth_weight=0.35,
    )

    assert np.allclose(target_weights, [0.35, 0.30])
    assert np.isclose(birth_weight, 0.35)


def test_input_validation_rejects_shape_and_probability_errors():
    with pytest.raises(ValueError, match="target_labels"):
        multisensor_ddp_association_update(
            [1.0, 1.0],
            [SensorAssociationBlock("radar", [[0.0, 0.0]])],
            target_labels=("only-one-label",),
        )

    with pytest.raises(ValueError, match="probabilities"):
        multisensor_ddp_association_update(
            [1.0],
            [SensorAssociationBlock("camera", [[0.0]], detection_probabilities=1.5)],
        )

    with pytest.raises(ValueError, match="duplicate sensor_id"):
        multisensor_ddp_association_update(
            [1.0],
            [
                SensorAssociationBlock("camera", [[0.0]]),
                SensorAssociationBlock("camera", [[0.0]]),
            ],
        )


@pytest.mark.parametrize("bad_value", _TEMPORAL_PAYLOADS)
def test_predict_ddp_base_weights_rejects_temporal_numeric_payloads(bad_value):
    with pytest.raises(ValueError, match="datetime64|timedelta64"):
        predict_ddp_base_weights([bad_value])

    with pytest.raises(ValueError, match="datetime64|timedelta64"):
        predict_ddp_base_weights([1.0], survival_probabilities=bad_value)

    with pytest.raises(ValueError, match="datetime64|timedelta64"):
        predict_ddp_base_weights([1.0], birth_weight=bad_value)


@pytest.mark.parametrize("bad_value", _TEMPORAL_PAYLOADS)
def test_multisensor_update_rejects_temporal_numeric_payloads(bad_value):
    valid_block = SensorAssociationBlock("radar", [[0.0]])

    with pytest.raises(ValueError, match="datetime64|timedelta64"):
        multisensor_ddp_association_update([bad_value], [valid_block])

    with pytest.raises(ValueError, match="datetime64|timedelta64"):
        multisensor_ddp_association_update([1.0], [valid_block], birth_weight=bad_value)

    with pytest.raises(ValueError, match="datetime64|timedelta64"):
        multisensor_ddp_association_update(
            [1.0], [valid_block], prior_strength=bad_value
        )

    invalid_sensor_blocks = [
        SensorAssociationBlock("radar", [[bad_value]]),
        SensorAssociationBlock("radar", [[0.0]], detection_probabilities=bad_value),
        SensorAssociationBlock("radar", [[0.0]], birth_log_weights=bad_value),
        SensorAssociationBlock("radar", [[0.0]], clutter_log_weights=bad_value),
        SensorAssociationBlock("radar", [[0.0]], concentration=bad_value),
    ]
    for block in invalid_sensor_blocks:
        with pytest.raises(ValueError, match="datetime64|timedelta64"):
            multisensor_ddp_association_update([1.0], [block])
