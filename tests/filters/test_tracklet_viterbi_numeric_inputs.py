import numpy as np
import pytest
from pyrecest.filters.tracklet_viterbi import (
    TrackletAssociationCandidate,
    TrackletViterbiConfig,
    solve_fixed_lag_tracklet_viterbi,
    solve_tracklet_viterbi,
)


@pytest.mark.parametrize(
    "time_s",
    [
        np.nan,
        np.inf,
        -np.inf,
        np.complex64(1.0 + 2.0j),
        np.array(1.0 + 0.0j),
        np.ma.array(1.0, mask=True),
    ],
)
def test_tracklet_candidate_rejects_invalid_timestamps(time_s):
    with pytest.raises(ValueError, match="time_s"):
        TrackletAssociationCandidate("invalid-time", time_s=time_s)


def test_tracklet_scalar_controls_reject_masked_values():
    masked = np.ma.array(1.0, mask=True)

    with pytest.raises(ValueError, match="unary_cost"):
        TrackletAssociationCandidate("masked-cost", unary_cost=masked)
    with pytest.raises(ValueError, match="motion_weight"):
        TrackletViterbiConfig(motion_weight=masked)
    with pytest.raises(ValueError, match="lag_s"):
        solve_fixed_lag_tracklet_viterbi(
            [[TrackletAssociationCandidate("candidate")]],
            lag_s=masked,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_name"),
    [
        (
            "previous_position",
            np.array([0.0 + 1.0j, 0.0]),
            "previous candidate position",
        ),
        (
            "current_position",
            np.array([1.0 + 2.0j, 0.0]),
            "current candidate position",
        ),
        (
            "previous_velocity",
            np.array([1.0 + 2.0j, 0.0]),
            "previous candidate velocity",
        ),
        (
            "current_velocity",
            np.array([1.0 + 2.0j, 0.0]),
            "current candidate velocity",
        ),
        (
            "current_position",
            np.ma.array([1.0, 0.0], mask=[True, False]),
            "current candidate position",
        ),
        (
            "current_position",
            np.array([np.nan, 0.0]),
            "current candidate position",
        ),
    ],
)
def test_tracklet_motion_rejects_invalid_numeric_vectors(
    field, value, expected_name
):
    previous_position = np.array([0.0, 0.0])
    current_position = np.array([1.0, 0.0])
    previous_velocity = np.array([0.0, 0.0])
    current_velocity = np.array([1.0, 0.0])

    if field == "previous_position":
        previous_position = value
    elif field == "current_position":
        current_position = value
    elif field == "previous_velocity":
        previous_velocity = value
    else:
        current_velocity = value

    frames = [
        [
            TrackletAssociationCandidate(
                "previous",
                time_s=0.0,
                position=previous_position,
                velocity=previous_velocity,
            )
        ],
        [
            TrackletAssociationCandidate(
                "current",
                time_s=1.0,
                position=current_position,
                velocity=current_velocity,
            )
        ],
    ]
    config = TrackletViterbiConfig(
        motion_weight=1.0,
        transition_velocity_std=1.0,
        missed_detection_cost=100.0,
    )

    with pytest.raises(ValueError, match=expected_name):
        solve_tracklet_viterbi(frames, config=config)


def test_tracklet_motion_accepts_clear_mask_vectors():
    previous = TrackletAssociationCandidate(
        "previous",
        time_s=np.ma.array(0.0, mask=False),
        position=np.ma.array([0.0, 0.0], mask=[False, False]),
        velocity=np.ma.array([1.0, 0.0], mask=[False, False]),
    )
    current = TrackletAssociationCandidate(
        "current",
        time_s=np.ma.array(1.0, mask=False),
        position=np.ma.array([1.0, 0.0], mask=[False, False]),
        velocity=np.ma.array([1.0, 0.0], mask=[False, False]),
    )

    result = solve_tracklet_viterbi(
        [[previous], [current]],
        config=TrackletViterbiConfig(
            motion_weight=1.0,
            transition_velocity_std=1.0,
            missed_detection_cost=100.0,
        ),
    )

    assert result.path[0] is previous
    assert result.path[1] is current
    assert np.isfinite(result.total_cost)
