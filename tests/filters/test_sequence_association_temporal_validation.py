import numpy as np
import pytest
from pyrecest.filters import (
    SequenceAssociationNode,
    solve_top_k_viterbi_sequence_associations,
    solve_viterbi_sequence_association,
)

_TEMPORAL_VALUES = (
    pytest.param(np.timedelta64(1, "ns"), id="timedelta-ns"),
    pytest.param(np.timedelta64(1, "us"), id="timedelta-us"),
    pytest.param(
        np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
        id="datetime-ns",
    ),
    pytest.param(
        np.datetime64("1970-01-01T00:00:00.000001", "us"),
        id="datetime-us",
    ),
    pytest.param(
        np.array(np.timedelta64(1, "ns"), dtype=object),
        id="object-timedelta",
    ),
    pytest.param(
        np.array(
            np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
            dtype=object,
        ),
        id="object-datetime",
    ),
)


@pytest.mark.parametrize("temporal_value", _TEMPORAL_VALUES)
def test_sequence_node_rejects_temporal_indices_and_costs(temporal_value):
    with pytest.raises(ValueError, match="frame_index must be an integer"):
        SequenceAssociationNode(temporal_value, 0)

    with pytest.raises(ValueError, match="candidate_index must be an integer"):
        SequenceAssociationNode(0, temporal_value)

    with pytest.raises(ValueError, match="unary_cost must be a scalar numeric cost"):
        SequenceAssociationNode(0, 0, unary_cost=temporal_value)


@pytest.mark.parametrize("temporal_value", _TEMPORAL_VALUES)
def test_sequence_solver_rejects_temporal_counts_and_costs(temporal_value):
    frames = [
        [SequenceAssociationNode(0, 0)],
        [SequenceAssociationNode(1, 0)],
    ]

    with pytest.raises(
        ValueError,
        match="top_k_terminal_paths must be a positive integer",
    ):
        solve_top_k_viterbi_sequence_associations(
            frames,
            lambda _previous, _current, _context: 0.0,
            top_k_terminal_paths=temporal_value,
        )

    with pytest.raises(
        ValueError,
        match="transition_cost must be a scalar numeric cost",
    ):
        solve_viterbi_sequence_association(
            frames,
            lambda _previous, _current, _context: temporal_value,
        )
