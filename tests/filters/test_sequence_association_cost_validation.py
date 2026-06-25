import pytest
from pyrecest.filters import SequenceAssociationNode, solve_viterbi_sequence_association


def test_sequence_association_rejects_boolean_unary_cost():
    with pytest.raises(ValueError, match="unary_cost must be a scalar numeric cost"):
        SequenceAssociationNode(frame_index=0, candidate_index=0, unary_cost=True)


def test_sequence_association_rejects_text_unary_cost():
    with pytest.raises(ValueError, match="unary_cost must be a scalar numeric cost"):
        SequenceAssociationNode(frame_index=0, candidate_index=0, unary_cost="1.0")


def test_sequence_association_rejects_boolean_transition_cost():
    frames = [
        [SequenceAssociationNode(frame_index=0, candidate_index=0)],
        [SequenceAssociationNode(frame_index=1, candidate_index=0)],
    ]

    def transition_cost(_previous, _current, _context):
        return True

    with pytest.raises(
        ValueError, match="transition_cost must be a scalar numeric cost"
    ):
        solve_viterbi_sequence_association(frames, transition_cost)


def test_sequence_association_rejects_text_transition_cost():
    frames = [
        [SequenceAssociationNode(frame_index=0, candidate_index=0)],
        [SequenceAssociationNode(frame_index=1, candidate_index=0)],
    ]

    def transition_cost(_previous, _current, _context):
        return "1.0"

    with pytest.raises(
        ValueError, match="transition_cost must be a scalar numeric cost"
    ):
        solve_viterbi_sequence_association(frames, transition_cost)
