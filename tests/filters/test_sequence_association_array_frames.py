import numpy as np
from pyrecest.filters import (
    SequenceAssociationNode,
    solve_viterbi_sequence_association,
)


def test_viterbi_accepts_numpy_object_array_of_frames():
    frames = np.empty(2, dtype=object)
    frames[0] = (
        SequenceAssociationNode(0, 0, payload="A"),
        SequenceAssociationNode(0, 1, payload="B", unary_cost=2.0),
    )
    frames[1] = (
        SequenceAssociationNode(1, 0, payload="A"),
        SequenceAssociationNode(1, 1, payload="B", unary_cost=2.0),
    )

    path = solve_viterbi_sequence_association(
        frames,
        lambda previous, current, _context: (
            0.0 if previous.payload == current.payload else 10.0
        ),
    )

    assert path.candidate_indices == (0, 0)
    assert path.payloads == ("A", "A")
