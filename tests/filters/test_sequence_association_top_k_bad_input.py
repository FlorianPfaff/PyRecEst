import pytest
from pyrecest.filters import (
    SequenceAssociationNode,
    solve_top_k_viterbi_sequence_associations,
)


class UncoercibleScalar:
    def __array__(self, dtype=None):
        del dtype
        raise TypeError("cannot convert")


def test_top_k_terminal_paths_reports_value_error_for_uncoercible_scalar():
    frames = [[SequenceAssociationNode(0, 0)]]

    with pytest.raises(
        ValueError, match="top_k_terminal_paths must be a positive integer"
    ):
        solve_top_k_viterbi_sequence_associations(
            frames,
            lambda _previous, _current, _context: 0.0,
            top_k_terminal_paths=UncoercibleScalar(),
        )
