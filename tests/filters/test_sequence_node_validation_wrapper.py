import numpy as np
import pytest
from pyrecest.filters import SequenceAssociationNode


def test_public_sequence_node_rejects_array_like_missed_detection_flag():
    with pytest.raises(ValueError, match="is_missed_detection must be a bool"):
        SequenceAssociationNode(
            frame_index=0,
            candidate_index=3,
            is_missed_detection=np.array([True, False]),
        )
