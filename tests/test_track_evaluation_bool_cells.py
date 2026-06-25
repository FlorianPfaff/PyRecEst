import numpy as np
from pyrecest.utils.track_evaluation import normalize_track_matrix, track_pair_set


def test_boolean_track_cells_are_missing():
    matrix = normalize_track_matrix([[False, True, np.bool_(False), np.array(True)]])
    assert matrix.tolist() == [[None, None, None, None]]
    assert track_pair_set([[False, True]]) == set()
