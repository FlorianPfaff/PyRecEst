from decimal import Decimal
from fractions import Fraction

import numpy as np
from pyrecest.utils.track_evaluation import normalize_track_matrix, track_lengths


def test_fractional_exact_observation_values_are_missing():
    track_matrix = np.array(
        [
            [Decimal("1.5"), Fraction(3, 2)],
            [
                np.array(Decimal("2.5"), dtype=object),
                np.array(Fraction(5, 2), dtype=object),
            ],
        ],
        dtype=object,
    )

    normalized = normalize_track_matrix(track_matrix)

    assert normalized.tolist() == [[None, None], [None, None]]
    assert track_lengths(track_matrix).tolist() == [0, 0]


def test_integral_exact_observation_values_are_preserved():
    track_matrix = np.array([[Decimal("1"), Fraction(4, 2)]], dtype=object)

    normalized = normalize_track_matrix(track_matrix)

    assert normalized.tolist() == [[1, 2]]
