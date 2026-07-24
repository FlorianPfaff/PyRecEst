from decimal import Decimal
from fractions import Fraction

import numpy as np
from pyrecest.utils.track_completion import (
    CompletionCandidate,
    enumerate_fragment_completion_paths,
)


def _provider_with(candidate):
    def provider(*args):
        return [candidate]

    return provider


def _assert_candidate_rejected(candidate):
    try:
        enumerate_fragment_completion_paths(
            [[0, None]],
            direction="suffix",
            candidate_provider=_provider_with(candidate),
        )
    except ValueError as exc:
        assert "candidate observations must be non-negative integers" in str(exc)
    else:
        raise AssertionError("candidate observation was accepted")


def test_text_candidate_observations_are_rejected():
    candidate_text = str(1)
    invalid_candidates = (
        candidate_text,
        np.str_(candidate_text),
        np.array(candidate_text),
        np.array(candidate_text, dtype=object),
        CompletionCandidate(candidate_text),
        CompletionCandidate(np.array(candidate_text)),
    )

    for candidate in invalid_candidates:
        _assert_candidate_rejected(candidate)


def test_fractional_object_candidate_observations_are_rejected():
    fractional_candidates = (
        Decimal("1.5"),
        Fraction(3, 2),
        np.array(Decimal("1.5"), dtype=object),
        np.array(Fraction(3, 2), dtype=object),
        CompletionCandidate(Decimal("1.5")),
        CompletionCandidate(np.array(Fraction(3, 2), dtype=object)),
    )

    for candidate in fractional_candidates:
        _assert_candidate_rejected(candidate)
