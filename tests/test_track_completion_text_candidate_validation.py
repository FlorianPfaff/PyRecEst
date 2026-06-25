import numpy as np
from pyrecest.utils.track_completion import (
    CompletionCandidate,
    enumerate_fragment_completion_paths,
)


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

    for invalid_candidate in invalid_candidates:
        provider = lambda *_: [invalid_candidate]
        try:
            enumerate_fragment_completion_paths(
                [[0, None]], direction="suffix", candidate_provider=provider
            )
        except ValueError as exc:
            assert "candidate observations must be non-negative integers" in str(exc)
        else:
            raise AssertionError("text-valued candidate observation was accepted")
