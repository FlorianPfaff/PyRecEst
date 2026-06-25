import numpy as np
import pytest
from pyrecest.utils.track_completion import (
    CompletionCandidate,
    enumerate_fragment_completion_paths,
)


def _candidate_provider_with_score(score):
    def provider(session, observation, target_session):
        del session, observation, target_session
        return [CompletionCandidate(1, score=score)]

    return provider


def test_completion_candidate_scores_reject_non_real_values():
    invalid_scores = (
        True,
        np.bool_(False),
        "1.0",
        b"1.0",
        np.array("1.0"),
        1.0 + 0.0j,
        np.array(1.0 + 0.0j),
        float("nan"),
        float("inf"),
        [1.0],
    )

    for invalid_score in invalid_scores:
        with pytest.raises(
            ValueError, match="candidate scores must be a finite real scalar"
        ):
            enumerate_fragment_completion_paths(
                [[0, None]],
                direction="suffix",
                candidate_provider=_candidate_provider_with_score(invalid_score),
            )


def test_score_path_rejects_non_real_values():
    invalid_scores = ("2.0", True, np.inf, [1.0])

    for invalid_score in invalid_scores:
        with pytest.raises(
            ValueError, match="path scores must be a finite real scalar"
        ):
            enumerate_fragment_completion_paths(
                [[0, None]],
                direction="suffix",
                candidate_provider=_candidate_provider_with_score(0.25),
                score_path=lambda steps, value=invalid_score: value,
            )


def test_completion_scores_accept_finite_real_numpy_scalars():
    paths = enumerate_fragment_completion_paths(
        [[0, None]],
        direction="suffix",
        candidate_provider=_candidate_provider_with_score(np.array(0.25)),
        score_path=lambda steps: np.float64(sum(step.score for step in steps) + 0.5),
    )

    assert len(paths) == 1
    assert paths[0].score == pytest.approx(0.75)
