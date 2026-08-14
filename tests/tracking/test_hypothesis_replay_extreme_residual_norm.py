"""Regression tests for scale-stable hypothesis replay residual norms."""

from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import HypothesisReplay, score_hypothesis_replay


def test_replay_scoring_preserves_large_finite_fallback_residual_norm() -> None:
    replay = HypothesisReplay(
        hypothesis_id="large-finite-residual",
        records=[{"residual": np.array([1.0e200, 1.0e200])}],
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        score = score_hypothesis_replay(replay)

    assert score.finite_residual_count == 1
    assert score.robust_sum_residual == pytest.approx(5.0)
    assert score.total_score == pytest.approx(0.05)
