from __future__ import annotations

import numpy as np
import pytest
from pyrecest.evaluation.selection import (
    protected_tail_topk_mask,
    quantile_tail_mask,
    quantile_tail_threshold,
    retained_count_from_fraction,
    tail_rescue_quota_count,
    tail_rescue_topk_mask,
    top_count_mask,
    top_fraction_mask,
)


def test_top_count_mask_is_deterministic_with_ties() -> None:
    mask = top_count_mask([1.0, 2.0, 2.0, 0.5], 2)

    assert mask.tolist() == [False, True, True, False]


def test_top_count_mask_uses_tie_break_scores() -> None:
    mask = top_count_mask([2.0, 2.0, 1.0], 1, tie_break_scores=[0.1, 0.9, 1.0])

    assert mask.tolist() == [False, True, False]


def test_top_fraction_mask_uses_ceil_retained_count() -> None:
    assert retained_count_from_fraction(10, 0.21) == 3
    assert top_fraction_mask(np.arange(10), 0.21).sum() == 3


def test_quantile_tail_mask_selects_lower_tail() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0])

    assert quantile_tail_threshold(values, 0.5) == pytest.approx(1.5)
    assert quantile_tail_mask(values, 0.5).tolist() == [True, True, False, False]


def test_quantile_tail_mask_selects_upper_tail() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0])

    assert quantile_tail_threshold(values, 0.25, tail="upper") == pytest.approx(2.25)
    assert quantile_tail_mask(values, 0.25, tail="upper").tolist() == [
        False,
        False,
        False,
        True,
    ]


def test_protected_tail_topk_mask_preserves_proportional_tail_capacity() -> None:
    primary = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    tail_score = np.asarray([0.0, 0.0, 30.0, 20.0, 10.0, 1.0])
    reliability = np.asarray([10.0, 9.0, 0.0, 1.0, 8.0, 7.0])

    mask = protected_tail_topk_mask(
        primary,
        tail_score,
        reliability,
        0.5,
        tail_quantile=0.5,
    )

    assert mask.tolist() == [True, False, True, True, False, False]


def test_tail_rescue_topk_mask_swaps_in_missing_tail_items() -> None:
    primary = np.asarray([10.0, 9.0, 8.0, 7.0, 1.0, 0.5])
    tail_score = np.asarray([0.0, 0.0, 0.0, 0.0, 100.0, 90.0])
    reliability = np.asarray([10.0, 9.0, 8.0, 7.0, 0.0, 1.0])

    mask = tail_rescue_topk_mask(
        primary,
        tail_score,
        reliability,
        0.5,
        tail_quantile=0.5,
        rescue_fraction=1.0 / 3.0,
    )

    assert mask.sum() == 3
    assert mask[4]
    assert not mask[2]


def test_tail_rescue_quota_count_validates_fraction() -> None:
    assert tail_rescue_quota_count(10, rescue_fraction=0.2) == 2
    with pytest.raises(ValueError, match="rescue_fraction"):
        tail_rescue_quota_count(10, rescue_fraction=0.0)
