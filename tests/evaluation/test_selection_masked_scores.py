"""Regression tests for masked score inputs in selection helpers."""

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.evaluation import sanitized_score_vector, top_count_mask


@pytest.mark.parametrize(
    "scores",
    (
        np.ma.array([0.1, 999.0], mask=[False, True]),
        [0.1, np.ma.masked],
        np.array([0.1, np.ma.masked], dtype=object),
    ),
)
def test_sanitized_score_vector_rejects_masked_values(scores):
    with pytest.raises(ValueError, match="scores must contain real numeric values"):
        sanitized_score_vector(scores)


def test_top_count_mask_does_not_select_hidden_masked_payload():
    scores = np.ma.array([0.1, 999.0], mask=[False, True])

    with pytest.raises(ValueError, match="scores must contain real numeric values"):
        top_count_mask(scores, 1)


def test_clear_mask_score_arrays_remain_supported():
    scores = np.ma.array([0.1, 0.9], mask=[False, False])

    npt.assert_allclose(sanitized_score_vector(scores), [0.1, 0.9])
    npt.assert_array_equal(top_count_mask(scores, 1), [False, True])
