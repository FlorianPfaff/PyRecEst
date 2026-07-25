import numpy as np
import pytest
from pyrecest.evaluation.selection import sanitized_score_vector, top_count_mask


def test_sanitized_score_vector_rejects_boolean_scores():
    with pytest.raises(ValueError, match="real numeric"):
        sanitized_score_vector([True, False])


def test_sanitized_score_vector_rejects_text_scores():
    with pytest.raises(ValueError, match="real numeric"):
        sanitized_score_vector(["0.1", "0.9"])


def test_sanitized_score_vector_rejects_object_scores_with_none():
    with pytest.raises(ValueError, match="real numeric"):
        sanitized_score_vector([None, 0.5])


def test_top_count_mask_rejects_boolean_scores():
    with pytest.raises(ValueError, match="real numeric"):
        top_count_mask([True, False], 1)


def test_sanitized_score_vector_keeps_numeric_sanitization_contract():
    scores = sanitized_score_vector([np.nan, -2.0, 3.5])

    np.testing.assert_allclose(scores, [0.0, 0.0, 3.5])
