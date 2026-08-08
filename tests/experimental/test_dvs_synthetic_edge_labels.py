import numpy as np
import pytest
from pyrecest.experimental.dvs import (
    edge_probabilities_from_activity,
    uniform_edge_probabilities,
)
from pyrecest.experimental.dvs.synthetic import summarize_edge_counts


def test_activity_probabilities_reject_unknown_edge_labels() -> None:
    with pytest.raises(ValueError, match="edge_labels"):
        edge_probabilities_from_activity(
            ["left", "diagonal"],
            np.array([1.0, 1000.0]),
            background_activity=0.0,
        )


def test_uniform_probabilities_and_counts_reject_invalid_edge_labels() -> None:
    with pytest.raises(ValueError, match="edge_labels"):
        uniform_edge_probabilities(["left", 1])

    with pytest.raises(ValueError, match="edge_labels"):
        summarize_edge_counts(["left", "unknown"], np.array([2, 3]))


def test_numpy_string_edge_labels_remain_supported() -> None:
    labels = [np.str_("left"), np.str_("right"), np.str_("top"), np.str_("bottom")]

    probabilities = uniform_edge_probabilities(labels)
    counts = summarize_edge_counts(labels, np.array([1, 2, 3, 4]))

    assert probabilities == {
        "left": 0.25,
        "right": 0.25,
        "top": 0.25,
        "bottom": 0.25,
    }
    assert counts == {"left": 1, "right": 2, "top": 3, "bottom": 4}
