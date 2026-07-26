import numpy as np
import pytest
from pyrecest.utils.metrics import gospa_distance, mospa_distance, ospa_distance

_POINTS = np.array([[0.0]])


@pytest.mark.parametrize("invalid_distance", [-1.0, np.nan])
@pytest.mark.parametrize(
    "evaluate",
    [
        pytest.param(
            lambda distance_fn: ospa_distance(
                _POINTS,
                _POINTS,
                cutoff=3.0,
                distance_fn=distance_fn,
            ),
            id="ospa",
        ),
        pytest.param(
            lambda distance_fn: gospa_distance(
                _POINTS,
                _POINTS,
                cutoff=3.0,
                distance_fn=distance_fn,
            ),
            id="gospa",
        ),
        pytest.param(
            lambda distance_fn: mospa_distance(
                [_POINTS],
                [_POINTS],
                cutoff=3.0,
                distance_fn=distance_fn,
            ),
            id="mospa",
        ),
    ],
)
def test_set_metrics_reject_invalid_pairwise_distances(evaluate, invalid_distance):
    with pytest.raises(ValueError, match="pairwise distances must be non-negative"):
        evaluate(lambda _estimate, _truth: invalid_distance)


def test_positive_infinite_distance_is_clipped_to_cutoff():
    distance = ospa_distance(
        _POINTS,
        _POINTS,
        cutoff=3.0,
        distance_fn=lambda _estimate, _truth: np.inf,
    )

    assert distance == pytest.approx(3.0)
