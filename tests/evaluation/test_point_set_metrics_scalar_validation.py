import numpy as np
import pytest
from pyrecest.evaluation.point_set_metrics import (
    distance_quantiles,
    point_set_geometry_summary,
    precision_recall_curve,
    precision_recall_fscore,
)

_POINTS = np.array([[0.0]])


@pytest.mark.parametrize(
    "threshold",
    [True, False, "0.25", np.array([0.25]), np.array([[0.25]])],
)
def test_point_set_metric_thresholds_reject_non_numeric_scalars(threshold):
    with pytest.raises(ValueError, match="Distance thresholds"):
        precision_recall_fscore(_POINTS, _POINTS, threshold)

    with pytest.raises(ValueError, match="Distance thresholds"):
        precision_recall_curve(_POINTS, _POINTS, (threshold,))

    with pytest.raises(ValueError, match="Distance thresholds"):
        point_set_geometry_summary(_POINTS, _POINTS, thresholds=(threshold,))


@pytest.mark.parametrize(
    "quantile",
    [True, False, "0.5", np.array([0.5]), np.array([[0.5]]), np.nan, np.inf],
)
def test_distance_quantiles_reject_non_numeric_scalar_probabilities(quantile):
    with pytest.raises(ValueError, match="Quantiles"):
        distance_quantiles(_POINTS, _POINTS, quantiles=(quantile,))


@pytest.mark.parametrize("quantile", [-0.1, 1.1])
def test_distance_quantiles_reject_out_of_range_probabilities(quantile):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        distance_quantiles(_POINTS, _POINTS, quantiles=(quantile,))


def test_point_set_metric_numeric_scalar_arrays_remain_supported():
    metrics = precision_recall_fscore(_POINTS, _POINTS, np.array(0.25))
    assert metrics["threshold"] == pytest.approx(0.25)

    quantiles = distance_quantiles(_POINTS, _POINTS, quantiles=(np.array(0.5),))
    assert quantiles == {0.5: pytest.approx(0.0)}
