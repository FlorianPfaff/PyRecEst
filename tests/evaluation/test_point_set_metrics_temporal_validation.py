import numpy as np
import pytest
from pyrecest.evaluation.point_set_metrics import (
    deterministic_subsample,
    distance_quantiles,
    nearest_neighbor_distances,
    point_set_geometry_summary,
    precision_recall_curve,
    precision_recall_fscore,
)

_POINTS = np.array([[0.0], [1.0]])
_TEMPORAL_TWO = np.timedelta64(2, "ns")
_TEMPORAL_ONE = np.timedelta64(1, "ns")
_DATETIME_TWO = np.datetime64("1970-01-01T00:00:00.000000002", "ns")


@pytest.mark.parametrize(
    "temporal_scalar",
    [
        _TEMPORAL_TWO,
        _DATETIME_TWO,
        np.array(_TEMPORAL_TWO),
        np.array(_DATETIME_TWO),
        np.array(_TEMPORAL_TWO, dtype=object),
        np.array(_DATETIME_TWO, dtype=object),
    ],
)
def test_point_set_metric_integer_controls_reject_temporal_scalars(temporal_scalar):
    with pytest.raises(ValueError, match="positive integer"):
        nearest_neighbor_distances(
            _POINTS,
            _POINTS,
            query_chunk_size=temporal_scalar,
        )

    with pytest.raises(ValueError, match="max_points"):
        deterministic_subsample(_POINTS, max_points=temporal_scalar)


@pytest.mark.parametrize(
    "temporal_scalar",
    [
        _TEMPORAL_TWO,
        _DATETIME_TWO,
        np.array(_TEMPORAL_TWO),
        np.array(_DATETIME_TWO),
        np.array(_TEMPORAL_TWO, dtype=object),
        np.array(_DATETIME_TWO, dtype=object),
    ],
)
def test_point_set_metric_thresholds_reject_temporal_scalars(temporal_scalar):
    with pytest.raises(ValueError, match="Distance thresholds"):
        precision_recall_fscore(_POINTS, _POINTS, temporal_scalar)

    with pytest.raises(ValueError, match="Distance thresholds"):
        precision_recall_curve(_POINTS, _POINTS, (temporal_scalar,))

    with pytest.raises(ValueError, match="Distance thresholds"):
        point_set_geometry_summary(_POINTS, _POINTS, thresholds=(temporal_scalar,))


@pytest.mark.parametrize(
    "temporal_scalar",
    [
        _TEMPORAL_ONE,
        np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
        np.array(_TEMPORAL_ONE),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001", "ns")),
        np.array(_TEMPORAL_ONE, dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001", "ns"), dtype=object),
    ],
)
def test_distance_quantiles_reject_temporal_scalars(temporal_scalar):
    with pytest.raises(ValueError, match="Quantiles"):
        distance_quantiles(_POINTS, _POINTS, quantiles=(temporal_scalar,))
