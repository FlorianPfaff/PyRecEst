import numpy as np
import pytest
from pyrecest.evaluation.point_set_metrics import (
    chamfer_distance,
    deterministic_subsample,
    distance_quantiles,
    nearest_neighbor_distances,
    point_set_geometry_summary,
    precision_recall_curve,
    precision_recall_fscore,
)


def test_nearest_neighbor_distances_are_directed():
    query = np.array([[0.0, 0.0], [2.0, 0.0]])
    reference = np.array([[0.0, 0.0], [1.0, 0.0]])

    np.testing.assert_allclose(
        nearest_neighbor_distances(query, reference, query_chunk_size=1), [0.0, 1.0]
    )


def test_chamfer_distance_supports_l1_and_l2_conventions():
    points_a = np.array([[0.0, 0.0], [1.0, 0.0]])
    points_b = np.array([[0.0, 0.0], [3.0, 0.0]])

    assert chamfer_distance(points_a, points_b, query_chunk_size=1) == pytest.approx(
        1.5
    )
    assert chamfer_distance(
        points_a, points_b, squared=True, query_chunk_size=1
    ) == pytest.approx(2.5)
    assert chamfer_distance(
        points_a, points_b, symmetric=False, query_chunk_size=1
    ) == pytest.approx(0.5)


def test_precision_recall_fscore_matches_threshold_definition():
    estimate = np.array([[0.0, 0.0], [1.0, 0.0]])
    reference = np.array([[0.0, 0.0], [3.0, 0.0]])

    metrics = precision_recall_fscore(estimate, reference, 0.25, query_chunk_size=1)

    assert metrics == {
        "threshold": pytest.approx(0.25),
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "f_score": pytest.approx(0.5),
    }


def test_precision_recall_curve_reuses_distances_for_multiple_thresholds():
    estimate = np.array([[0.0, 0.0], [1.0, 0.0]])
    reference = np.array([[0.0, 0.0], [3.0, 0.0]])

    rows = precision_recall_curve(estimate, reference, (0.25, 2.0), query_chunk_size=1)

    assert rows[0]["f_score"] == pytest.approx(0.5)
    assert rows[1]["f_score"] == pytest.approx(1.0)


def test_point_set_geometry_summary_matches_legacy_geometry_columns():
    estimate = np.array([[0.0, 0.0], [1.0, 0.0]])
    reference = np.array([[0.0, 0.0], [3.0, 0.0]])

    summary, threshold_rows = point_set_geometry_summary(
        estimate, reference, thresholds=(0.25,), query_chunk_size=1
    )

    assert summary["accuracy_mean"] == pytest.approx(0.5)
    assert summary["completion_mean"] == pytest.approx(1.0)
    assert summary["chamfer_l1"] == pytest.approx(1.5)
    assert summary["chamfer_l2"] == pytest.approx(2.5)
    assert threshold_rows[0]["precision"] == pytest.approx(0.5)
    assert threshold_rows[0]["recall"] == pytest.approx(0.5)


def test_distance_quantiles_are_reported_by_requested_probability():
    query = np.array([[0.0], [1.0], [3.0]])
    reference = np.array([[0.0]])

    quantiles = distance_quantiles(
        query, reference, quantiles=(0.0, 0.5, 1.0), query_chunk_size=1
    )

    assert quantiles == {
        0.0: pytest.approx(0.0),
        0.5: pytest.approx(1.0),
        1.0: pytest.approx(3.0),
    }


def test_deterministic_subsample_returns_sorted_indices_and_is_reproducible():
    points = np.arange(30.0).reshape(10, 3)

    subset_a, indices_a = deterministic_subsample(points, max_points=4, seed=7)
    subset_b, indices_b = deterministic_subsample(points, max_points=4, seed=7)

    np.testing.assert_array_equal(indices_a, indices_b)
    np.testing.assert_array_equal(subset_a, subset_b)
    np.testing.assert_array_equal(subset_a, points[indices_a])
    assert np.all(indices_a[:-1] <= indices_a[1:])


def test_deterministic_subsample_validates_max_points_integrality():
    points = np.arange(30.0).reshape(10, 3)

    subset, indices = deterministic_subsample(points, max_points=4.0, seed=7)
    assert subset.shape == (4, 3)
    assert indices.shape == (4,)

    all_points, all_indices = deterministic_subsample(points, max_points=-1)
    np.testing.assert_array_equal(all_points, points)
    np.testing.assert_array_equal(all_indices, np.arange(points.shape[0]))

    for max_points in (True, 1.5, np.nan, np.array([3])):
        with pytest.raises(ValueError, match="max_points"):
            deterministic_subsample(points, max_points=max_points)


def test_invalid_point_sets_raise_clear_errors():
    with pytest.raises(ValueError, match="shape"):
        nearest_neighbor_distances(np.array([0.0, 1.0]), np.array([[0.0], [1.0]]))

    with pytest.raises(ValueError, match="at least one point"):
        nearest_neighbor_distances(np.empty((0, 2)), np.array([[0.0, 1.0]]))

    with pytest.raises(ValueError, match="same point dimension"):
        nearest_neighbor_distances(np.array([[0.0, 1.0]]), np.array([[0.0, 1.0, 2.0]]))

    with pytest.raises(ValueError, match="non-negative"):
        precision_recall_fscore(np.array([[0.0]]), np.array([[0.0]]), -1.0)


def test_nonfinite_thresholds_raise_clear_errors():
    points = np.array([[0.0]])

    for threshold in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite and non-negative"):
            precision_recall_fscore(points, points, threshold)
        with pytest.raises(ValueError, match="finite and non-negative"):
            precision_recall_curve(points, points, (threshold,))
        with pytest.raises(ValueError, match="finite and non-negative"):
            point_set_geometry_summary(points, points, thresholds=(threshold,))
