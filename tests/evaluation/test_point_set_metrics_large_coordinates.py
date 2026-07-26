import numpy as np
import pyrecest.evaluation.point_set_metrics as point_set_metrics


def test_dense_fallback_preserves_large_coordinate_differences(monkeypatch):
    monkeypatch.setattr(
        point_set_metrics,
        "_nearest_neighbor_distances_ckdtree",
        lambda *args, **kwargs: None,
    )

    base = 1.0e150
    delta = 1.0e140
    query = np.array(
        [
            [base, 0.0],
            [base + 4.0 * delta, 0.0],
        ]
    )
    reference = np.array(
        [
            [base + delta, 0.0],
            [base + 7.0 * delta, 0.0],
        ]
    )

    distances, indices = point_set_metrics.nearest_neighbor_distances(
        query,
        reference,
        query_chunk_size=1,
        return_indices=True,
    )

    pairwise_distances = np.linalg.norm(
        query[:, None, :] - reference[None, :, :], axis=2
    )
    expected_indices = np.argmin(pairwise_distances, axis=1)
    expected_distances = pairwise_distances[np.arange(query.shape[0]), expected_indices]

    np.testing.assert_allclose(distances, expected_distances)
    np.testing.assert_array_equal(indices, expected_indices)
    assert np.all(distances > 0.0)
