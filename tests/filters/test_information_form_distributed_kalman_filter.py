"""Tests for the information-form distributed Kalman node."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, linalg
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.information_form_distributed_kalman_filter import IdkfNode
from pyrecest.filters.kalman_filter import KalmanFilter


def _three_node_fixture():
    system_matrix = array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    process_noise = eye(4)
    explicit_input = array([1.0, 2.0, 3.0, 4.0])
    means = (
        3.0 * array([1.0, 1.0, 1.0, 1.0]),
        array([1.0, 1.0, 1.0, 1.0]),
        array([1.0, 2.0, 3.0, 4.0]),
    )
    covariances = (
        eye(4),
        array(
            [
                [3.0, 0.1, 0.0, 0.0],
                [0.1, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.1],
                [0.0, 0.0, 0.1, 3.0],
            ]
        ),
        array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ]
        ),
    )
    return system_matrix, process_noise, explicit_input, means, covariances


def _centralized_prior(means, covariances):
    information_matrix = sum(linalg.inv(covariance) for covariance in covariances)
    information_vector = sum(
        linalg.solve(covariance, mean) for mean, covariance in zip(means, covariances)
    )
    return GaussianDistribution(
        linalg.solve(information_matrix, information_vector),
        linalg.inv(information_matrix),
        check_validity=False,
    )


def test_three_nodes_prediction_matches_centralized_kalman_filter():
    system_matrix, process_noise, explicit_input, means, covariances = (
        _three_node_fixture()
    )

    central_filter = KalmanFilter(_centralized_prior(means, covariances))
    central_filter.predict_linear(system_matrix, process_noise, explicit_input)

    nodes = [
        IdkfNode.from_local_gaussian(
            node_id,
            (mean, covariance),
            covariances,
        )
        for node_id, mean, covariance in zip((1, 2, 3), means, covariances)
    ]
    nodes[0].predict_linear(system_matrix, process_noise, explicit_input)
    nodes[1].predict_linear(system_matrix, process_noise)
    nodes[2].predict_linear(system_matrix, process_noise)

    fused_node = nodes[0].fused_copy(nodes[1]).fuse_with(nodes[2])

    assert allclose(fused_node.filter_state.mu, central_filter.filter_state.mu)
    assert allclose(fused_node.filter_state.C, central_filter.filter_state.C)


def test_local_updates_and_bank_fusion_match_centralized_kalman_filter():
    _system_matrix, _process_noise, _explicit_input, means, covariances = (
        _three_node_fixture()
    )
    measurement_models = (
        (
            array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            array([[2.0, 0.2], [0.2, 1.5]]),
        ),
        (
            array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            array([[1.2, 0.1], [0.1, 1.1]]),
        ),
        (
            array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
            array([[1.4, 0.0], [0.0, 0.9]]),
        ),
    )
    measurements = (array([2.0, 2.0]), array([5.0, 5.0]), array([1.0, 3.0]))

    central_filter = KalmanFilter(_centralized_prior(means, covariances))
    for measurement, (measurement_matrix, meas_noise) in zip(
        measurements,
        measurement_models,
    ):
        central_filter.update_linear(measurement, measurement_matrix, meas_noise)

    nodes = [
        IdkfNode.from_local_gaussian(
            node_id,
            (mean, covariance),
            covariances,
            measurement_matrix=measurement_matrix,
            meas_noise=meas_noise,
            measurement_models=measurement_models,
        )
        for node_id, mean, covariance, (measurement_matrix, meas_noise) in zip(
            (1, 2, 3),
            means,
            covariances,
            measurement_models,
        )
    ]
    for node, measurement in zip(nodes, measurements):
        node.update_linear(measurement)

    fused_node = nodes[0].fused_copy(nodes[1]).fuse_with(nodes[2])

    assert allclose(fused_node.filter_state.mu, central_filter.filter_state.mu)
    assert allclose(fused_node.filter_state.C, central_filter.filter_state.C)


def test_duplicate_contribution_is_not_double_counted():
    system_matrix, process_noise, explicit_input, means, covariances = (
        _three_node_fixture()
    )
    node_1 = IdkfNode.from_local_gaussian(1, (means[0], covariances[0]), covariances)
    node_2 = IdkfNode.from_local_gaussian(2, (means[1], covariances[1]), covariances)

    node_1.predict_linear(system_matrix, process_noise, explicit_input)
    node_2.predict_linear(system_matrix, process_noise)
    node_1.receive_contribution(node_2.export_contribution())
    before = node_1.information_vector_sum()

    changed = node_1.receive_contribution(node_2.export_contribution())
    after = node_1.information_vector_sum()

    assert changed is False
    assert allclose(before, after)
