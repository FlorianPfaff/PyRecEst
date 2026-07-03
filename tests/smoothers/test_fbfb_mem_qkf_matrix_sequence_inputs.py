import numpy.testing as npt
from pyrecest.backend import array, diag, eye
from pyrecest.smoothers import (
    ForwardBackwardForwardBackwardMEMQKFSmoother,
    MEMQKFTrackerState,
)


def _state(kinematic_state, covariance, shape_state, shape_covariance):
    return MEMQKFTrackerState(
        array(kinematic_state),
        array(covariance),
        array(shape_state),
        array(shape_covariance),
    )


def test_fbfb_accepts_python_matrix_measurement_noise_for_all_scans():
    smoother = ForwardBackwardForwardBackwardMEMQKFSmoother(shape_smoothing="none")
    filtered_states = [
        _state(
            [0.0, 0.0],
            0.5 * eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [2.0, 0.0],
            0.5 * eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0],
            eye(2),
            [0.0, 1.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]
    measurements = [array([[2.0], [0.0]]), array([[3.0], [0.0]])]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states,
        predicted_states,
        measurements=measurements,
        system_matrices=eye(2),
        shape_system_matrices=eye(3),
        meas_noise_covs=[[0.05, 0.0], [0.0, 0.05]],
        initial_shape_state=array([0.0, 1.0, 1.0]),
        initial_shape_covariance=diag(array([0.2, 0.2, 0.2])),
    )

    assert len(smoothed_states) == 2
    assert len(smoother_gains[0]) == 1
    npt.assert_allclose(smoothed_states[0].kinematic_state, array([1.0, 0.0]))
    npt.assert_allclose(smoothed_states[0].covariance, 0.375 * eye(2))
