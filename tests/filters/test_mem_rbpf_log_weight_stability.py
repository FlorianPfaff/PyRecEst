import numpy as np
import pytest
from pyrecest import backend
from pyrecest.backend import array, diag, eye, zeros
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="MEM-RBPF particle-weight stability is currently exercised on NumPy only",
)


def test_mem_rbpf_particle_weights_preserve_tiny_covariance_information():
    tracker = MEMRBPFTracker(
        kinematic_state=array([0.0, 0.0, 0.0, 0.0]),
        covariance=eye(4),
        shape_state=array([0.0, 1.0, 1.0]),
        shape_covariance=diag(array([1e-3, 1e-3, 1e-3])),
        meas_noise_cov=zeros((2, 2)),
        sys_noise=zeros((4, 4)),
        shape_sys_noise=zeros((3, 3)),
        multiplicative_noise_cov=1e-200 * eye(2),
        n_particles=2,
        resampling_threshold=0,
    )
    tracker.theta = array([0.0, 0.0])
    tracker.axis = array([[1.0, 1.0], [np.sqrt(2.0), np.sqrt(2.0)]])
    tracker.axis_covariances = zeros((2, 2, 2))
    tracker.weights = array([0.5, 0.5])

    tracker._update_particle_weights(
        centered=zeros((1, 2)),
        meas_noise_cov=zeros((2, 2)),
        mult_var=1e-200,
    )

    np.testing.assert_allclose(np.asarray(tracker.weights), [2.0 / 3.0, 1.0 / 3.0])
