import numpy as np
from pyrecest.distributions import GaussianDistribution
from beartype import beartype
from pyrecest.filters.abstract_axial_filter import AbstractAxialFilter

class AxialKalmanFilter(AbstractAxialFilter):
    def __init__(self, initial_state=None):
        if initial_state is None:
            initial_state = GaussianDistribution(np.array([1, 0, 0, 0]), np.eye(4, 4))
        self.filter_state = initial_state

    @property
    def filter_state(self):
        return self._filter_state
        
    @filter_state.setter
    @beartype
    def filter_state(self, new_state: GaussianDistribution):
        assert new_state.mu.shape == (2,) or new_state.mu.shape == (4,), "mu must be a 2d or 4d vector"
        assert abs(np.linalg.norm(new_state.mu) - 1) < 1e-5, "mean must be a unit vector"
        self._filter_state = new_state
        # TODO: define composition operator

    @beartype
    def predict_identity(self, gauss_w: GaussianDistribution):
        assert abs(np.linalg.norm(gauss_w.mu) - 1) < 1e-5, "mean must be a unit vector"
        mu_ = self.composition_operator(self.gauss.mu, gauss_w.mu)
        C_ = self.gauss.C + gauss_w.C
        self.filter_state = GaussianDistribution(mu_, C_)

    @beartype
    def update_identity(self, gauss_v: GaussianDistribution, z: np.ndarray):
        assert abs(np.linalg.norm(gauss_v.mu) - 1) < 1e-5, "mean must be a unit vector"
        assert gauss_v.mu.shape[0] == self.gauss.mu.shape[0]
        assert np.shape(z) == np.shape(self.gauss.mu)
        assert abs(np.linalg.norm(z) - 1) < 1e-5, "measurement must be a unit vector"

        mu_v_conj = np.concatenate(([gauss_v.mu[0]], -gauss_v.mu[1:]))
        z = self.composition_operator(mu_v_conj, z)

        if np.dot(z, self.gauss.mu) < 0:
            z = -z

        d = self.dim
        H = np.eye(d, d)
        IS = H @ self.filter_state.C @ H.T + gauss_v.C
        K = self.filter_state.C @ H.T @ np.linalg.inv(IS)
        IM = z - H @ self.filter_state.mu
        mu = self.filter_state.mu + K @ IM
        C = (np.eye(d, d) - K @ H) @ self.filter_state.C

        mu = mu / np.linalg.norm(mu)
        self.filter_state = GaussianDistribution(mu, C)

