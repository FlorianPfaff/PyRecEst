import copy

import numpy as np
from beartype import beartype
from pyrecest.distributions import GaussianDistribution

from .abstract_se2_filter import AbstractSE2Filter


class SE2UKF(AbstractSE2Filter):
    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    @beartype
    def filter_state(self, new_state: GaussianDistribution):
        assert np.size(new_state.mu) == 4
        assert np.shape(new_state.C) == (4, 4), "Wrong dimension."
        assert (
            np.linalg.norm(new_state.mu[:2]) == 1
        ), "First two entries of estimate must be normalized"
        self._filter_state = copy.deepcopy(new_state)
