from pyrecest.distributions.hypersphere_subset.bingham_distribution import BinghamDistribution
import numpy as np
from beartype import beartype

class BinghamFilter(AbstractAxialFilter):

    def __init__(self):
        B_ = BinghamDistribution(np.array([-1, -1, -1, 0]), np.eye(4))
        self.filter_state = B_

    @property
    def filter_state(self):
        return self.B

    @filter_state.setter
    @beartype
    def filter_state(self, dist: BinghamDistribution):
        assert dist.dim == 2 or dist.dim == 4, 'Only 2D and 4D distributions are supported'
        self._filter_state = dist
        # TODO composition operator for BinghamDistribution

    @beartype
    def predict_identity(self, sys_noise: BinghamDistribution):
        self.B = self.B.compose(sys_noise)
        