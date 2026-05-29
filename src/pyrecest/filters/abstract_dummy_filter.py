from .abstract_filter import AbstractFilter


class AbstractDummyFilter(AbstractFilter):
    """Abstract dummy filter that does nothing on predictions and updates.

    Subclasses should call super().__init__ with the initial distribution.
    """

    def __init__(self, initial_filter_state):
        AbstractFilter.__init__(self, initial_filter_state)

    @property
    def dist(self):
        return self._filter_state

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        # Do nothing - the dummy filter state is fixed at initialization
        pass

    def set_state(self, dist):
        if not hasattr(dist, "dim"):
            raise ValueError("dist must expose a dim attribute.")
        if dist.dim != self.dim:
            raise ValueError(f"dist must have dimension {self.dim}, got {dist.dim}.")
        # Do nothing

    def predict_identity(self, noise_distribution):
        pass

    def predict_nonlinear(self, f, *args, **kwargs):
        pass

    def predict_nonlinear_via_transition_density(self, transition_density, *args):
        pass

    def update_identity(self, noise_distribution, measurement):
        pass

    def update_nonlinear(self, likelihood, measurement=None):
        pass

    def get_estimate(self):
        return self.dist

    def get_point_estimate(self):
        return self.dist.sample(1)[0]
