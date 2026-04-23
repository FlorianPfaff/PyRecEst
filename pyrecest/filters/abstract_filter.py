"""Abstract base class for all filters"""

import copy
from abc import ABC, abstractmethod

from pyrecest.utils.history_recorder import HistoryRecorder


class AbstractFilter(ABC):
    """Abstract base class for all filters."""

    @abstractmethod
    def __init__(self, initial_filter_state):
        self._filter_state = initial_filter_state
        self.history = HistoryRecorder()

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        assert self._filter_state is None or isinstance(
            new_state, type(self.filter_state)
        ), "New distribution has to be of the same class as (or inherit from) the previous density."
        self._filter_state = copy.deepcopy(new_state)

    def get_point_estimate(self):
        """Get a point estimate"""
        return self.filter_state.mean()

    @property
    def dim(self) -> int:
        """Convenience function to get the dimension of the filter.
        Overwrite if the filter is not directly based on a distribution."""
        return self.filter_state.dim

    def record_history(self, name, value, pad_with_nan=False, copy_value=True):
        """Append a value to a named history and return the updated history."""
        return self.history.record(
            name, value, pad_with_nan=pad_with_nan, copy_value=copy_value
        )

    def clear_history(self, name=None):
        """Clear a named history or all registered histories."""
        self.history.clear(name)

    def record_filter_state(self, history_name="filter_state"):
        """Store a deep-copied snapshot of the current filter state."""
        return self.record_history(history_name, self.filter_state)

    def record_point_estimate(self, history_name="point_estimate"):
        """Store the current point estimate as a padded numeric history."""
        return self.record_history(
            history_name,
            self.get_point_estimate(),
            pad_with_nan=True,
            copy_value=False,
        )

    def plot_filter_state(self):
        """Plot the filter state."""
        self.filter_state.plot()
