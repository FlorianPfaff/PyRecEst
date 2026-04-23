from abc import ABC

from pyrecest.utils.history_recorder import HistoryRecorder


class AbstractTrackerWithLogging(ABC):
    def __init__(self, log_prior_estimates=False, log_posterior_estimates=False):
        self.log_prior_estimates = log_prior_estimates
        self.log_posterior_estimates = log_posterior_estimates
        self.history = HistoryRecorder()
        self.prior_estimates_over_time = None
        self.posterior_estimates_over_time = None
        self.prior_extents_over_time = None
        self.posterior_extents_over_time = None

        if log_prior_estimates:
            self.prior_estimates_over_time = self.history.register(
                "prior_estimates", pad_with_nan=True
            )

        if log_posterior_estimates:
            self.posterior_estimates_over_time = self.history.register(
                "posterior_estimates", pad_with_nan=True
            )

    def record_history(self, name, value, pad_with_nan=False, copy_value=True):
        """Append a value to a named history and return the updated history."""
        history = self.history.record(
            name, value, pad_with_nan=pad_with_nan, copy_value=copy_value
        )
        if name == "prior_estimates":
            self.prior_estimates_over_time = history
        elif name == "posterior_estimates":
            self.posterior_estimates_over_time = history
        elif name == "prior_extents":
            self.prior_extents_over_time = history
        elif name == "posterior_extents":
            self.posterior_extents_over_time = history
        return history

    def clear_history(self, name=None):
        """Clear a named history or all registered histories."""
        self.history.clear(name)
        if name is None or name == "prior_estimates":
            if "prior_estimates" in self.history:
                self.prior_estimates_over_time = self.history["prior_estimates"]
        if name is None or name == "posterior_estimates":
            if "posterior_estimates" in self.history:
                self.posterior_estimates_over_time = self.history["posterior_estimates"]
        if name is None or name == "prior_extents":
            if "prior_extents" in self.history:
                self.prior_extents_over_time = self.history["prior_extents"]
        if name is None or name == "posterior_extents":
            if "posterior_extents" in self.history:
                self.posterior_extents_over_time = self.history["posterior_extents"]

    def _record_estimates(self, history_name, curr_ests):
        """Record a numeric history with NaN padding for varying dimensions."""
        return self.record_history(
            history_name, curr_ests, pad_with_nan=True, copy_value=False
        )

    def _store_estimates(self, curr_ests, estimates_over_time):
        """Backward-compatible helper for appending padded estimate histories."""
        return HistoryRecorder.append_padded(curr_ests, estimates_over_time)
