# pylint: disable=no-name-in-module,no-member,too-many-public-methods
"""Fixed-lag helpers for out-of-sequence measurement processing.

The helpers in this module stay close to existing PyRecEst filter APIs.  They
buffer timestamped predict/update calls and, when a delayed measurement arrives
inside the fixed-lag window, replay the affected event history in chronological
order.
"""

import copy
from dataclasses import dataclass

from pyrecest.backend import asarray, atleast_1d, atleast_2d, float64, linalg, transpose
from pyrecest.distributions import GaussianDistribution

from .kalman_filter import KalmanFilter


def _safe_deepcopy(value):
    """Best-effort deepcopy that keeps non-copyable callables by reference."""
    try:
        return copy.deepcopy(value)
    except (TypeError, AttributeError, RuntimeError):
        return value


def _coerce_time(time):
    try:
        time = float(time)
    except (TypeError, ValueError) as exc:
        raise ValueError("time must be a real scalar") from exc
    if time != time or time in (float("inf"), float("-inf")):
        raise ValueError("time must be finite")
    return time


@dataclass(frozen=True)
class TimestampedItem:
    """A value stored with an ordered scalar timestamp."""

    time: float
    value: object
    sequence: int = 0


class FixedLagBuffer:
    """Chronologically ordered buffer with optional fixed-lag trimming.

    Parameters
    ----------
    max_lag : float, optional
        Maximum retained lag relative to the largest timestamp in the buffer.
    maxlen : int, optional
        Maximum number of retained entries after lag trimming.
    copy_values : bool, optional
        If true, values are deep-copied when inserted and returned.
    """

    def __init__(self, max_lag=None, maxlen=None, *, copy_values=True):
        self.max_lag = None if max_lag is None else float(max_lag)
        if self.max_lag is not None and self.max_lag < 0.0:
            raise ValueError("max_lag must be nonnegative or None")
        self.maxlen = None if maxlen is None else int(maxlen)
        if self.maxlen is not None and self.maxlen <= 0:
            raise ValueError("maxlen must be positive or None")
        self.copy_values = bool(copy_values)
        self._items = []
        self._next_sequence = 0

    def __len__(self):
        return len(self._items)

    @property
    def items(self):
        """Return buffered items in chronological order."""
        return tuple(self._copy_item(item) for item in self._items)

    @property
    def latest_time(self):
        """Return the largest buffered timestamp, or ``None`` if empty."""
        if not self._items:
            return None
        return max(item.time for item in self._items)

    @property
    def cutoff_time(self):
        """Return the current fixed-lag cutoff, or ``None`` if unavailable."""
        if self.max_lag is None or self.latest_time is None:
            return None
        return self.latest_time - self.max_lag

    def append(self, time, value):
        """Append ``value`` at ``time`` and return its timestamped record."""
        record = TimestampedItem(
            _coerce_time(time),
            _safe_deepcopy(value) if self.copy_values else value,
            self._next_sequence,
        )
        self._next_sequence += 1
        self._items.append(record)
        self._items.sort(key=lambda item: (item.time, item.sequence))
        self._trim()
        return self._copy_item(record)

    def clear(self):
        """Remove all buffered entries."""
        self._items.clear()

    def is_out_of_sequence(self, time):
        """Return true if ``time`` precedes the current latest timestamp."""
        latest_time = self.latest_time
        return latest_time is not None and _coerce_time(time) < latest_time

    def is_within_lag(self, time):
        """Return true if ``time`` is acceptable under the fixed-lag window."""
        latest_time = self.latest_time
        if latest_time is None or self.max_lag is None:
            return True
        return _coerce_time(time) >= latest_time - self.max_lag

    def latest_at_or_before(self, time):
        """Return the latest record with timestamp at or before ``time``."""
        query_time = _coerce_time(time)
        candidates = [item for item in self._items if item.time <= query_time]
        if not candidates:
            return None
        return self._copy_item(candidates[-1])

    def items_after(self, time):
        """Return all records with timestamp strictly greater than ``time``."""
        query_time = _coerce_time(time)
        return tuple(self._copy_item(item) for item in self._items if item.time > query_time)

    def items_at_or_after(self, time):
        """Return all records with timestamp greater than or equal to ``time``."""
        query_time = _coerce_time(time)
        return tuple(self._copy_item(item) for item in self._items if item.time >= query_time)

    def _copy_item(self, item):
        return TimestampedItem(
            item.time,
            _safe_deepcopy(item.value) if self.copy_values else item.value,
            item.sequence,
        )

    def _trim(self):
        if not self._items:
            return
        if self.max_lag is not None:
            cutoff_time = self.latest_time - self.max_lag
            self._items = [item for item in self._items if item.time >= cutoff_time]
        if self.maxlen is not None and len(self._items) > self.maxlen:
            self._items = self._items[-self.maxlen :]


@dataclass(frozen=True)
class MeasurementRecord:
    """Timestamped measurement plus optional model and metadata."""

    time: float
    measurement: object
    measurement_model: object = None
    metadata: dict | None = None
    sequence: int = 0


class MeasurementTimeBuffer:
    """Small helper for tracking measurement timestamps and OOSM status."""

    def __init__(self, max_lag=None, maxlen=None, *, copy_values=True):
        self._buffer = FixedLagBuffer(max_lag=max_lag, maxlen=maxlen, copy_values=copy_values)

    def __len__(self):
        return len(self._buffer)

    @property
    def latest_time(self):
        return self._buffer.latest_time

    @property
    def cutoff_time(self):
        return self._buffer.cutoff_time

    @property
    def measurements(self):
        """Return buffered measurements in chronological order."""
        return tuple(item.value for item in self._buffer.items)

    def add(self, time, measurement, measurement_model=None, **metadata):
        """Store a measurement and return the resulting record."""
        sequence = self._buffer._next_sequence  # pylint: disable=protected-access
        record = MeasurementRecord(
            time=_coerce_time(time),
            measurement=measurement,
            measurement_model=measurement_model,
            metadata=dict(metadata) if metadata else None,
            sequence=sequence,
        )
        buffered = self._buffer.append(time, record)
        stored = buffered.value
        return MeasurementRecord(
            time=stored.time,
            measurement=stored.measurement,
            measurement_model=stored.measurement_model,
            metadata=_safe_deepcopy(stored.metadata),
            sequence=buffered.sequence,
        )

    append = add

    def is_out_of_sequence(self, time):
        return self._buffer.is_out_of_sequence(time)

    def is_within_lag(self, time):
        return self._buffer.is_within_lag(time)

    def clear(self):
        self._buffer.clear()


@dataclass(frozen=True)
class OutOfSequenceResult:
    """Result returned by an out-of-sequence replay helper."""

    time: float
    final_time: float
    out_of_sequence: bool
    replayed_event_count: int
    accepted: bool = True
    diagnostics: dict | None = None
    filter_state: object = None


@dataclass(frozen=True)
class _ReplayEvent:
    time: float
    sequence: int
    method_name: str
    args: tuple
    kwargs: dict


class _EventReplayMixin:
    """Shared fixed-lag event replay logic for filter wrappers."""

    def _setup_replay(self, filter_object, initial_time=0.0, max_lag=None):
        self._filter_object = filter_object
        self._initial_time = _coerce_time(initial_time)
        self._latest_time = self._initial_time
        self.max_lag = None if max_lag is None else float(max_lag)
        if self.max_lag is not None and self.max_lag < 0.0:
            raise ValueError("max_lag must be nonnegative or None")
        self._initial_state = _safe_deepcopy(self._filter_object.filter_state)
        self._events = []
        self._next_sequence = 0

    @property
    def filter(self):
        """Return the wrapped filter object."""
        return self._filter_object

    @property
    def filter_state(self):
        """Return a snapshot of the wrapped filter state."""
        return _safe_deepcopy(self._filter_object.filter_state)

    @property
    def initial_time(self):
        return self._initial_time

    @property
    def current_time(self):
        return self._latest_time

    @property
    def event_count(self):
        return len(self._events)

    @property
    def events(self):
        """Return buffered replay events in chronological order."""
        return tuple(_safe_deepcopy(event) for event in self._events)

    def _insert_and_apply(self, time, method_name, args=(), kwargs=None):
        event_time = _coerce_time(time)
        if self.max_lag is not None and event_time < self._latest_time - self.max_lag:
            raise ValueError("out-of-sequence event lies outside the fixed-lag window")

        out_of_sequence = event_time < self._latest_time
        event = _ReplayEvent(
            event_time,
            self._next_sequence,
            method_name,
            tuple(_safe_deepcopy(arg) for arg in args),
            _safe_deepcopy(kwargs or {}),
        )
        self._next_sequence += 1
        self._events.append(event)
        self._events.sort(key=lambda item: (item.time, item.sequence))

        if out_of_sequence:
            captured_result = self._replay(capture_event=event)
            replayed_event_count = len([item for item in self._events if item.time >= event.time])
        else:
            captured_result = self._apply_event(event)
            self._latest_time = max(self._latest_time, event_time)
            replayed_event_count = 0

        self._trim_to_lag()
        diagnostics = captured_result if isinstance(captured_result, dict) else None
        accepted = True if diagnostics is None else bool(diagnostics.get("accepted", True))
        return OutOfSequenceResult(
            time=event_time,
            final_time=self._latest_time,
            out_of_sequence=out_of_sequence,
            replayed_event_count=replayed_event_count,
            accepted=accepted,
            diagnostics=diagnostics,
            filter_state=self.filter_state,
        )

    def _apply_event(self, event):
        method = getattr(self._filter_object, event.method_name)
        return method(*event.args, **event.kwargs)

    def _replay(self, capture_event=None):
        self._filter_object.filter_state = _safe_deepcopy(self._initial_state)
        self._latest_time = self._initial_time
        captured_result = None
        for event in self._events:
            result = self._apply_event(event)
            self._latest_time = max(self._latest_time, event.time)
            if event is capture_event:
                captured_result = result
        return captured_result

    def _trim_to_lag(self):
        if self.max_lag is None or not self._events:
            return
        cutoff_time = self._latest_time - self.max_lag
        if self._events[0].time >= cutoff_time:
            return

        current_state = self.filter_state
        self._filter_object.filter_state = _safe_deepcopy(self._initial_state)
        remaining_events = []
        base_time = self._initial_time
        for event in self._events:
            if event.time < cutoff_time:
                self._apply_event(event)
                base_time = event.time
            else:
                remaining_events.append(event)
        self._initial_state = self.filter_state
        self._initial_time = base_time
        self._events = remaining_events
        self._filter_object.filter_state = current_state


class OutOfSequenceKalmanUpdater(_EventReplayMixin):
    """Fixed-lag OOSM processor for :class:`KalmanFilter`."""

    def __init__(self, kalman_filter, initial_time=0.0, max_lag=None):
        filter_object = kalman_filter if isinstance(kalman_filter, KalmanFilter) else KalmanFilter(kalman_filter)
        self._setup_replay(filter_object, initial_time=initial_time, max_lag=max_lag)

    def predict_linear(self, time, system_matrix, sys_noise_cov, sys_input=None):
        """Record/apply a timestamped linear-Gaussian prediction."""
        return self._insert_and_apply(time, "predict_linear", (system_matrix, sys_noise_cov, sys_input))

    def predict_model(self, time, transition_model):
        """Record/apply a timestamped structural transition-model prediction."""
        return self._insert_and_apply(time, "predict_model", (transition_model,))

    def update_linear(
        self,
        time,
        measurement,
        measurement_matrix,
        meas_noise,
        *,
        return_diagnostics=False,
        scale=1.0,
        action="updated",
    ):
        """Record/apply a timestamped linear-Gaussian measurement update."""
        return self._insert_and_apply(
            time,
            "update_linear",
            (measurement, measurement_matrix, meas_noise),
            {"return_diagnostics": return_diagnostics, "scale": scale, "action": action},
        )

    def update_linear_robust(
        self,
        time,
        measurement,
        measurement_matrix,
        meas_noise,
        *,
        robust_update="student-t",
        gate_threshold=None,
        student_t_dof=4.0,
        huber_threshold=2.0,
        inflation_alpha=1.0,
        return_diagnostics=False,
    ):
        """Record/apply a timestamped robust linear-Gaussian update."""
        return self._insert_and_apply(
            time,
            "update_linear_robust",
            (measurement, measurement_matrix, meas_noise),
            {
                "robust_update": robust_update,
                "gate_threshold": gate_threshold,
                "student_t_dof": student_t_dof,
                "huber_threshold": huber_threshold,
                "inflation_alpha": inflation_alpha,
                "return_diagnostics": return_diagnostics,
            },
        )

    def update_model(self, time, measurement_model, measurement, *, return_diagnostics=False, scale=1.0, action="updated"):
        """Record/apply a timestamped structural measurement-model update."""
        return self._insert_and_apply(
            time,
            "update_model",
            (measurement_model, measurement),
            {"return_diagnostics": return_diagnostics, "scale": scale, "action": action},
        )

    def update_model_robust(self, time, measurement_model, measurement, **kwargs):
        """Record/apply a timestamped robust structural measurement-model update."""
        return self._insert_and_apply(time, "update_model_robust", (measurement_model, measurement), kwargs)


class OutOfSequenceParticleUpdater(_EventReplayMixin):
    """Fixed-lag OOSM processor for particle filters.

    Replaying stochastic transition models draws new process noise.  For
    bitwise reproducibility, use deterministic transitions or caller-controlled
    random seeds.
    """

    def __init__(self, particle_filter, initial_time=0.0, max_lag=None):
        if not hasattr(particle_filter, "filter_state"):
            raise TypeError("particle_filter must expose a filter_state property")
        self._setup_replay(particle_filter, initial_time=initial_time, max_lag=max_lag)

    def predict_model(self, time, transition_model):
        """Record/apply a timestamped particle transition-model prediction."""
        return self._insert_and_apply(time, "predict_model", (transition_model,))

    def predict_nonlinear(
        self,
        time,
        f,
        noise_distribution=None,
        function_is_vectorized=True,
        shift_instead_of_add=None,
    ):
        """Record/apply a timestamped nonlinear particle prediction."""
        kwargs = {"noise_distribution": noise_distribution, "function_is_vectorized": function_is_vectorized}
        if shift_instead_of_add is not None:
            kwargs["shift_instead_of_add"] = shift_instead_of_add
        return self._insert_and_apply(time, "predict_nonlinear", (f,), kwargs)

    def update_model(self, time, measurement_model, measurement=None):
        """Record/apply a timestamped particle measurement-model update."""
        return self._insert_and_apply(time, "update_model", (measurement_model,), {"measurement": measurement})

    def update_nonlinear_using_likelihood(self, time, likelihood, measurement=None):
        """Record/apply a timestamped likelihood-based particle update."""
        return self._insert_and_apply(time, "update_nonlinear_using_likelihood", (likelihood,), {"measurement": measurement})

    update_with_likelihood = update_nonlinear_using_likelihood


def _as_vector(x, name):
    x = atleast_1d(asarray(x, dtype=float64))
    if len(x.shape) != 1:
        raise ValueError(f"{name} must be one-dimensional after coercion")
    return x


def _as_matrix(x, name):
    x = atleast_2d(asarray(x, dtype=float64))
    if len(x.shape) != 2:
        raise ValueError(f"{name} must be two-dimensional after coercion")
    return x


def retrodict_linear_gaussian(
    mean,
    covariance,
    system_matrix,
    sys_input=None,
    sys_noise_cov=None,
    *,
    remove_process_noise=False,
):
    """Retrodict a linear-Gaussian state through a square transition matrix.

    For ``x_next = F x_prev + u + w``, this computes the Gaussian on
    ``x_prev`` implied by the supplied Gaussian on ``x_next``.  By default the
    covariance is transformed as ``inv(F) P_next inv(F).T``.  If
    ``remove_process_noise`` is true, ``sys_noise_cov`` is subtracted first.
    """
    mean = _as_vector(mean, "mean")
    covariance = _as_matrix(covariance, "covariance")
    system_matrix = _as_matrix(system_matrix, "system_matrix")
    if system_matrix.shape[0] != system_matrix.shape[1]:
        raise ValueError("system_matrix must be square for linear retrodiction")
    if mean.shape[0] != system_matrix.shape[0]:
        raise ValueError("mean has incompatible shape")
    if covariance.shape != (mean.shape[0], mean.shape[0]):
        raise ValueError("covariance has incompatible shape")

    shifted_mean = mean
    if sys_input is not None:
        sys_input = _as_vector(sys_input, "sys_input")
        if sys_input.shape != mean.shape:
            raise ValueError("sys_input has incompatible shape")
        shifted_mean = shifted_mean - sys_input

    effective_covariance = covariance
    if remove_process_noise:
        if sys_noise_cov is None:
            raise ValueError("sys_noise_cov is required when remove_process_noise is true")
        sys_noise_cov = _as_matrix(sys_noise_cov, "sys_noise_cov")
        if sys_noise_cov.shape != covariance.shape:
            raise ValueError("sys_noise_cov has incompatible shape")
        effective_covariance = covariance - sys_noise_cov

    previous_mean = linalg.solve(system_matrix, shifted_mean)
    left_solved = linalg.solve(system_matrix, effective_covariance)
    previous_covariance = transpose(linalg.solve(system_matrix, transpose(left_solved)))
    previous_covariance = 0.5 * (previous_covariance + transpose(previous_covariance))
    return previous_mean, previous_covariance


def retrodict_linear_gaussian_state(
    state,
    system_matrix,
    sys_input=None,
    sys_noise_cov=None,
    *,
    remove_process_noise=False,
):
    """Return a :class:`GaussianDistribution` retrodicted one linear step."""
    if not isinstance(state, GaussianDistribution):
        raise TypeError("state must be a GaussianDistribution")
    previous_mean, previous_covariance = retrodict_linear_gaussian(
        state.mu,
        state.C,
        system_matrix,
        sys_input=sys_input,
        sys_noise_cov=sys_noise_cov,
        remove_process_noise=remove_process_noise,
    )
    return GaussianDistribution(previous_mean, previous_covariance, check_validity=False)
