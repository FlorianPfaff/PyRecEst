from __future__ import annotations

import numpy as np
from mem_qkf.trackers_elliptical.mem_qkf_ipls import (
    FixedIntervalMEMIPLSSmoother,
    _IPLSSnapshot,
)


class ScanOnlyMEMIPLSSmoother(FixedIntervalMEMIPLSSmoother):
    """Current-scan IPLS wrapper for MEM-QKF.

    This variant reuses the fixed-interval IPLS scan-factor approximation, but
    applies it only to the current measurement set.  After every ``update`` it
    emits the current step immediately through ``pop_ready_states()`` and writes
    the iterated posterior back into the live MEM-QKF state before the next
    ``predict``.  It is therefore usable as a stronger online filter rather than
    as a temporal smoother.
    """

    outputs_delayed_states = True

    def __init__(self, *args, lag: int = 0, **kwargs):
        if int(lag) != 0:
            raise ValueError("ScanOnlyMEMIPLSSmoother only supports lag=0")
        self.lag = 0
        super().__init__(*args, **kwargs)
        self.lag = 0
        self._ready_queue: list[tuple[int, np.ndarray]] = []

    def update(self, Z: np.ndarray):
        FixedIntervalMEMIPLSSmoother.update(self, Z)
        return self._run_scan_only_smoothing()

    def pop_ready_states(self) -> list[tuple[int, np.ndarray]]:
        ready = self._ready_queue
        self._ready_queue = []
        return ready

    def finalize_smoothing(self) -> list[tuple[int, np.ndarray]]:
        ready = self._ready_queue
        self._ready_queue = []
        return ready

    def _run_scan_only_smoothing(self) -> np.ndarray:
        step = self._current_step
        local = self._filtered_history[step].copy()
        for _ in range(self.n_iterations):
            local = self._posterior_linearized_scan_update(
                prior=self._prior_history[step],
                measurements=self._measurements[step],
                linearization=local,
            )

        self._filtered_history[step] = local.copy()
        self._smoothed_history[step] = local.copy()
        self._apply_snapshot_to_tracker(local)

        state = self._public_state_from_internal(local.mean)
        if step > self._last_emitted_step:
            self._ready_queue.append((step, state.copy()))
            self._last_emitted_step = step
        return state

    def _apply_snapshot_to_tracker(self, snapshot: _IPLSSnapshot) -> None:
        self.m = snapshot.mean[:4].copy()
        self.C_m = self._regularize_covariance(snapshot.cov[:4, :4])
        self.tracker.shape_state = snapshot.mean[4:].copy()
        self.tracker.shape_covariance = self._regularize_covariance(
            snapshot.cov[4:, 4:]
        )


ZeroLagMEMIPLSSmoother = ScanOnlyMEMIPLSSmoother
