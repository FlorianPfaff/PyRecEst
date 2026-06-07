"""Common delayed-output interface for online fixed-lag smoothers.

Some online smoothers produce an estimate only after a finite look-ahead window
has become available.  They therefore need a small queue of ``(step, state)``
records and a finalization path that flushes the end-of-trajectory tail.  This
module keeps that API in one place without prescribing any particular smoothing
algorithm or state representation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

DelayedStateOutput: TypeAlias = tuple[int, Any]


class DelayedStateOutputMixin:
    """Mixin implementing the delayed-output queue used by fixed-lag smoothers.

    Subclasses should call :meth:`_initialize_delayed_state_outputs` during
    initialization.  The mixin also initializes lazily, which keeps it usable in
    lightweight test doubles and with classes restored through ``__new__``.

    The public convention is intentionally simple: ready outputs are returned as
    ``(step_index, state)`` tuples.  The state object can be a backend array,
    NumPy array, dataclass, tracker state, or any other payload selected by the
    concrete smoother.
    """

    outputs_delayed_states = True

    def _initialize_delayed_state_outputs(
        self,
        *,
        last_emitted_step: int = -1,
    ) -> None:
        """Reset the delayed-output queue and emitted-step cursor."""

        self._ready_queue: list[DelayedStateOutput] = []
        self._last_emitted_step = int(last_emitted_step)

    def _ensure_delayed_state_outputs_initialized(self) -> None:
        if not hasattr(self, "_ready_queue"):
            self._ready_queue = []
        if not hasattr(self, "_last_emitted_step"):
            self._last_emitted_step = -1

    @property
    def last_emitted_step(self) -> int:
        """Index of the latest queued or finalized delayed output."""

        self._ensure_delayed_state_outputs_initialized()
        return int(self._last_emitted_step)

    def _queue_delayed_state(self, step: int, state: Any) -> bool:
        """Queue ``state`` for ``step`` if it has not already been emitted.

        Returns ``True`` when a new item was queued and ``False`` when ``step``
        is negative or not newer than the current delayed-output cursor.
        """

        self._ensure_delayed_state_outputs_initialized()
        step = int(step)
        if step < 0 or step <= int(self._last_emitted_step):
            return False
        self._ready_queue.append((step, state))
        self._last_emitted_step = step
        return True

    def pop_ready_states(self) -> list[DelayedStateOutput]:
        """Return and clear newly available delayed outputs."""

        self._ensure_delayed_state_outputs_initialized()
        ready = list(self._ready_queue)
        self._ready_queue = []
        return ready

    def _finalize_delayed_state_outputs(
        self,
        current_step: int,
        state_for_step: Callable[[int], Any | None],
    ) -> list[DelayedStateOutput]:
        """Flush queued outputs and append the un-emitted trajectory tail.

        ``state_for_step`` is called for every step after the current delayed
        cursor up to and including ``current_step``.  Returning ``None`` skips a
        step, which lets subclasses handle windows that cannot be smoothed.
        """

        self._ensure_delayed_state_outputs_initialized()
        current_step = int(current_step)
        if current_step < 0:
            return self.pop_ready_states()

        ready = self.pop_ready_states()
        start_step = int(self._last_emitted_step) + 1
        for step in range(start_step, current_step + 1):
            state = state_for_step(step)
            if state is None:
                continue
            ready.append((step, state))
            self._last_emitted_step = step

        if int(self._last_emitted_step) < current_step:
            self._last_emitted_step = current_step
        return ready
