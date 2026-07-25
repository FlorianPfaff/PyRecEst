from __future__ import annotations

import pytest
from pyrecest.smoothers import DelayedStateOutputMixin


class _DummyDelayedOutputSmoother(DelayedStateOutputMixin):
    def __init__(self):
        self._initialize_delayed_state_outputs()


def test_finalize_failure_preserves_queue_and_cursor_for_retry() -> None:
    smoother = _DummyDelayedOutputSmoother()
    smoother._queue_delayed_state(1, "queued")

    def failing_state_for_step(step: int) -> str:
        if step == 3:
            raise RuntimeError("tail generation failed")
        return f"tail-{step}"

    with pytest.raises(RuntimeError, match="tail generation failed"):
        smoother._finalize_delayed_state_outputs(3, failing_state_for_step)

    assert smoother._ready_queue == [(1, "queued")]
    assert smoother.last_emitted_step == 1

    finalized = smoother._finalize_delayed_state_outputs(
        3,
        lambda step: f"tail-{step}",
    )

    assert finalized == [(1, "queued"), (2, "tail-2"), (3, "tail-3")]
    assert smoother.pop_ready_states() == []
    assert smoother.last_emitted_step == 3
