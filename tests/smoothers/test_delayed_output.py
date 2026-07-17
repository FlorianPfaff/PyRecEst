from __future__ import annotations

import numpy as np
import pytest

from pyrecest.smoothers import DelayedStateOutputMixin


class _DummyDelayedOutputSmoother(DelayedStateOutputMixin):
    def __init__(self):
        self._initialize_delayed_state_outputs()


def test_delayed_output_queue_returns_items_once() -> None:
    smoother = _DummyDelayedOutputSmoother()

    assert smoother._queue_delayed_state(0, "zero")
    assert smoother._queue_delayed_state(1, "one")
    assert not smoother._queue_delayed_state(1, "duplicate")

    assert smoother.pop_ready_states() == [(0, "zero"), (1, "one")]
    assert smoother.pop_ready_states() == []
    assert smoother.last_emitted_step == 1


def test_finalize_flushes_queued_states_and_unemitted_tail() -> None:
    smoother = _DummyDelayedOutputSmoother()
    smoother._queue_delayed_state(2, "queued")

    finalized = smoother._finalize_delayed_state_outputs(
        4,
        lambda step: f"tail-{step}",
    )

    assert finalized == [(2, "queued"), (3, "tail-3"), (4, "tail-4")]
    assert smoother.pop_ready_states() == []
    assert smoother.last_emitted_step == 4


def test_finalize_allows_missing_tail_states() -> None:
    smoother = _DummyDelayedOutputSmoother()

    finalized = smoother._finalize_delayed_state_outputs(
        2,
        lambda step: None if step == 1 else f"state-{step}",
    )

    assert finalized == [(0, "state-0"), (2, "state-2")]
    assert smoother.last_emitted_step == 2


def test_delayed_output_mixin_initializes_lazily() -> None:
    smoother = DelayedStateOutputMixin()

    assert smoother.pop_ready_states() == []
    assert smoother._queue_delayed_state(0, "state")
    assert smoother.pop_ready_states() == [(0, "state")]
    assert smoother.last_emitted_step == 0


@pytest.mark.parametrize(
    "step",
    [True, False, 1.5, "2", np.bool_(True), np.array(2.0)],
)
def test_delayed_output_queue_rejects_noninteger_steps(step) -> None:
    smoother = _DummyDelayedOutputSmoother()

    with pytest.raises(ValueError, match="step must be an integer"):
        smoother._queue_delayed_state(step, "state")

    assert smoother.pop_ready_states() == []
    assert smoother.last_emitted_step == -1


def test_delayed_output_rejects_masked_step_cursors() -> None:
    masked_step = np.ma.array(2, mask=True)
    smoother = _DummyDelayedOutputSmoother()

    with pytest.raises(ValueError, match="step must be an integer"):
        smoother._queue_delayed_state(masked_step, "state")
    with pytest.raises(ValueError, match="last_emitted_step must be an integer"):
        smoother._initialize_delayed_state_outputs(last_emitted_step=masked_step)
    with pytest.raises(ValueError, match="current_step must be an integer"):
        smoother._finalize_delayed_state_outputs(masked_step, lambda step: step)

    assert smoother.pop_ready_states() == []
    assert smoother.last_emitted_step == -1


def test_delayed_output_cursor_arguments_require_integer_scalars() -> None:
    smoother = _DummyDelayedOutputSmoother()

    with pytest.raises(ValueError, match="last_emitted_step must be an integer"):
        smoother._initialize_delayed_state_outputs(last_emitted_step=True)

    with pytest.raises(ValueError, match="current_step must be an integer"):
        smoother._finalize_delayed_state_outputs(1.5, lambda step: step)


def test_delayed_output_accepts_numpy_integer_steps() -> None:
    smoother = _DummyDelayedOutputSmoother()

    assert smoother._queue_delayed_state(np.int64(2), "state")
    assert smoother._queue_delayed_state(np.ma.array(3, mask=False), "masked-state")
    assert smoother.pop_ready_states() == [(2, "state"), (3, "masked-state")]
    assert smoother.last_emitted_step == 3
