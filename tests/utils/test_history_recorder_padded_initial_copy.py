import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.utils import HistoryRecorder


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="JAX arrays are immutable and cannot expose mutable input aliasing.",
)
def test_padded_initial_value_does_not_alias_caller_array():
    initial_value = array([1.0, 2.0])
    recorder = HistoryRecorder()

    registered_history = recorder.register(
        "estimate",
        initial_value=initial_value,
        pad_with_nan=True,
    )
    initial_value[0] = 99.0

    assert registered_history is recorder["estimate"]
    assert float(recorder["estimate"][0, 0]) == pytest.approx(1.0)
    assert float(recorder["estimate"][1, 0]) == pytest.approx(2.0)
