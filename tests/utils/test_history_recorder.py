import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
from pyrecest.utils import HistoryRecorder


def _to_numpy(value):
    return np.asarray(backend.to_numpy(value), dtype=float)


def test_padded_history_records_array_like_values():
    recorder = HistoryRecorder()

    history = recorder.record("state", [1.0, 2.0], pad_with_nan=True)
    history = recorder.record("state", (3.0,), pad_with_nan=True)

    npt.assert_allclose(
        _to_numpy(history),
        np.array([[1.0, 3.0], [2.0, np.nan]]),
        equal_nan=True,
    )


def test_padded_history_accepts_scalar_values():
    recorder = HistoryRecorder()

    history = recorder.record("scalar", 1.5, pad_with_nan=True)
    history = recorder.record("scalar", 2.5, pad_with_nan=True)

    npt.assert_allclose(_to_numpy(history), np.array([[1.5, 2.5]]))


def test_padded_history_registers_array_like_initial_value():
    recorder = HistoryRecorder()

    history = recorder.register("state", [1.0, 2.0], pad_with_nan=True)

    npt.assert_allclose(_to_numpy(history), np.array([[1.0], [2.0]]))
