import numpy as np
import pytest

from pyrecest import backend
from pyrecest.utils.history_recorder import HistoryRecorder


def _as_numpy(value):
    return backend.to_numpy(value)


def _swapped_uint16_dtype():
    return np.dtype("uint16").newbyteorder("S")


def test_padded_history_accepts_non_native_unsigned_integer_arrays():
    recorder = HistoryRecorder()
    initial = np.array([1, 2], dtype=_swapped_uint16_dtype())

    history = recorder.register("counts", initial, pad_with_nan=True)

    assert _as_numpy(history).tolist() == [[1.0], [2.0]]

    updated = recorder.record("counts", np.array([3], dtype=_swapped_uint16_dtype()))

    expected = np.array([[1.0, 3.0], [2.0, np.nan]])
    assert np.allclose(_as_numpy(updated), expected, equal_nan=True)


def test_padded_history_still_rejects_unicode_arrays():
    recorder = HistoryRecorder()

    with pytest.raises(TypeError, match="padded history values must be real numeric"):
        recorder.register("bad", np.array(["1"], dtype=np.dtype("U1")), pad_with_nan=True)
