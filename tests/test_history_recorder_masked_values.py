import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
import pytest
from pyrecest.utils import HistoryRecorder


def _to_numpy(value):
    return np.asarray(backend.to_numpy(value), dtype=float)


@pytest.mark.parametrize(
    "value",
    [
        np.ma.array([1.0, 999.0], mask=[False, True]),
        [np.ma.array(1.0, mask=True)],
        np.array([np.ma.masked], dtype=object),
    ],
)
def test_padded_history_record_rejects_masked_values(value):
    recorder = HistoryRecorder()

    with pytest.raises(ValueError, match="masked"):
        recorder.record("state", value, pad_with_nan=True)


@pytest.mark.parametrize(
    "value",
    [
        np.ma.array([1.0, 999.0], mask=[False, True]),
        [np.ma.array(1.0, mask=True)],
        np.array([np.ma.masked], dtype=object),
    ],
)
def test_padded_history_registration_rejects_masked_values(value):
    recorder = HistoryRecorder()

    with pytest.raises(ValueError, match="masked"):
        recorder.register("state", value, pad_with_nan=True)


def test_padded_history_accepts_fully_unmasked_masked_arrays():
    recorder = HistoryRecorder()

    history = recorder.record(
        "state",
        np.ma.array([1.0, 2.0], mask=False),
        pad_with_nan=True,
    )

    npt.assert_allclose(_to_numpy(history), np.array([[1.0], [2.0]]))
