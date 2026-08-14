import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
from pyrecest.utils import HistoryRecorder


def _to_numpy(value):
    return np.asarray(backend.to_numpy(value), dtype=float)


def test_empty_padded_record_preserves_time_axis():
    recorder = HistoryRecorder()
    recorder.register("estimate", pad_with_nan=True)

    recorder.record("estimate", backend.array([1.0, 2.0]), pad_with_nan=True)
    recorder.record("estimate", backend.array([]), pad_with_nan=True)
    history = recorder.record("estimate", backend.array([3.0]), pad_with_nan=True)

    npt.assert_allclose(
        _to_numpy(history),
        np.array(
            [
                [1.0, np.nan, 3.0],
                [2.0, np.nan, np.nan],
            ]
        ),
        equal_nan=True,
    )
