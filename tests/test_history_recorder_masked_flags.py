import numpy as np
import pytest
from pyrecest.utils import HistoryRecorder


@pytest.mark.parametrize(
    "masked_flag",
    [
        np.ma.array(True, mask=True),
        np.ma.array(False, mask=True),
    ],
)
def test_register_rejects_masked_pad_with_nan(masked_flag):
    recorder = HistoryRecorder()

    with pytest.raises(TypeError, match="pad_with_nan must be a boolean"):
        recorder.register("state", pad_with_nan=masked_flag)


@pytest.mark.parametrize(
    "masked_flag",
    [
        np.ma.array(True, mask=True),
        np.ma.array(False, mask=True),
    ],
)
def test_record_rejects_masked_pad_with_nan(masked_flag):
    recorder = HistoryRecorder()

    with pytest.raises(TypeError, match="pad_with_nan must be a boolean"):
        recorder.record("state", 1.0, pad_with_nan=masked_flag)


@pytest.mark.parametrize(
    "masked_flag",
    [
        np.ma.array(True, mask=True),
        np.ma.array(False, mask=True),
    ],
)
def test_record_rejects_masked_copy_value(masked_flag):
    recorder = HistoryRecorder()

    with pytest.raises(TypeError, match="copy_value must be a boolean"):
        recorder.record("state", 1.0, copy_value=masked_flag)
