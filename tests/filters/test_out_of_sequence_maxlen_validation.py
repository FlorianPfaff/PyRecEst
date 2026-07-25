import numpy as np
import pytest
from pyrecest.filters import FixedLagBuffer, MeasurementTimeBuffer

_BUFFER_TYPES = (FixedLagBuffer, MeasurementTimeBuffer)


@pytest.mark.parametrize("buffer_type", _BUFFER_TYPES)
@pytest.mark.parametrize(
    "maxlen",
    [True, 1.5, 2.0, "2", [2], np.array([2]), 0, -1],
)
def test_fixed_lag_buffers_reject_invalid_maxlen(buffer_type, maxlen):
    with pytest.raises(ValueError, match="maxlen must be a positive integer or None"):
        buffer_type(maxlen=maxlen)


@pytest.mark.parametrize("buffer_type", _BUFFER_TYPES)
@pytest.mark.parametrize("maxlen", [2, np.int64(2), np.array(2, dtype=np.int64)])
def test_fixed_lag_buffers_accept_integer_scalar_maxlen(buffer_type, maxlen):
    buffer = buffer_type(maxlen=maxlen)

    for time in range(3):
        if buffer_type is FixedLagBuffer:
            buffer.append(time, time)
        else:
            buffer.add(time, time)

    assert len(buffer) == 2
    if buffer_type is FixedLagBuffer:
        assert [item.value for item in buffer.items] == [1, 2]
    else:
        assert [record.measurement for record in buffer.measurements] == [1, 2]
