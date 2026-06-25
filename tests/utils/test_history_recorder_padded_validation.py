import unittest

import numpy as np
from pyrecest.utils import HistoryRecorder


class HistoryRecorderPaddedValidationTest(unittest.TestCase):
    def test_padded_history_rejects_boolean_values(self):
        recorder = HistoryRecorder()

        for value in ([True], np.array([False]), np.array([1.0, True], dtype=object)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "padded history values"):
                    recorder.record("state", value, pad_with_nan=True)

    def test_padded_history_rejects_text_values(self):
        recorder = HistoryRecorder()

        for value in (["1.0"], np.array(["2.0"]), np.array([1.0, "2.0"], dtype=object)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "padded history values"):
                    recorder.record("state", value, pad_with_nan=True)

    def test_padded_history_rejects_complex_initial_values(self):
        recorder = HistoryRecorder()

        with self.assertRaisesRegex(TypeError, "padded history values"):
            recorder.register(
                "state",
                np.array([1.0 + 2.0j]),
                pad_with_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
