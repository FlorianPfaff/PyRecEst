import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyrecest.backend as backend
import pyrecest.reproducibility as reproducibility
from pyrecest.reproducibility import (
    _get_pytorch_cuda_random_state,
    _normalize_seed,
    _set_pytorch_cuda_random_state,
    preserve_backend_random_state,
)


class ReproducibilityValidationTest(unittest.TestCase):
    def test_normalize_seed_rejects_text_values(self):
        for value in (
            "1",
            np.array("1"),
            bytes([49]),
            bytearray([49]),
            np.bytes_(bytes([49])),
        ):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(
                    ValueError,
                    "seed must be a non-negative integer or None",
                ):
                    _normalize_seed(value)

    def test_normalize_seed_rejects_boolean_values(self):
        for value in (True, np.bool_(True), np.array(True)):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(
                    ValueError,
                    "seed must be a non-negative integer or None",
                ):
                    _normalize_seed(value)

    def test_normalize_seed_preserves_numeric_scalar_support(self):
        self.assertIsNone(_normalize_seed(None))
        self.assertEqual(_normalize_seed(1), 1)
        self.assertEqual(_normalize_seed(2.0), 2)
        self.assertEqual(_normalize_seed(np.array(3)), 3)
        self.assertEqual(_normalize_seed(np.array(4.0)), 4)

    def test_pytorch_cuda_random_state_helpers_snapshot_and_restore(self):
        raw_state = [bytearray(b"cuda-state")]
        fake_cuda = mock.Mock()
        fake_cuda.is_available.return_value = True
        fake_cuda.get_rng_state_all.return_value = raw_state
        fake_torch = SimpleNamespace(cuda=fake_cuda)

        with mock.patch.object(backend, "__backend_name__", "pytorch", create=True):
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                saved_state = _get_pytorch_cuda_random_state()
                _set_pytorch_cuda_random_state(saved_state)

        self.assertEqual(saved_state, raw_state)
        self.assertIsNot(saved_state, raw_state)
        self.assertIsNot(saved_state[0], raw_state[0])
        fake_cuda.set_rng_state_all.assert_called_once_with(saved_state)

    def test_preserve_backend_random_state_restores_cuda_before_numpy(self):
        events = []
        backend_state = ("backend-state",)
        cuda_state = [("cuda-state",)]
        numpy_state = ("numpy-state",)

        with mock.patch.object(
            reproducibility,
            "get_backend_random_state",
            return_value=backend_state,
        ), mock.patch.object(
            reproducibility,
            "set_backend_random_state",
            side_effect=lambda state: events.append(("backend", state)),
        ), mock.patch.object(
            reproducibility,
            "_get_pytorch_cuda_random_state",
            return_value=cuda_state,
        ), mock.patch.object(
            reproducibility,
            "_set_pytorch_cuda_random_state",
            side_effect=lambda state: events.append(("cuda", state)),
        ), mock.patch.object(
            np.random,
            "get_state",
            return_value=numpy_state,
        ), mock.patch.object(
            np.random,
            "set_state",
            side_effect=lambda state: events.append(("numpy", state)),
        ):
            with preserve_backend_random_state():
                pass

        self.assertEqual(
            events,
            [
                ("backend", backend_state),
                ("cuda", cuda_state),
                ("numpy", numpy_state),
            ],
        )


if __name__ == "__main__":
    unittest.main()
