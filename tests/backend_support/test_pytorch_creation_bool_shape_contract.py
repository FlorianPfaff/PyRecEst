import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_creation_helpers_reject_boolean_shapes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "pytorch"

bad_shapes = (
    True,
    np.bool_(False),
    np.array(True, dtype=np.bool_),
    [True, 2],
    [np.bool_(True), 2],
    [True, False],
    np.array([True, False], dtype=np.bool_),
    np.array([True, 2], dtype=object),
    torch.tensor(True),
    torch.tensor([True, False]),
)

for creation_backend in (backend, raw_backend):
    for helper_name, extra_args in (
        ("empty", ()),
        ("zeros", ()),
        ("ones", ()),
        ("full", (7,)),
    ):
        helper = getattr(creation_backend, helper_name)
        for bad_shape in bad_shapes:
            try:
                helper(bad_shape, *extra_args)
            except TypeError:
                pass
            else:
                raise AssertionError(f"{helper_name} accepted boolean shape {bad_shape!r}")
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_creation_helpers_reject_boolean_shapes_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

bad_shapes = (
    True,
    np.bool_(False),
    np.array(True, dtype=np.bool_),
    [True, 2],
    [np.bool_(True), 2],
    [True, False],
    np.array([True, False], dtype=np.bool_),
    np.array([True, 2], dtype=object),
    torch.tensor(True),
    torch.tensor([True, False]),
)

for helper_name, extra_args in (
    ("empty", ()),
    ("zeros", ()),
    ("ones", ()),
    ("full", (7,)),
):
    helper = getattr(raw_backend, helper_name)
    for bad_shape in bad_shapes:
        try:
            helper(bad_shape, *extra_args)
        except TypeError:
            pass
        else:
            raise AssertionError(f"raw {helper_name} accepted boolean shape {bad_shape!r}")
""",
    )

    assert result.returncode == 0, result.stderr
