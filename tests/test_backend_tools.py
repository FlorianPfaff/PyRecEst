import os

import pytest
import pyrecest


def test_backend_tools_report_active_backend():
    active = pyrecest.get_backend_name()

    assert pyrecest.is_backend(active)
    pyrecest.assert_backend(active)


def test_assert_backend_rejects_unexpected_backend():
    active = pyrecest.get_backend_name()
    unexpected = "jax" if active != "jax" else "numpy"

    with pytest.raises(RuntimeError):
        pyrecest.assert_backend(unexpected)


def test_warn_if_backend_env_changed(monkeypatch):
    active = pyrecest.get_backend_name()
    changed = "jax" if active != "jax" else "numpy"
    monkeypatch.setenv("PYRECEST_BACKEND", changed)

    with pytest.warns(RuntimeWarning):
        pyrecest.warn_if_backend_env_changed()

    monkeypatch.setenv("PYRECEST_BACKEND", active)
    pyrecest.warn_if_backend_env_changed()
