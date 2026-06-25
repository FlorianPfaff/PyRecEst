"""Small helpers for inspecting the process-global PyRecEst backend."""

from __future__ import annotations

import os
import warnings


def get_backend_name() -> str:
    """Return the backend selected at import time."""
    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    return backend.__backend_name__  # pylint: disable=no-member


def is_backend(expected: str) -> bool:
    """Return whether the active backend equals ``expected``."""
    return get_backend_name() == expected


def _normalize_expected_backend_names(
    expected: str | tuple[str, ...],
) -> tuple[str, ...]:
    message = "expected must name at least one backend."
    if isinstance(expected, str):
        names = (expected,)
    else:
        try:
            names = tuple(expected)
        except TypeError as exc:
            raise ValueError(message) from exc
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(message)
    return names


def assert_backend(expected: str | tuple[str, ...]) -> None:
    """Raise ``RuntimeError`` unless the active backend matches ``expected``.

    Parameters
    ----------
    expected : str or tuple[str, ...]
        Allowed backend name or names.
    """
    active = get_backend_name()
    expected_names = _normalize_expected_backend_names(expected)
    if active not in expected_names:
        allowed = ", ".join(expected_names)
        raise RuntimeError(
            f"Expected PyRecEst backend {allowed}; active backend is {active}."
        )


def warn_if_backend_env_changed() -> None:
    """Warn when ``PYRECEST_BACKEND`` no longer matches the imported backend.

    Backend selection is process-global and import-time only. Changing the
    environment variable after importing :mod:`pyrecest` does not switch the
    already constructed backend facade.
    """
    active = get_backend_name()
    requested = os.environ.get("PYRECEST_BACKEND", active)
    if requested != active:
        warnings.warn(
            "PYRECEST_BACKEND was changed after pyrecest was imported. "
            f"The active backend remains {active!r}; the environment now requests "
            f"{requested!r}. Start a new Python process to switch backends.",
            RuntimeWarning,
            stacklevel=2,
        )
