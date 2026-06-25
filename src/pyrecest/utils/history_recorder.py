"""Utilities for recording named histories for filters and trackers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import nan
from typing import Any

from pyrecest import backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, asarray, full, hstack, pad


@dataclass
class _HistoryEntry:
    values: Any
    pad_with_nan: bool


def _validate_bool_flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    try:
        value_array = asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a boolean") from exc
    if getattr(value_array, "shape", None) == ():
        scalar = value_array.item()
        if isinstance(scalar, bool):
            return bool(scalar)
    raise TypeError(f"{name} must be a boolean")


_INVALID_PADDED_HISTORY_DTYPE_PREFIXES = (
    "<u",
    ">u",
    "|u",
    "=u",
    "<s",
    ">s",
    "|s",
    "=s",
)


def _is_invalid_padded_history_dtype(dtype: Any) -> bool:
    dtype_name = str(dtype).lower()
    return (
        "bool" in dtype_name
        or "complex" in dtype_name
        or "object" in dtype_name
        or "str" in dtype_name
        or "bytes" in dtype_name
        or "datetime" in dtype_name
        or "timedelta" in dtype_name
        or dtype_name.startswith(_INVALID_PADDED_HISTORY_DTYPE_PREFIXES)
    )


def _as_padded_numeric_array(curr_ests):
    message = "padded history values must be real numeric"
    try:
        raw_curr_ests = asarray(curr_ests)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(message) from exc

    if _is_invalid_padded_history_dtype(getattr(raw_curr_ests, "dtype", None)):
        raise TypeError(message)

    try:
        return asarray(raw_curr_ests, dtype=float)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(message) from exc


class HistoryRecorder:
    """Record and retrieve named histories.

    Histories come in two flavors:

    * padded numeric histories, which are stored as a 2-D backend array and can
      grow in their first dimension over time while earlier columns are padded
      with NaNs, and
    * generic histories, which are stored as Python lists of deep-copied values.
    """

    def __init__(self):
        self._entries: dict[str, _HistoryEntry] = {}

    def register(self, name: str, initial_value=None, pad_with_nan: bool = False):
        """Register a named history and return its storage object."""
        pad_with_nan = _validate_bool_flag(pad_with_nan, "pad_with_nan")
        if name in self._entries:
            raise ValueError(f"History '{name}' is already registered.")

        if initial_value is None:
            initial_value = array([[]]) if pad_with_nan else []
        elif pad_with_nan:
            initial_value = self._ensure_2d(initial_value)
        else:
            initial_value = copy.deepcopy(initial_value)

        self._entries[name] = _HistoryEntry(initial_value, pad_with_nan)
        return self._entries[name].values

    def record(
        self,
        name: str,
        value,
        pad_with_nan: bool | None = None,
        copy_value: bool = True,
    ):
        """Append a value to the named history and return the updated history."""
        copy_value = _validate_bool_flag(copy_value, "copy_value")
        if pad_with_nan is not None:
            pad_with_nan = _validate_bool_flag(pad_with_nan, "pad_with_nan")

        if name not in self._entries:
            self.register(
                name, pad_with_nan=False if pad_with_nan is None else pad_with_nan
            )

        entry = self._entries[name]
        if pad_with_nan is not None and entry.pad_with_nan != pad_with_nan:
            raise ValueError(
                f"History '{name}' was registered with pad_with_nan={entry.pad_with_nan}."
            )

        if entry.pad_with_nan:
            entry.values = self.append_padded(value, entry.values)
        else:
            if not isinstance(entry.values, list):
                raise TypeError(
                    f"History '{name}' is expected to be list-backed, got {type(entry.values)}."
                )
            entry.values.append(copy.deepcopy(value) if copy_value else value)

        return entry.values

    def clear(self, name: str | None = None):
        """Clear a named history or all histories in place."""
        if name is None:
            for history_name in list(self._entries):
                self.clear(history_name)
            return None

        entry = self._entries[name]
        entry.values = array([[]]) if entry.pad_with_nan else []
        return entry.values

    def get(self, name: str, default=None):
        """Return the stored history for *name*."""
        entry = self._entries.get(name)
        if entry is None:
            return default
        return entry.values

    def items(self):
        """Iterate over `(name, history)` pairs."""
        for name, entry in self._entries.items():
            yield name, entry.values

    def keys(self):
        """Return the registered history names."""
        return self._entries.keys()

    def values(self):
        """Return the stored histories."""
        for entry in self._entries.values():
            yield entry.values

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str):
        return self._entries[name].values

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _ensure_2d(curr_ests):
        curr_ests = _as_padded_numeric_array(curr_ests)
        if curr_ests.ndim == 0:
            curr_ests = curr_ests.reshape(1, 1)
        elif curr_ests.ndim != 2 or curr_ests.shape[1] != 1:
            curr_ests = curr_ests.reshape(-1, 1)
        return curr_ests

    @staticmethod
    def append_padded(curr_ests, estimates_over_time):
        """Append a column to a possibly growing 2-D history array."""
        curr_ests = HistoryRecorder._ensure_2d(curr_ests)

        m, t = estimates_over_time.shape
        n = curr_ests.shape[0]

        if n <= m:
            curr_ests = pad(
                curr_ests, ((0, m - n), (0, 0)), mode="constant", constant_values=nan
            )
            estimates_over_time_new = hstack((estimates_over_time, curr_ests))
        else:
            estimates_over_time_new = full((n, t + 1), nan)
            if backend.__backend_name__ != "jax":
                estimates_over_time_new[:m, :t] = estimates_over_time
                estimates_over_time_new[:, -1] = curr_ests.flatten()
            else:
                estimates_over_time_new = estimates_over_time_new.at[:m, :t].set(
                    estimates_over_time
                )
                estimates_over_time_new = estimates_over_time_new.at[:, -1].set(
                    curr_ests.flatten()
                )

        return estimates_over_time_new
