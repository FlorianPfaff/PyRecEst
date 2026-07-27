"""Validation wiring for reusable transition-model capability flags."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._additive_noise_sample_count_validation import (
    install_additive_noise_sample_count_validation,
)
from .likelihood import (
    DensityTransitionModel,
    SampleableTransitionModel,
    _validate_sample_count,
)
from .validation import _validate_bool_flag


def _get_function_is_vectorized(model: SampleableTransitionModel) -> bool:
    return model._function_is_vectorized


def _set_function_is_vectorized(
    model: SampleableTransitionModel,
    value: Any,
) -> None:
    model._function_is_vectorized = _validate_bool_flag(
        value,
        "function_is_vectorized",
    )


def _is_temporal_sample_count(value: Any) -> bool:
    """Return whether a scalar count contains a NumPy temporal value."""

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if value_array.shape != ():
        return False
    if value_array.dtype.kind in {"M", "m"}:
        return True
    return isinstance(value_array.item(), (np.datetime64, np.timedelta64))


def _requested_sample_count(value: Any) -> int:
    if _is_temporal_sample_count(value):
        raise ValueError("n must be a nonnegative integer.")
    return _validate_sample_count(value)


def _patch_sampler_count_check(model_cls) -> None:
    original = model_cls.sample_next
    if getattr(original, "_pyrecest_sampler_count_checked", False):
        return

    def checked_sample_next(self, state, n=1):
        normalized_n = _requested_sample_count(n)
        has_sampler = getattr(self, "_sample_next", None) is not None
        has_count_argument = (
            getattr(self, "_sample_next_count_call_mode", None) is not None
        )
        if has_sampler and not has_count_argument and normalized_n != 1:
            raise TypeError("sample count is not supported by this sampler.")
        return original(self, state, n=normalized_n)

    checked_sample_next._pyrecest_sampler_count_checked = True
    model_cls.sample_next = checked_sample_next


def install_sampleable_transition_validation() -> None:
    """Validate transition-model capability flags and sample counts."""

    SampleableTransitionModel.function_is_vectorized = property(
        _get_function_is_vectorized,
        _set_function_is_vectorized,
        doc="Whether ``sample_next`` accepts a batch of states.",
    )
    _patch_sampler_count_check(SampleableTransitionModel)
    _patch_sampler_count_check(DensityTransitionModel)

    from .additive_noise import (  # pylint: disable=import-outside-toplevel
        AdditiveNoiseMeasurementModel,
        AdditiveNoiseTransitionModel,
    )

    install_additive_noise_sample_count_validation(
        AdditiveNoiseTransitionModel,
        AdditiveNoiseMeasurementModel,
    )
