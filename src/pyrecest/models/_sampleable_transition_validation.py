"""Validation wiring for reusable transition-model capability flags."""

from __future__ import annotations

from typing import Any

from .likelihood import (
    DensityTransitionModel,
    SampleableTransitionModel,
    _validate_sample_count,
)
from .validation import _validate_bool_flag

_SAMPLEABLE_SAMPLE_NEXT = SampleableTransitionModel.sample_next
_DENSITY_SAMPLE_NEXT = DensityTransitionModel.sample_next


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


def _validated_sampler_count(model: Any, n: Any) -> int:
    n = _validate_sample_count(n)
    if model._sample_next_count_call_mode is None and n != 1:
        raise ValueError(
            "sample_next callback does not accept an n argument; only n=1 is supported."
        )
    return n


def _sampleable_sample_next(model: SampleableTransitionModel, state: Any, n: int = 1) -> Any:
    n = _validated_sampler_count(model, n)
    return _SAMPLEABLE_SAMPLE_NEXT(model, state, n=n)


def _density_sample_next(model: DensityTransitionModel, state: Any, n: int = 1) -> Any:
    if model._sample_next is not None:
        n = _validated_sampler_count(model, n)
    return _DENSITY_SAMPLE_NEXT(model, state, n=n)


def install_sampleable_transition_validation() -> None:
    """Install runtime validation hooks for transition-model capabilities."""

    SampleableTransitionModel.function_is_vectorized = property(
        _get_function_is_vectorized,
        _set_function_is_vectorized,
        doc="Whether ``sample_next`` accepts a batch of states.",
    )
    SampleableTransitionModel.sample_next = _sampleable_sample_next
    DensityTransitionModel.sample_next = _density_sample_next
