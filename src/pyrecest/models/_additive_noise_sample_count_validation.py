"""Validation wiring for additive-noise model sample counts."""

from __future__ import annotations

from functools import wraps
from typing import Any

from .likelihood import _validate_sample_count


def _patch_sample_count(model_cls: type, method_name: str) -> None:
    """Validate ``n`` before an additive-noise sampler is invoked."""

    original = getattr(model_cls, method_name)
    if getattr(original, "_pyrecest_sample_count_checked", False):
        return

    @wraps(original)
    def checked_sample(self, state: Any, n: int = 1, *args: Any, **kwargs: Any):
        return original(self, state, _validate_sample_count(n), *args, **kwargs)

    setattr(checked_sample, "_pyrecest_sample_count_checked", True)
    setattr(model_cls, method_name, checked_sample)


def install_additive_noise_sample_count_validation(
    transition_model_cls: type,
    measurement_model_cls: type,
) -> None:
    """Install consistent sample-count validation on additive-noise models."""

    _patch_sample_count(transition_model_cls, "sample_next")
    _patch_sample_count(measurement_model_cls, "sample_measurement")
