"""Named pairwise feature helpers for probabilistic association models."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    all,
    asarray,
    clip,
    exp,
    float64,
    isfinite,
    isinf,
    isnan,
    log,
    stack,
    where,
)

_COST_MODE = Literal["negative_log_probability", "one_minus_probability"]
FeatureTransform = Callable[[Mapping[str, Any]], Any]
_PROBABILITY_CLIP_ERROR = "probability_clip must be a finite real scalar in (0, 0.5)"
_PREDICTED_PROBABILITY_ERROR = "predicted probabilities must be real numeric"
_PAIRWISE_COST_ERROR = "predicted pairwise costs must be real numeric"
_TEXT_SCALAR_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)
_NON_REAL_SCALAR_TYPES = (
    bool,
    np.bool_,
    complex,
    np.complexfloating,
    np.datetime64,
    np.timedelta64,
    *_TEXT_SCALAR_TYPES,
)
_TEMPORAL_DTYPE_KINDS = {"M", "m"}


@dataclass(frozen=True, init=False)
class NamedPairwiseFeatureSchema:
    """Named schema for building pairwise feature tensors from components."""

    feature_names: tuple[str, ...]
    transforms: Mapping[str, FeatureTransform]

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        transforms: Mapping[str, FeatureTransform] | None = None,
    ) -> None:
        normalized_names = _normalize_feature_names(feature_names)
        normalized_transforms = _normalize_feature_transforms(
            normalized_names, transforms
        )
        object.__setattr__(self, "feature_names", normalized_names)
        object.__setattr__(self, "transforms", MappingProxyType(normalized_transforms))

    def __len__(self) -> int:
        return len(self.feature_names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.feature_names)

    def feature_index(self, feature_name: str) -> int:
        try:
            return self.feature_names.index(feature_name)
        except ValueError as exc:
            raise KeyError(f"Unknown feature name {feature_name!r}") from exc

    def build_tensor(self, components: Mapping[str, Any]) -> Any:
        return pairwise_feature_tensor(
            components, self.feature_names, transforms=self.transforms
        )


def pairwise_feature_tensor(
    components: Mapping[str, Any],
    feature_names: Sequence[str] | NamedPairwiseFeatureSchema,
    transforms: Mapping[str, FeatureTransform] | None = None,
) -> Any:
    """Build a pairwise feature tensor from named component planes.

    All feature planes must have the same shape and contain real numeric values.
    Non-finite values are converted to finite sentinels: ``nan -> 0``,
    ``+inf -> 1e6``, and ``-inf -> -1e6``.
    """
    if isinstance(feature_names, NamedPairwiseFeatureSchema):
        if transforms is not None:
            raise ValueError(
                "transforms must be omitted when feature_names is a schema"
            )
        return pairwise_feature_tensor(
            components, feature_names.feature_names, transforms=feature_names.transforms
        )

    normalized_names = _normalize_feature_names(feature_names)
    normalized_transforms = _normalize_feature_transforms(normalized_names, transforms)
    feature_planes = [
        _component_feature(components, feature_name, normalized_transforms)
        for feature_name in normalized_names
    ]
    reference_shape = feature_planes[0].shape
    for feature_name, feature_plane in zip(normalized_names, feature_planes):
        if feature_plane.shape != reference_shape:
            raise ValueError(
                f"Feature {feature_name!r} has shape {feature_plane.shape}, expected {reference_shape}"
            )
    return stack(feature_planes, axis=-1)


@dataclass(frozen=True, init=False)
class CalibratedPairwiseAssociationModel:
    """Association model wrapper that keeps a named pairwise feature schema."""

    model: Any
    schema: NamedPairwiseFeatureSchema
    probability_clip: float

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str] | None = None,
        *,
        schema: NamedPairwiseFeatureSchema | None = None,
        transforms: Mapping[str, FeatureTransform] | None = None,
        probability_clip: float = 1.0e-12,
    ) -> None:
        if schema is not None and feature_names is not None:
            raise ValueError("Provide either schema or feature_names, not both")
        if schema is not None and transforms is not None:
            raise ValueError("transforms must be attached to the provided schema")
        if schema is None:
            if feature_names is None:
                raise ValueError("feature_names or schema is required")
            schema = NamedPairwiseFeatureSchema(feature_names, transforms=transforms)
        probability_clip = _normalize_probability_clip(probability_clip)

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "probability_clip", probability_clip)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.schema.feature_names

    def build_feature_tensor(self, components: Mapping[str, Any]) -> Any:
        return self.schema.build_tensor(components)

    def predict_match_probability(self, features_or_components: Any) -> Any:
        features = self._features_from_components_or_tensor(features_or_components)
        if hasattr(self.model, "predict_match_probability"):
            probabilities = self.model.predict_match_probability(features)
        elif hasattr(self.model, "predict_proba"):
            probabilities = self._predict_proba_probability(features)
        elif hasattr(self.model, "pairwise_cost_matrix"):
            costs = _as_real_numeric_backend_array(
                self.model.pairwise_cost_matrix(features),
                message=_PAIRWISE_COST_ERROR,
            )
            probabilities = exp(-costs)
        else:
            raise TypeError(
                "model must expose predict_match_probability, predict_proba, or pairwise_cost_matrix"
            )
        return clip(_finite_probability_array(probabilities), 0.0, 1.0)

    def pairwise_probability_matrix_from_components(
        self, components: Mapping[str, Any]
    ) -> Any:
        return self.predict_match_probability(components)

    def pairwise_cost_matrix_from_components(
        self,
        components: Mapping[str, Any],
        *,
        mode: _COST_MODE = "negative_log_probability",
    ) -> Any:
        return self.pairwise_cost_matrix(components, mode=mode)

    def pairwise_cost_matrix(
        self,
        features_or_components: Any,
        *,
        mode: _COST_MODE = "negative_log_probability",
    ) -> Any:
        probabilities = clip(
            self.predict_match_probability(features_or_components),
            self.probability_clip,
            1.0 - self.probability_clip,
        )
        if mode == "negative_log_probability":
            return -log(probabilities)
        if mode == "one_minus_probability":
            return 1.0 - probabilities
        raise ValueError(f"Unsupported cost mode: {mode}")

    def _features_from_components_or_tensor(self, features_or_components: Any) -> Any:
        if isinstance(features_or_components, Mapping):
            return self.build_feature_tensor(features_or_components)
        return features_or_components

    def _predict_proba_probability(self, features: Any) -> Any:
        flattened_features, original_shape = _flatten_prediction_features(features)
        probabilities = _as_real_numeric_backend_array(
            self.model.predict_proba(flattened_features),
            message=_PREDICTED_PROBABILITY_ERROR,
        )
        if probabilities.ndim >= 2 and probabilities.shape[-1] == 2:
            probabilities = probabilities[
                ..., self._predict_proba_positive_class_index()
            ]
        elif probabilities.ndim != 1:
            raise ValueError(
                "predict_proba must return probabilities with shape (n_samples,) or (n_samples, 2)"
            )
        if original_shape == ():
            return asarray(probabilities[0], dtype=float64)
        return probabilities.reshape(original_shape)

    def _predict_proba_positive_class_index(self) -> int:
        classes = _class_labels_to_list(getattr(self.model, "classes_", None))
        if len(classes) != 2:
            return 1
        for class_index, class_label in enumerate(classes):
            if _is_positive_binary_label(class_label):
                return class_index
        return 1


def _normalize_probability_clip(value: Any) -> float:
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(_PROBABILITY_CLIP_ERROR) from exc
    if value_array.shape != ():
        raise ValueError(_PROBABILITY_CLIP_ERROR)

    scalar = value_array.item()
    if isinstance(
        scalar, (bool, np.bool_, complex, np.complexfloating, *_TEXT_SCALAR_TYPES)
    ):
        raise ValueError(_PROBABILITY_CLIP_ERROR)
    if not isinstance(scalar, Real):
        raise ValueError(_PROBABILITY_CLIP_ERROR)

    probability_clip = float(scalar)
    if not np.isfinite(probability_clip) or not 0.0 < probability_clip < 0.5:
        raise ValueError(_PROBABILITY_CLIP_ERROR)
    return probability_clip


def _finite_probability_array(probabilities: Any) -> Any:
    probabilities = _as_real_numeric_backend_array(
        probabilities,
        message=_PREDICTED_PROBABILITY_ERROR,
    )
    if not all(isfinite(probabilities)):
        raise ValueError("predicted probabilities must be finite")
    if not all(probabilities >= 0.0) or not all(probabilities <= 1.0):
        raise ValueError("predicted probabilities must lie in [0, 1]")
    return probabilities


def _as_real_numeric_backend_array(values: Any, *, message: str) -> Any:
    if _contains_non_real_values(values):
        raise ValueError(message)
    try:
        return asarray(values, dtype=float64)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(message) from exc


def _contains_non_real_values(values: Any) -> bool:
    if isinstance(values, _NON_REAL_SCALAR_TYPES):
        return True
    if _has_temporal_dtype(values):
        return True
    try:
        value_array = np.asarray(values)
    except (TypeError, ValueError, RuntimeError):
        value_array = None
    if value_array is not None and _has_temporal_dtype(value_array):
        return True
    try:
        object_values = np.asarray(values, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(value, _NON_REAL_SCALAR_TYPES) for value in object_values)


def _has_temporal_dtype(values: Any) -> bool:
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        return False
    try:
        return np.dtype(dtype).kind in _TEMPORAL_DTYPE_KINDS
    except TypeError:
        return False


def _is_positive_binary_label(class_label: Any) -> bool:
    try:
        if bool(class_label == 1):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(class_label, str):
        normalized_label = class_label.strip().casefold()
        return normalized_label in {"1", "true"}
    return False


def _class_labels_to_list(classes: Any) -> list[Any]:
    if classes is None:
        return []
    if hasattr(classes, "detach"):
        classes = classes.detach().cpu()
    if hasattr(classes, "reshape"):
        try:
            classes = classes.reshape(-1)
        except TypeError:
            pass
    if hasattr(classes, "tolist"):
        labels = classes.tolist()
    else:
        try:
            labels = list(classes)
        except TypeError:
            return []
    if isinstance(labels, list):
        return labels
    return [labels]


def _normalize_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_names, str):
        raise ValueError("feature_names must be a sequence of names, not a string")
    normalized_names = tuple(feature_names)
    if not normalized_names:
        raise ValueError("At least one feature is required")
    for feature_name in normalized_names:
        if not isinstance(feature_name, str) or not feature_name:
            raise ValueError("feature_names must contain non-empty strings")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("feature_names must not contain duplicates")
    return normalized_names


def _normalize_feature_transforms(
    feature_names: tuple[str, ...],
    transforms: Mapping[str, FeatureTransform] | None,
) -> dict[str, FeatureTransform]:
    if transforms is None:
        return {}
    normalized_transforms = dict(transforms)
    unknown_transforms = set(normalized_transforms) - set(feature_names)
    if unknown_transforms:
        raise KeyError(
            f"Transforms supplied for unknown features: {sorted(unknown_transforms)!r}"
        )
    for feature_name, transform in normalized_transforms.items():
        if not callable(transform):
            raise TypeError(f"Transform for feature {feature_name!r} must be callable")
    return normalized_transforms


def _component_feature(
    components: Mapping[str, Any],
    feature_name: str,
    transforms: Mapping[str, FeatureTransform],
) -> Any:
    if feature_name in transforms:
        values = transforms[feature_name](components)
    else:
        if feature_name not in components:
            raise KeyError(f"Pairwise components do not contain {feature_name!r}")
        values = components[feature_name]
    return _finite_feature_plane(values, feature_name)


def _finite_feature_plane(values: Any, feature_name: str) -> Any:
    values = _as_real_numeric_backend_array(
        values,
        message=f"Feature {feature_name!r} must contain real numeric values",
    )
    if values.ndim == 0:
        raise ValueError(f"Feature {feature_name!r} must be at least one-dimensional")
    finite_values = where(isnan(values), 0.0, values)
    finite_values = where(
        isinf(finite_values),
        where(finite_values > 0.0, 1.0e6, -1.0e6),
        finite_values,
    )
    return finite_values


def _flatten_prediction_features(features: Any) -> tuple[Any, tuple[int, ...]]:
    features = asarray(features, dtype=float64)
    if features.ndim == 0:
        raise ValueError("features must be at least one-dimensional")
    if features.ndim == 1:
        return features[None, :], ()
    return features.reshape(-1, features.shape[-1]), features.shape[:-1]
