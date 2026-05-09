"""Named pairwise feature helpers for probabilistic association models."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    asarray,
    clip,
    exp,
    float64,
    isinf,
    isnan,
    log,
    stack,
    where,
)

_COST_MODE = Literal["negative_log_probability", "one_minus_probability"]
FeatureTransform = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, init=False)
class NamedPairwiseFeatureSchema:
    """Named schema for building pairwise feature tensors from components.

    Parameters
    ----------
    feature_names:
        Ordered names of the feature planes to stack on the final tensor axis.
        If no transform is registered for a name, the same key is looked up in
        ``components`` when building a tensor.
    transforms:
        Optional mapping from feature name to a callable receiving the full
        component mapping and returning the feature plane. This keeps derived
        features domain-specific without baking their semantics into PyRecEst.
    """

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
        object.__setattr__(
            self, "transforms", MappingProxyType(normalized_transforms)
        )

    def __len__(self) -> int:
        return len(self.feature_names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.feature_names)

    def feature_index(self, feature_name: str) -> int:
        """Return the last-axis index of a named feature."""
        try:
            return self.feature_names.index(feature_name)
        except ValueError as exc:
            raise KeyError(f"Unknown feature name {feature_name!r}") from exc

    def build_tensor(self, components: Mapping[str, Any]) -> Any:
        """Build a ``(..., n_features)`` tensor from named pairwise components."""
        return pairwise_feature_tensor(
            components, self.feature_names, transforms=self.transforms
        )


def pairwise_feature_tensor(
    components: Mapping[str, Any],
    feature_names: Sequence[str] | NamedPairwiseFeatureSchema,
    transforms: Mapping[str, FeatureTransform] | None = None,
) -> Any:
    """Build a pairwise feature tensor from named component planes.

    All feature planes must have the same shape. Non-finite values are converted
    to finite sentinels: ``nan -> 0``, ``+inf -> 1e6``, and ``-inf -> -1e6``.
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
    normalized_transforms = _normalize_feature_transforms(
        normalized_names, transforms
    )
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
    """Association model wrapper that keeps a named pairwise feature schema.

    The wrapped model may expose ``predict_match_probability``,
    ``predict_proba``, or ``pairwise_cost_matrix``. Component mappings are first
    converted to tensors via ``schema`` before being passed to the model.
    """

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
            schema = NamedPairwiseFeatureSchema(
                feature_names, transforms=transforms
            )
        if not 0.0 < probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, 0.5)")

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "probability_clip", float(probability_clip))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered feature names expected by the wrapped model."""
        return self.schema.feature_names

    def build_feature_tensor(self, components: Mapping[str, Any]) -> Any:
        """Build a feature tensor using this model's schema."""
        return self.schema.build_tensor(components)

    def predict_match_probability(self, features_or_components: Any) -> Any:
        """Return calibrated match probabilities for tensors or components."""
        features = self._features_from_components_or_tensor(features_or_components)
        if hasattr(self.model, "predict_match_probability"):
            probabilities = self.model.predict_match_probability(features)
        elif hasattr(self.model, "predict_proba"):
            probabilities = self._predict_proba_probability(features)
        elif hasattr(self.model, "pairwise_cost_matrix"):
            costs = asarray(self.model.pairwise_cost_matrix(features), dtype=float64)
            probabilities = exp(-costs)
        else:
            raise TypeError(
                "model must expose predict_match_probability, predict_proba, or pairwise_cost_matrix"
            )
        return clip(asarray(probabilities, dtype=float64), 0.0, 1.0)

    def pairwise_probability_matrix_from_components(
        self, components: Mapping[str, Any]
    ) -> Any:
        """Convert components into calibrated pairwise probabilities."""
        return self.predict_match_probability(components)

    def pairwise_cost_matrix_from_components(
        self,
        components: Mapping[str, Any],
        *,
        mode: _COST_MODE = "negative_log_probability",
    ) -> Any:
        """Convert components into calibrated assignment costs."""
        return self.pairwise_cost_matrix(components, mode=mode)

    def pairwise_cost_matrix(
        self,
        features_or_components: Any,
        *,
        mode: _COST_MODE = "negative_log_probability",
    ) -> Any:
        """Convert features or components into an assignment cost matrix."""
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
        probabilities = asarray(
            self.model.predict_proba(flattened_features), dtype=float64
        )
        if probabilities.ndim >= 2 and probabilities.shape[-1] == 2:
            probabilities = probabilities[..., 1]
        elif probabilities.ndim != 1:
            raise ValueError(
                "predict_proba must return probabilities with shape (n_samples,) or (n_samples, 2)"
            )
        if original_shape == ():
            return asarray(probabilities[0], dtype=float64)
        return probabilities.reshape(original_shape)


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
    values = asarray(values, dtype=float64)
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
