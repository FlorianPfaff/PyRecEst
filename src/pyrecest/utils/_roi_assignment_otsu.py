"""Runtime patches for ROI-assignment threshold semantics."""

from __future__ import annotations


def _patch_minimum_similarity_threshold(roi_assignment_module) -> None:
    """Validate histogram bin counts before minimum-threshold early returns."""

    original_minimum = roi_assignment_module.minimum_similarity_threshold
    if getattr(original_minimum, "_pyrecest_positive_nbins_validation", False):
        return

    def minimum_similarity_threshold(similarities, *, nbins: int = 256) -> float:
        """Estimate a threshold by locating a valley between the two strongest modes."""

        nbins = roi_assignment_module._as_positive_integer(nbins, "nbins")
        return original_minimum(similarities, nbins=nbins)

    minimum_similarity_threshold.__name__ = getattr(
        original_minimum,
        "__name__",
        "minimum_similarity_threshold",
    )
    minimum_similarity_threshold.__doc__ = getattr(original_minimum, "__doc__", None)
    minimum_similarity_threshold._pyrecest_positive_nbins_validation = True
    roi_assignment_module.minimum_similarity_threshold = minimum_similarity_threshold


def patch_otsu_similarity_threshold(roi_assignment_module) -> None:
    """Patch ROI threshold helpers for strict Otsu splitting and input validation."""

    original_otsu = roi_assignment_module.otsu_similarity_threshold
    if getattr(original_otsu, "_pyrecest_strict_foreground_split", False) and getattr(
        original_otsu, "_pyrecest_positive_nbins_validation", False
    ):
        _patch_minimum_similarity_threshold(roi_assignment_module)
        return

    def otsu_similarity_threshold(similarities, *, nbins: int = 256) -> float:
        """Estimate a threshold using Otsu's method on one-dimensional similarities."""

        nbins = roi_assignment_module._as_positive_integer(nbins, "nbins")
        values = roi_assignment_module.asarray(
            similarities,
            dtype=roi_assignment_module.float64,
        )
        values = values[roi_assignment_module.isfinite(values)]
        if values.shape[0] == 0:
            return 0.0

        value_min = float(roi_assignment_module.amin(values))
        value_max = float(roi_assignment_module.amax(values))
        if value_min == value_max:
            return value_min

        histogram, bin_edges = roi_assignment_module._histogram(
            values,
            bins=nbins,
            value_min=value_min,
            value_max=value_max,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        weight_background = roi_assignment_module.cumsum(histogram)
        histogram_weighted_centers = histogram * bin_centers
        weighted_sum_background = roi_assignment_module.cumsum(
            histogram_weighted_centers
        )
        total_weight = weight_background[-1]
        total_weighted_sum = weighted_sum_background[-1]
        weight_foreground = total_weight - weight_background
        weighted_sum_foreground = total_weighted_sum - weighted_sum_background

        valid_mask = (weight_background > 0) & (weight_foreground > 0)
        if not bool(roi_assignment_module.any(valid_mask)):
            return 0.0

        background_denominator = roi_assignment_module.where(
            valid_mask,
            weight_background,
            1.0,
        )
        foreground_denominator = roi_assignment_module.where(
            valid_mask,
            weight_foreground,
            1.0,
        )
        mean_background = roi_assignment_module.where(
            valid_mask,
            weighted_sum_background / background_denominator,
            0.0,
        )
        mean_foreground = roi_assignment_module.where(
            valid_mask,
            weighted_sum_foreground / foreground_denominator,
            0.0,
        )

        between_class_variance = roi_assignment_module.where(
            valid_mask,
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2,
            0.0,
        )

        best_index = int(roi_assignment_module.argmax(between_class_variance))
        return float(bin_centers[best_index])

    otsu_similarity_threshold.__name__ = getattr(
        original_otsu,
        "__name__",
        "otsu_similarity_threshold",
    )
    otsu_similarity_threshold.__doc__ = getattr(original_otsu, "__doc__", None)
    otsu_similarity_threshold._pyrecest_strict_foreground_split = True
    otsu_similarity_threshold._pyrecest_positive_nbins_validation = True
    roi_assignment_module.otsu_similarity_threshold = otsu_similarity_threshold
    _patch_minimum_similarity_threshold(roi_assignment_module)
