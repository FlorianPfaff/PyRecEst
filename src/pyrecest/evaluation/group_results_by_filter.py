import math

import numpy as np


def _parameter_sort_key(parameter):
    if parameter is None:
        return (0, "")
    if np.ma.isMaskedArray(parameter) and bool(
        np.any(np.ma.getmaskarray(parameter))
    ):
        return (3, str(parameter))
    try:
        parameter_array = np.asarray(parameter)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return (3, str(parameter))
    if parameter_array.shape != ():
        return (3, str(parameter))
    try:
        numeric_parameter = float(parameter_array.item())
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return (3, str(parameter))
    if math.isnan(numeric_parameter):
        return (2, "")
    return (1, numeric_parameter)


def group_results_by_filter(data):
    # Sort the data by 'parameter', treating None as negative infinity for sorting purposes
    sorted_data = sorted(
        data,
        key=lambda x: _parameter_sort_key(x["parameter"]),
    )

    output_dict = {}
    group_sizes = {}
    for entry in sorted_data:
        name = entry["name"]
        # Remove the 'name' key-value pair from the entry
        entry_values = {k: v for k, v in entry.items() if k != "name"}

        if name not in output_dict:
            output_dict[name] = {k: [v] for k, v in entry_values.items()}
            group_sizes[name] = 1
            continue

        grouped_values = output_dict[name]
        n_existing = group_sizes[name]
        for key in tuple(grouped_values):
            grouped_values[key].append(entry_values.get(key))
        for key, value in entry_values.items():
            if key not in grouped_values:
                grouped_values[key] = [None] * n_existing + [value]
        group_sizes[name] += 1

    return output_dict
