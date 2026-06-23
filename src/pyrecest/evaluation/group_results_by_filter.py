def _parameter_sort_key(parameter):
    if parameter is None:
        return (0, "")
    try:
        return (1, float(parameter))
    except (TypeError, ValueError, OverflowError):
        return (2, str(parameter))


def group_results_by_filter(data):
    # Sort the data by 'parameter', treating None as negative infinity for sorting purposes
    sorted_data = sorted(
        data,
        key=lambda x: _parameter_sort_key(x["parameter"]),
    )

    output_dict = {}
    for entry in sorted_data:
        name = entry["name"]
        # Remove the 'name' key-value pair from the entry
        entry_values = {k: v for k, v in entry.items() if k != "name"}

        # Check if the name already exists in the output_dict
        if name in output_dict:
            for key, value in entry_values.items():
                # Append values to the existing lists
                output_dict[name][key].append(value)
        else:
            # Initialize the entry in the output_dict with lists for each value
            output_dict[name] = {k: [v] for k, v in entry_values.items()}

    return output_dict
