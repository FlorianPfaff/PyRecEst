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

        if name not in output_dict:
            output_dict[name] = {k: [v] for k, v in entry_values.items()}
            continue

        grouped_values = output_dict[name]
        existing_row_count = len(next(iter(grouped_values.values()), []))

        # Backfill columns that first appear in a later row so every column remains aligned.
        for key in entry_values:
            if key not in grouped_values:
                grouped_values[key] = [None] * existing_row_count

        # Append one value per known column, using None for fields omitted by this row.
        for key in grouped_values:
            grouped_values[key].append(entry_values.get(key))

    return output_dict
