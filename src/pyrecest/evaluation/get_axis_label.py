def get_axis_label(manifold_name):
    if "circleSymm" in manifold_name:
        error_label = "Error in radian"

    elif "circle" in manifold_name or "hypertorus" in manifold_name:
        error_label = "Error in radian"

    elif "hypersphereSymmetric" in manifold_name:
        error_label = "Angular error in radian"

    elif "hypersphere" in manifold_name:
        error_label = "Error (orthodromic distance) in radian"

    elif "se2bounded" in manifold_name:
        error_label = "Error in radian"

    elif "se2" in manifold_name or "se2linear" in manifold_name:
        error_label = "Error in meters"

    elif "se3bounded" in manifold_name:
        error_label = "Error in radian"

    elif "se3" in manifold_name or "se3linear" in manifold_name:
        error_label = "Error in meters"

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and "MTT" not in manifold_name:
        error_label = "Error in meters"

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and "MTT" in manifold_name:
        error_label = "OSPA error in meters"

    else:
        raise ValueError("Mode not recognized")

    return error_label
