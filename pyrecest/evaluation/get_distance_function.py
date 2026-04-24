from pyrecest.backend import arccos, dot, linalg
from pyrecest.distributions import AbstractHypertoroidalDistribution


def get_distance_function(
    manifold_name, additional_params=None, nSymm=None, symmetryOffsets=None
):
    if nSymm is not None or symmetryOffsets is not None or "Symm" in manifold_name:
        raise NotImplementedError("Not implemented yet")

    if "circle" in manifold_name or "hypertorus" in manifold_name:

        def distance_function(xest, xtrue):
            return linalg.norm(
                AbstractHypertoroidalDistribution.angular_error(xest, xtrue)
            )

    elif "hypersphere" in manifold_name:

        def distance_function(x1, x2):
            return arccos(dot(x1, x2))

    elif "hypersphereSymmetric" in manifold_name:

        def distance_function(x1, x2):
            return min(arccos(dot(x1, x2)), arccos(dot(x1, -x2)))

    elif "se2" in manifold_name or "se2linear" in manifold_name:

        def distance_function(x1, x2):
            return linalg.norm(x1[1:3, :] - x2[1:3, :])

    elif "se2bounded" in manifold_name:

        def distance_function(xest, xtrue):
            return linalg.norm(
                AbstractHypertoroidalDistribution.angular_error(xest[0, :], xtrue[0, :])
            )

    elif "se3" in manifold_name or "se3linear" in manifold_name:

        def distance_function(x1, x2):
            return linalg.norm(x1[4:7, :] - x2[4:7, :])

    elif "se3bounded" in manifold_name:

        def distance_function(x1, x2):
            return min(arccos(dot(x1[:4], x2[:4])), arccos(dot(x1[:4], -x2[:4])))

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and "MTT" not in manifold_name:

        def distance_function(x1, x2):
            return linalg.norm(x1 - x2)

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and "MTT" in manifold_name:
        additional_params.get("cutoff_distance", 1000000)

        def distance_function(x1, x2):
            raise NotImplementedError("Not implemented yet")

    else:
        raise ValueError("Mode not recognized")

    return distance_function
