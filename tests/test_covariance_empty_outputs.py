from pyrecest.backend import zeros
from pyrecest.utils import pairwise_covariance_shape_components


def test_pairwise_covariance_empty_outputs_are_independent_objects():
    shape_cost, logdet_cost, shape_similarity = pairwise_covariance_shape_components(
        zeros((2, 2, 0)), zeros((2, 2, 4))
    )

    assert shape_cost.shape == (0, 4)
    assert logdet_cost.shape == (0, 4)
    assert shape_similarity.shape == (0, 4)
    assert len({id(shape_cost), id(logdet_cost), id(shape_similarity)}) == 3
