import numpy as np
import pytest
from pyrecest.utils import LogisticPairwiseAssociationModel


def test_one_feature_logistic_model_accepts_scalar_prediction_input():
    model = LogisticPairwiseAssociationModel(
        standardize=False,
        class_weight=None,
        l2_regularization=1.0,
        max_iterations=25,
    )
    model.fit(np.array([-1.0, 0.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))

    scalar_probability = model.predict_match_probability(0.25)
    vector_probability = model.predict_match_probability([0.25])

    assert np.asarray(scalar_probability).shape == ()
    assert np.allclose(
        float(np.asarray(scalar_probability)),
        float(np.asarray(vector_probability)),
    )


def test_multifeature_logistic_model_rejects_scalar_prediction_input_cleanly():
    model = LogisticPairwiseAssociationModel(
        standardize=False,
        class_weight=None,
        l2_regularization=1.0,
        max_iterations=25,
    )
    model.fit(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        np.array([0, 0, 1, 1]),
    )

    with pytest.raises(ValueError, match="scalar prediction input"):
        model.predict_match_probability(0.25)
