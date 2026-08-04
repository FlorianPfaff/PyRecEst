import numpy.testing as npt
import pytest
from pyrecest.backend import array, zeros
from pyrecest.utils import LogisticPairwiseAssociationModel


def test_failed_refit_preserves_previous_fitted_state():
    model = LogisticPairwiseAssociationModel(class_weight=None)
    original_features = array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = array([0, 0, 1, 1])
    model.fit(original_features, labels)

    probabilities_before = model.predict_match_probability(original_features)
    n_features_before = model.n_features_in_
    n_iter_before = model.n_iter_
    converged_before = model.converged_
    class_weights_before = model.class_weights_

    replacement_features = array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(
        ValueError, match="At least one example must receive positive weight"
    ):
        model.fit(
            replacement_features,
            labels,
            sample_weight=zeros(labels.shape),
        )

    assert model.n_features_in_ == n_features_before
    assert model.n_iter_ == n_iter_before
    assert model.converged_ is converged_before
    assert model.class_weights_ == class_weights_before
    npt.assert_allclose(
        model.predict_match_probability(original_features),
        probabilities_before,
    )
