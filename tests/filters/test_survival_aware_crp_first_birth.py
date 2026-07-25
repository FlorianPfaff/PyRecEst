import pytest
from pyrecest.filters.survival_aware_crp import SurvivalAwareCRPAssociationPrior


@pytest.mark.parametrize("concentration", [0.0, -0.25])
def test_first_target_generated_cluster_uses_unit_pitman_yor_weight(concentration):
    prior = SurvivalAwareCRPAssociationPrior(
        concentration=concentration,
        discount=0.5,
    )

    existing_weights, birth_weight, clutter_weight = (
        prior.predictive_assignment_weights(
            [],
            base_birth_weight=0.25,
            clutter_weight=0.75,
        )
    )
    probabilities = prior.predictive_assignment_probabilities(
        [],
        base_birth_weight=0.25,
        clutter_weight=0.75,
    )

    assert existing_weights == ()
    assert birth_weight == pytest.approx(0.25)
    assert clutter_weight == pytest.approx(0.75)
    assert probabilities.as_tuple == pytest.approx((0.25, 0.75))


def test_subsequent_births_still_use_concentration_and_discount():
    prior = SurvivalAwareCRPAssociationPrior(
        concentration=-0.25,
        discount=0.5,
    )

    assert prior.birth_weight(1, base_birth_weight=2.0) == pytest.approx(0.5)
