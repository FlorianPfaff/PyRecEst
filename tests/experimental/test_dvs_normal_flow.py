import numpy as np
import pytest
from pyrecest.experimental.dvs.normal_flow import (
    INFER_POLARITY_CONTRAST_SIGN,
    event_polarity_sign,
    infer_polarity_contrast_sign,
    normalize_polarity_contrast_sign,
    polarity_consistency_for_signed_flow,
    polarity_weight_for_signed_flow,
    polarity_weights_for_signed_flows,
    signed_scalar_sign,
)


def test_signed_scalar_sign_uses_dead_zone():
    assert signed_scalar_sign(1e-3) == 1.0
    assert signed_scalar_sign(-1e-3) == -1.0
    assert signed_scalar_sign(1e-13) == 0.0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, "1.0", 1.0 + 0.0j, True])
def test_signed_scalar_sign_rejects_invalid_values(bad_value):
    with pytest.raises(ValueError, match="value"):
        signed_scalar_sign(bad_value)


@pytest.mark.parametrize(
    "bad_tolerance", [-1e-12, np.nan, np.inf, "1e-12", 1e-12 + 0.0j]
)
def test_signed_scalar_sign_rejects_invalid_zero_tolerance(bad_tolerance):
    with pytest.raises(ValueError, match="zero_tolerance"):
        signed_scalar_sign(1.0, zero_tolerance=bad_tolerance)


def test_event_polarity_sign_supports_zero_one_and_signed_conventions():
    assert event_polarity_sign(1) == 1.0
    assert event_polarity_sign(0) == -1.0
    assert event_polarity_sign(-1) == -1.0
    assert event_polarity_sign(True) == 1.0
    assert event_polarity_sign(False) == -1.0


@pytest.mark.parametrize("bad_polarity", [np.nan, np.inf, "1", 1.0 + 0.0j])
def test_event_polarity_sign_rejects_invalid_polarities(bad_polarity):
    with pytest.raises(ValueError, match="event_polarity"):
        event_polarity_sign(bad_polarity)


def test_normalize_polarity_contrast_sign():
    assert normalize_polarity_contrast_sign(None) is None
    assert normalize_polarity_contrast_sign("infer") == INFER_POLARITY_CONTRAST_SIGN
    assert normalize_polarity_contrast_sign(2.0) == 1.0
    assert normalize_polarity_contrast_sign(-3.0) == -1.0
    with pytest.raises(ValueError, match="polarity_contrast_sign"):
        normalize_polarity_contrast_sign("auto")
    with pytest.raises(ValueError, match="polarity_contrast_sign"):
        normalize_polarity_contrast_sign(0.0)


@pytest.mark.parametrize("bad_sign", [np.nan, np.inf, True, 1.0 + 0.0j])
def test_normalize_polarity_contrast_sign_rejects_invalid_numeric_values(bad_sign):
    with pytest.raises(ValueError, match="polarity_contrast_sign"):
        normalize_polarity_contrast_sign(bad_sign)


def test_infer_polarity_contrast_sign_from_batch():
    signed_flows = np.array([1.0, -0.5, 0.0])
    polarities = np.array([1.0, 0.0, 1.0])

    assert infer_polarity_contrast_sign(signed_flows, polarities, "infer") == 1.0
    assert infer_polarity_contrast_sign(-signed_flows, polarities, "infer") == -1.0
    assert infer_polarity_contrast_sign(signed_flows, polarities, None) is None
    assert infer_polarity_contrast_sign(signed_flows, polarities, -1.0) == -1.0


def test_infer_polarity_contrast_sign_rejects_invalid_batches():
    with pytest.raises(ValueError, match="signed_normal_flows"):
        infer_polarity_contrast_sign([1.0, np.nan], [1.0, 0.0], "infer")
    with pytest.raises(ValueError, match="event_polarities"):
        infer_polarity_contrast_sign([1.0, -1.0], ["1", "0"], "infer")


def test_polarity_consistency_and_weight_for_signed_flow():
    assert polarity_consistency_for_signed_flow(1.0, 1.0, 1.0)
    assert polarity_consistency_for_signed_flow(-1.0, 0.0, 1.0)
    assert not polarity_consistency_for_signed_flow(1.0, 0.0, 1.0)
    assert polarity_consistency_for_signed_flow(0.0, 1.0, 1.0) is None
    assert (
        polarity_weight_for_signed_flow(
            1.0,
            0.0,
            1.0,
            polarity_mismatch_weight=0.2,
        )
        == 0.2
    )
    assert (
        polarity_weight_for_signed_flow(
            1.0,
            1.0,
            1.0,
            polarity_mismatch_weight=0.2,
        )
        == 1.0
    )
    with pytest.raises(ValueError, match="polarity_mismatch_weight"):
        polarity_weight_for_signed_flow(1.0, 1.0, 1.0, polarity_mismatch_weight=1.5)


@pytest.mark.parametrize("bad_weight", [np.nan, np.inf, "0.5", 0.5 + 0.0j, -0.1, 1.1])
def test_polarity_weight_rejects_invalid_mismatch_weight(bad_weight):
    with pytest.raises(ValueError, match="polarity_mismatch_weight"):
        polarity_weight_for_signed_flow(
            1.0, 0.0, 1.0, polarity_mismatch_weight=bad_weight
        )


def test_polarity_weights_for_signed_flows_resolves_batch_sign():
    weights = polarity_weights_for_signed_flows(
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        polarity_mismatch_weight=0.2,
    )
    np.testing.assert_allclose(weights, np.array([1.0, 1.0, 1.0]))


def test_polarity_weights_for_signed_flows_rejects_invalid_mismatch_weight_when_disabled():
    with pytest.raises(ValueError, match="polarity_mismatch_weight"):
        polarity_weights_for_signed_flows(
            [1.0], [1.0], None, polarity_mismatch_weight=np.nan
        )
