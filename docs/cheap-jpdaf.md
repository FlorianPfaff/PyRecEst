# Cheap joint probabilistic data association

`CheapJPDAF` (also `CJPDAF` and
`CheapJointProbabilisticDataAssociationFilter`) is a NumPy-only,
linear-Gaussian alternative to the exact `JPDAF`. It retains soft association
and Gaussian mixture moment matching without enumerating joint events.
It is a Fitzgerald-style approximation with an explicit normalization repair,
not a numerically equivalent fast implementation of exact JPDA.

## Usage

```python
import numpy as np
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import CheapJPDAF, KalmanFilter

tracker = CheapJPDAF(
    [
        KalmanFilter(GaussianDistribution(np.array([-1.0, 0.0]), np.eye(2))),
        KalmanFilter(GaussianDistribution(np.array([1.0, 0.0]), np.eye(2))),
    ],
    association_param={
        "detection_probability": 0.95,
        "clutter_intensity": 1e-3,
        "gating_distance_threshold": 9.21,
    },
)
tracker.predict_linear(np.eye(2), 0.01 * np.eye(2))
measurements = np.array([[-0.8, 0.9], [0.1, -0.1]])  # (measurement_dim, n_meas)
tracker.update_linear(measurements, np.eye(2), 0.1 * np.eye(2))

beta = tracker.latest_association_probabilities
assert np.allclose(beta.sum(axis=1), 1.0)
assert np.all(beta[:, 1:].sum(axis=0) <= 1.0 + 1e-12)
print(tracker.get_point_estimate())
```

The constructor, `filter_state`, `predict_linear`, and `update_linear` follow
`JPDAF`. Measurement covariances may be shared `(d, d)` or per measurement
`(d, d, n_meas)`. Clutter intensity may be a positive finite scalar or one
positive finite value per measurement. Detection probability must satisfy
`0 < P_D < 1`. Gating compares **squared** Mahalanobis distance with the
threshold; the inherited default is the 99.9% chi-square quantile for the
measurement dimension. `max_enumerated_events` is ignored by this class;
it remains an enforced limit in exact `JPDAF`.

## Association and normalization convention

Let `L[i,j]` be the predicted Gaussian likelihood, set to zero outside the
gate, and let `kappa[j]` be the clutter intensity. Define dimensionless odds

```text
w[i,j] = P_D * L[i,j] / ((1 - P_D) * kappa[j])
r[i]   = sum_j w[i,j]
c[j]   = sum_i w[i,j]
beta[i,j+1] = w[i,j] / (1 + r[i] + c[j] - w[i,j])
beta[i,0]   = 1 - sum_j beta[i,j+1]
```

For scalar clutter, this is the usual likelihood-form denominator
`T[i] + M[j] - L[i,j] + B` with `B=(1-P_D)*kappa/P_D`.
The row/column competition formula is reproduced in equations (49)-(52) of
[US patent application 20160245949](https://patents.justia.com/patent/20160245949),
which attributes it to Fitzgerald. Its original separate miss expression,
`1/(1+r[i])`, generally does **not** normalize a track's weights when tracks
compete. PyRecEst deliberately uses the complementary mass instead, and does
not claim to reproduce that unnormalized original variant.

The denominator is at least `1+r[i]` and at least `1+c[j]`. Thus each row's
detection mass is at most one, and each measurement's total allocation is at
most one. Adding the complementary missed-detection mass gives a valid
per-track mixture. These bounds do not make the marginals exact Bayesian
association probabilities.

For numerical stability, the implementation uses log likelihood ratios and
prefix/suffix log sums for the other tracks' likelihoods. It avoids both a
single global likelihood rescaling and subtraction of a dominant entry from
a column sum. The miss is evaluated through the positive-term identity

```text
competition[i,j] = c[j] - w[i,j]
beta[i,0] = 1/(1+r[i])
          + sum_j (w[i,j]/(1+r[i]))
                  * competition[i,j]/(1+r[i]+competition[i,j])
```

rather than floating-point subtraction from one. This preserves tiny miss
probabilities. No optional dependency or iterative association solver is added.

The approximation coincides with the existing exact JPDAF association model
for a single track, a single measurement, or disjoint single-track validation
components. General ambiguous multi-track/multi-measurement cases differ.
For example, with a 2-by-2 matrix of unit odds, cheap JPDA assigns each track
`[miss=1/2, measurement_1=1/4, measurement_2=1/4]`, whereas exact JPDA gives
`[3/7, 2/7, 2/7]`.

## State update and diagnostics

The update reuses the exact JPDAF's per-pair Kalman hypotheses and mixture
moment matching, including the between-hypothesis covariance term. All-gated
and empty-measurement scans leave the prior unchanged. Like exact JPDAF here,
the miss prior is `1-P_D`: the gate does not introduce an additional gate
probability `P_G`. No track-birth or deletion mechanism is added.

`find_association_probabilities(...)` returns `(beta, greedy_assignment)`.
`beta[:,0]` contains misses, and the remaining columns follow measurement order.
The second value is a **track-order greedy likelihood-ratio diagnostic**, with
`-1` for misses and no measurement reused. It is not a MAP event and is not used
by `update_linear`. Ties with a miss are resolved as misses; equal detection
scores use measurement order. Consequently, the diagnostic may depend on track
order even though the soft marginals are permutation-equivariant.

The diagnostic is stored in `latest_greedy_association` and returned by
`find_association(...)`. `latest_map_association` remains `None`, so callers
cannot accidentally interpret the diagnostic as an exact MAP result. Empty
banks retain the exact class's legacy probability-array shape `(0,1)`.

## Cost and limitations

After pairwise Gaussian likelihoods are available, marginals and the greedy
diagnostic require **O(n_targets * n_meas)** work and memory, not a combinatorial
number of events. The whole update has this target/measurement scaling when
state and measurement dimensions are fixed. Gaussian linear algebra still
contributes its usual dimension-dependent cost.

Competition can move extra mass to the missed-detection component; the result
is intentionally conservative in ambiguous examples, but is not a calibrated
probability guarantee. It does not solve track coalescence or establish a
trajectory-level association model. Use exact `JPDAF` as a small-problem
reference rather than expecting identical output in dense scenes.

Focused regressions are in
`tests/filters/test_cheap_joint_probabilistic_data_association_filter.py` and
cover reference probabilities, normalization and allocation bounds, exact
special cases, numerical extremes, diagnostic semantics, event-limit
independence, covariance moment matching, and input/backend contracts.
