# Robust linear-Gaussian MAP smoothing

PyRecEst provides a sparse batch and fixed-lag maximum-a-posteriori (MAP)
smoother for linear-Gaussian state sequences. It is useful when a Kalman or IMM
replay has already produced an initial trajectory, but individual measurement
factors may contain gross outliers that should be downweighted rather than
accepted quadratically.

The state model is

\[
x_{k+1} = F_k x_k + b_k + w_k,
\qquad w_k \sim \mathcal N(0, Q_k),
\]

and each measurement factor is

\[
z_j = H_j x_{k(j)} + d_j + v_j,
\qquad v_j \sim \mathcal N(0, R_j).
\]

Prior and process factors remain quadratic. A measurement factor marked
`robust=True` applies the configured loss to the norm of its whitened residual.
Using one weight per vector factor makes the robustification invariant to a
rotation of the measurement coordinates.

## Batch example

```python
import numpy as np

from pyrecest.smoothers import (
    LinearGaussianMeasurementFactor,
    RobustLinearGaussianMapConfig,
    robust_linear_gaussian_map_smooth,
)

initial_states = np.array(
    [
        [0.0, 1.0],
        [1.2, 1.0],
        [20.0, 1.0],  # deliberately poor initial position
        [3.0, 1.0],
    ]
)
transition = np.array([[1.0, 1.0], [0.0, 1.0]])
transition_matrices = np.repeat(transition[None, :, :], 3, axis=0)
process_covariances = np.repeat(
    np.diag([0.05, 0.05])[None, :, :],
    3,
    axis=0,
)
measurements = tuple(
    LinearGaussianMeasurementFactor(
        state_index=index,
        measurement=np.array([position]),
        observation_matrix=np.array([[1.0, 0.0]]),
        covariance=np.array([[1.0]]),
    )
    for index, position in enumerate((0.0, 1.0, 25.0, 3.0))
)

result = robust_linear_gaussian_map_smooth(
    initial_states,
    prior_mean=np.array([0.0, 1.0]),
    prior_covariance=np.diag([0.1, 0.1]),
    transition_matrices=transition_matrices,
    process_covariances=process_covariances,
    measurements=measurements,
    config=RobustLinearGaussianMapConfig(
        loss="huber",
        loss_scale=2.0,
    ),
)

print(result.states)
print(result.measurement_sqrt_weights)
```

`measurement_sqrt_weights` contains the final square-root IRLS weight for each
measurement factor. A value below one indicates that the configured robust loss
downweighted that factor.

## Fixed-lag operation

Use `fixed_lag_robust_linear_gaussian_map_smooth(...)` when each output state
may use only a bounded interval of future data. Supply one timestamp and one
anchor covariance per initial state. The function solves the window beginning
at every state and returns the first state of each window, together with one
window summary per output.

A lag that covers all remaining timestamps gives the same first-state estimate
as the corresponding batch problem with the same first-state prior.

## Losses

The supported losses are:

- `linear`: ordinary quadratic least squares;
- `huber`: quadratic near zero and linear in the tail;
- `soft_l1`: a differentiable approximation to an absolute-value tail;
- `cauchy`: logarithmic tail growth;
- `arctan`: a bounded redescending loss.

Set `robust=False` on a trusted measurement factor to keep it quadratic even
when the problem uses a robust loss.

## Covariance semantics

The current result deliberately reports `covariances=None`. Computing MAP
marginal covariances requires selected blocks of the inverse final approximate
Hessian. Reusing filtered covariances would mislabel them as smoother
uncertainty, so callers should retain filtered covariances separately until a
marginal-covariance API is added.

## Numerical implementation

The implementation assembles a SciPy sparse least-squares system and performs
iteratively reweighted solves with a monotone backtracking line search. Public
inputs are normalized to finite real NumPy arrays; this smoother is therefore a
NumPy/SciPy facility rather than a backend-portable JAX or PyTorch primitive.
