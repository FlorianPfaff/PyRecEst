# Robust Kalman Updates

Outlier-prone measurements can be handled by combining linear-Gaussian updates
with normalized innovation squared (NIS) diagnostics and robust covariance
inflation.

```python
from pyrecest.backend import array, diag, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter

state = GaussianDistribution(array([0.0, 0.0]), diag(array([1.0, 1.0])))
kalman_filter = KalmanFilter(state)

measurement = array([20.0, 20.0])
measurement_matrix = eye(2)
measurement_noise = eye(2)

diagnostics = kalman_filter.update_linear_robust(
    measurement,
    measurement_matrix,
    measurement_noise,
    robust_update="student-t",
    student_t_dof=4.0,
    return_diagnostics=True,
)

print(diagnostics["nis"], diagnostics["scale"], diagnostics["action"])
```

Use hard gating when a measurement should be rejected rather than downweighted:

```python
diagnostics = kalman_filter.update_linear_robust(
    measurement,
    measurement_matrix,
    measurement_noise,
    robust_update="none",
    gate_threshold=9.21,
    return_diagnostics=True,
)

if not diagnostics["accepted"]:
    print("measurement rejected")
```

For repeatable experiments, log the returned diagnostics together with the
posterior state. The `nis` value is computed against the nominal measurement
covariance; robust updates then inflate the covariance used for the actual state
update.
