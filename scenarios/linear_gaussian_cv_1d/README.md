# Linear Gaussian Constant-Velocity Scenario

This is the smallest scenario in the zoo. It follows a one-dimensional
constant-velocity state model:

```text
x_k = F x_{k-1} + w_k
z_k = H x_k + v_k
```

with fixed measurements. The expected final estimate is used as a golden
output for scenario and CLI regression tests.
