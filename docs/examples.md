# Examples

The source repository includes executable examples in `examples/basic/`.

Run examples from the repository root after installing PyRecEst or after
installing a development checkout. The CI package job smoke-runs the core
Kalman and Gaussian examples from an installed wheel so packaging errors are
caught outside editable mode.

## Choose An Example By Task

| Task                                                  | Example                                                                                      | Backend notes                                         | Good for                                                |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------|
| Linear Gaussian filtering                             | `examples/basic/kalman_filter.py`                                                            | Portable baseline.                                    | First smoke test, API familiarization.                  |
| Linear Gaussian filtering with reusable model objects | `examples/basic/kalman_filter_with_models.py`                                                | Portable where model-object APIs are supported.       | Applications with shared transition/measurement models. |
| Nonlinear additive-noise filtering                    | `examples/basic/ukf_with_models.py`                                                          | Follows `UnscentedKalmanFilter` backend limitations.  | Nonlinear Euclidean examples.                           |
| Particle filtering                                    | `examples/basic/particle_filter_with_models.py`                                              | Requires sampleable transition and likelihood models. | Monte Carlo filter loops.                               |
| Multi-target tracking with clutter                    | `examples/basic/multi_target_tracking.py`                                                    | NumPy backend.                                        | Missed detections, clutter, and labeled tracks.         |
| 3D ellipsoid tracking with MEM-EKF*                    | `examples/basic/mem_ekf_star_3d_tracking.py`                                                   | NumPy simulation; eager backend tracker operations.    | Full spatial orientation, axes, and kinematics.         |
| Directional distribution algebra                      | `examples/basic/von_mises_fisher_multiplication.py`                                          | Manifold/domain-specific APIs.                        | Unit-sphere probability operations.                     |
| User-defined protocols                                | `examples/basic/custom_distribution_protocol.py`, `examples/basic/custom_filter_protocol.py` | Backend-independent protocol shape.                   | Extending PyRecEst without subclassing.                 |

## Scenario Examples

The `scenarios/` directory contains data-first examples with TOML configuration
and JSON golden outputs. Use these when you want a reproducible regression case
rather than an exploratory script. The first scenario is
`scenarios/linear_gaussian_cv_1d`.

## Basic Examples

### 3D MEM-EKF* Ellipsoid Tracking

`examples/basic/mem_ekf_star_3d_tracking.py` tracks a translating and rotating
ellipsoid from noisy points sampled uniformly throughout its volume. It starts
with errors in orientation, axes, position, and velocity, and reports the final
estimation errors. The [3D MEM-EKF* guide](mem-ekf-star-3d.md) explains the body
rotation error and covariance conventions.

```bash
python examples/basic/mem_ekf_star_3d_tracking.py
python examples/basic/mem_ekf_star_3d_tracking.py --output tracking-3d.png
```

### Gaussian Multiplication

`examples/basic/gaussian_multiplication.py` multiplies several two-dimensional
Gaussian distributions and checks the result against the closed-form
information representation.

```bash
python examples/basic/gaussian_multiplication.py
```

### Kalman Filter

`examples/basic/kalman_filter.py` runs a one-dimensional constant-velocity
Kalman filter.

```bash
python examples/basic/kalman_filter.py
```

### Kalman Filter With Model Objects

`examples/basic/kalman_filter_with_models.py` runs the same
one-dimensional constant-velocity Kalman filter as `kalman_filter.py`, but
defines reusable linear-Gaussian transition and measurement model objects and
passes them to `predict_model()` and `update_model()`.

```bash
python examples/basic/kalman_filter_with_models.py
```

### Unscented Kalman Filter With Model Objects

`examples/basic/ukf_with_models.py` demonstrates the additive-noise nonlinear
model-object API with an unscented Kalman filter. The transition model stores
the state propagation function and process-noise covariance; the measurement
model stores the measurement function and measurement-noise covariance.

```bash
python examples/basic/ukf_with_models.py
```

This example follows the current backend limitations of
`UnscentedKalmanFilter`.

### Particle Filter With Model Objects

`examples/basic/particle_filter_with_models.py` demonstrates a particle-filter
loop with a sampleable transition model and a likelihood-based measurement
model.

```bash
python examples/basic/particle_filter_with_models.py
```

### Multi-Target Tracking

`examples/basic/multi_target_tracking.py` runs a small linear/Gaussian
multi-Bernoulli-tracker scenario with two labeled targets, missed detections,
and clutter measurements.

```bash
python examples/basic/multi_target_tracking.py
```

This example currently requires the NumPy backend.

### von Mises-Fisher Multiplication

`examples/basic/von_mises_fisher_multiplication.py` multiplies two
von Mises-Fisher distributions on the unit sphere and verifies the analytic
product relation.

```bash
python examples/basic/von_mises_fisher_multiplication.py
```

### Custom Distribution Protocol

`examples/basic/custom_distribution_protocol.py` shows a small user-defined
scalar distribution that satisfies the public dimension protocols without
inheriting from a PyRecEst distribution base class.

```bash
python examples/basic/custom_distribution_protocol.py
```

### Custom Filter Protocol

`examples/basic/custom_filter_protocol.py` shows a small user-defined recursive
filter that satisfies the public dimension protocol and follows common PyRecEst
filter naming conventions.

```bash
python examples/basic/custom_filter_protocol.py
```

## Backend Selection

Select a non-default backend by setting `PYRECEST_BACKEND` before running the
script. For example, on a bash-compatible shell:

```bash
PYRECEST_BACKEND=pytorch python examples/basic/kalman_filter.py
```

Install the matching optional dependency extra before using a non-default
backend.
