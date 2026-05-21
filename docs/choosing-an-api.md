# Choosing an API

Use this page as a first-pass map from an estimation task to the most relevant
PyRecEst components. The exact class names and argument conventions are covered
in the API overview, examples, and reference pages.

| Task                                | Start with                               | Notes                                                                                            |
|-------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------|
| Linear Euclidean Gaussian filtering | `KalmanFilter`                           | Use `predict_linear()` and `update_linear()` for matrix-based transition and measurement models. |
| Nonlinear Euclidean filtering       | Unscented or particle-filter examples    | Prefer model-object examples when the transition and measurement models should be reusable.      |
| Circular or angular states          | Circular distributions and filters       | Use circular/hypertoroidal distributions when wrap-around behavior matters.                      |
| Spherical or directional states     | Hyperspherical distributions and filters | Use directional distributions for unit-vector or sphere-valued states.                           |
| Multi-target tracking               | Multi-target tracker examples            | Check the measurement-set shape convention before wiring detections into a tracker.              |
| Extended object tracking            | EOT-oriented filters and trackers        | Start with NumPy unless the target API is documented and tested for another backend.             |
| Backend-portable code               | `pyrecest.backend` imports               | Set `PYRECEST_BACKEND` before importing PyRecEst and run focused tests on each backend you need. |
| Custom estimators or distributions  | Public protocol docs                     | Use the protocol examples when you do not need to inherit from a PyRecEst base class.            |

## Estimator Selection

| State and model assumptions                                         | Recommended starting point                   | Why                                                                                  |
|---------------------------------------------------------------------|----------------------------------------------|--------------------------------------------------------------------------------------|
| Linear transition and measurement models with Gaussian noise        | `KalmanFilter`                               | Closed-form, fast, and the most portable baseline across backends.                   |
| Smooth nonlinear Euclidean models with moderate dimension           | `UnscentedKalmanFilter`                      | Avoids manual Jacobians while keeping a compact Gaussian state.                      |
| Strong nonlinearity, multimodality, or likelihood-only measurements | particle-filter examples                     | Represents non-Gaussian states and arbitrary likelihoods at the cost of sample size. |
| Gridded low-dimensional densities                                   | grid or Fourier filters                      | Useful when deterministic density support is more important than sample efficiency.  |
| Manifold-valued states                                              | manifold-specific filters and distributions  | Preserves wrap-around, unit-norm, or group constraints explicitly.                   |
| Detection sets with clutter and missed detections                   | GNN, JPDAF, Multi-Bernoulli, or MHT trackers | Separates association hypotheses from per-target filtering.                          |

## Recommended First Examples

1. Run `examples/basic/kalman_filter.py` to verify installation and learn the
   linear Gaussian conventions.
2. Run `examples/basic/kalman_filter_with_models.py` or
   `examples/basic/ukf_with_models.py` when you want reusable model objects.
3. Run `examples/basic/multi_target_tracking.py` when your measurements are sets
   of detections rather than one detection per time step.
4. Compare the same small workflow under `PYRECEST_BACKEND=numpy`, `pytorch`, and
   `jax` before relying on backend portability.
