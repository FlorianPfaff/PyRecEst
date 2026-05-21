# Distribution Taxonomy

PyRecEst contains many distribution classes because recursive estimation often
depends on both the geometry of the state space and the numerical
representation of the density.  Use this page as a high-level map before
opening the full API reference.

## By State Space

| Domain                              | Typical use                                                            | Common representation families                                     |
|-------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|
| Euclidean / linear spaces           | positions, velocities, sensor biases, additive errors                  | Gaussian, Gaussian mixtures, Dirac/particle, grid                  |
| Circular states                     | headings, bearings, phases                                             | von Mises, wrapped normal, circular Fourier, circular Dirac        |
| Hypertoroidal states                | multiple coupled angles                                                | hypertoroidal wrapped/Gaussian-like, Fourier, grid, Dirac          |
| Spherical and hyperspherical states | unit vectors, directions, orientations represented as unit quaternions | von Mises-Fisher, Bingham, hyperspherical Dirac/grid               |
| Hyperhemispherical states           | antipodal or sign-normalized directional quantities                    | Bingham-related and hemisphere-specific grid/Dirac forms           |
| SE(2), SE(3), SO(3) style spaces    | rigid-body pose or attitude estimation                                 | manifold-specific analytic, grid, or particle-style approximations |
| Cartesian-product states            | mixed Euclidean, angular, and directional components                   | product distributions and hypercylindrical distributions           |

## By Numerical Representation

| Representation               | Best fit                                                            | Main tradeoff                                                    |
|------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------|
| Analytic parametric density  | Compact unimodal states with closed-form operations                 | Less expressive for multimodal or irregular posteriors.          |
| Mixture density              | Multimodal states with a finite number of modes                     | Component management and pruning become important.               |
| Dirac / particle density     | Nonlinear models, arbitrary likelihoods, sample-based prediction    | Accuracy depends on sample count and resampling quality.         |
| Grid density                 | Low-dimensional deterministic density calculations                  | Memory and computation grow quickly with dimension.              |
| Fourier / harmonic density   | Periodic or spherical domains where spectral operations are natural | Backend support and truncation behavior are route-specific.      |
| Moment-matched approximation | Converting rich representations back to Gaussian-like summaries     | Higher-order or multimodal structure is intentionally discarded. |

## Conversion Guidance

Use `pyrecest.distributions.conversion.convert_distribution(...)` when moving
between representations.  Conversion behavior is route-specific:

- Euclidean Gaussian-to-particle and particle-to-Gaussian routes are the most
  portable baseline.
- Grid, Fourier, spherical-harmonics, and manifold routes may depend on NumPy,
  SciPy, or domain-specific optional dependencies.
- Moment matching is useful for summaries and Gaussian filters, but it should
  not be treated as a lossless conversion.

When adding a new distribution class, document both the state-space geometry and
the numerical representation.  If the class participates in conversion, add a
focused conversion test and update the backend API matrix when the route is
intended to be backend-portable.
