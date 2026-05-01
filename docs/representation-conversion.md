# Representation Conversion

PyRecEst distributions often support several representations of the same
uncertainty state: analytic densities, Dirac/particle sets, grids, Fourier
series, mixtures, and moment-matched approximations.

Use `convert_distribution` to make these conversions explicit and discoverable.
The conversion API is re-exported from `pyrecest.distributions`.

```python
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution, convert_distribution

prior = GaussianDistribution(array([0.0, 0.0]), eye(2))
particles = convert_distribution(prior, "particles", n_particles=1000)
```

Some PyRecEst distribution base classes also expose convenience wrappers such as
`convert_to(...)` and `approximate_as(...)`; these delegate to the same generic
conversion gateway when available.

```python
particles = prior.approximate_as("particles", n_particles=1000)
gaussian = particles.approximate_as("gaussian")
```

## Target-centric conversions

Conversions are target-centric. A target class can expose
`from_distribution(distribution, ...)`, and the generic conversion gateway will
call it.

This keeps domain-specific approximation logic close to the representation that
owns it. For example:

- `LinearDiracDistribution.from_distribution(...)` samples a source density.
- `CircularGridDistribution.from_distribution(...)` evaluates a circular density
  on a grid.
- `CircularFourierDistribution.from_distribution(...)` builds Fourier
  coefficients from samples or grid values.
- `GaussianDistribution.from_distribution(...)` performs Gaussian moment
  matching when the source exposes `mean()` and `covariance()`.

## Metadata

Use `return_info=True` when you need to know how the conversion was performed.

```python
result = convert_distribution(
    prior,
    "particles",
    n_particles=1000,
    return_info=True,
)

particles = result.distribution
print(result.method)
print(result.exact)
```

Identity conversions are exact. Sampling, grid approximation, Fourier
truncation, and moment matching are reported as approximate unless a converter
explicitly marks them as exact.

## Custom conversions

Third-party representations can register conversions without editing central
dispatch code.

```python
from pyrecest.distributions import register_conversion


@register_conversion(MyDistribution, MyParticleDistribution)
def my_distribution_to_particles(distribution, n_particles):
    return MyParticleDistribution(distribution.sample(n_particles))
```

After registration, the normal gateway works:

```python
particles = convert_distribution(source, MyParticleDistribution, n_particles=1000)
```

## String aliases

The conversion gateway accepts concrete classes and a small set of built-in
aliases. Useful aliases include:

- `"particles"`, `"dirac"`, and `"samples"` for domain-aware Dirac/particle
  representations;
- `"gaussian"` and `"moment_matched_gaussian"` for Gaussian moment matching;
- `"grid"` for domain-aware circular, hypertoroidal, hyperspherical, or
  hyperhemispherical grid representations;
- `"fourier"` for circular or hypertoroidal Fourier representations;
- explicit aliases such as `"linear_dirac"`, `"circular_grid"`,
  `"hypertoroidal_grid"`, and `"circular_fourier"`.

Aliases are case-insensitive, and hyphens or spaces are normalized to
underscores. Custom aliases can be registered with `register_conversion_alias`:

```python
from pyrecest.distributions import register_conversion_alias

register_conversion_alias("my_particles", MyParticleDistribution)
```

## Common parameters

Common conversion parameters depend on the target representation:

- `n_particles` for Dirac/particle approximations.
- `no_of_gridpoints` for one-dimensional circular grids.
- `n_grid_points` for hypertoroidal grids.
- `no_of_grid_points` and `grid_type` for hypersphere-subset grids.
- `n` for Fourier representations.
