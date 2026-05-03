# Representation Conversion

PyRecEst distributions often support several representations of the same
uncertainty state: analytic densities, Dirac/particle sets, grids, Fourier
series, mixtures, and moment-matched approximations.

Use `convert_distribution` to make these conversions explicit and discoverable.

```python
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.conversion import convert_distribution

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
- SO(3) tangent-Gaussian conversions are registered separately and use local
  tangent-space moment matching.

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
from pyrecest.distributions.conversion import register_conversion


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
  `"hypertoroidal_grid"`, and `"circular_fourier"`;
- SO(3)-specific aliases such as `"so3_dirac"`, `"so3_particles"`,
  `"so3_tangent_gaussian"`, `"so3_gaussian"`, `"so3_product_dirac"`,
  `"so3_product_particles"`, `"so3_product_tangent_gaussian"`, and
  `"so3_product_gaussian"`.

For SO(3) and SO(3)^K distributions, `"particles"`/`"dirac"` resolve to
SO(3)-specific Dirac representations, and `"gaussian"` resolves to the
corresponding tangent-Gaussian approximation.

Aliases are case-insensitive, and hyphens or spaces are normalized to
underscores. Custom aliases can be registered with `register_conversion_alias`:

```python
from pyrecest.distributions.conversion import register_conversion_alias

register_conversion_alias("my_particles", MyParticleDistribution)
```

## Support matrix

The table below documents the routes supported by the generic conversion
gateway. A route is supported when the alias resolver can select a target class
and that target class either implements `from_distribution(...)`, is already the
source type, or has an explicitly registered converter.

| Source representation | Target / alias | Concrete target selected | Required parameters | Method | Exact? |
| --- | --- | --- | --- | --- | --- |
| Any distribution | same concrete class | requested class | none | identity conversion | yes |
| Any registered source type | registered target class | registered target class | converter-specific | registered converter | converter-specific |
| Any distribution accepted by a registered alias | registered alias | alias target or resolver result | alias/converter-specific | registered converter or target `from_distribution(...)` | converter-specific |
| `AbstractLinearDistribution` | `"particles"`, `"dirac"`, `"samples"`, `"linear_dirac"` | `LinearDiracDistribution` | `n_particles` | sampling through `from_distribution(...)` | no |
| distribution exposing `mean()` and `covariance()` | `"gaussian"`, `"normal"`, `"moment_matched_gaussian"` | `GaussianDistribution` | none | moment matching through `GaussianDistribution.from_distribution(...)` | no, except identity |
| `GaussianDistribution` | `LinearDiracDistribution` or linear particle aliases | `LinearDiracDistribution` | `n_particles` | Gaussian sampling | no |
| `LinearDiracDistribution` | Gaussian aliases | `GaussianDistribution` | none | weighted empirical mean/covariance | no |
| `AbstractCircularDistribution` | `"particles"`, `"dirac"`, `"samples"`, `"circular_dirac"` | `CircularDiracDistribution` | `n_particles` | circular sampling | no |
| `AbstractCircularDistribution` | `"grid"`, `"circular_grid"` | `CircularGridDistribution` | `no_of_gridpoints` | evaluate density on circular grid | no |
| `AbstractCircularDistribution` | `"fourier"`, `"circular_fourier"` | `CircularFourierDistribution` | `n` | Fourier approximation | no |
| `AbstractHypertoroidalDistribution` | `"particles"`, `"dirac"`, `"samples"`, `"hypertoroidal_dirac"` | `HypertoroidalDiracDistribution` | `n_particles` | hypertoroidal sampling | no |
| `AbstractHypertoroidalDistribution` | `"grid"`, `"hypertoroidal_grid"` | `HypertoroidalGridDistribution` | `n_grid_points` | evaluate density on hypertoroidal grid | no |
| `AbstractHypertoroidalDistribution` | `"fourier"`, `"hypertoroidal_fourier"` | `HypertoroidalFourierDistribution` | target-specific Fourier parameters | Fourier approximation | no |
| `AbstractHypersphericalDistribution` | `"particles"`, `"dirac"`, `"samples"`, `"hyperspherical_dirac"` | `HypersphericalDiracDistribution` | `n_particles` | hyperspherical sampling | no |
| `AbstractHypersphericalDistribution` | `"grid"`, `"hyperspherical_grid"` | `HypersphericalGridDistribution` | `no_of_grid_points`, `grid_type` | evaluate density on hyperspherical grid | no |
| `AbstractHyperhemisphericalDistribution` | `"particles"`, `"dirac"`, `"samples"`, `"hyperhemispherical_dirac"` | `HyperhemisphericalDiracDistribution` | `n_particles` | hyperhemispherical sampling | no |
| `AbstractHyperhemisphericalDistribution` | `"grid"`, `"hyperhemispherical_grid"` | `HyperhemisphericalGridDistribution` | `no_of_grid_points`, `grid_type` | evaluate density on hyperhemispherical grid | no |

The `"samples"` alias currently follows the same route as `"particles"` and
returns a weighted Dirac representation, not a raw unweighted sample array. Use
the source distribution's `sample(...)` method directly when raw samples are
needed.

## Common parameters

Common conversion parameters depend on the target representation:

| Parameter | Used by | Meaning |
| --- | --- | --- |
| `n_particles` | Dirac/particle targets | Number of samples used to form the weighted Dirac approximation. |
| `no_of_gridpoints` | `CircularGridDistribution` | Number of circular grid points. |
| `n_grid_points` | `HypertoroidalGridDistribution` | Number of grid points per toroidal dimension. |
| `no_of_grid_points` | hypersphere-subset grid targets | Number of hyperspherical or hyperhemispherical grid points. |
| `grid_type` | hypersphere-subset grid targets | Grid construction method, such as a Leopardi-style grid where supported. |
| `n` | Fourier targets | Fourier truncation/order parameter used by the target representation. |
| `covariance_regularization` | SO(3) tangent-Gaussian targets | Optional diagonal regularization when fitting tangent-Gaussian approximations from few Dirac particles. |
| `return_info` | conversion gateway | Return a `ConversionResult` instead of only the converted distribution. |
| `copy_if_same` | conversion gateway | Return a deep copy for identity conversion instead of the original object. |

## Error messages

Alias-related conversion errors distinguish between two cases:

- an unknown alias, such as `"not_a_representation"`;
- a known alias that is not valid for the source distribution, such as asking a
  linear Gaussian distribution for a domain-aware `"grid"` representation.

For known-but-unsupported aliases, the error message names the source type and
lists aliases that are supported for that source. For unknown aliases, the error
message lists known built-in aliases, aliases supported by the current source,
and any custom aliases registered with `register_conversion_alias(...)`.

Use an explicit target class when you intentionally want to try a conversion
outside the domain-aware alias system, or register a custom alias for a new
representation family.
