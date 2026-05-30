def test_partitioned_so3_product_particle_filter_is_public_lazy_export():
    from pyrecest.filters import PartitionedSO3ProductParticleFilter
    from pyrecest.filters.partitioned_so3_product_particle_filter import (
        PartitionedSO3ProductParticleFilter as DirectPartitionedSO3ProductParticleFilter,
    )

    assert (
        PartitionedSO3ProductParticleFilter is DirectPartitionedSO3ProductParticleFilter
    )
