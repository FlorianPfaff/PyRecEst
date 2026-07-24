"""Small multisensor DDP/HDP-style association example."""

from __future__ import annotations

import numpy as np
from pyrecest.experimental import (
    SensorAssociationBlock,
    multisensor_ddp_association_update,
)


def main() -> None:
    target_weights = [0.45, 0.45]
    target_labels = ("vehicle-left", "vehicle-right")

    radar = SensorAssociationBlock(
        "radar",
        log_likelihoods=np.log([[0.92, 0.03], [0.04, 0.86], [0.02, 0.03]]),
        clutter_log_weights=np.log([0.01, 0.01, 0.60]),
        birth_log_weights=np.log([0.02, 0.02, 0.05]),
        concentration=4.0,
    )
    camera = SensorAssociationBlock(
        "camera",
        log_likelihoods=np.log([[0.88, 0.04], [0.05, 0.83]]),
        clutter_log_weights=np.log([0.02, 0.02]),
        birth_log_weights=np.log([0.03, 0.03]),
        concentration=4.0,
    )

    result = multisensor_ddp_association_update(
        target_weights,
        [radar, camera],
        target_labels=target_labels,
        birth_weight=0.10,
        prior_strength=0.5,
    )

    for sensor_id, posterior in result.sensor_posteriors.items():
        print(sensor_id, posterior.hard_assignments)

    print(
        "updated target weights:",
        dict(zip(target_labels, result.updated_target_weights, strict=True)),
    )
    print("updated birth weight:", result.updated_birth_weight)
    print("expected clutter count:", result.expected_clutter_count)


if __name__ == "__main__":
    main()
