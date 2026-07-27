import numpy as np
import pytest
from pyrecest.tracking.tracklet_graph import (
    Tracklet,
    TrackletGraphConfig,
    build_tracklet_adjacency,
)


def _overlapping_tracklets():
    return [
        Tracklet("left", 0.0, 2.0, [0.0], [1.0]),
        Tracklet("right", 1.0, 3.0, [1.0], [2.0]),
    ]


def _edge_cost(_left, _right):
    return 0.0


def test_config_rejects_non_boolean_allow_overlap():
    for value in ("false", 0, 1, [False], np.array([False])):
        with pytest.raises(ValueError, match="allow_overlap must be a boolean"):
            TrackletGraphConfig(allow_overlap=value)


def test_config_accepts_scalar_numpy_boolean():
    config = TrackletGraphConfig(allow_overlap=np.array(True))
    assert config.allow_overlap is True


def test_direct_adjacency_builder_rejects_non_boolean_allow_overlap():
    with pytest.raises(ValueError, match="allow_overlap must be a boolean"):
        build_tracklet_adjacency(
            _overlapping_tracklets(),
            _edge_cost,
            allow_overlap="false",
        )


def test_boolean_allow_overlap_controls_overlapping_edges():
    tracklets = _overlapping_tracklets()

    blocked = build_tracklet_adjacency(tracklets, _edge_cost, allow_overlap=False)
    allowed = build_tracklet_adjacency(tracklets, _edge_cost, allow_overlap=True)

    assert blocked["left"] == []
    assert allowed["left"] == [("right", 0.0)]
