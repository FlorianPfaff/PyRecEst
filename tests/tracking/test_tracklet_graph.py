from __future__ import annotations

import numpy as np
from pyrecest.tracking.tracklet_graph import (
    Tracklet,
    TrackletGraphConfig,
    constant_velocity_edge_cost,
    diverse_k_best_tracklet_paths,
    k_best_tracklet_paths,
    path_jaccard,
    tracklet_paths_to_dicts,
)


def _tracklet(
    identifier: str,
    start: float,
    end: float,
    x0: float,
    x1: float,
    *,
    cost: float = 0.0,
    label: str = "a",
) -> Tracklet:
    return Tracklet(
        id=identifier,
        start_time=start,
        end_time=end,
        start_state=np.array([x0, 0.0]),
        end_state=np.array([x1, 0.0]),
        cost=cost,
        metadata={"label": label},
    )


def test_k_best_tracklet_paths_prefers_feasible_low_cost_chain() -> None:
    tracklets = [
        _tracklet("a", 0.0, 1.0, 0.0, 1.0, cost=-5.0),
        _tracklet("b", 2.0, 3.0, 2.0, 3.0, cost=-5.0),
        _tracklet("c", 2.0, 3.0, 80.0, 81.0),
    ]
    edge_cost = constant_velocity_edge_cost(max_gap=5.0, max_speed=10.0)

    paths = k_best_tracklet_paths(
        tracklets, edge_cost_fn=edge_cost, config=TrackletGraphConfig(top_k=3)
    )

    assert paths[0].tracklet_ids == ("a", "b")
    assert ("a", "c") not in [path.tracklet_ids for path in paths]
    assert paths[0].length == 2


def test_switch_penalty_and_node_cost_are_reflected() -> None:
    tracklets = [
        _tracklet("a", 0.0, 1.0, 0.0, 1.0, cost=-5.0, label="one"),
        _tracklet("b", 2.0, 3.0, 2.0, 3.0, cost=-5.0, label="two"),
    ]
    edge_cost = constant_velocity_edge_cost(
        max_gap=5.0,
        max_speed=10.0,
        switch_metadata_key="label",
        switch_penalty=7.0,
    )

    paths = k_best_tracklet_paths(
        tracklets, edge_cost_fn=edge_cost, config=TrackletGraphConfig(top_k=3)
    )
    path = next(item for item in paths if item.tracklet_ids == ("a", "b"))

    assert path.edge_cost >= 7.0


def test_diverse_path_selection_penalizes_overlap() -> None:
    tracklets = [
        _tracklet("a", 0.0, 1.0, 0.0, 1.0),
        _tracklet("b", 2.0, 3.0, 2.0, 3.0),
        _tracklet("c", 4.0, 5.0, 4.0, 5.0),
        _tracklet("x", 0.0, 1.0, 100.0, 101.0, cost=0.1),
        _tracklet("y", 2.0, 3.0, 102.0, 103.0, cost=0.1),
    ]
    edge_cost = constant_velocity_edge_cost(max_gap=5.0, max_speed=10.0)

    paths = diverse_k_best_tracklet_paths(
        tracklets,
        edge_cost_fn=edge_cost,
        config=TrackletGraphConfig(top_k=2, diversity_weight=100.0),
    )

    assert len(paths) == 2
    assert path_jaccard(paths[0], paths[1]) < 1.0
    assert "jaccard_to_previous" in paths[1].metadata


def test_tracklet_paths_to_dicts_includes_materialized_times() -> None:
    tracklets = {
        "a": _tracklet("a", 0.0, 1.0, 0.0, 1.0, cost=-5.0),
        "b": _tracklet("b", 2.0, 3.0, 2.0, 3.0, cost=-5.0),
    }
    edge_cost = constant_velocity_edge_cost(max_gap=5.0, max_speed=10.0)
    path = k_best_tracklet_paths(
        list(tracklets.values()),
        edge_cost_fn=edge_cost,
        config=TrackletGraphConfig(top_k=1),
    )[0]

    rows = tracklet_paths_to_dicts([path], tracklets=tracklets)

    assert rows[0]["rank"] == 0
    assert rows[0]["tracklet_ids"] == "a;b"
    assert rows[0]["start_time"] == 0.0
    assert rows[0]["end_time"] == 3.0


def test_tracklet_validation_rejects_bad_state() -> None:
    try:
        Tracklet("bad", 1.0, 0.0, np.zeros(2), np.zeros(2))
    except ValueError as exc:
        assert "end_time" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
