import numpy as np
from pyrecest.evaluation.get_extract_mean import get_extract_mean


class Track:
    def __init__(self, mean):
        self.mu = np.asarray(mean)


class TrackerWithTracksAttribute:
    def __init__(self):
        self.tracks = {
            "track-a": Track([1.0, 2.0]),
            "track-b": Track([3.0, 4.0]),
        }


class TrackerWithGetTracks:
    def get_tracks(self):
        return {
            10: Track([5.0, 6.0]),
            20: Track([7.0, 8.0]),
        }


def test_mtt_mean_extracts_mapping_values_from_tracks_attribute():
    extract_mean = get_extract_mean("euclidean", mtt_scenario=True)

    means = extract_mean(TrackerWithTracksAttribute())

    assert len(means) == 2
    np.testing.assert_array_equal(means[0], [1.0, 2.0])
    np.testing.assert_array_equal(means[1], [3.0, 4.0])


def test_mtt_mean_extracts_mapping_values_from_get_tracks():
    extract_mean = get_extract_mean("euclidean", mtt_scenario=True)

    means = extract_mean(TrackerWithGetTracks())

    assert len(means) == 2
    np.testing.assert_array_equal(means[0], [5.0, 6.0])
    np.testing.assert_array_equal(means[1], [7.0, 8.0])


def test_mtt_mean_extracts_mapping_values_from_plain_mapping():
    extract_mean = get_extract_mean("euclidean", mtt_scenario=True)
    tracker_state = {
        "track-a": Track([9.0, 10.0]),
        "track-b": Track([11.0, 12.0]),
    }

    means = extract_mean(tracker_state)

    assert isinstance(means, list)
    assert len(means) == 2
    np.testing.assert_array_equal(means[0], [9.0, 10.0])
    np.testing.assert_array_equal(means[1], [11.0, 12.0])
