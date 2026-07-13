from pyrecest.utils import HistoryRecorder


def test_generic_history_wraps_single_initial_value_in_live_list():
    recorder = HistoryRecorder()
    initial_value = {"kind": "initial"}

    history = recorder.register("events", initial_value=initial_value)
    initial_value["kind"] = "mutated"

    assert history == [{"kind": "initial"}]
    assert recorder.record("events", {"kind": "next"}) is history
    assert history == [{"kind": "initial"}, {"kind": "next"}]
