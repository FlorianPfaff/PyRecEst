from pyrecest.utils.track_evaluation import score_complete_tracks, score_track_links


def test_disjoint_track_identity_sets_have_zero_f1():
    complete_scores = score_complete_tracks([[0, 1]], [[2, 3]])

    assert complete_scores["complete_track_true_positives"] == 0
    assert complete_scores["complete_track_false_positives"] == 1
    assert complete_scores["complete_track_false_negatives"] == 1
    assert complete_scores["complete_track_precision"] == 0.0
    assert complete_scores["complete_track_recall"] == 0.0
    assert complete_scores["complete_track_f1"] == 0.0

    link_scores = score_track_links([[0, 1]], [[2, 3]])

    assert link_scores["track_link_true_positives"] == 0
    assert link_scores["track_link_false_positives"] == 1
    assert link_scores["track_link_false_negatives"] == 1
    assert link_scores["track_link_precision"] == 0.0
    assert link_scores["track_link_recall"] == 0.0
    assert link_scores["track_link_f1"] == 0.0


def test_empty_track_identity_sets_keep_perfect_precision_recall_and_f1():
    scores = score_complete_tracks([[None, None]], [[None, None]])

    assert scores["complete_tracks"] == 0
    assert scores["reference_complete_tracks"] == 0
    assert scores["complete_track_precision"] == 1.0
    assert scores["complete_track_recall"] == 1.0
    assert scores["complete_track_f1"] == 1.0
