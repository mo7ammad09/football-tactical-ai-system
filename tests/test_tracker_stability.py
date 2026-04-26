from collections import defaultdict

import numpy as np
import supervision as sv

from src.trackers.tracker import Tracker


def _tracker_without_model() -> Tracker:
    tracker = Tracker.__new__(Tracker)
    tracker.person_confidence = 0.25
    tracker.ball_confidence = 0.08
    tracker.role_stability_window = 5
    tracker._role_votes = defaultdict(list)
    tracker._frame_index = 0
    tracker._next_display_id = 1
    tracker._display_id_by_tracker_id = {}
    tracker._last_seen_by_display_id = {}
    tracker.identity_lost_buffer = 90
    tracker.identity_long_buffer = 3600
    tracker.appearance_match_threshold = 0.28
    return tracker


def test_normalizes_common_role_labels():
    tracker = _tracker_without_model()

    assert tracker._normalize_class_name("players") == "player"
    assert tracker._normalize_class_name("Football-Player") == "player"
    assert tracker._normalize_class_name("assistant_referee") == "referee"
    assert tracker._normalize_class_name("Sports-Ball") == "ball"


def test_track_role_uses_recent_majority_to_reduce_class_flicker():
    tracker = _tracker_without_model()

    roles = [
        tracker._stable_role_for_track(7, role)
        for role in ["referee", "referee", "player", "referee"]
    ]

    assert roles[-1] == "referee"


def test_filter_detections_uses_role_specific_confidence_thresholds():
    tracker = _tracker_without_model()
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [20, 0, 30, 10],
                [40, 0, 44, 4],
            ],
            dtype=float,
        ),
        confidence=np.array([0.2, 0.3, 0.09], dtype=float),
        class_id=np.array([0, 1, 2], dtype=int),
    )
    cls_names = {0: "player", 1: "referee", 2: "ball"}

    filtered = tracker._filter_detections(detections, cls_names)

    assert filtered.class_id.tolist() == [1, 2]


def test_display_id_relinks_new_raw_track_near_recent_player():
    tracker = _tracker_without_model()

    first_display_id = tracker._display_id_for_track(
        raw_track_id=101,
        bbox=np.array([100, 100, 140, 220], dtype=float),
        role="player",
        appearance=None,
        used_display_ids=set(),
    )
    tracker._remember_display_track(
        first_display_id,
        np.array([100, 100, 140, 220], dtype=float),
        "player",
        None,
    )
    tracker._frame_index = 1

    relinked_display_id = tracker._display_id_for_track(
        raw_track_id=430,
        bbox=np.array([104, 102, 144, 222], dtype=float),
        role="player",
        appearance=None,
        used_display_ids=set(),
    )

    assert relinked_display_id == first_display_id


def test_display_id_does_not_relink_player_to_recent_referee():
    tracker = _tracker_without_model()
    referee_display_id = tracker._display_id_for_track(
        raw_track_id=10,
        bbox=np.array([100, 100, 140, 220], dtype=float),
        role="referee",
        appearance=None,
        used_display_ids=set(),
    )
    tracker._remember_display_track(
        referee_display_id,
        np.array([100, 100, 140, 220], dtype=float),
        "referee",
        None,
    )
    tracker._frame_index = 1

    player_display_id = tracker._display_id_for_track(
        raw_track_id=11,
        bbox=np.array([104, 102, 144, 222], dtype=float),
        role="player",
        appearance=None,
        used_display_ids=set(),
    )

    assert player_display_id != referee_display_id


def test_display_id_relinks_long_gap_with_matching_appearance():
    tracker = _tracker_without_model()
    appearance = np.zeros(128, dtype=float)
    appearance[5] = 1.0

    first_display_id = tracker._display_id_for_track(
        raw_track_id=21,
        bbox=np.array([100, 100, 140, 220], dtype=float),
        role="player",
        appearance=appearance,
        used_display_ids=set(),
    )
    tracker._remember_display_track(
        first_display_id,
        np.array([100, 100, 140, 220], dtype=float),
        "player",
        appearance,
    )
    tracker._frame_index = tracker.identity_lost_buffer + 30

    relinked_display_id = tracker._display_id_for_track(
        raw_track_id=99,
        bbox=np.array([700, 100, 740, 220], dtype=float),
        role="player",
        appearance=appearance,
        used_display_ids=set(),
    )

    assert relinked_display_id == first_display_id


def test_display_id_does_not_relink_long_gap_with_different_appearance():
    tracker = _tracker_without_model()
    first_appearance = np.zeros(128, dtype=float)
    first_appearance[5] = 1.0
    second_appearance = np.zeros(128, dtype=float)
    second_appearance[70] = 1.0

    first_display_id = tracker._display_id_for_track(
        raw_track_id=21,
        bbox=np.array([100, 100, 140, 220], dtype=float),
        role="player",
        appearance=first_appearance,
        used_display_ids=set(),
    )
    tracker._remember_display_track(
        first_display_id,
        np.array([100, 100, 140, 220], dtype=float),
        "player",
        first_appearance,
    )
    tracker._frame_index = tracker.identity_lost_buffer + 30

    new_display_id = tracker._display_id_for_track(
        raw_track_id=99,
        bbox=np.array([700, 100, 740, 220], dtype=float),
        role="player",
        appearance=second_appearance,
        used_display_ids=set(),
    )

    assert new_display_id != first_display_id
