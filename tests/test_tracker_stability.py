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
