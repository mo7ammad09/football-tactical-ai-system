from src.trackers.tracker import Tracker


def test_bytetrack_kwargs_for_new_supervision_names():
    kwargs = Tracker._bytetrack_kwargs_for_signature(
        {
            "track_activation_threshold",
            "lost_track_buffer",
            "minimum_matching_threshold",
            "minimum_consecutive_frames",
            "frame_rate",
        }
    )

    assert kwargs == {
        "track_activation_threshold": 0.2,
        "lost_track_buffer": 90,
        "minimum_matching_threshold": 0.7,
        "minimum_consecutive_frames": 1,
        "frame_rate": 30,
    }


def test_bytetrack_kwargs_for_old_supervision_names():
    kwargs = Tracker._bytetrack_kwargs_for_signature(
        {"track_thresh", "track_buffer", "match_thresh", "frame_rate"}
    )

    assert kwargs == {
        "track_thresh": 0.2,
        "track_buffer": 90,
        "match_thresh": 0.7,
        "frame_rate": 30,
    }
