from collections import Counter

import numpy as np

from src.processing.batch_analyzer import (
    _apply_identity_merge_map,
    _build_auto_identity_merge_map,
    _rebuild_player_outputs,
)


def _profile(track_id, start, end, embedding, team=1):
    return {
        "id": track_id,
        "role_counts": Counter({"player": 10}),
        "team_counts": Counter({team: 10}),
        "frames_seen": 10,
        "first_sample": start,
        "last_sample": end,
        "first_source_frame_idx": start,
        "last_source_frame_idx": end,
        "first_bbox": [100, 100, 140, 220],
        "last_bbox": [110, 100, 150, 220],
        "reid_embedding": np.asarray(embedding, dtype=float),
        "reid_count": 10,
        "color_signature": None,
        "color_count": 0,
    }


def test_auto_identity_reconciliation_links_non_overlapping_reid_tracklets():
    base_embedding = np.zeros(8, dtype=float)
    base_embedding[2] = 1.0
    similar_embedding = np.zeros(8, dtype=float)
    similar_embedding[2] = 0.98
    similar_embedding[3] = 0.02
    different_embedding = np.zeros(8, dtype=float)
    different_embedding[6] = 1.0

    merge_map, links = _build_auto_identity_merge_map(
        {
            12: _profile(12, 0, 100, base_embedding),
            430: _profile(430, 160, 260, similar_embedding),
            77: _profile(77, 300, 400, different_embedding),
        }
    )

    assert merge_map == {430: 12}
    assert links[0]["evidence"] == "osnet_reid"


def test_auto_identity_reconciliation_does_not_merge_overlapping_tracklets():
    embedding = np.zeros(8, dtype=float)
    embedding[1] = 1.0

    merge_map, links = _build_auto_identity_merge_map(
        {
            12: _profile(12, 0, 100, embedding),
            430: _profile(430, 80, 180, embedding),
        }
    )

    assert merge_map == {}
    assert links == []


def test_rebuild_player_outputs_after_auto_merge_map():
    states = [
        {
            "sample_number": 1,
            "source_frame_idx": 0,
            "players": {12: {"role": "player", "team": 1, "bbox": [0, 0, 1, 1]}},
            "referees": {},
        },
        {
            "sample_number": 2,
            "source_frame_idx": 30,
            "players": {430: {"role": "player", "team": 1, "bbox": [0, 0, 1, 1]}},
            "referees": {},
        },
    ]
    _apply_identity_merge_map(
        {"players": [state["players"] for state in states], "referees": [{}, {}]},
        {430: 12},
    )

    player_stats, spans = _rebuild_player_outputs(states)

    assert player_stats == [
        {
            "id": 12,
            "name": "Player 12",
            "role": "player",
            "team": 1,
            "frames_seen": 2,
            "distance_km": None,
            "max_speed_kmh": None,
            "distance_speed_confidence": 0.0,
        }
    ]
    assert spans[0]["merged_from_ids"] == [430]
