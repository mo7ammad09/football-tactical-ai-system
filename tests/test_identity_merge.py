from src.processing.batch_analyzer import (
    _apply_identity_merge_map,
    _normalize_identity_merge_map,
)


def test_normalize_identity_merge_map_keeps_valid_positive_changes():
    assert _normalize_identity_merge_map(
        {
            "430": "12",
            "5": "5",
            "bad": "7",
            "0": "8",
            "9": "-1",
        }
    ) == {430: 12}


def test_apply_identity_merge_map_rekeys_players_and_referees():
    batch_tracks = {
        "players": [
            {
                430: {"bbox": [0, 0, 10, 10]},
                12: {"bbox": [0, 0, 5, 5]},
                9: {"bbox": [10, 10, 20, 20]},
            },
            {
                430: {"bbox": [1, 1, 11, 11]},
            },
        ],
        "referees": [
            {
                88: {"bbox": [0, 0, 3, 3]},
            },
            {},
        ],
        "ball": [{}, {}],
    }

    conflicts = _apply_identity_merge_map(batch_tracks, {430: 12, 88: 7})

    assert conflicts == 1
    assert 430 not in batch_tracks["players"][0]
    assert batch_tracks["players"][0][12]["merged_from_id"] == 430
    assert batch_tracks["players"][1][12]["merged_from_id"] == 430
    assert batch_tracks["referees"][0][7]["merged_from_id"] == 88
