import numpy as np

from src.team_assigner.team_assigner import TeamAssigner


class DummyKMeans:
    def __init__(self):
        self.cluster_centers_ = np.array(
            [
                [10.0, 20.0, 30.0],
                [200.0, 210.0, 220.0],
            ]
        )


def test_get_player_team_uses_recent_confident_majority(monkeypatch):
    assigner = TeamAssigner()
    assigner.kmeans = DummyKMeans()

    calls = {"count": 0}

    def fake_get_player_color(_frame, _bbox):
        calls["count"] += 1
        return np.array([10.0, 20.0, 30.0])

    monkeypatch.setattr(assigner, "get_player_color", fake_get_player_color)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 40, 80]

    team_first = assigner.get_player_team(frame, bbox, player_id=7)
    team_second = assigner.get_player_team(frame, bbox, player_id=7)

    assert team_first == 1
    assert team_second == 1
    assert calls["count"] == 2


def test_get_player_team_keeps_previous_when_color_is_ambiguous(monkeypatch):
    assigner = TeamAssigner()
    assigner.kmeans = DummyKMeans()

    colors = [
        np.array([10.0, 20.0, 30.0]),
        np.array([105.0, 115.0, 125.0]),
    ]

    def fake_get_player_color(_frame, _bbox):
        return colors.pop(0)

    monkeypatch.setattr(assigner, "get_player_color", fake_get_player_color)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 40, 80]

    assert assigner.get_player_team(frame, bbox, player_id=7) == 1
    assert assigner.get_player_team(frame, bbox, player_id=7) == 1
