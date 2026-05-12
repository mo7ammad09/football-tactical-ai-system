from src.ball_assigner.ball_assigner import BallAssigner


def test_assign_ball_to_closest_player():
    players = {
        10: {"bbox": [100, 100, 140, 220]},
        11: {"bbox": [300, 100, 340, 220]},
    }
    ball_bbox = [120, 180, 130, 190]

    assigner = BallAssigner(max_player_ball_distance=80)
    assigned = assigner.assign_ball_to_player(players, ball_bbox)

    assert assigned == 10


def test_assign_ball_returns_minus_one_when_far():
    players = {
        10: {"bbox": [100, 100, 140, 220]},
    }
    ball_bbox = [1000, 1000, 1010, 1010]

    assigner = BallAssigner(max_player_ball_distance=50)
    assigned = assigner.assign_ball_to_player(players, ball_bbox)

    assert assigned == -1
