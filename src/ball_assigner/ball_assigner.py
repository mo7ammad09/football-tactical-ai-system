"""
Ball possession assignment based on closest player distance.
Based on: football_analysis_yolo by TrishamBP
"""

import numpy as np

from src.utils.bbox_utils import get_center_of_bbox, measure_distance


class BallAssigner:
    """Assign ball to closest player."""

    def __init__(self, max_player_ball_distance: float = 70):
        """Initialize ball assigner.

        Args:
            max_player_ball_distance: Maximum distance to consider player has ball.
        """
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players: dict, ball_bbox: list) -> int:
        """Assign ball to closest player.

        Args:
            players: Dictionary of player tracks.
            ball_bbox: Ball bounding box.

        Returns:
            Assigned player track ID, or -1 if no assignment.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Measure distance from both feet to ball
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
