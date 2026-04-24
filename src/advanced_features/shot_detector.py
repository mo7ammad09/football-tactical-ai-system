"""
Shot detection module.
Detects shots on goal using ball trajectory and player position analysis.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ShotType(Enum):
    """Types of shots."""
    NORMAL = "normal"
    HEADER = "header"
    VOLLEY = "volley"
    FREE_KICK = "free_kick"
    PENALTY = "penalty"
    OWN_GOAL = "own_goal"


class ShotOutcome(Enum):
    """Shot outcomes."""
    GOAL = "goal"
    SAVED = "saved"
    BLOCKED = "blocked"
    OFF_TARGET = "off_target"
    POST = "post"


@dataclass
class Shot:
    """Represents a shot event."""
    frame: int
    player_id: int
    team: int
    position: Tuple[float, float]
    shot_type: ShotType
    outcome: ShotOutcome
    xg: float  # Expected goals probability
    speed: float  # Ball speed
    defenders_between: int  # Number of defenders between ball and goal
    distance_to_goal: float


class ShotDetector:
    """Detect shots from tracking data."""

    def __init__(
        self,
        goal_left: Tuple[float, float] = (0, 34),  # Left goal center
        goal_right: Tuple[float, float] = (105, 34),  # Right goal center
        shot_speed_threshold: float = 15.0,  # m/s
        max_shot_distance: float = 40.0  # meters
    ):
        """Initialize shot detector.

        Args:
            goal_left: Left goal position
            goal_right: Right goal position
            shot_speed_threshold: Ball speed threshold for shot
            max_shot_distance: Maximum shot distance
        """
        self.goal_left = goal_left
        self.goal_right = goal_right
        self.shot_speed_threshold = shot_speed_threshold
        self.max_shot_distance = max_shot_distance

    def detect_shots(
        self,
        tracks: Dict,
        team_ball_control: List[int]
    ) -> List[Shot]:
        """Detect all shots in the match.

        Args:
            tracks: Tracking data
            team_ball_control: Ball possession per frame

        Returns:
            List of Shot objects
        """
        shots = []
        
        # Analyze ball trajectory
        ball_positions = []
        for frame_num in range(len(tracks["ball"])):
            ball = tracks["ball"][frame_num]
            if ball and 1 in ball:
                pos = ball[1].get("position", (0, 0))
                ball_positions.append((frame_num, pos))
        
        # Detect shot events (rapid ball acceleration toward goal)
        for i in range(1, len(ball_positions) - 1):
            frame_prev, pos_prev = ball_positions[i - 1]
            frame_curr, pos_curr = ball_positions[i]
            frame_next, pos_next = ball_positions[i + 1]
            
            # Calculate ball speed
            dt = frame_curr - frame_prev
            if dt == 0:
                continue
            
            speed = np.sqrt((pos_curr[0] - pos_prev[0])**2 + (pos_curr[1] - pos_prev[1])**2) / dt
            
            # Check if speed indicates a shot
            if speed > self.shot_speed_threshold:
                # Determine direction toward goal
                team = team_ball_control[frame_curr] if frame_curr < len(team_ball_control) else 0
                
                # Determine which goal is being attacked
                if team == 1:
                    target_goal = self.goal_right
                else:
                    target_goal = self.goal_left
                
                distance_to_goal = np.sqrt((pos_curr[0] - target_goal[0])**2 + (pos_curr[1] - target_goal[1])**2)
                
                # Check if shot is within reasonable distance
                if distance_to_goal < self.max_shot_distance:
                    shot = self._classify_shot(
                        frame_curr, pos_curr, speed, distance_to_goal,
                        target_goal, team, tracks
                    )
                    if shot:
                        shots.append(shot)
        
        return shots

    def _classify_shot(
        self,
        frame: int,
        position: Tuple[float, float],
        speed: float,
        distance_to_goal: float,
        target_goal: Tuple[float, float],
        team: int,
        tracks: Dict
    ) -> Optional[Shot]:
        """Classify a detected shot.

        Args:
            frame: Frame number
            position: Ball position
            speed: Ball speed
            distance_to_goal: Distance to goal
            target_goal: Target goal position
            team: Team shooting
            tracks: Tracking data

        Returns:
            Shot object or None
        """
        # Get shooter
        shooter = None
        if frame < len(tracks["players"]):
            for player_id, track in tracks["players"][frame].items():
                if track.get("has_ball", False):
                    shooter = player_id
                    break
        
        if shooter is None:
            return None
        
        # Calculate xG (simplified model)
        xg = self._calculate_xg(position, distance_to_goal, target_goal)
        
        # Count defenders between ball and goal
        defenders = self._count_defenders(position, target_goal, team, tracks, frame)
        
        # Determine shot type
        if distance_to_goal < 11:  # Penalty area
            shot_type = ShotType.PENALTY if distance_to_goal < 11 else ShotType.NORMAL
        else:
            shot_type = ShotType.NORMAL
        
        # Determine outcome (simplified - would need more frames to determine)
        outcome = ShotOutcome.OFF_TARGET  # Default
        
        return Shot(
            frame=frame,
            player_id=shooter,
            team=team,
            position=position,
            shot_type=shot_type,
            outcome=outcome,
            xg=xg,
            speed=speed,
            defenders_between=defenders,
            distance_to_goal=distance_to_goal
        )

    def _calculate_xg(
        self,
        position: Tuple[float, float],
        distance: float,
        goal: Tuple[float, float]
    ) -> float:
        """Calculate expected goals probability.

        Args:
            position: Shot position
            distance: Distance to goal
            goal: Goal position

        Returns:
            xG probability (0-1)
        """
        # Simplified xG model based on distance
        # Real xG models are much more complex
        max_xg = 0.95
        min_xg = 0.01
        
        # Exponential decay with distance
        xg = max_xg * np.exp(-distance / 15)
        
        # Adjust for angle
        angle = abs(position[1] - goal[1]) / max(distance, 1)
        angle_factor = max(0, 1 - angle / 2)
        
        xg *= angle_factor
        
        return np.clip(xg, min_xg, max_xg)

    def _count_defenders(
        self,
        position: Tuple[float, float],
        goal: Tuple[float, float],
        attacking_team: int,
        tracks: Dict,
        frame: int
    ) -> int:
        """Count defenders between ball and goal.

        Args:
            position: Ball position
            goal: Goal position
            attacking_team: Team attacking
            tracks: Tracking data
            frame: Frame number

        Returns:
            Number of defenders
        """
        if frame >= len(tracks["players"]):
            return 0
        
        defenders = 0
        
        for player_id, track in tracks["players"][frame].items():
            if track.get("team") != attacking_team:
                player_pos = track.get("position", (0, 0))
                
                # Check if defender is between ball and goal
                if self._is_between(player_pos, position, goal):
                    defenders += 1
        
        return defenders

    def _is_between(
        self,
        point: Tuple[float, float],
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> bool:
        """Check if point is between start and end.

        Args:
            point: Point to check
            start: Start point
            end: End point

        Returns:
            True if point is between
        """
        # Simplified check - check if point is in the bounding box
        min_x = min(start[0], end[0])
        max_x = max(start[0], end[0])
        min_y = min(start[1], end[1])
        max_y = max(start[1], end[1])
        
        return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

    def get_shot_stats(self, shots: List[Shot]) -> Dict:
        """Calculate shot statistics.

        Args:
            shots: List of shots

        Returns:
            Statistics dictionary
        """
        if not shots:
            return {}
        
        total = len(shots)
        goals = sum(1 for s in shots if s.outcome == ShotOutcome.GOAL)
        on_target = sum(1 for s in shots if s.outcome in [ShotOutcome.GOAL, ShotOutcome.SAVED])
        
        by_team = {1: 0, 2: 0}
        for s in shots:
            if s.team in by_team:
                by_team[s.team] += 1
        
        total_xg = sum(s.xg for s in shots)
        
        return {
            "total_shots": total,
            "goals": goals,
            "on_target": on_target,
            "accuracy": on_target / total * 100 if total > 0 else 0,
            "by_team": by_team,
            "total_xg": total_xg,
            "avg_xg": total_xg / total if total > 0 else 0,
        }
