"""
Pass detection module.
Detects passes between players using ball trajectory analysis.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PassType(Enum):
    """Types of passes."""
    SHORT = "short"
    LONG = "long"
    THROUGH = "through"
    CROSS = "cross"
    BACK = "back"
    FAILED = "failed"


@dataclass
class Pass:
    """Represents a pass event."""
    start_frame: int
    end_frame: int
    from_player: int
    to_player: Optional[int]
    from_team: int
    to_team: int
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    pass_type: PassType
    success: bool
    distance: float


class PassDetector:
    """Detect passes from tracking data."""

    def __init__(
        self,
        min_pass_distance: float = 3.0,  # meters
        max_pass_distance: float = 60.0,  # meters
        max_pass_duration: int = 30,  # frames
        ball_speed_threshold: float = 5.0  # m/s
    ):
        """Initialize pass detector.

        Args:
            min_pass_distance: Minimum pass distance
            max_pass_distance: Maximum pass distance
            max_pass_duration: Maximum pass duration in frames
            ball_speed_threshold: Ball speed threshold for pass detection
        """
        self.min_pass_distance = min_pass_distance
        self.max_pass_distance = max_pass_distance
        self.max_pass_duration = max_pass_duration
        self.ball_speed_threshold = ball_speed_threshold

    def detect_passes(
        self,
        tracks: Dict,
        team_ball_control: List[int]
    ) -> List[Pass]:
        """Detect all passes in the match.

        Args:
            tracks: Tracking data
            team_ball_control: Ball possession per frame

        Returns:
            List of Pass objects
        """
        passes = []
        
        # Get ball positions
        ball_positions = []
        for frame_num in range(len(tracks["ball"])):
            ball = tracks["ball"][frame_num]
            if ball and 1 in ball:
                pos = ball[1].get("position", (0, 0))
                ball_positions.append((frame_num, pos))
        
        # Detect pass events
        i = 0
        while i < len(ball_positions) - 1:
            frame1, pos1 = ball_positions[i]
            
            # Look for ball movement
            for j in range(i + 1, min(i + self.max_pass_duration, len(ball_positions))):
                frame2, pos2 = ball_positions[j]
                
                distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                duration = frame2 - frame1
                
                if distance > self.min_pass_distance:
                    # Potential pass detected
                    pass_event = self._classify_pass(
                        frame1, frame2, pos1, pos2, distance, duration,
                        tracks, team_ball_control
                    )
                    if pass_event:
                        passes.append(pass_event)
                    i = j
                    break
            else:
                i += 1
        
        return passes

    def _classify_pass(
        self,
        start_frame: int,
        end_frame: int,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        distance: float,
        duration: int,
        tracks: Dict,
        team_ball_control: List[int]
    ) -> Optional[Pass]:
        """Classify a detected pass event.

        Args:
            start_frame: Start frame
            end_frame: End frame
            start_pos: Start position
            end_pos: End position
            distance: Pass distance
            duration: Pass duration
            tracks: Tracking data
            team_ball_control: Ball possession

        Returns:
            Pass object or None
        """
        # Get players at start and end
        from_player = self._get_ball_carrier(start_frame, tracks)
        to_player = self._get_ball_carrier(end_frame, tracks)
        
        if from_player is None:
            return None
        
        # Determine teams
        from_team = team_ball_control[start_frame] if start_frame < len(team_ball_control) else 0
        to_team = team_ball_control[end_frame] if end_frame < len(team_ball_control) else 0
        
        # Determine pass type
        if distance > 30:
            pass_type = PassType.LONG
        elif end_pos[0] > start_pos[0] + 10:  # Forward pass
            pass_type = PassType.THROUGH
        elif abs(end_pos[1] - start_pos[1]) > 20:  # Wide pass
            pass_type = PassType.CROSS
        elif end_pos[0] < start_pos[0]:  # Back pass
            pass_type = PassType.BACK
        else:
            pass_type = PassType.SHORT
        
        # Check if successful
        success = from_team == to_team and to_player is not None
        
        return Pass(
            start_frame=start_frame,
            end_frame=end_frame,
            from_player=from_player,
            to_player=to_player,
            from_team=from_team,
            to_team=to_team,
            start_pos=start_pos,
            end_pos=end_pos,
            pass_type=pass_type,
            success=success,
            distance=distance
        )

    def _get_ball_carrier(self, frame_num: int, tracks: Dict) -> Optional[int]:
        """Get player with ball at specific frame.

        Args:
            frame_num: Frame number
            tracks: Tracking data

        Returns:
            Player ID or None
        """
        if frame_num >= len(tracks["players"]):
            return None
        
        for player_id, track in tracks["players"][frame_num].items():
            if track.get("has_ball", False):
                return player_id
        
        return None

    def get_pass_stats(self, passes: List[Pass]) -> Dict:
        """Calculate pass statistics.

        Args:
            passes: List of passes

        Returns:
            Statistics dictionary
        """
        if not passes:
            return {}
        
        total = len(passes)
        successful = sum(1 for p in passes if p.success)
        
        by_type = {}
        for p in passes:
            pt = p.pass_type.value
            by_type[pt] = by_type.get(pt, 0) + 1
        
        by_team = {1: 0, 2: 0}
        for p in passes:
            if p.from_team in by_team:
                by_team[p.from_team] += 1
        
        return {
            "total_passes": total,
            "successful_passes": successful,
            "success_rate": successful / total * 100 if total > 0 else 0,
            "by_type": by_type,
            "by_team": by_team,
            "avg_distance": np.mean([p.distance for p in passes]) if passes else 0,
        }
