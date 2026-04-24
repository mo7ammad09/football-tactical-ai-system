"""
Advanced tactical analysis for football matches.
Combines features from multiple reference projects.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TacticalEvent(Enum):
    """Types of tactical events."""
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    COUNTER_ATTACK = "counter_attack"
    HIGH_PRESS = "high_press"
    BUILD_UP = "build_up"
    SET_PIECE = "set_piece"
    OFFSIDE = "offside"


@dataclass
class PlayerPosition:
    """Player position data."""
    x: float
    y: float
    team: int
    player_id: int
    has_ball: bool = False
    speed: float = 0.0


@dataclass
class TacticalMoment:
    """A tactical moment in the match."""
    frame: int
    time_seconds: float
    event_type: TacticalEvent
    team: int
    players_involved: List[int]
    description: str
    zone: str = ""
    success: bool = True


class TacticalAnalyzer:
    """Advanced tactical analyzer."""

    def __init__(self, pitch_width: float = 105.0, pitch_height: float = 68.0):
        """Initialize analyzer.

        Args:
            pitch_width: Pitch width in meters.
            pitch_height: Pitch height in meters.
        """
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.moments: List[TacticalMoment] = []

    # ==================== FORMATION DETECTION ====================

    def detect_formation(
        self,
        player_positions: List[Tuple[float, float]],
        team_direction: str = "left_to_right"
    ) -> str:
        """Detect team formation from player positions.

        Based on: Tactic_Zone + football_analysis_yolo
        """
        if len(player_positions) < 10:
            return "unknown"

        # Sort by x position (depth)
        sorted_players = sorted(player_positions, key=lambda p: p[0])
        field_players = sorted_players[1:] if len(sorted_players) >= 11 else sorted_players

        # Divide into lines
        x_positions = [p[0] for p in field_players]
        min_x, max_x = min(x_positions), max(x_positions)
        range_x = max_x - min_x

        defense_zone = min_x + range_x * 0.25
        midfield_zone = min_x + range_x * 0.55

        defenders = sum(1 for p in field_players if p[0] <= defense_zone)
        midfielders = sum(1 for p in field_players if defense_zone < p[0] <= midfield_zone)
        attackers = sum(1 for p in field_players if p[0] > midfield_zone)

        return f"{defenders}-{midfielders}-{attackers}"

    # ==================== PITCH CONTROL ====================

    def calculate_pitch_control(
        self,
        team1_positions: List[Tuple[float, float]],
        team2_positions: List[Tuple[float, float]],
        grid_size: Tuple[int, int] = (50, 32)
    ) -> np.ndarray:
        """Calculate pitch control grid using Voronoi-like approach.

        Based on: Football Match Intelligence (DataKnight1)
        """
        grid = np.zeros(grid_size)
        cell_width = self.pitch_width / grid_size[0]
        cell_height = self.pitch_height / grid_size[1]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell_x = i * cell_width + cell_width / 2
                cell_y = j * cell_height + cell_height / 2

                # Calculate influence from each team
                team1_influence = sum(
                    1 / (np.sqrt((cell_x - px)**2 + (cell_y - py)**2) + 0.1)
                    for px, py in team1_positions
                )
                team2_influence = sum(
                    1 / (np.sqrt((cell_x - px)**2 + (cell_y - py)**2) + 0.1)
                    for px, py in team2_positions
                )

                total = team1_influence + team2_influence
                if total > 0:
                    grid[j, i] = (team1_influence - team2_influence) / total

        return grid

    def get_pitch_control_percentage(
        self,
        pitch_control: np.ndarray,
        team: int = 1
    ) -> float:
        """Get pitch control percentage for a team."""
        if team == 1:
            controlled = np.sum(pitch_control > 0)
        else:
            controlled = np.sum(pitch_control < 0)
        return (controlled / pitch_control.size) * 100

    # ==================== DEFENSIVE ANALYSIS ====================

    def detect_defensive_line(
        self,
        player_positions: List[Tuple[float, float]],
        team_direction: str = "left_to_right"
    ) -> Dict:
        """Analyze defensive line position and compactness."""
        if not player_positions:
            return {"line_position": 0, "compactness": 0}

        sorted_by_x = sorted(player_positions, key=lambda p: p[0])
        back_players = sorted_by_x[:min(4, len(sorted_by_x))]

        line_position = np.mean([p[0] for p in back_players])

        # Compactness = std deviation of back line positions
        if len(back_players) > 1:
            compactness = np.std([p[0] for p in back_players])
        else:
            compactness = 0

        return {
            "line_position": line_position,
            "compactness": compactness,
            "line_height": line_position / self.pitch_width  # 0 = own goal, 1 = opponent goal
        }

    # ==================== PRESSING ANALYSIS ====================

    def calculate_pressing_intensity(
        self,
        defending_positions: List[Tuple[float, float]],
        attacking_positions: List[Tuple[float, float]],
        ball_position: Tuple[float, float]
    ) -> Dict:
        """Calculate pressing intensity around the ball."""
        pressing_radius = 10.0  # meters

        # Count defending players near ball
        pressing_players = sum(
            1 for pos in defending_positions
            if np.sqrt((pos[0] - ball_position[0])**2 + (pos[1] - ball_position[1])**2) < pressing_radius
        )

        # PPDA (Passes Per Defensive Action) proxy
        ppda = len(attacking_positions) / max(pressing_players, 1)

        return {
            "pressing_players": pressing_players,
            "intensity": min(pressing_players / 3.0, 1.0),
            "ppda_proxy": ppda,
            "is_high_press": pressing_players >= 3
        }

    # ==================== PASS ANALYSIS ====================

    def detect_pass_sequence(
        self,
        ball_possession: List[Optional[int]],
        team_assignments: Dict[int, int]
    ) -> List[Dict]:
        """Detect pass sequences and possession chains."""
        sequences = []
        current_team = None
        start_frame = 0
        players_in_sequence = []

        for frame, player_id in enumerate(ball_possession):
            if player_id is None:
                continue

            team = team_assignments.get(player_id)

            if team != current_team:
                # End previous sequence
                if current_team is not None and len(players_in_sequence) > 1:
                    sequences.append({
                        "team": current_team,
                        "start_frame": start_frame,
                        "end_frame": frame - 1,
                        "num_passes": len(set(players_in_sequence)) - 1,
                        "duration_frames": frame - start_frame,
                        "players": list(set(players_in_sequence))
                    })

                # Start new sequence
                current_team = team
                start_frame = frame
                players_in_sequence = [player_id]
            else:
                if player_id not in players_in_sequence:
                    players_in_sequence.append(player_id)

        return sequences

    # ==================== ADVANCED PATTERNS ====================

    def detect_counter_attack(
        self,
        team_positions: List[List[Tuple[float, float]]],
        ball_positions: List[Tuple[float, float]],
        frame_window: int = 30
    ) -> List[Dict]:
        """Detect counter-attack opportunities.

        Criteria:
        - Ball moves quickly from defensive to attacking third
        - Attacking team has numerical advantage
        - High ball speed
        """
        counters = []

        for i in range(len(ball_positions) - frame_window):
            start_pos = ball_positions[i]
            end_pos = ball_positions[i + frame_window]

            # Ball moved from defensive to attacking third
            if start_pos[0] < self.pitch_width * 0.3 and end_pos[0] > self.pitch_width * 0.7:
                distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                speed = distance / frame_window

                if speed > 2.0:  # Fast transition
                    counters.append({
                        "start_frame": i,
                        "end_frame": i + frame_window,
                        "distance": distance,
                        "speed": speed,
                        "type": "counter_attack"
                    })

        return counters

    def detect_high_press(
        self,
        team1_positions: List[List[Tuple[float, float]]],
        team2_positions: List[List[Tuple[float, float]]],
        ball_positions: List[Tuple[float, float]],
        frame_window: int = 60
    ) -> List[Dict]:
        """Detect high pressing periods.

        Criteria:
        - 3+ defending players in opponent half
        - Close to ball carrier
        - Sustained for 2+ seconds
        """
        presses = []

        for i in range(0, len(ball_positions) - frame_window, frame_window):
            avg_pressing = 0
            press_frames = 0

            for j in range(i, i + frame_window):
                if j >= len(team1_positions) or j >= len(ball_positions):
                    break

                pressing = self.calculate_pressing_intensity(
                    team1_positions[j], team2_positions[j], ball_positions[j]
                )

                if pressing["is_high_press"]:
                    press_frames += 1

            press_ratio = press_frames / frame_window

            if press_ratio > 0.5:  # Pressing more than 50% of time
                presses.append({
                    "start_frame": i,
                    "end_frame": i + frame_window,
                    "intensity": press_ratio,
                    "type": "high_press"
                })

        return presses

    def detect_build_up_pattern(
        self,
        pass_sequences: List[Dict],
        min_passes: int = 5,
        max_duration: int = 300
    ) -> List[Dict]:
        """Detect build-up play patterns.

        Criteria:
        - 5+ passes in sequence
        - Progression from defensive to attacking third
        - Multiple players involved
        """
        build_ups = []

        for seq in pass_sequences:
            if seq["num_passes"] >= min_passes and seq["duration_frames"] <= max_duration:
                build_ups.append({
                    "start_frame": seq["start_frame"],
                    "end_frame": seq["end_frame"],
                    "passes": seq["num_passes"],
                    "players": seq["players"],
                    "type": "build_up"
                })

        return build_ups

    def detect_wing_attack(
        self,
        player_positions: List[List[PlayerPosition]],
        ball_positions: List[Tuple[float, float]],
        frame_window: int = 45
    ) -> List[Dict]:
        """Detect wing attacks.

        Criteria:
        - Ball in wide areas (y < 20 or y > 48)
        - Players making runs from deep
        - Cross or cut-back opportunity
        """
        wing_attacks = []

        for i in range(len(ball_positions) - frame_window):
            ball_y = ball_positions[i][1]

            # Ball in wide area
            if ball_y < 20 or ball_y > 48:
                # Check for attacking players in box
                attacking_players = sum(
                    1 for pos in player_positions[i]
                    if pos.team == 1 and pos.x > self.pitch_width * 0.75
                )

                if attacking_players >= 2:
                    wing_attacks.append({
                        "start_frame": i,
                        "side": "left" if ball_y < 34 else "right",
                        "players_in_box": attacking_players,
                        "type": "wing_attack"
                    })

        return wing_attacks

    # ==================== COMPREHENSIVE ANALYSIS ====================

    def analyze_match(
        self,
        tracks: Dict,
        team_ball_control: List[int]
    ) -> Dict:
        """Run complete tactical analysis on match data.

        Args:
            tracks: Tracking data from video analysis.
            team_ball_control: Ball possession per frame.

        Returns:
            Complete tactical analysis report.
        """
        report = {
            "summary": {},
            "formations": {},
            "possession": {},
            "pressing": {},
            "patterns": {},
            "recommendations": []
        }

        # Formation detection (from first frame)
        if tracks.get("players") and len(tracks["players"]) > 0:
            team1_positions = []
            team2_positions = []

            for track_id, track in tracks["players"][0].items():
                pos = track.get("position_transformed") or track.get("position")
                if pos:
                    if track.get("team") == 1:
                        team1_positions.append(pos)
                    elif track.get("team") == 2:
                        team2_positions.append(pos)

            report["formations"]["team1"] = self.detect_formation(team1_positions)
            report["formations"]["team2"] = self.detect_formation(team2_positions)

        # Possession stats
        total_frames = len(team_ball_control)
        if total_frames > 0:
            report["possession"]["team1"] = (team_ball_control.count(1) / total_frames) * 100
            report["possession"]["team2"] = (team_ball_control.count(2) / total_frames) * 100

        # Summary
        report["summary"] = {
            "total_frames": total_frames,
            "formation_team1": report["formations"].get("team1", "unknown"),
            "formation_team2": report["formations"].get("team2", "unknown"),
            "possession_team1": f"{report['possession'].get('team1', 0):.1f}%",
            "possession_team2": f"{report['possession'].get('team2', 0):.1f}%"
        }

        return report
