"""
Interactive tactical board and visualizations.
Based on: Football Match Intelligence (DataKnight1)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import matplotlib.patches as patches


class TacticalBoard:
    """Interactive tactical board for football analysis."""

    def __init__(self, width: float = 105.0, height: float = 68.0):
        """Initialize tactical board.

        Args:
            width: Pitch width in meters.
            height: Pitch height in meters.
        """
        self.width = width
        self.height = height

    def draw_pitch(
        self,
        figsize: Tuple[int, int] = (14, 9),
        color: str = "#1a5f2a",
        line_color: str = "white",
        orientation: str = "horizontal"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Draw football pitch.

        Args:
            figsize: Figure size.
            color: Pitch color.
            line_color: Line color.
            orientation: "horizontal" or "vertical".

        Returns:
            (fig, ax) matplotlib objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Pitch background
        pitch = Rectangle(
            (0, 0), self.width, self.height,
            linewidth=0, edgecolor="none", facecolor=color
        )
        ax.add_patch(pitch)

        # Lines
        lw = 2  # line width

        # Outer boundary
        ax.plot([0, self.width], [0, 0], color=line_color, linewidth=lw)
        ax.plot([0, self.width], [self.height, self.height], color=line_color, linewidth=lw)
        ax.plot([0, 0], [0, self.height], color=line_color, linewidth=lw)
        ax.plot([self.width, self.width], [0, self.height], color=line_color, linewidth=lw)

        # Center line
        ax.plot([self.width/2, self.width/2], [0, self.height], color=line_color, linewidth=lw)

        # Center circle
        center_circle = plt.Circle((self.width/2, self.height/2), 9.15, fill=False, color=line_color, linewidth=lw)
        ax.add_patch(center_circle)
        ax.plot(self.width/2, self.height/2, "wo", markersize=8)

        # Penalty areas
        # Left
        ax.plot([0, 16.5], [self.height/2 - 20.16, self.height/2 - 20.16], color=line_color, linewidth=lw)
        ax.plot([0, 16.5], [self.height/2 + 20.16, self.height/2 + 20.16], color=line_color, linewidth=lw)
        ax.plot([16.5, 16.5], [self.height/2 - 20.16, self.height/2 + 20.16], color=line_color, linewidth=lw)

        # Right
        ax.plot([self.width - 16.5, self.width], [self.height/2 - 20.16, self.height/2 - 20.16], color=line_color, linewidth=lw)
        ax.plot([self.width - 16.5, self.width], [self.height/2 + 20.16, self.height/2 + 20.16], color=line_color, linewidth=lw)
        ax.plot([self.width - 16.5, self.width - 16.5], [self.height/2 - 20.16, self.height/2 + 20.16], color=line_color, linewidth=lw)

        # Goal areas
        # Left
        ax.plot([0, 5.5], [self.height/2 - 9.16, self.height/2 - 9.16], color=line_color, linewidth=lw)
        ax.plot([0, 5.5], [self.height/2 + 9.16, self.height/2 + 9.16], color=line_color, linewidth=lw)
        ax.plot([5.5, 5.5], [self.height/2 - 9.16, self.height/2 + 9.16], color=line_color, linewidth=lw)

        # Right
        ax.plot([self.width - 5.5, self.width], [self.height/2 - 9.16, self.height/2 - 9.16], color=line_color, linewidth=lw)
        ax.plot([self.width - 5.5, self.width], [self.height/2 + 9.16, self.height/2 + 9.16], color=line_color, linewidth=lw)
        ax.plot([self.width - 5.5, self.width - 5.5], [self.height/2 - 9.16, self.height/2 + 9.16], color=line_color, linewidth=lw)

        # Goals
        ax.plot([-2, 0], [self.height/2 - 3.66, self.height/2 - 3.66], color=line_color, linewidth=lw)
        ax.plot([-2, 0], [self.height/2 + 3.66, self.height/2 + 3.66], color=line_color, linewidth=lw)
        ax.plot([-2, -2], [self.height/2 - 3.66, self.height/2 + 3.66], color=line_color, linewidth=lw)

        ax.plot([self.width, self.width + 2], [self.height/2 - 3.66, self.height/2 - 3.66], color=line_color, linewidth=lw)
        ax.plot([self.width, self.width + 2], [self.height/2 + 3.66, self.height/2 + 3.66], color=line_color, linewidth=lw)
        ax.plot([self.width + 2, self.width + 2], [self.height/2 - 3.66, self.height/2 + 3.66], color=line_color, linewidth=lw)

        # Corner arcs
        corner_radius = 1
        for x, y in [(0, 0), (0, self.height), (self.width, 0), (self.width, self.height)]:
            if x == 0 and y == 0:
                arc = patches.Arc((x, y), corner_radius*2, corner_radius*2, angle=0, theta1=0, theta2=90, color=line_color, linewidth=lw)
            elif x == 0 and y == self.height:
                arc = patches.Arc((x, y), corner_radius*2, corner_radius*2, angle=90, theta1=0, theta2=90, color=line_color, linewidth=lw)
            elif x == self.width and y == 0:
                arc = patches.Arc((x, y), corner_radius*2, corner_radius*2, angle=270, theta1=0, theta2=90, color=line_color, linewidth=lw)
            else:
                arc = patches.Arc((x, y), corner_radius*2, corner_radius*2, angle=180, theta1=0, theta2=90, color=line_color, linewidth=lw)
            ax.add_patch(arc)

        ax.set_xlim(-5, self.width + 5)
        ax.set_ylim(-5, self.height + 5)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig, ax

    def draw_players(
        self,
        ax: plt.Axes,
        players: Dict[int, Dict],
        ball_position: Optional[Tuple[float, float]] = None,
        show_numbers: bool = True,
        show_speed: bool = False
    ) -> None:
        """Draw players on tactical board.

        Args:
            ax: Matplotlib axes.
            players: {track_id: {position: (x, y), team: 1/2, has_ball: bool, speed: float}}.
            ball_position: Ball position.
            show_numbers: Show player numbers.
            show_speed: Show speed indicators.
        """
        team_colors = {1: "#0066cc", 2: "#cc0000"}

        for track_id, player in players.items():
            pos = player.get("position", (0, 0))
            team = player.get("team", 1)
            has_ball = player.get("has_ball", False)
            speed = player.get("speed", 0)

            color = team_colors.get(team, "gray")
            size = 400 if has_ball else 250
            edge_color = "yellow" if has_ball else "white"
            edge_width = 3 if has_ball else 1.5

            ax.scatter(pos[0], pos[1], s=size, c=color, edgecolors=edge_color,
                      linewidth=edge_width, zorder=5, alpha=0.9)

            if show_numbers:
                ax.text(pos[0], pos[1], str(track_id), ha="center", va="center",
                       color="white", fontsize=8, fontweight="bold", zorder=6)

            if show_speed and speed > 0:
                ax.annotate(f"{speed:.1f}", xy=(pos[0], pos[1]),
                           xytext=(pos[0] + 3, pos[1] + 3),
                           fontsize=7, color="yellow", fontweight="bold")

        # Draw ball
        if ball_position:
            ax.scatter(ball_position[0], ball_position[1], s=200, c="white",
                      edgecolors="black", linewidth=2, zorder=7, marker="o")

    def draw_passing_options(
        self,
        ax: plt.Axes,
        player_position: Tuple[float, float],
        teammates: List[Tuple[float, float]],
        opponents: List[Tuple[float, float]],
        max_distance: float = 30.0
    ) -> None:
        """Draw passing options for a player.

        Args:
            ax: Matplotlib axes.
            player_position: Ball carrier position.
            teammates: Teammate positions.
            opponents: Opponent positions.
            max_distance: Maximum pass distance.
        """
        for teammate in teammates:
            distance = np.sqrt((player_position[0] - teammate[0])**2 +
                             (player_position[1] - teammate[1])**2)

            if distance > max_distance:
                continue

            # Check if pass is blocked
            blocked = False
            for opponent in opponents:
                if self._is_pass_blocked(player_position, teammate, opponent):
                    blocked = True
                    break

            color = "lime" if not blocked else "red"
            alpha = 0.8 if not blocked else 0.3
            style = "-" if not blocked else "--"

            ax.plot([player_position[0], teammate[0]],
                   [player_position[1], teammate[1]],
                   color=color, linestyle=style, alpha=alpha, linewidth=2, zorder=3)

            if not blocked:
                circle = Circle(teammate, 1.5, fill=False, color="lime", linewidth=2, zorder=4)
                ax.add_patch(circle)

    def draw_heat_zone(
        self,
        ax: plt.Axes,
        positions: List[Tuple[float, float]],
        team: int = 1,
        alpha: float = 0.3
    ) -> None:
        """Draw heat zone for team/player.

        Args:
            ax: Matplotlib axes.
            positions: List of positions.
            team: Team ID (affects color).
            alpha: Transparency.
        """
        color = "blue" if team == 1 else "red"

        if len(positions) < 3:
            return

        # Create heat map using scatter density
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]

        ax.hexbin(x, y, gridsize=15, cmap="Blues" if team == 1 else "Reds",
                 alpha=alpha, mincnt=1)

    def draw_movement_trail(
        self,
        ax: plt.Axes,
        positions: List[Tuple[float, float]],
        color: str = "yellow",
        alpha: float = 0.5
    ) -> None:
        """Draw movement trail for a player.

        Args:
            ax: Matplotlib axes.
            positions: List of positions over time.
            color: Trail color.
            alpha: Transparency.
        """
        if len(positions) < 2:
            return

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Gradient alpha based on recency
        for i in range(len(positions) - 1):
            a = alpha * (i / len(positions))
            ax.plot(xs[i:i+2], ys[i:i+2], color=color, alpha=a, linewidth=2, zorder=2)

    def _is_pass_blocked(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        opponent: Tuple[float, float],
        radius: float = 1.5
    ) -> bool:
        """Check if pass is blocked by opponent.

        Args:
            start: Pass start.
            end: Pass end.
            opponent: Opponent position.
            radius: Block radius.

        Returns:
            True if blocked.
        """
        line_vec = np.array([end[0] - start[0], end[1] - start[1]])
        point_vec = np.array([opponent[0] - start[0], opponent[1] - start[1]])

        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return False

        projection = np.dot(point_vec, line_vec) / line_len
        if projection < 0 or projection > line_len:
            return False

        closest_point = np.array(start) + (projection / line_len) * line_vec
        distance = np.linalg.norm(np.array(opponent) - closest_point)

        return distance < radius

    def save_frame(
        self,
        fig: plt.Figure,
        output_path: str,
        dpi: int = 150
    ) -> None:
        """Save tactical board frame.

        Args:
            fig: Matplotlib figure.
            output_path: Output path.
            dpi: DPI.
        """
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
        plt.close(fig)

    def create_formation_diagram(
        self,
        formation: str,
        team: int = 1,
        figsize: Tuple[int, int] = (10, 12)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create formation diagram.

        Args:
            formation: Formation string (e.g., "4-3-3").
            team: Team ID.
            figsize: Figure size.

        Returns:
            (fig, ax).
        """
        fig, ax = self.draw_pitch(figsize=figsize)

        # Parse formation
        parts = formation.split("-")
        if len(parts) != 3:
            return fig, ax

        defenders = int(parts[0])
        midfielders = int(parts[1])
        attackers = int(parts[2])

        color = "#0066cc" if team == 1 else "#cc0000"

        # Goalkeeper
        ax.scatter(5, self.height/2, s=400, c=color, edgecolors="white",
                  linewidth=2, zorder=5)
        ax.text(5, self.height/2, "GK", ha="center", va="center",
               color="white", fontsize=10, fontweight="bold")

        # Defenders
        y_positions = np.linspace(10, self.height - 10, defenders)
        for i, y in enumerate(y_positions):
            ax.scatter(20, y, s=350, c=color, edgecolors="white",
                      linewidth=2, zorder=5)
            ax.text(20, y, f"D{i+1}", ha="center", va="center",
                   color="white", fontsize=9, fontweight="bold")

        # Midfielders
        y_positions = np.linspace(10, self.height - 10, midfielders)
        for i, y in enumerate(y_positions):
            ax.scatter(45, y, s=350, c=color, edgecolors="white",
                      linewidth=2, zorder=5)
            ax.text(45, y, f"M{i+1}", ha="center", va="center",
                   color="white", fontsize=9, fontweight="bold")

        # Attackers
        y_positions = np.linspace(15, self.height - 15, attackers)
        for i, y in enumerate(y_positions):
            ax.scatter(75, y, s=350, c=color, edgecolors="white",
                      linewidth=2, zorder=5)
            ax.text(75, y, f"A{i+1}", ha="center", va="center",
                   color="white", fontsize=9, fontweight="bold")

        ax.set_title(f"Formation: {formation} (Team {team})",
                    fontsize=16, fontweight="bold", color="white", pad=20)

        return fig, ax
