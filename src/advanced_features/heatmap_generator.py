"""
Heatmap generation for player and team analysis.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class HeatmapGenerator:
    """Generate heatmaps for player positions and activity."""

    def __init__(self, pitch_width: float = 105.0, pitch_height: float = 68.0):
        """Initialize heatmap generator.

        Args:
            pitch_width: Pitch width in meters
            pitch_height: Pitch height in meters
        """
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height

    def generate_player_heatmap(
        self,
        positions: List[Tuple[float, float]],
        player_id: int,
        team: int,
        grid_size: Tuple[int, int] = (50, 32)
    ) -> np.ndarray:
        """Generate heatmap for a single player.

        Args:
            positions: List of player positions
            player_id: Player ID
            team: Team ID
            grid_size: Grid size for heatmap

        Returns:
            Heatmap array
        """
        heatmap = np.zeros(grid_size)
        
        if not positions:
            return heatmap
        
        # Count positions in each grid cell
        for x, y in positions:
            grid_x = int(x / self.pitch_width * grid_size[0])
            grid_y = int(y / self.pitch_height * grid_size[1])
            
            if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                heatmap[grid_y, grid_x] += 1
        
        # Smooth heatmap
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=1.5)
        
        return heatmap

    def generate_team_heatmap(
        self,
        tracks: Dict,
        team: int,
        grid_size: Tuple[int, int] = (50, 32)
    ) -> np.ndarray:
        """Generate combined heatmap for a team.

        Args:
            tracks: Tracking data
            team: Team ID
            grid_size: Grid size

        Returns:
            Heatmap array
        """
        heatmap = np.zeros(grid_size)
        
        # Collect all positions for team
        for frame_players in tracks.get("players", []):
            for player_id, track in frame_players.items():
                if track.get("team") == team:
                    pos = track.get("position_transformed") or track.get("position")
                    if pos:
                        grid_x = int(pos[0] / self.pitch_width * grid_size[0])
                        grid_y = int(pos[1] / self.pitch_height * grid_size[1])
                        
                        if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                            heatmap[grid_y, grid_x] += 1
        
        # Smooth
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=2)
        
        return heatmap

    def plot_heatmap(
        self,
        heatmap: np.ndarray,
        title: str = "Heatmap",
        cmap: str = "hot",
        alpha: float = 0.6
    ) -> plt.Figure:
        """Plot heatmap on pitch.

        Args:
            heatmap: Heatmap array
            title: Plot title
            cmap: Colormap
            alpha: Transparency

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw pitch
        pitch = Rectangle((0, 0), self.pitch_width, self.pitch_height,
                         linewidth=0, edgecolor="none", facecolor="#1a5f2a")
        ax.add_patch(pitch)
        
        # Plot heatmap
        extent = [0, self.pitch_width, 0, self.pitch_height]
        ax.imshow(heatmap, extent=extent, origin="lower", cmap=cmap, alpha=alpha)
        
        # Pitch lines
        ax.plot([0, self.pitch_width], [0, 0], "w-", linewidth=2)
        ax.plot([0, self.pitch_width], [self.pitch_height, self.pitch_height], "w-", linewidth=2)
        ax.plot([0, 0], [0, self.pitch_height], "w-", linewidth=2)
        ax.plot([self.pitch_width, self.pitch_width], [0, self.pitch_height], "w-", linewidth=2)
        ax.plot([self.pitch_width/2, self.pitch_width/2], [0, self.pitch_height], "w-", linewidth=2)
        
        # Center circle
        center_circle = plt.Circle((self.pitch_width/2, self.pitch_height/2), 9.15,
                                   fill=False, color="white", linewidth=2)
        ax.add_patch(center_circle)
        
        ax.set_xlim(-5, self.pitch_width + 5)
        ax.set_ylim(-5, self.pitch_height + 5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=16, color="white")
        ax.axis("off")
        
        return fig

    def generate_pressure_heatmap(
        self,
        tracks: Dict,
        team: int,
        grid_size: Tuple[int, int] = (50, 32)
    ) -> np.ndarray:
        """Generate pressure heatmap (where team applies pressure).

        Args:
            tracks: Tracking data
            team: Team ID
            grid_size: Grid size

        Returns:
            Pressure heatmap
        """
        pressure_map = np.zeros(grid_size)
        
        for frame_num, frame_players in enumerate(tracks.get("players", [])):
            # Get ball position
            ball_pos = None
            if frame_num < len(tracks.get("ball", [])):
                ball = tracks["ball"][frame_num]
                if ball and 1 in ball:
                    ball_pos = ball[1].get("position_transformed") or ball[1].get("position")
            
            if not ball_pos:
                continue
            
            # Check which team has ball
            ball_team = 0
            for player_id, track in frame_players.items():
                if track.get("has_ball", False):
                    ball_team = track.get("team", 0)
                    break
            
            # If opposing team has ball, calculate pressure
            if ball_team != team and ball_team != 0:
                for player_id, track in frame_players.items():
                    if track.get("team") == team:
                        pos = track.get("position_transformed") or track.get("position")
                        if pos:
                            distance_to_ball = np.sqrt((pos[0] - ball_pos[0])**2 + (pos[1] - ball_pos[1])**2)
                            
                            if distance_to_ball < 10:  # Within pressing distance
                                grid_x = int(pos[0] / self.pitch_width * grid_size[0])
                                grid_y = int(pos[1] / self.pitch_height * grid_size[1])
                                
                                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                                    pressure = 1 / (1 + distance_to_ball)
                                    pressure_map[grid_y, grid_x] += pressure
        
        # Smooth
        from scipy.ndimage import gaussian_filter
        pressure_map = gaussian_filter(pressure_map, sigma=2)
        
        return pressure_map
