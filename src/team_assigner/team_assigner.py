"""
Team assignment based on jersey colors using KMeans clustering.
Based on: football_analysis_yolo by TrishamBP
"""

from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """Assign players to teams based on jersey color clustering."""

    def __init__(self):
        """Initialize team assigner."""
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_votes: Dict[int, List[int]] = defaultdict(list)
        self.kmeans = None
        self.team_vote_window = 12
        self.ambiguous_color_margin = 0.18

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """Get KMeans model for image color clustering.

        Args:
            image: Image array (H, W, 3).

        Returns:
            Fitted KMeans model.
        """
        # Reshape image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Extract dominant jersey color from player bounding box.

        Args:
            frame: Full video frame.
            bbox: [x1, y1, x2, y2].

        Returns:
            Dominant jersey color (BGR).
        """
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(bbox[0])))
        y1 = max(0, min(height - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(width, int(bbox[2])))
        y2 = max(y1 + 1, min(height, int(bbox[3])))
        image = frame[y1:y2, x1:x2]

        # Use top half (jersey area)
        top_half_image = image[0:int(image.shape[0] / 2), :]
        if top_half_image.size == 0 or top_half_image.shape[0] < 2 or top_half_image.shape[1] < 2:
            return np.mean(image.reshape(-1, 3), axis=0)

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get player cluster (non-background)
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame: np.ndarray, player_detections: dict) -> None:
        """Assign team colors by clustering all player jersey colors.

        Args:
            frame: Reference frame.
            player_detections: Player detections for frame.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def predict_team_from_color(self, player_color: np.ndarray) -> tuple[int, float]:
        """Predict team and confidence margin from jersey color."""
        centers = np.asarray(self.kmeans.cluster_centers_, dtype=float)
        distances = np.linalg.norm(centers - player_color.reshape(1, -1), axis=1)
        order = np.argsort(distances)
        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else best_idx
        team_id = best_idx + 1
        margin = (
            float(distances[second_idx] - distances[best_idx])
            / max(float(distances[second_idx]), 1e-6)
        )
        return team_id, margin

    def _stable_team_for_player(self, player_id: int, team_id: int) -> int:
        """Smooth team assignment over recent confident observations."""
        votes = self.player_team_votes[int(player_id)]
        votes.append(int(team_id))
        if len(votes) > self.team_vote_window:
            del votes[0: len(votes) - self.team_vote_window]

        stable_team = Counter(votes).most_common(1)[0][0]
        self.player_team_dict[int(player_id)] = int(stable_team)
        return int(stable_team)

    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        """Get smoothed team for a player.

        Args:
            frame: Current frame.
            player_bbox: Player bounding box.
            player_id: Player track ID.

        Returns:
            Team ID (1 or 2).
        """
        player_color = self.get_player_color(frame, player_bbox)
        team_id, margin = self.predict_team_from_color(player_color)

        if margin < self.ambiguous_color_margin:
            return int(self.player_team_dict.get(player_id, 0))

        return self._stable_team_for_player(player_id, team_id)
