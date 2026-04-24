"""
Team assignment based on jersey colors using KMeans clustering.
Based on: football_analysis_yolo by TrishamBP
"""

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """Assign players to teams based on jersey color clustering."""

    def __init__(self):
        """Initialize team assigner."""
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

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
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Use top half (jersey area)
        top_half_image = image[0:int(image.shape[0] / 2), :]

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

    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        """Get team for specific player (with caching).

        Args:
            frame: Current frame.
            player_bbox: Player bounding box.
            player_id: Player track ID.

        Returns:
            Team ID (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = int(team_id)

        return int(team_id)
