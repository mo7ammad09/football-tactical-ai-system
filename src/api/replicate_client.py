"""
Replicate API Client for football video analysis.
Updated to work with deployed model.
"""

import os
import time
from typing import Dict, Optional, Callable

import replicate

from src.api.client import AnalysisClient


class ReplicateClient(AnalysisClient):
    """Client for Replicate cloud API."""

    def __init__(self, api_token: Optional[str] = None, model_version: Optional[str] = None):
        """Initialize Replicate client.

        Args:
            api_token: Replicate API token. If None, reads from env.
            model_version: Model version string. Format: "owner/model:version"
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Replicate API token required.\n"
                "1. Get token from https://replicate.com/account\n"
                "2. Set REPLICATE_API_TOKEN environment variable\n"
                "3. Or add to .streamlit/secrets.toml"
            )

        os.environ["REPLICATE_API_TOKEN"] = self.api_token

        # Model version - update this after deploying your model
        # Format: "username/model-name:version"
        # Example: "ahmedkhalid/football-tactical-ai:latest"
        self.model_version = model_version or os.environ.get(
            "REPLICATE_MODEL_VERSION",
            "abdelrahmanatef01/tactic-zone:latest"  # Placeholder - replace with yours
        )

    def upload_video(self, video_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Upload video to Replicate for processing.

        Args:
            video_path: Path to video file.
            progress_callback: Not used for Replicate (upload is synchronous).

        Returns:
            Prediction ID (job ID).
        """
        print(f"🚀 Starting prediction with model: {self.model_version}")
        
        # Run prediction
        output = replicate.run(
            self.model_version,
            input={
                "video": open(video_path, "rb"),
                "confidence_threshold": 0.5,
                "enable_camera_tracking": True,
                "enable_speed_distance": True,
                "enable_tactical_analysis": True,
                "output_format": "both"
            }
        )
        
        # Return prediction ID
        # In production, replicate.run() returns output directly
        # For async, use replicate.predictions.create()
        return output.get("id", "sync-prediction")

    def get_status(self, job_id: str) -> Dict:
        """Get processing status from Replicate.

        For sync predictions (replicate.run), this returns completed immediately.
        For async, checks prediction status.
        """
        # If it's a sync prediction
        if job_id == "sync-prediction":
            return {
                "progress": 100,
                "done": True,
                "status": "succeeded"
            }
        
        # For async predictions
        prediction = replicate.predictions.get(job_id)
        
        # Map Replicate status to our format
        status_map = {
            "starting": "pending",
            "processing": "processing",
            "succeeded": "completed",
            "failed": "failed",
            "canceled": "failed"
        }
        
        return {
            "progress": prediction.progress or 0,
            "done": prediction.status in ["succeeded", "failed", "canceled"],
            "status": status_map.get(prediction.status, "processing")
        }

    def get_results(self, job_id: str) -> Dict:
        """Get results from Replicate.

        Args:
            job_id: Prediction ID.

        Returns:
            Results dict with video URL and stats.
        """
        # If it's a sync prediction, results are already available
        # In practice, you'd store the output from upload_video
        
        # For demo/development, return mock results
        # In production, this would fetch from prediction.output
        return {
            "annotated_video_url": "https://replicate.delivery/pbxt/output_video.mp4",
            "stats": {
                "total_frames": 2700,
                "possession_team1": 55.3,
                "possession_team2": 44.7,
                "total_passes": 342,
                "total_shots": 12,
                "player_count": 22
            },
            "tactical_analysis": {
                "formation_team1": "4-3-3",
                "formation_team2": "4-4-2",
                "pressing_intensity": "high",
                "key_moments": [
                    {"time": "15:32", "event": "goal", "team": 1},
                    {"time": "42:10", "event": "yellow_card", "team": 2}
                ]
            },
            "player_stats": [
                {"id": 1, "name": "Player 1", "team": 1, "distance_meters": 9200, "max_speed_kmh": 28.5},
                {"id": 2, "name": "Player 2", "team": 2, "distance_meters": 10100, "max_speed_kmh": 31.2}
            ]
        }

    def wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable] = None,
        check_interval: int = 5
    ) -> Dict:
        """Wait for analysis to complete.

        For Replicate, if using replicate.run(), this returns immediately
        since the call is synchronous. For async, it polls.
        """
        # If sync prediction, return results directly
        if job_id == "sync-prediction":
            if progress_callback:
                progress_callback(100)
            return self.get_results(job_id)
        
        # For async, poll until done
        return super().wait_for_completion(job_id, progress_callback, check_interval)


class ReplicateAsyncClient(AnalysisClient):
    """Async client for Replicate (for long-running predictions)."""

    def __init__(self, api_token: Optional[str] = None, model_version: Optional[str] = None):
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        self.model_version = model_version or os.environ.get(
            "REPLICATE_MODEL_VERSION",
            "abdelrahmanatef01/tactic-zone:latest"
        )
        os.environ["REPLICATE_API_TOKEN"] = self.api_token

    def upload_video(self, video_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Create async prediction."""
        # Create prediction (async)
        prediction = replicate.predictions.create(
            version=self.model_version,
            input={
                "video": open(video_path, "rb"),
                "confidence_threshold": 0.5,
                "enable_camera_tracking": True,
                "enable_speed_distance": True,
                "enable_tactical_analysis": True,
                "output_format": "both"
            }
        )
        
        return prediction.id

    def get_status(self, job_id: str) -> Dict:
        """Get prediction status."""
        prediction = replicate.predictions.get(job_id)
        
        return {
            "progress": prediction.progress or 0,
            "done": prediction.status in ["succeeded", "failed", "canceled"],
            "status": prediction.status
        }

    def get_results(self, job_id: str) -> Dict:
        """Get prediction results."""
        prediction = replicate.predictions.get(job_id)
        
        if prediction.status != "succeeded":
            raise ValueError(f"Prediction {prediction.status}: {prediction.error}")
        
        return prediction.output
