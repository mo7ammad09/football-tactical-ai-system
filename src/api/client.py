"""
API Client abstraction layer.
Supports both Replicate API and custom GPU server.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable
from pathlib import Path

import requests


class AnalysisClient(ABC):
    """Abstract client for football video analysis."""

    @abstractmethod
    def upload_video(self, video_path: str, progress_callback: Optional[Callable] = None) -> str:
        """Upload video and return job ID.

        Args:
            video_path: Path to video file.
            progress_callback: Optional callback for upload progress.

        Returns:
            Job ID for tracking.
        """
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> Dict:
        """Get processing status.

        Args:
            job_id: Job ID from upload.

        Returns:
            Dict with 'progress' (0-100), 'done' (bool), 'status' (str).
        """
        pass

    @abstractmethod
    def get_results(self, job_id: str) -> Dict:
        """Get final results.

        Args:
            job_id: Job ID.

        Returns:
            Dict with 'annotated_video_url', 'stats', 'tactical_analysis'.
        """
        pass

    def wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable] = None,
        check_interval: int = 5
    ) -> Dict:
        """Wait for analysis to complete.

        Args:
            job_id: Job ID.
            progress_callback: Callback(progress_percent).
            check_interval: Seconds between status checks.

        Returns:
            Final results.
        """
        while True:
            status = self.get_status(job_id)

            if progress_callback:
                progress_callback(status.get("progress", 0))

            if status.get("done", False):
                return self.get_results(job_id)

            time.sleep(check_interval)
