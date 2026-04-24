"""
Custom GPU Server Client for football video analysis.
"""

from __future__ import annotations

from typing import Dict, Optional, Callable

import requests

from src.api.client import AnalysisClient


class ServerClient(AnalysisClient):
    """Client for custom GPU server."""

    def __init__(self, server_url: str, api_key: Optional[str] = None):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def upload_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None,
        model_path: Optional[str] = None,
        model_file_path: Optional[str] = None,
        analysis_fps: float = 1.0,
        max_frames: int = 5400,
        resize_width: int = 960,
    ) -> str:
        """Upload video to server and start job."""
        data = {
            "analysis_fps": str(analysis_fps),
            "max_frames": str(max_frames),
            "resize_width": str(resize_width),
        }
        if model_path:
            data["model_path"] = model_path

        with open(video_path, "rb") as vf:
            files = {"file": vf}

            model_file = None
            if model_file_path:
                model_file = open(model_file_path, "rb")
                files["model_file"] = model_file

            try:
                response = requests.post(
                    f"{self.server_url}/upload",
                    files=files,
                    data=data,
                    headers=self.headers,
                    timeout=300,
                )
                response.raise_for_status()
                return response.json()["job_id"]
            finally:
                if model_file is not None:
                    model_file.close()

    def get_status(self, job_id: str) -> Dict:
        response = requests.get(
            f"{self.server_url}/status/{job_id}",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        payload["done"] = payload.get("status") in {"completed", "failed"}
        return payload

    def get_results(self, job_id: str) -> Dict:
        response = requests.get(
            f"{self.server_url}/results/{job_id}",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_video_url(self, job_id: str) -> str:
        return f"{self.server_url}/video/{job_id}"
