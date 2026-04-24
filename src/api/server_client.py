"""
Custom GPU Server Client for football video analysis.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import Dict, Optional, Callable

import requests

from src.api.client import AnalysisClient


class ServerClient(AnalysisClient):
    """Client for custom GPU server."""

    CHUNK_SIZE_BYTES = 24 * 1024 * 1024  # 24 MB
    CHUNK_UPLOAD_THRESHOLD_BYTES = 80 * 1024 * 1024  # 80 MB

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
        video_size = os.path.getsize(video_path)
        if video_size >= self.CHUNK_UPLOAD_THRESHOLD_BYTES:
            return self._upload_video_chunked(
                video_path=video_path,
                model_path=model_path,
                model_file_path=model_file_path,
                analysis_fps=analysis_fps,
                max_frames=max_frames,
                resize_width=resize_width,
                progress_callback=progress_callback,
            )

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
                if response.status_code == 413:
                    return self._upload_video_chunked(
                        video_path=video_path,
                        model_path=model_path,
                        model_file_path=model_file_path,
                        analysis_fps=analysis_fps,
                        max_frames=max_frames,
                        resize_width=resize_width,
                        progress_callback=progress_callback,
                    )
                response.raise_for_status()
                return response.json()["job_id"]
            finally:
                if model_file is not None:
                    model_file.close()

    def _upload_file_chunks(
        self,
        *,
        upload_id: str,
        file_path: str,
        file_kind: str,
        progress_callback: Optional[Callable],
        progress_start: int,
        progress_end: int,
    ) -> int:
        """Upload a file in chunks and return total_chunks."""
        file_size = os.path.getsize(file_path)
        total_chunks = max(1, math.ceil(file_size / self.CHUNK_SIZE_BYTES))

        with open(file_path, "rb") as f:
            for idx in range(total_chunks):
                chunk_data = f.read(self.CHUNK_SIZE_BYTES)
                files = {"chunk": ("chunk.bin", chunk_data, "application/octet-stream")}
                data = {
                    "upload_id": upload_id,
                    "file_kind": file_kind,
                    "chunk_index": str(idx),
                    "total_chunks": str(total_chunks),
                }
                resp = requests.post(
                    f"{self.server_url}/upload_chunk",
                    files=files,
                    data=data,
                    headers=self.headers,
                    timeout=300,
                )
                resp.raise_for_status()

                if progress_callback is not None:
                    ratio = (idx + 1) / total_chunks
                    progress_value = int(progress_start + (progress_end - progress_start) * ratio)
                    progress_callback(progress_value)

        return total_chunks

    def _upload_video_chunked(
        self,
        *,
        video_path: str,
        model_path: Optional[str],
        model_file_path: Optional[str],
        analysis_fps: float,
        max_frames: int,
        resize_width: int,
        progress_callback: Optional[Callable],
    ) -> str:
        """Chunked upload flow for large payloads."""
        upload_id = str(uuid.uuid4())

        video_total_chunks = self._upload_file_chunks(
            upload_id=upload_id,
            file_path=video_path,
            file_kind="video",
            progress_callback=progress_callback,
            progress_start=5,
            progress_end=70,
        )

        model_total_chunks = 0
        model_file_name = None
        if model_file_path:
            model_file_name = os.path.basename(model_file_path)
            model_total_chunks = self._upload_file_chunks(
                upload_id=upload_id,
                file_path=model_file_path,
                file_kind="model",
                progress_callback=progress_callback,
                progress_start=70,
                progress_end=90,
            )

        data = {
            "upload_id": upload_id,
            "video_file_name": os.path.basename(video_path),
            "video_total_chunks": str(video_total_chunks),
            "analysis_fps": str(analysis_fps),
            "max_frames": str(max_frames),
            "resize_width": str(resize_width),
            "model_total_chunks": str(model_total_chunks),
        }
        if model_path:
            data["model_path"] = model_path
        if model_file_name:
            data["model_file_name"] = model_file_name

        resp = requests.post(
            f"{self.server_url}/upload_finalize",
            data=data,
            headers=self.headers,
            timeout=600,
        )
        resp.raise_for_status()

        if progress_callback is not None:
            progress_callback(95)
        return resp.json()["job_id"]

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
