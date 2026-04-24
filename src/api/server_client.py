"""
Custom GPU Server Client for football video analysis.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from typing import Dict, Optional, Callable

import requests

from src.api.client import AnalysisClient


class ServerClient(AnalysisClient):
    """Client for custom GPU server."""

    CHUNK_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB (more resilient to proxy resets)
    CHUNK_UPLOAD_THRESHOLD_BYTES = 80 * 1024 * 1024  # 80 MB
    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
    CHUNK_MAX_RETRIES = 8
    REQUEST_MAX_RETRIES = 5
    BACKOFF_BASE_SECONDS = 1.5

    def __init__(self, server_url: str, api_key: Optional[str] = None):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        max_retries: int,
        retry_label: str,
        raise_for_status: bool = True,
        **kwargs,
    ) -> requests.Response:
        """Send HTTP request with retry for transient network/proxy failures."""
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = requests.request(method=method, url=url, **kwargs)
                if response.status_code in self.RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    wait_s = self.BACKOFF_BASE_SECONDS * (2 ** attempt)
                    time.sleep(wait_s)
                    continue
                if raise_for_status:
                    response.raise_for_status()
                return response
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt >= max_retries - 1:
                    break
                wait_s = self.BACKOFF_BASE_SECONDS * (2 ** attempt)
                time.sleep(wait_s)

        if last_exc is not None:
            raise requests.ConnectionError(f"{retry_label} failed after {max_retries} attempts: {last_exc}") from last_exc
        raise RuntimeError(f"{retry_label} failed after {max_retries} attempts")

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
                response = self._request_with_retry(
                    "POST",
                    f"{self.server_url}/upload",
                    max_retries=self.REQUEST_MAX_RETRIES,
                    retry_label="Direct upload",
                    raise_for_status=False,
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
                self._request_with_retry(
                    "POST",
                    f"{self.server_url}/upload_chunk",
                    max_retries=self.CHUNK_MAX_RETRIES,
                    retry_label=f"{file_kind} chunk {idx + 1}/{total_chunks}",
                    files=files,
                    data=data,
                    headers=self.headers,
                    timeout=300,
                )

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

        resp = self._request_with_retry(
            "POST",
            f"{self.server_url}/upload_finalize",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label="Finalize upload",
            data=data,
            headers=self.headers,
            timeout=600,
        )

        if progress_callback is not None:
            progress_callback(95)
        return resp.json()["job_id"]

    def get_status(self, job_id: str) -> Dict:
        response = self._request_with_retry(
            "GET",
            f"{self.server_url}/status/{job_id}",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label=f"Status for job {job_id}",
            headers=self.headers,
            timeout=30,
        )
        payload = response.json()
        payload["done"] = payload.get("status") in {"completed", "failed"}
        return payload

    def get_results(self, job_id: str) -> Dict:
        response = self._request_with_retry(
            "GET",
            f"{self.server_url}/results/{job_id}",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label=f"Results for job {job_id}",
            headers=self.headers,
            timeout=30,
        )
        return response.json()

    def get_video_url(self, job_id: str) -> str:
        return f"{self.server_url}/video/{job_id}"
