"""RunPod Serverless API client."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

import requests

from src.api.client import AnalysisClient
from src.api.object_storage_client import ObjectStorageClient


class RunPodServerlessClient(AnalysisClient):
    """Client for queue-based RunPod Serverless endpoints."""

    TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "CANCELED", "TIMED_OUT"}
    FAILED_STATUSES = {"FAILED", "CANCELLED", "CANCELED", "TIMED_OUT"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        storage_client: Optional[ObjectStorageClient] = None,
        base_url: str = "https://api.runpod.ai/v2",
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required")
        if not self.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID is required")

        self.base_url = base_url.rstrip("/")
        self.storage = storage_client or ObjectStorageClient()
        self.session = requests.Session()
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._last_outputs: Dict[str, Dict[str, Any]] = {}

    def _url(self, suffix: str) -> str:
        return f"{self.base_url}/{self.endpoint_id}/{suffix.lstrip('/')}"

    def _emit_progress(
        self,
        progress_callback: Optional[Callable],
        percent: float,
        transferred_bytes: Optional[int] = None,
        total_bytes: Optional[int] = None,
        phase: str = "upload",
    ) -> None:
        """Report progress while keeping older one-argument callbacks compatible."""
        if not progress_callback:
            return
        try:
            progress_callback(percent, transferred_bytes, total_bytes, phase)
        except TypeError:
            progress_callback(percent)

    def _normalize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize worker output into the app's result shape."""
        if "output" in output and isinstance(output["output"], dict):
            output = output["output"]
        if output.get("error"):
            raise ValueError(str(output["error"]))
        if str(output.get("status", "")).lower() == "failed":
            raise ValueError(str(output.get("message") or "RunPod worker reported failed status"))
        return output

    def _failure_message(self, data: Dict[str, Any], fallback: str) -> str:
        """Extract the most useful error message from a RunPod status payload."""
        output = data.get("output")
        if isinstance(output, dict):
            for key in ("error", "message", "detail"):
                value = output.get(key)
                if value:
                    return str(value)
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if value:
                return str(value)
        return fallback

    def start_from_storage(
        self,
        *,
        video_object_key: str,
        model_object_key: Optional[str] = None,
        model_path: Optional[str] = None,
        analysis_fps: float = 3.0,
        output_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        resize_width: int = 1280,
    ) -> str:
        """Submit a serverless job using existing object storage keys."""
        payload: Dict[str, Any] = {
            "input": {
                "video_object_key": video_object_key,
                "analysis_fps": float(analysis_fps),
                "output_fps": float(output_fps) if output_fps is not None else None,
                "resize_width": int(resize_width),
                "max_frames": int(max_frames) if max_frames is not None else None,
            }
        }
        if model_object_key:
            payload["input"]["model_object_key"] = model_object_key
        if model_path:
            payload["input"]["model_path"] = model_path

        response = self.session.post(self._url("run"), headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        job_id = data.get("id")
        if not job_id:
            raise ValueError(f"RunPod did not return a job id: {data}")
        return str(job_id)

    def upload_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None,
        model_path: Optional[str] = None,
        model_file_path: Optional[str] = None,
        analysis_fps: float = 3.0,
        output_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        resize_width: int = 1280,
    ) -> str:
        """Upload input artifacts to object storage and submit a RunPod job."""
        video_total_bytes = os.path.getsize(video_path)
        self._emit_progress(progress_callback, 5, 0, video_total_bytes, "video_upload")

        def video_progress(upload_percent: float, transferred: int, total: int) -> None:
            self._emit_progress(
                progress_callback,
                5 + (float(upload_percent) * 0.65),
                transferred,
                total,
                "video_upload",
            )

        video_info = self.storage.upload_file(
            video_path,
            file_kind="video",
            content_type="video/mp4",
            progress_callback=video_progress,
        )
        self._emit_progress(progress_callback, 70, phase="video_uploaded")

        model_object_key = None
        if model_file_path:
            def model_progress(upload_percent: float, transferred: int, total: int) -> None:
                self._emit_progress(
                    progress_callback,
                    70 + (float(upload_percent) * 0.15),
                    transferred,
                    total,
                    "model_upload",
                )

            model_info = self.storage.upload_file(
                model_file_path,
                file_kind="model",
                content_type="application/octet-stream",
                progress_callback=model_progress,
            )
            model_object_key = model_info["object_key"]
            self._emit_progress(progress_callback, 85, phase="model_uploaded")

        self._emit_progress(progress_callback, 90, phase="submitting")
        job_id = self.start_from_storage(
            video_object_key=str(video_info["object_key"]),
            model_object_key=model_object_key,
            model_path=model_path,
            analysis_fps=analysis_fps,
            output_fps=output_fps,
            max_frames=max_frames,
            resize_width=resize_width,
        )
        self._emit_progress(progress_callback, 95, phase="submitted")
        return job_id

    def get_status(self, job_id: str) -> Dict:
        response = self.session.get(self._url(f"status/{job_id}"), headers=self.headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        raw_status = str(data.get("status", "UNKNOWN")).upper()

        progress_map = {
            "IN_QUEUE": 10,
            "IN_PROGRESS": 50,
            "COMPLETED": 100,
            "FAILED": 0,
            "CANCELLED": 0,
            "CANCELED": 0,
            "TIMED_OUT": 0,
        }
        if raw_status == "COMPLETED":
            try:
                self._last_outputs[job_id] = self._normalize_output(data)
            except ValueError as exc:
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "raw_status": raw_status,
                    "progress": 0,
                    "done": True,
                    "message": str(exc),
                    "raw": data,
                }

        message = self._failure_message(data, raw_status) if raw_status in self.FAILED_STATUSES else raw_status

        return {
            "job_id": job_id,
            "status": "completed" if raw_status == "COMPLETED" else "failed" if raw_status in self.FAILED_STATUSES else "processing",
            "raw_status": raw_status,
            "progress": progress_map.get(raw_status, 25),
            "done": raw_status in self.TERMINAL_STATUSES,
            "message": message,
            "raw": data,
        }

    def get_results(self, job_id: str) -> Dict:
        if job_id in self._last_outputs:
            return self._last_outputs[job_id]

        response = self.session.get(self._url(f"status/{job_id}"), headers=self.headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        raw_status = str(data.get("status", "UNKNOWN")).upper()
        if raw_status != "COMPLETED":
            raise ValueError(f"RunPod job is not complete: {raw_status}")

        output = self._normalize_output(data)
        self._last_outputs[job_id] = output
        return output

    def wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable] = None,
        check_interval: int = 10,
    ) -> Dict:
        while True:
            status = self.get_status(job_id)
            if progress_callback:
                progress_callback(status.get("progress", 0))
            if status.get("done"):
                if status.get("status") == "failed":
                    raise ValueError(status.get("message", "RunPod job failed"))
                return self.get_results(job_id)
            time.sleep(check_interval)
