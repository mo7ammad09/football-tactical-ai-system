"""
Custom GPU Server Client for football video analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from src.api.client import AnalysisClient


class ServerClient(AnalysisClient):
    """Client for custom GPU server."""

    CHUNK_SIZE_BYTES = 2 * 1024 * 1024  # fallback chunk API size
    CHUNK_UPLOAD_THRESHOLD_BYTES = 80 * 1024 * 1024
    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
    CHUNK_MAX_RETRIES = 8
    REQUEST_MAX_RETRIES = 5
    BACKOFF_BASE_SECONDS = 1.5

    STORAGE_PART_SIZE_BYTES = 8 * 1024 * 1024
    STORAGE_DIRECT_UPLOAD_THRESHOLD_BYTES = 32 * 1024 * 1024
    STORAGE_MAX_WORKERS = 4
    STORAGE_HEALTH_TTL_SECONDS = 120
    STORAGE_STATE_FILE = Path("temp/.object_storage_upload_state.json")
    STORAGE_MODEL_CACHE_KEY = "__model_object_cache__"

    def __init__(self, server_url: str, api_key: Optional[str] = None):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.headers: Dict[str, str] = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._storage_enabled_cache: Optional[bool] = None
        self._storage_cache_at = 0.0
        self._session = requests.Session()

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        max_retries: int,
        retry_label: str,
        raise_for_status: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        """Send HTTP request with retry for transient network/proxy failures."""
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self._session.request(method=method, url=url, **kwargs)
                if response.status_code in self.RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    wait_s = self.BACKOFF_BASE_SECONDS * (2**attempt)
                    time.sleep(wait_s)
                    continue
                if raise_for_status:
                    response.raise_for_status()
                return response
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt >= max_retries - 1:
                    break
                wait_s = self.BACKOFF_BASE_SECONDS * (2**attempt)
                time.sleep(wait_s)

        if last_exc is not None:
            raise requests.ConnectionError(
                f"{retry_label} failed after {max_retries} attempts: {last_exc}"
            ) from last_exc
        raise RuntimeError(f"{retry_label} failed after {max_retries} attempts")

    def _is_object_storage_enabled(self) -> bool:
        """Check whether server-side object storage flow is available."""
        now = time.time()
        if (
            self._storage_enabled_cache is not None
            and now - self._storage_cache_at < self.STORAGE_HEALTH_TTL_SECONDS
        ):
            return self._storage_enabled_cache

        enabled = False
        try:
            resp = self._request_with_retry(
                "GET",
                f"{self.server_url}/storage/health",
                max_retries=2,
                retry_label="Storage health",
                raise_for_status=False,
                headers=self.headers,
                timeout=10,
            )
            if resp.ok:
                enabled = bool(resp.json().get("enabled"))
        except Exception:
            enabled = False

        self._storage_enabled_cache = enabled
        self._storage_cache_at = now
        return enabled

    def _load_storage_state(self) -> Dict[str, Any]:
        """Load local resume state for multipart uploads."""
        try:
            if not self.STORAGE_STATE_FILE.exists():
                return {}
            return json.loads(self.STORAGE_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_storage_state(self, state: Dict[str, Any]) -> None:
        """Persist local resume state atomically."""
        self.STORAGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.STORAGE_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")
        tmp.replace(self.STORAGE_STATE_FILE)

    def _file_sha256(self, file_path: str) -> str:
        """Compute SHA256 for a file path."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _get_cached_model_object_key(self, model_file_path: str) -> Optional[str]:
        """Return cached object key for model file, if present."""
        state = self._load_storage_state()
        cache = state.get(self.STORAGE_MODEL_CACHE_KEY, {})
        if not isinstance(cache, dict):
            return None

        fp = self._file_sha256(model_file_path)
        entry = cache.get(fp)
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            value = entry.get("object_key")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _set_cached_model_object_key(self, model_file_path: str, object_key: str) -> None:
        """Save model object key in local cache."""
        state = self._load_storage_state()
        cache = state.get(self.STORAGE_MODEL_CACHE_KEY)
        if not isinstance(cache, dict):
            cache = {}

        fp = self._file_sha256(model_file_path)
        cache[fp] = {
            "object_key": object_key,
            "updated_at": int(time.time()),
        }
        state[self.STORAGE_MODEL_CACHE_KEY] = cache
        self._save_storage_state(state)

    def _storage_object_exists(self, object_key: str) -> bool:
        """Check whether storage object key exists on server."""
        payload = {"object_key": object_key}
        try:
            resp = self._request_with_retry(
                "POST",
                f"{self.server_url}/storage/object_exists",
                max_retries=2,
                retry_label="Storage object_exists",
                raise_for_status=False,
                json=payload,
                headers=self.headers,
                timeout=20,
            )
            if not resp.ok:
                return False
            return bool(resp.json().get("exists"))
        except Exception:
            return False

    def _resume_key(self, file_path: str, file_kind: str) -> str:
        """Stable key used for resumable multipart state."""
        stat = os.stat(file_path)
        raw = (
            f"{self.server_url}|{file_kind}|{os.path.abspath(file_path)}|"
            f"{stat.st_size}|{int(stat.st_mtime)}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _storage_init_upload(self, file_path: str, file_kind: str) -> Dict[str, Any]:
        """Create multipart upload on object storage."""
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        payload = {
            "file_name": os.path.basename(file_path),
            "file_kind": file_kind,
            "content_type": content_type,
            "part_size_mb": max(2, self.STORAGE_PART_SIZE_BYTES // (1024 * 1024)),
        }
        resp = self._request_with_retry(
            "POST",
            f"{self.server_url}/storage/multipart/init",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label=f"Init storage upload ({file_kind})",
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        return resp.json()

    def _storage_get_direct_upload_url(self, file_path: str, file_kind: str) -> Dict[str, Any]:
        """Get pre-signed direct-upload URL for small files."""
        payload = {
            "file_name": os.path.basename(file_path),
            "file_kind": file_kind,
        }
        resp = self._request_with_retry(
            "POST",
            f"{self.server_url}/storage/direct_upload_url",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label=f"Direct upload URL ({file_kind})",
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        return resp.json()

    def _storage_list_parts(self, upload_id: str, object_key: str) -> Optional[Dict[int, str]]:
        """List uploaded parts. Returns None when upload session is not found."""
        payload = {"upload_id": upload_id, "object_key": object_key}
        resp = self._request_with_retry(
            "POST",
            f"{self.server_url}/storage/multipart/list_parts",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label="List storage parts",
            raise_for_status=False,
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        uploaded: Dict[int, str] = {}
        for part in resp.json().get("parts", []):
            uploaded[int(part["part_number"])] = str(part["etag"]).strip('"')
        return uploaded

    def _storage_presign_part(self, upload_id: str, object_key: str, part_number: int) -> str:
        """Request pre-signed URL for one object-storage part."""
        payload = {
            "upload_id": upload_id,
            "object_key": object_key,
            "part_number": part_number,
        }
        resp = self._request_with_retry(
            "POST",
            f"{self.server_url}/storage/multipart/presign_part",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label=f"Presign part {part_number}",
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        return resp.json()["url"]

    def _upload_one_storage_part(
        self,
        *,
        file_path: str,
        upload_id: str,
        object_key: str,
        part_number: int,
        part_size_bytes: int,
        file_size: int,
    ) -> Tuple[int, str]:
        """Upload one object-storage part with retries."""
        offset = (part_number - 1) * part_size_bytes
        length = min(part_size_bytes, file_size - offset)
        if length <= 0:
            raise ValueError(f"Invalid length for part {part_number}")

        with open(file_path, "rb") as f:
            f.seek(offset)
            payload = f.read(length)

        last_exc: Optional[Exception] = None
        for attempt in range(self.CHUNK_MAX_RETRIES):
            try:
                url = self._storage_presign_part(upload_id, object_key, part_number)
                put_resp = requests.put(url, data=payload, timeout=300)
                if put_resp.status_code in self.RETRYABLE_STATUS_CODES and attempt < self.CHUNK_MAX_RETRIES - 1:
                    wait_s = self.BACKOFF_BASE_SECONDS * (2**attempt)
                    time.sleep(wait_s)
                    continue
                put_resp.raise_for_status()
                etag = str(put_resp.headers.get("ETag", "")).strip('"')
                if not etag:
                    etag = hashlib.md5(payload).hexdigest()
                return part_number, etag
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                if attempt >= self.CHUNK_MAX_RETRIES - 1:
                    break
                wait_s = self.BACKOFF_BASE_SECONDS * (2**attempt)
                time.sleep(wait_s)

        raise requests.ConnectionError(
            f"Failed to upload storage part {part_number} after {self.CHUNK_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def _upload_file_to_object_storage(
        self,
        *,
        file_path: str,
        file_kind: str,
        progress_callback: Optional[Callable[[int], None]],
        progress_start: int,
        progress_end: int,
    ) -> str:
        """Multipart upload to object storage with resume and parallel uploads."""
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            raise ValueError(f"File is empty: {file_path}")

        if file_size <= self.STORAGE_DIRECT_UPLOAD_THRESHOLD_BYTES:
            info = self._storage_get_direct_upload_url(file_path=file_path, file_kind=file_kind)
            put_url = info["upload_url"]
            with open(file_path, "rb") as f:
                put_resp = requests.put(put_url, data=f, timeout=300)
            put_resp.raise_for_status()
            if progress_callback is not None:
                progress_callback(progress_end)
            return info["object_key"]

        state = self._load_storage_state()
        key = self._resume_key(file_path, file_kind)
        entry = state.get(key)

        upload_id = None
        object_key = None
        part_size_bytes = self.STORAGE_PART_SIZE_BYTES
        uploaded_parts: Dict[int, str] = {}

        if entry:
            upload_id = entry.get("upload_id")
            object_key = entry.get("object_key")
            part_size_bytes = int(entry.get("part_size_bytes", part_size_bytes))
            listed = self._storage_list_parts(upload_id, object_key) if upload_id and object_key else None
            if listed is None:
                entry = None
            else:
                uploaded_parts = listed
                if not uploaded_parts:
                    saved_parts = entry.get("completed_parts", {})
                    uploaded_parts = {int(k): str(v) for k, v in saved_parts.items()}

        if not entry:
            init_data = self._storage_init_upload(file_path=file_path, file_kind=file_kind)
            upload_id = init_data["upload_id"]
            object_key = init_data["object_key"]
            part_size_bytes = int(init_data.get("part_size_bytes", part_size_bytes))
            entry = {
                "upload_id": upload_id,
                "object_key": object_key,
                "part_size_bytes": part_size_bytes,
                "completed_parts": {},
                "updated_at": int(time.time()),
            }
            state[key] = entry
            self._save_storage_state(state)

        if not upload_id or not object_key:
            raise RuntimeError("Invalid storage upload session state")

        total_parts = max(1, math.ceil(file_size / part_size_bytes))
        pending_parts = [p for p in range(1, total_parts + 1) if p not in uploaded_parts]
        completed_count = len(uploaded_parts)

        if progress_callback is not None:
            ratio = completed_count / total_parts
            progress_callback(int(progress_start + (progress_end - progress_start) * ratio))

        if pending_parts:
            workers = min(self.STORAGE_MAX_WORKERS, len(pending_parts))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(
                        self._upload_one_storage_part,
                        file_path=file_path,
                        upload_id=upload_id,
                        object_key=object_key,
                        part_number=part_number,
                        part_size_bytes=part_size_bytes,
                        file_size=file_size,
                    ): part_number
                    for part_number in pending_parts
                }
                for future in as_completed(futures):
                    part_number, etag = future.result()
                    uploaded_parts[part_number] = etag
                    completed_count += 1

                    entry["completed_parts"] = {str(k): v for k, v in uploaded_parts.items()}
                    entry["updated_at"] = int(time.time())
                    state[key] = entry
                    self._save_storage_state(state)

                    if progress_callback is not None:
                        ratio = completed_count / total_parts
                        progress_callback(int(progress_start + (progress_end - progress_start) * ratio))

        parts_payload = [
            {"part_number": part_number, "etag": uploaded_parts[part_number]}
            for part_number in sorted(uploaded_parts)
        ]
        complete_payload = {
            "upload_id": upload_id,
            "object_key": object_key,
            "parts": parts_payload,
        }
        self._request_with_retry(
            "POST",
            f"{self.server_url}/storage/multipart/complete",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label=f"Complete storage upload ({file_kind})",
            json=complete_payload,
            headers=self.headers,
            timeout=120,
        )

        state.pop(key, None)
        self._save_storage_state(state)
        return object_key

    def _upload_one_chunk_to_api(
        self,
        *,
        file_path: str,
        upload_id: str,
        file_kind: str,
        chunk_index: int,
        total_chunks: int,
        chunk_size: int,
    ) -> int:
        """Upload one fallback API chunk with retries."""
        offset = chunk_index * chunk_size
        length = min(chunk_size, os.path.getsize(file_path) - offset)
        if length <= 0:
            raise ValueError(f"Invalid chunk length for index {chunk_index}")

        with open(file_path, "rb") as f:
            f.seek(offset)
            payload = f.read(length)

        files = {"chunk": ("chunk.bin", payload, "application/octet-stream")}
        data = {
            "upload_id": upload_id,
            "file_kind": file_kind,
            "chunk_index": str(chunk_index),
            "total_chunks": str(total_chunks),
        }
        self._request_with_retry(
            "POST",
            f"{self.server_url}/upload_chunk",
            max_retries=self.CHUNK_MAX_RETRIES,
            retry_label=f"{file_kind} api chunk {chunk_index + 1}/{total_chunks}",
            files=files,
            data=data,
            headers=self.headers,
            timeout=300,
        )
        return chunk_index

    def _upload_file_chunks(
        self,
        *,
        upload_id: str,
        file_path: str,
        file_kind: str,
        progress_callback: Optional[Callable[[int], None]],
        progress_start: int,
        progress_end: int,
    ) -> int:
        """Upload a file in API chunks in parallel and return total_chunks."""
        file_size = os.path.getsize(file_path)
        total_chunks = max(1, math.ceil(file_size / self.CHUNK_SIZE_BYTES))

        workers = min(self.STORAGE_MAX_WORKERS, total_chunks)
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    self._upload_one_chunk_to_api,
                    file_path=file_path,
                    upload_id=upload_id,
                    file_kind=file_kind,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    chunk_size=self.CHUNK_SIZE_BYTES,
                ): idx
                for idx in range(total_chunks)
            }
            for future in as_completed(futures):
                future.result()
                done += 1
                if progress_callback is not None:
                    ratio = done / total_chunks
                    progress_value = int(progress_start + (progress_end - progress_start) * ratio)
                    progress_callback(progress_value)

        return total_chunks

    def _upload_video_via_storage(
        self,
        *,
        video_path: str,
        model_path: Optional[str],
        model_file_path: Optional[str],
        analysis_fps: float,
        max_frames: int,
        resize_width: int,
        progress_callback: Optional[Callable[[int], None]],
    ) -> str:
        """Upload files to object storage then start job from storage keys."""
        video_object_key = self._upload_file_to_object_storage(
            file_path=video_path,
            file_kind="video",
            progress_callback=progress_callback,
            progress_start=5,
            progress_end=75,
        )

        model_object_key = None
        if model_file_path:
            cached_model_key = self._get_cached_model_object_key(model_file_path)
            if cached_model_key and self._storage_object_exists(cached_model_key):
                model_object_key = cached_model_key
                if progress_callback is not None:
                    progress_callback(90)
            else:
                model_object_key = self._upload_file_to_object_storage(
                    file_path=model_file_path,
                    file_kind="model",
                    progress_callback=progress_callback,
                    progress_start=75,
                    progress_end=90,
                )
                self._set_cached_model_object_key(model_file_path, model_object_key)

        payload: Dict[str, Any] = {
            "video_object_key": video_object_key,
            "analysis_fps": float(analysis_fps),
            "max_frames": int(max_frames),
            "resize_width": int(resize_width),
        }
        if model_path:
            payload["model_path"] = model_path
        if model_object_key:
            payload["model_object_key"] = model_object_key

        resp = self._request_with_retry(
            "POST",
            f"{self.server_url}/jobs/from_storage",
            max_retries=self.REQUEST_MAX_RETRIES,
            retry_label="Start job from storage",
            json=payload,
            headers=self.headers,
            timeout=120,
        )
        if progress_callback is not None:
            progress_callback(95)
        return resp.json()["job_id"]

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

        if self._is_object_storage_enabled():
            return self._upload_video_via_storage(
                video_path=video_path,
                model_path=model_path,
                model_file_path=model_file_path,
                analysis_fps=analysis_fps,
                max_frames=max_frames,
                resize_width=resize_width,
                progress_callback=progress_callback,
            )

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
        """Fallback chunked upload flow via API endpoints."""
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
