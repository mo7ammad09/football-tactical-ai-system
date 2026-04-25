"""RunPod Serverless handler for automatic GPU analysis."""

from __future__ import annotations

import os
import sys
import uuid
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import runpod
except ImportError:  # pragma: no cover - local tests may not install runpod
    runpod = None

# Ensure official src imports work in both the repo and the worker image.
HANDLER_DIR = Path(__file__).resolve().parent
sys.path.extend([str(HANDLER_DIR), str(HANDLER_DIR.parent)])

from src.api.object_storage_client import ObjectStorageClient
from src.processing.batch_analyzer import run_batch_analysis


DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/abdullah_yolov5.pt")
TMP_ROOT = Path(os.environ.get("RUNPOD_TMP_ROOT", "/tmp/football-ai"))


def _download_url(url: str, target_path: Path) -> None:
    """Download a legacy URL input to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(target_path))


def _artifact_response(storage: Optional[ObjectStorageClient], job_id: str, name: str, path: Path, content_type: str) -> Dict[str, Any]:
    """Upload an artifact when storage is configured and return metadata."""
    artifact: Dict[str, Any] = {
        "local_path": str(path),
        "content_type": content_type,
        "object_key": None,
        "public_url": None,
    }
    if storage is None:
        return artifact

    uploaded = storage.upload_file(
        str(path),
        file_kind="artifact",
        job_id=job_id,
        content_type=content_type,
    )
    artifact.update(uploaded)
    return artifact


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod queue-based serverless entrypoint."""
    input_data = event.get("input", {}) or {}
    job_id = str(event.get("id") or input_data.get("job_id") or uuid.uuid4())

    analysis_fps = float(input_data.get("analysis_fps", 3.0))
    resize_width = int(input_data.get("resize_width", 1280))
    max_frames_raw = input_data.get("max_frames")
    max_frames = int(max_frames_raw) if max_frames_raw else None
    batch_size = int(input_data.get("batch_size") or os.environ.get("PROCESSING_BATCH_SIZE", "16"))

    work_dir = TMP_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "input_video.mp4"
    output_dir = work_dir / "outputs"

    storage: Optional[ObjectStorageClient] = None
    try:
        storage = ObjectStorageClient()
    except Exception:
        storage = None

    try:
        video_object_key = input_data.get("video_object_key")
        video_url = input_data.get("video_url")
        if video_object_key:
            if storage is None:
                raise ValueError("Object storage is required when video_object_key is provided")
            storage.download_file(video_object_key, str(video_path))
        elif video_url:
            _download_url(video_url, video_path)
        else:
            raise ValueError("video_object_key is required for production serverless jobs")

        model_path = input_data.get("model_path") or DEFAULT_MODEL_PATH
        model_object_key = input_data.get("model_object_key")
        if model_object_key:
            if storage is None:
                raise ValueError("Object storage is required when model_object_key is provided")
            model_path = str(work_dir / "model.pt")
            storage.download_file(model_object_key, model_path)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found in worker: {model_path}")

        result = run_batch_analysis(
            job_id=job_id,
            video_path=str(video_path),
            model_path=str(model_path),
            output_dir=output_dir,
            analysis_fps=analysis_fps,
            resize_width=resize_width,
            max_frames=max_frames,
            batch_size=batch_size,
        )
        report = result["report"]
        paths = result["paths"]

        artifacts = {
            "annotated_video": _artifact_response(storage, job_id, "annotated_video", paths["annotated_video"], "video/mp4"),
            "report_json": _artifact_response(storage, job_id, "report_json", paths["report_json"], "application/json"),
            "report_csv": _artifact_response(storage, job_id, "report_csv", paths["report_csv"], "text/csv"),
        }
        report["artifacts"] = artifacts
        report["annotated_video_url"] = artifacts["annotated_video"].get("public_url")
        report["report_json_url"] = artifacts["report_json"].get("public_url")
        report["report_csv_url"] = artifacts["report_csv"].get("public_url")
        report["status"] = "completed"
        return report

    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "job_id": job_id,
        }


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("runpod package is required to start the serverless worker")
    runpod.serverless.start({"handler": handler})
