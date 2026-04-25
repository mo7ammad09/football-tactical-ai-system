"""
FastAPI server for GPU-based video analysis.
This runs on the GPU server and processes videos.
"""

from __future__ import annotations

import os
import csv
import json
import queue
import shutil
import sqlite3
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - exercised when server deps are not installed locally
    boto3 = None
    BotoConfig = None

    class ClientError(Exception):
        """Fallback when botocore is unavailable."""

# Import our analysis modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.trackers.tracker import Tracker
from src.team_assigner.team_assigner import TeamAssigner
from src.ball_assigner.ball_assigner import BallAssigner
from src.camera_movement.camera_movement_estimator import CameraMovementEstimator
from src.view_transformer.view_transformer import ViewTransformer
from src.utils.video_utils import get_video_properties, iter_video_frames_sampled


app = FastAPI(
    title="Football Tactical AI API",
    description="GPU-powered football match analysis API",
    version="1.1.0"
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Enforce API key when GPU_API_KEY/REMOTE_GPU_API_KEY is configured."""
    if not GPU_API_KEY:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    bearer = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else ""
    api_key = request.headers.get("x-api-key", "").strip()

    if bearer != GPU_API_KEY and api_key != GPU_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


def _connect_db() -> sqlite3.Connection:
    """Open a SQLite connection for persistent job state."""
    JOB_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(JOB_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_job_db() -> None:
    """Create the job table if needed."""
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                message TEXT NOT NULL,
                video_path TEXT NOT NULL,
                output_path TEXT,
                settings_json TEXT NOT NULL,
                results_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )


def _row_to_job(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a DB row to the runtime job dictionary."""
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "progress": int(row["progress"]),
        "message": row["message"],
        "video_path": row["video_path"],
        "output_path": row["output_path"],
        "settings": json.loads(row["settings_json"]),
        "results": json.loads(row["results_json"]) if row["results_json"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _insert_job(job: Dict[str, Any]) -> None:
    """Persist a newly-created job."""
    _init_job_db()
    now = time.time()
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, progress, message, video_path, output_path,
                settings_json, results_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["job_id"],
                job["status"],
                int(job["progress"]),
                job["message"],
                job["video_path"],
                job.get("output_path"),
                json.dumps(job["settings"], ensure_ascii=True),
                json.dumps(job["results"], ensure_ascii=True) if job.get("results") else None,
                now,
                now,
            ),
        )


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Load a job by ID."""
    _init_job_db()
    with _connect_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def _update_job(job_id: str, **fields: Any) -> None:
    """Update selected job fields."""
    if not fields:
        return

    columns = []
    values: List[Any] = []
    for key, value in fields.items():
        if key == "settings":
            columns.append("settings_json = ?")
            values.append(json.dumps(value, ensure_ascii=True))
        elif key == "results":
            columns.append("results_json = ?")
            values.append(json.dumps(value, ensure_ascii=True))
        elif key in {"status", "progress", "message", "video_path", "output_path"}:
            columns.append(f"{key} = ?")
            values.append(value)
        else:
            raise ValueError(f"Unsupported job field: {key}")

    columns.append("updated_at = ?")
    values.append(time.time())
    values.append(job_id)

    _init_job_db()
    with _connect_db() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(columns)} WHERE job_id = ?", values)


def _worker_loop() -> None:
    """Single-GPU processing loop."""
    while True:
        job_id = job_queue.get()
        try:
            process_video(job_id)
        finally:
            job_queue.task_done()


def _ensure_worker_started() -> None:
    """Start the background worker once."""
    global worker_started
    with worker_lock:
        if worker_started:
            return
        thread = threading.Thread(target=_worker_loop, name="football-gpu-worker", daemon=True)
        thread.start()
        worker_started = True


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize persistent job state and the single worker."""
    _init_job_db()
    _ensure_worker_started()


def _normalize_color(color: Any, fallback: tuple[int, int, int] = (0, 0, 255)) -> tuple[int, int, int]:
    """Convert model/KMeans colors into OpenCV BGR tuples."""
    if color is None:
        return fallback
    try:
        values = list(color)
        if len(values) < 3:
            return fallback
        return tuple(int(max(0, min(255, v))) for v in values[:3])
    except Exception:
        return fallback


def _create_video_writer(output_path: Path, first_frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    """Create a browser-friendly OpenCV video writer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    for codec in ("avc1", "mp4v", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, max(1.0, float(fps)), (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise ValueError(f"Could not initialize video writer for: {output_path}")


def _draw_review_frame(
    tracker: Tracker,
    frame: np.ndarray,
    player_track: Dict,
    referee_track: Dict,
    ball_track: Dict,
    team_ball_counts: Dict[int, int],
    processed_frames: int,
) -> np.ndarray:
    """Draw lightweight review annotations for one sampled frame."""
    out = frame.copy()

    for track_id, player in player_track.items():
        color = _normalize_color(player.get("team_color"), (0, 0, 255))
        out = tracker.draw_ellipse(out, player["bbox"], color, track_id)
        if player.get("has_ball", False):
            out = tracker.draw_triangle(out, player["bbox"], (0, 0, 255))

    for _, referee in referee_track.items():
        out = tracker.draw_ellipse(out, referee["bbox"], (0, 255, 255))

    for _, ball in ball_track.items():
        out = tracker.draw_triangle(out, ball["bbox"], (0, 255, 0))

    height, width = out.shape[:2]
    panel_w = min(520, max(320, width // 3))
    panel_h = 110
    x1 = max(0, width - panel_w - 24)
    y1 = max(0, height - panel_h - 24)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    counted = team_ball_counts.get(1, 0) + team_ball_counts.get(2, 0)
    if counted > 0:
        team1 = team_ball_counts.get(1, 0) / counted * 100
        team2 = team_ball_counts.get(2, 0) / counted * 100
        cv2.putText(out, f"Team 1 possession: {team1:.1f}%", (x1 + 16, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(out, f"Team 2 possession: {team2:.1f}%", (x1 + 16, y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        cv2.putText(out, "Possession: unavailable", (x1 + 16, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.putText(out, f"Frame sample: {processed_frames}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def _estimate_sample_count(video_path: str, analysis_fps: float, max_frames: Optional[int]) -> Optional[int]:
    """Estimate sampled frames for progress reporting."""
    props = get_video_properties(video_path)
    duration = props.get("duration_seconds") or 0
    if duration <= 0:
        return max_frames
    estimated = max(1, int(duration * analysis_fps))
    if max_frames:
        estimated = min(estimated, max_frames)
    return estimated


def _write_report_files(job_id: str, report: Dict[str, Any], player_stats: List[Dict[str, Any]]) -> Dict[str, Path]:
    """Write JSON and CSV report artifacts."""
    report_dir = Path("outputs") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{job_id}_report.json"
    csv_path = report_dir / f"{job_id}_players.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = ["id", "name", "team", "frames_seen", "distance_km", "max_speed_kmh", "distance_speed_confidence"]
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in player_stats:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return {"report_json": json_path, "report_csv": csv_path}

# Default model path
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
CHUNK_ROOT = Path("uploads") / "_chunked"
JOB_DB_PATH = Path(os.environ.get("JOB_DB_PATH", "jobs.sqlite3"))
OBJECT_STORAGE_BUCKET = os.environ.get("OBJECT_STORAGE_BUCKET", "").strip()
OBJECT_STORAGE_REGION = os.environ.get("OBJECT_STORAGE_REGION", "us-east-1").strip() or "us-east-1"
OBJECT_STORAGE_ENDPOINT_URL = os.environ.get("OBJECT_STORAGE_ENDPOINT_URL", "").strip() or None
OBJECT_STORAGE_PREFIX = os.environ.get("OBJECT_STORAGE_PREFIX", "football-ai").strip().strip("/")
OBJECT_STORAGE_URL_EXPIRES = int(os.environ.get("OBJECT_STORAGE_URL_EXPIRES", "3600"))
OBJECT_STORAGE_PUBLIC_BASE_URL = os.environ.get("OBJECT_STORAGE_PUBLIC_BASE_URL", "").strip().rstrip("/")
GPU_API_KEY = (
    os.environ.get("GPU_API_KEY")
    or os.environ.get("REMOTE_GPU_API_KEY")
    or ""
).strip()
PROCESSING_BATCH_SIZE = int(os.environ.get("PROCESSING_BATCH_SIZE", "16"))

job_queue: "queue.Queue[str]" = queue.Queue()
worker_started = False
worker_lock = threading.Lock()


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: int
    message: str


class MultipartInitRequest(BaseModel):
    """Request to start multipart upload in object storage."""
    file_name: str
    file_kind: str = "video"
    content_type: Optional[str] = None
    part_size_mb: int = 8


class MultipartPresignPartRequest(BaseModel):
    """Request a pre-signed URL for uploading one part."""
    upload_id: str
    object_key: str
    part_number: int


class MultipartListPartsRequest(BaseModel):
    """Request uploaded parts list for resume support."""
    upload_id: str
    object_key: str


class StorageObjectExistsRequest(BaseModel):
    """Check whether an object key exists in object storage."""
    object_key: str


class StorageDirectUploadRequest(BaseModel):
    """Request pre-signed URL for direct single-part upload."""
    file_name: str
    file_kind: str = "video"


class MultipartPart(BaseModel):
    """Completed part metadata."""
    part_number: int = Field(..., ge=1)
    etag: str


class MultipartCompleteRequest(BaseModel):
    """Complete multipart upload."""
    upload_id: str
    object_key: str
    parts: List[MultipartPart]


class StartFromStorageRequest(BaseModel):
    """Start a job from object storage keys."""
    video_object_key: str
    model_object_key: Optional[str] = None
    model_path: Optional[str] = None
    analysis_fps: float = 3.0
    max_frames: Optional[int] = None
    resize_width: int = 1280


def _is_object_storage_enabled() -> bool:
    """Return True when object storage has minimum config."""
    return bool(OBJECT_STORAGE_BUCKET and boto3 is not None)


def _storage_error() -> HTTPException:
    """Standard error for missing object storage config."""
    return HTTPException(
        status_code=503,
        detail=(
            "Object storage is not configured. Set OBJECT_STORAGE_BUCKET "
            "(and credentials / endpoint vars) on the GPU server."
        ),
    )


@lru_cache(maxsize=1)
def _get_s3_client():
    """Create and cache S3-compatible client."""
    if not _is_object_storage_enabled():
        raise RuntimeError("Object storage is not configured or boto3 is not installed")

    access_key = os.environ.get("OBJECT_STORAGE_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("OBJECT_STORAGE_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = os.environ.get("OBJECT_STORAGE_SESSION_TOKEN") or os.environ.get("AWS_SESSION_TOKEN")

    client_kwargs: Dict[str, Any] = {
        "service_name": "s3",
        "region_name": OBJECT_STORAGE_REGION,
        "endpoint_url": OBJECT_STORAGE_ENDPOINT_URL,
        "config": BotoConfig(signature_version="s3v4"),
    }
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        client_kwargs["aws_session_token"] = session_token

    return boto3.client(**client_kwargs)


def _object_key(file_kind: str, file_name: str) -> str:
    """Build canonical object key."""
    safe_name = _safe_name(file_name, "upload.bin")
    return f"{OBJECT_STORAGE_PREFIX}/{file_kind}/{uuid.uuid4()}_{safe_name}"


def _download_from_object_storage(object_key: str, local_path: Path) -> None:
    """Download object to local path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _get_s3_client().download_file(OBJECT_STORAGE_BUCKET, object_key, str(local_path))


def _artifact_object_key(job_id: str, local_path: Path) -> str:
    """Return the object key used for final artifacts."""
    prefix = OBJECT_STORAGE_PREFIX.strip("/")
    safe_name = _safe_name(local_path.name, "artifact.bin")
    return f"{prefix}/artifacts/{job_id}/{safe_name}" if prefix else f"artifacts/{job_id}/{safe_name}"


def _upload_artifact(job_id: str, local_path: Path, content_type: str) -> Dict[str, Optional[str]]:
    """Upload an output artifact to object storage when configured."""
    if not _is_object_storage_enabled():
        return {"object_key": None, "public_url": None}

    object_key = _artifact_object_key(job_id, local_path)
    extra_args = {"ContentType": content_type}
    _get_s3_client().upload_file(str(local_path), OBJECT_STORAGE_BUCKET, object_key, ExtraArgs=extra_args)

    public_url = f"{OBJECT_STORAGE_PUBLIC_BASE_URL}/{object_key}" if OBJECT_STORAGE_PUBLIC_BASE_URL else None
    return {"object_key": object_key, "public_url": public_url}


def _safe_name(filename: str, fallback: str) -> str:
    """Return a safe basename."""
    safe_name = Path(filename or fallback).name
    return safe_name if safe_name else fallback


def _resolve_existing_model_path(requested_model_path: Optional[str]) -> str:
    """Resolve model path or raise a clear HTTP 400 before creating the job."""
    candidate = (requested_model_path or "").strip()
    if candidate:
        if not os.path.exists(candidate):
            raise HTTPException(
                status_code=400,
                detail=f"Model path does not exist on GPU server: {candidate}",
            )
        return candidate

    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH

    raise HTTPException(
        status_code=400,
        detail=(
            "No model available on GPU server. "
            "Upload a .pt model file or provide a valid model_path."
        ),
    )


async def _save_upload_file(upload: UploadFile, dest_path: Path) -> None:
    """Save UploadFile to disk in streaming mode."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _assemble_chunks(chunk_dir: Path, total_chunks: int, output_path: Path) -> None:
    """Assemble chunk files into a single output file."""
    if total_chunks < 1:
        raise ValueError("total_chunks must be >= 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out_f:
        for idx in range(total_chunks):
            part_path = chunk_dir / f"{idx:06d}.part"
            if not part_path.exists():
                raise FileNotFoundError(f"Missing chunk {idx} in {chunk_dir}")
            with open(part_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)


def _create_job(
    background_tasks: BackgroundTasks,
    video_path: Path,
    chosen_model_path: str,
    analysis_fps: float,
    max_frames: Optional[int],
    resize_width: int,
    video_object_key: Optional[str] = None,
    model_object_key: Optional[str] = None,
) -> Dict[str, str]:
    """Create and enqueue a processing job."""
    job_id = str(uuid.uuid4())
    normalized_max_frames = int(max_frames) if max_frames else None

    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Job queued for GPU processing",
        "video_path": str(video_path),
        "output_path": None,
        "results": None,
        "settings": {
            "model_path": chosen_model_path,
            "analysis_fps": max(0.1, float(analysis_fps)),
            "max_frames": normalized_max_frames if normalized_max_frames and normalized_max_frames > 0 else None,
            "resize_width": max(256, int(resize_width)),
            "video_object_key": video_object_key,
            "model_object_key": model_object_key,
        },
    }

    _insert_job(job)
    _ensure_worker_started()
    job_queue.put(job_id)
    return {"job_id": job_id, "status": "pending"}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_path: Optional[str] = Form(default=None),
    analysis_fps: float = Form(default=3.0),
    max_frames: int = Form(default=0),
    resize_width: int = Form(default=1280),
    model_file: Optional[UploadFile] = File(default=None),
):
    """Upload video for analysis.

    Supports either:
    - model_path: existing model on GPU server
    - model_file: uploaded model file (.pt)
    """
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    safe_video_name = _safe_name(file.filename or "video.mp4", "video.mp4")
    video_path = upload_dir / f"{uuid.uuid4()}_{safe_video_name}"
    await _save_upload_file(file, video_path)

    if model_file is not None and model_file.filename:
        model_dir = Path("uploaded_models")
        model_dir.mkdir(exist_ok=True)
        safe_name = _safe_name(model_file.filename, "model.pt")
        saved_model_path = model_dir / f"{uuid.uuid4()}_{safe_name}"
        await _save_upload_file(model_file, saved_model_path)
        chosen_model_path = str(saved_model_path)
    else:
        chosen_model_path = _resolve_existing_model_path(model_path)

    return _create_job(
        background_tasks=background_tasks,
        video_path=video_path,
        chosen_model_path=chosen_model_path,
        analysis_fps=analysis_fps,
        max_frames=max_frames,
        resize_width=resize_width,
    )


@app.post("/upload_chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    file_kind: str = Form(...),  # "video" or "model"
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk: UploadFile = File(...),
):
    """Upload one chunk for a large file."""
    if file_kind not in {"video", "model"}:
        raise HTTPException(status_code=400, detail="file_kind must be 'video' or 'model'")
    if chunk_index < 0 or total_chunks < 1 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="Invalid chunk_index/total_chunks")

    chunk_dir = CHUNK_ROOT / upload_id / file_kind
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"{chunk_index:06d}.part"
    await _save_upload_file(chunk, chunk_path)

    return {"ok": True, "chunk_index": chunk_index, "total_chunks": total_chunks}


@app.post("/upload_finalize")
async def upload_finalize(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    video_file_name: str = Form(...),
    video_total_chunks: int = Form(...),
    model_path: Optional[str] = Form(default=None),
    model_file_name: Optional[str] = Form(default=None),
    model_total_chunks: int = Form(default=0),
    analysis_fps: float = Form(default=3.0),
    max_frames: int = Form(default=0),
    resize_width: int = Form(default=1280),
):
    """Finalize chunked upload, assemble files, and start processing."""
    chunk_root = CHUNK_ROOT / upload_id
    video_chunk_dir = chunk_root / "video"

    if not video_chunk_dir.exists():
        raise HTTPException(status_code=400, detail="Video chunks not found")

    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        safe_video_name = _safe_name(video_file_name, "video.mp4")
        video_path = upload_dir / f"{uuid.uuid4()}_{safe_video_name}"
        _assemble_chunks(video_chunk_dir, int(video_total_chunks), video_path)

        if model_total_chunks > 0:
            model_chunk_dir = chunk_root / "model"
            if not model_file_name:
                raise HTTPException(status_code=400, detail="model_file_name is required when model_total_chunks > 0")
            if not model_chunk_dir.exists():
                raise HTTPException(status_code=400, detail="Model chunks not found")

            model_dir = Path("uploaded_models")
            model_dir.mkdir(exist_ok=True)
            safe_model_name = _safe_name(model_file_name, "model.pt")
            model_path_out = model_dir / f"{uuid.uuid4()}_{safe_model_name}"
            _assemble_chunks(model_chunk_dir, int(model_total_chunks), model_path_out)
            chosen_model_path = str(model_path_out)
        else:
            chosen_model_path = _resolve_existing_model_path(model_path)

        return _create_job(
            background_tasks=background_tasks,
            video_path=video_path,
            chosen_model_path=chosen_model_path,
            analysis_fps=analysis_fps,
            max_frames=max_frames,
            resize_width=resize_width,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to finalize upload: {exc}") from exc
    finally:
        shutil.rmtree(chunk_root, ignore_errors=True)


@app.get("/storage/health")
async def storage_health():
    """Object storage readiness."""
    return {
        "enabled": _is_object_storage_enabled(),
        "bucket": OBJECT_STORAGE_BUCKET or None,
        "endpoint": OBJECT_STORAGE_ENDPOINT_URL,
        "prefix": OBJECT_STORAGE_PREFIX,
    }


@app.post("/storage/multipart/init")
async def storage_multipart_init(payload: MultipartInitRequest):
    """Initialize multipart upload and return upload metadata."""
    if not _is_object_storage_enabled():
        raise _storage_error()

    if payload.file_kind not in {"video", "model"}:
        raise HTTPException(status_code=400, detail="file_kind must be 'video' or 'model'")

    part_size_mb = min(max(int(payload.part_size_mb), 2), 64)
    key = _object_key(payload.file_kind, payload.file_name)
    extra_args: Dict[str, Any] = {}
    if payload.content_type:
        extra_args["ContentType"] = payload.content_type

    resp = _get_s3_client().create_multipart_upload(
        Bucket=OBJECT_STORAGE_BUCKET,
        Key=key,
        **extra_args,
    )
    return {
        "upload_id": resp["UploadId"],
        "object_key": key,
        "bucket": OBJECT_STORAGE_BUCKET,
        "part_size_bytes": part_size_mb * 1024 * 1024,
    }


@app.post("/storage/multipart/presign_part")
async def storage_multipart_presign_part(payload: MultipartPresignPartRequest):
    """Generate pre-signed URL for one upload part."""
    if not _is_object_storage_enabled():
        raise _storage_error()
    if payload.part_number < 1:
        raise HTTPException(status_code=400, detail="part_number must be >= 1")

    url = _get_s3_client().generate_presigned_url(
        ClientMethod="upload_part",
        Params={
            "Bucket": OBJECT_STORAGE_BUCKET,
            "Key": payload.object_key,
            "UploadId": payload.upload_id,
            "PartNumber": payload.part_number,
        },
        ExpiresIn=OBJECT_STORAGE_URL_EXPIRES,
    )
    return {"url": url, "part_number": payload.part_number}


@app.post("/storage/multipart/list_parts")
async def storage_multipart_list_parts(payload: MultipartListPartsRequest):
    """List already-uploaded parts to support resume."""
    if not _is_object_storage_enabled():
        raise _storage_error()

    s3 = _get_s3_client()
    try:
        parts: List[Dict[str, Any]] = []
        marker = 0
        while True:
            resp = s3.list_parts(
                Bucket=OBJECT_STORAGE_BUCKET,
                Key=payload.object_key,
                UploadId=payload.upload_id,
                PartNumberMarker=marker,
                MaxParts=1000,
            )
            for part in resp.get("Parts", []):
                parts.append(
                    {
                        "part_number": int(part["PartNumber"]),
                        "etag": str(part["ETag"]).strip('"'),
                    }
                )

            if not resp.get("IsTruncated"):
                break
            marker = int(resp.get("NextPartNumberMarker", 0))

        return {"parts": parts}
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchUpload", "404"}:
            raise HTTPException(status_code=404, detail="Upload session not found") from exc
        raise HTTPException(status_code=400, detail=f"Cannot list uploaded parts: {exc}") from exc


@app.post("/storage/multipart/complete")
async def storage_multipart_complete(payload: MultipartCompleteRequest):
    """Complete multipart upload and return object reference."""
    if not _is_object_storage_enabled():
        raise _storage_error()
    if not payload.parts:
        raise HTTPException(status_code=400, detail="parts is required")

    parts = sorted(
        [
            {
                "PartNumber": int(part.part_number),
                "ETag": part.etag if part.etag.startswith('"') and part.etag.endswith('"') else f'"{part.etag}"',
            }
            for part in payload.parts
        ],
        key=lambda p: p["PartNumber"],
    )

    try:
        _get_s3_client().complete_multipart_upload(
            Bucket=OBJECT_STORAGE_BUCKET,
            Key=payload.object_key,
            UploadId=payload.upload_id,
            MultipartUpload={"Parts": parts},
        )
    except ClientError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot complete multipart upload: {exc}") from exc

    response = {
        "object_key": payload.object_key,
        "object_uri": f"s3://{OBJECT_STORAGE_BUCKET}/{payload.object_key}",
    }
    if OBJECT_STORAGE_PUBLIC_BASE_URL:
        response["public_url"] = f"{OBJECT_STORAGE_PUBLIC_BASE_URL}/{payload.object_key}"
    return response


@app.post("/storage/object_exists")
async def storage_object_exists(payload: StorageObjectExistsRequest):
    """Return whether object exists in storage."""
    if not _is_object_storage_enabled():
        raise _storage_error()
    try:
        _get_s3_client().head_object(Bucket=OBJECT_STORAGE_BUCKET, Key=payload.object_key)
        return {"exists": True}
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return {"exists": False}
        raise HTTPException(status_code=400, detail=f"Cannot check object key: {exc}") from exc


@app.post("/storage/direct_upload_url")
async def storage_direct_upload_url(payload: StorageDirectUploadRequest):
    """Return pre-signed URL for direct (single-part) upload."""
    if not _is_object_storage_enabled():
        raise _storage_error()
    if payload.file_kind not in {"video", "model"}:
        raise HTTPException(status_code=400, detail="file_kind must be 'video' or 'model'")

    object_key = _object_key(payload.file_kind, payload.file_name)
    try:
        url = _get_s3_client().generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": OBJECT_STORAGE_BUCKET, "Key": object_key},
            ExpiresIn=OBJECT_STORAGE_URL_EXPIRES,
        )
    except ClientError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot generate direct upload URL: {exc}") from exc

    return {
        "object_key": object_key,
        "upload_url": url,
        "expires_in": OBJECT_STORAGE_URL_EXPIRES,
    }


@app.post("/jobs/from_storage")
async def start_job_from_storage(
    payload: StartFromStorageRequest,
    background_tasks: BackgroundTasks,
):
    """Create job using video/model objects from storage."""
    if not _is_object_storage_enabled():
        raise _storage_error()

    video_name = Path(payload.video_object_key).name or "video.mp4"
    local_video_path = Path("uploads") / f"{uuid.uuid4()}_{video_name}"

    model_object_key = payload.model_object_key
    if model_object_key:
        model_name = Path(model_object_key).name or "model.pt"
        local_model_path = Path("uploaded_models") / f"{uuid.uuid4()}_{model_name}"
        chosen_model_path = str(local_model_path)
    else:
        chosen_model_path = _resolve_existing_model_path(payload.model_path)

    return _create_job(
        background_tasks=background_tasks,
        video_path=local_video_path,
        chosen_model_path=chosen_model_path,
        analysis_fps=payload.analysis_fps,
        max_frames=payload.max_frames,
        resize_width=payload.resize_width,
        video_object_key=payload.video_object_key,
        model_object_key=model_object_key,
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get processing status."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
    )


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not complete")

    return job["results"]


@app.get("/video/{job_id}")
async def get_video(job_id: str):
    """Download annotated video."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    video_path = job.get("output_path")

    if not video_path or not os.path.exists(video_path):
        results = job.get("results") or {}
        artifact = results.get("artifacts", {}).get("annotated_video", {})
        public_url = artifact.get("public_url")
        if public_url:
            return RedirectResponse(public_url)
        raise HTTPException(status_code=404, detail="Video not found")

    ext = Path(video_path).suffix.lower() or ".mp4"
    media_type = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
    return FileResponse(video_path, media_type=media_type, filename=f"{job_id}_analyzed{ext}")


@app.get("/artifact/{job_id}/{artifact_name}")
async def get_artifact(job_id: str, artifact_name: str):
    """Download a generated report artifact."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    results = job.get("results") or {}
    artifact = results.get("artifacts", {}).get(artifact_name)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    public_url = artifact.get("public_url")
    local_path = artifact.get("local_path")
    if local_path and os.path.exists(local_path):
        media_type = artifact.get("content_type") or "application/octet-stream"
        return FileResponse(local_path, media_type=media_type, filename=Path(local_path).name)
    if public_url:
        return RedirectResponse(public_url)

    raise HTTPException(status_code=404, detail="Artifact not found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "default_model_exists": os.path.exists(DEFAULT_MODEL_PATH),
        "object_storage_enabled": _is_object_storage_enabled(),
        "job_store": str(JOB_DB_PATH),
        "queue_size": job_queue.qsize(),
        "auth_enabled": bool(GPU_API_KEY),
    }


def process_video(job_id: str) -> None:
    """Process video in background."""
    job = _get_job(job_id)
    if job is None:
        return

    try:
        video_path = job["video_path"]
        settings = job["settings"]

        model_path = settings["model_path"]
        video_object_key = settings.get("video_object_key")
        model_object_key = settings.get("model_object_key")
        analysis_fps = settings["analysis_fps"]
        max_frames = settings["max_frames"]
        resize_width = settings["resize_width"]
        warnings: List[str] = []

        if video_object_key:
            _update_job(job_id, status="processing", message="Downloading video from object storage", progress=2)
            _download_from_object_storage(video_object_key, Path(video_path))

        if model_object_key:
            _update_job(job_id, status="processing", message="Downloading model from object storage", progress=4)
            _download_from_object_storage(model_object_key, Path(model_path))

        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            raise ValueError(f"Video file is missing or empty: {video_path}")

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found on GPU server: {model_path}. "
                "Upload a model file or provide a valid model path."
            )

        _update_job(job_id, status="processing", message="Initializing tracker", progress=8)
        tracker = Tracker(model_path)
        team_assigner = TeamAssigner()
        ball_assigner = BallAssigner()
        view_transformer = ViewTransformer()

        estimated_frames = _estimate_sample_count(video_path, float(analysis_fps), max_frames)
        props = get_video_properties(video_path)
        output_dir = Path("outputs")
        output_path = output_dir / f"{job_id}_output.mp4"
        writer: Optional[cv2.VideoWriter] = None

        processed_frames = 0
        player_frame_detections = 0
        referee_frame_detections = 0
        ball_detections = 0
        max_players_in_frame = 0
        team_ball_counts = {1: 0, 2: 0}
        last_team_control = 0
        player_summary: Dict[int, Dict[str, Any]] = {}
        calibration_checked = False
        field_calibration_confidence = 0.0

        batch: List[np.ndarray] = []
        _update_job(job_id, status="processing", message="Reading sampled frames", progress=12)

        def process_batch(frames: List[np.ndarray]) -> None:
            nonlocal writer
            nonlocal processed_frames
            nonlocal player_frame_detections
            nonlocal referee_frame_detections
            nonlocal ball_detections
            nonlocal max_players_in_frame
            nonlocal last_team_control
            nonlocal calibration_checked
            nonlocal field_calibration_confidence

            if not frames:
                return

            batch_tracks = tracker.get_object_tracks_for_frames(frames)
            tracker.add_position_to_tracks(batch_tracks)

            if not calibration_checked:
                transformed = 0
                total_positions = 0
                for player_track in batch_tracks["players"]:
                    for track in player_track.values():
                        total_positions += 1
                        if view_transformer.transform_point(track.get("position")) is not None:
                            transformed += 1
                field_calibration_confidence = transformed / total_positions if total_positions else 0.0
                calibration_checked = True
                if field_calibration_confidence < 0.5:
                    warnings.append(
                        "Field calibration confidence is low; distance, speed, formation, and tactical board metrics are disabled."
                    )

            if team_assigner.kmeans is None:
                for frame, player_track in zip(frames, batch_tracks["players"]):
                    if len(player_track) >= 2:
                        try:
                            team_assigner.assign_team_color(frame, player_track)
                        except Exception as exc:
                            warnings.append(f"Team color assignment failed on an early frame: {exc}")
                        break

            for local_idx, frame in enumerate(frames):
                player_track = batch_tracks["players"][local_idx]
                referee_track = batch_tracks["referees"][local_idx]
                ball_track = batch_tracks["ball"][local_idx]

                max_players_in_frame = max(max_players_in_frame, len(player_track))
                player_frame_detections += len(player_track)
                referee_frame_detections += len(referee_track)
                if ball_track:
                    ball_detections += 1

                for player_id, track in player_track.items():
                    team = 0
                    if team_assigner.kmeans is not None:
                        try:
                            team = team_assigner.get_player_team(frame, track["bbox"], player_id)
                        except Exception:
                            team = 0
                    track["team"] = int(team)
                    track["team_color"] = team_assigner.team_colors.get(team, (128, 128, 128))

                    summary = player_summary.setdefault(
                        int(player_id),
                        {
                            "id": int(player_id),
                            "name": f"Player {player_id}",
                            "team": int(team),
                            "frames_seen": 0,
                            "distance_km": None,
                            "max_speed_kmh": None,
                            "distance_speed_confidence": 0.0,
                        },
                    )
                    summary["frames_seen"] += 1
                    if team:
                        summary["team"] = int(team)

                ball_bbox = ball_track.get(1, {}).get("bbox") if ball_track else None
                assigned_player = -1
                if ball_bbox:
                    assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
                if assigned_player != -1 and assigned_player in player_track:
                    player_track[assigned_player]["has_ball"] = True
                    last_team_control = int(player_track[assigned_player].get("team", 0))

                if last_team_control in team_ball_counts:
                    team_ball_counts[last_team_control] += 1

                processed_frames += 1
                annotated = _draw_review_frame(
                    tracker=tracker,
                    frame=frame,
                    player_track=player_track,
                    referee_track=referee_track,
                    ball_track=ball_track,
                    team_ball_counts=team_ball_counts,
                    processed_frames=processed_frames,
                )
                if writer is None:
                    writer = _create_video_writer(output_path, annotated, float(analysis_fps))
                writer.write(annotated)

            if estimated_frames:
                progress = min(90, 12 + int((processed_frames / max(1, estimated_frames)) * 78))
            else:
                progress = min(90, 12 + processed_frames // 50)
            _update_job(
                job_id,
                status="processing",
                message=f"Processed {processed_frames} sampled frames",
                progress=progress,
            )

        for frame in iter_video_frames_sampled(
            video_path,
            target_fps=float(analysis_fps),
            max_frames=max_frames,
            resize_width=int(resize_width),
        ):
            batch.append(frame)
            if len(batch) >= max(1, PROCESSING_BATCH_SIZE):
                process_batch(batch)
                batch = []

        process_batch(batch)

        if writer is not None:
            writer.release()

        if processed_frames == 0:
            raise ValueError(
                f"Could not read video: {video_path}. Try MP4 (H.264) or re-upload the file."
            )

        if team_assigner.kmeans is None:
            warnings.append("Team assignment was unavailable because not enough players were detected in any sampled frame.")

        counted_possession = team_ball_counts[1] + team_ball_counts[2]
        possession_team1 = (team_ball_counts[1] / counted_possession * 100.0) if counted_possession else None
        possession_team2 = (team_ball_counts[2] / counted_possession * 100.0) if counted_possession else None
        possession_confidence = counted_possession / processed_frames if processed_frames else 0.0

        player_stats = sorted(player_summary.values(), key=lambda item: item["id"])
        stats = {
            "possession_team1": possession_team1,
            "possession_team2": possession_team2,
            "possession_confidence": possession_confidence,
            "total_passes": None,
            "total_shots": None,
            "player_count": len(player_summary),
            "max_players_in_frame": max_players_in_frame,
            "processed_frames": processed_frames,
            "analysis_fps": float(analysis_fps),
            "resize_width": int(resize_width),
            "source_duration_seconds": props.get("duration_seconds"),
            "source_fps": props.get("fps"),
            "ball_detection_frames": ball_detections,
            "player_frame_detections": player_frame_detections,
            "referee_frame_detections": referee_frame_detections,
        }

        confidence = {
            "field_calibration": field_calibration_confidence,
            "possession": possession_confidence,
            "distance_speed": 0.0,
            "formation": 0.0,
        }

        report = {
            "job_id": job_id,
            "annotated_video_url": f"/video/{job_id}",
            "stats": stats,
            "tactical_analysis": {
                "formation_team1": None,
                "formation_team2": None,
                "pressing_intensity": None,
                "key_moments": [],
                "status": "unavailable_without_field_calibration",
            },
            "player_stats": player_stats,
            "warnings": warnings,
            "confidence": confidence,
            "unavailable_metrics": [
                "passes",
                "shots",
                "formations",
                "distance_km",
                "max_speed_kmh",
            ],
        }

        report_paths = _write_report_files(job_id, report, player_stats)
        _update_job(job_id, status="processing", message="Uploading artifacts", progress=94, output_path=str(output_path))

        video_artifact = _upload_artifact(job_id, output_path, "video/mp4")
        json_artifact = _upload_artifact(job_id, report_paths["report_json"], "application/json")
        csv_artifact = _upload_artifact(job_id, report_paths["report_csv"], "text/csv")

        report["artifacts"] = {
            "annotated_video": {
                "local_path": str(output_path),
                "object_key": video_artifact.get("object_key"),
                "public_url": video_artifact.get("public_url"),
                "content_type": "video/mp4",
            },
            "report_json": {
                "local_path": str(report_paths["report_json"]),
                "object_key": json_artifact.get("object_key"),
                "public_url": json_artifact.get("public_url"),
                "content_type": "application/json",
            },
            "report_csv": {
                "local_path": str(report_paths["report_csv"]),
                "object_key": csv_artifact.get("object_key"),
                "public_url": csv_artifact.get("public_url"),
                "content_type": "text/csv",
            },
        }
        if video_artifact.get("public_url"):
            report["annotated_video_url"] = video_artifact["public_url"]
        report["report_json_url"] = json_artifact.get("public_url") or f"/artifact/{job_id}/report_json"
        report["report_csv_url"] = csv_artifact.get("public_url") or f"/artifact/{job_id}/report_csv"

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Analysis complete",
            output_path=str(output_path),
            results=report,
        )

    except Exception as e:
        _update_job(job_id, status="failed", message=f"Error: {str(e)}", progress=0)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("stubs", exist_ok=True)
    os.makedirs("uploaded_models", exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)
