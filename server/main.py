"""
FastAPI server for GPU-based video analysis.
This runs on the GPU server and processes videos.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import our analysis modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.trackers.tracker import Tracker
from src.team_assigner.team_assigner import TeamAssigner
from src.ball_assigner.ball_assigner import BallAssigner
from src.camera_movement.camera_movement_estimator import CameraMovementEstimator
from src.view_transformer.view_transformer import ViewTransformer
from src.speed_distance.speed_distance_estimator import SpeedDistanceEstimator
from src.utils.video_utils import read_video_sampled, save_video


app = FastAPI(
    title="Football Tactical AI API",
    description="GPU-powered football match analysis API",
    version="1.1.0"
)

# In-memory job storage (use Redis in production)
jobs: Dict[str, Dict[str, Any]] = {}

# Default model path
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
CHUNK_ROOT = Path("uploads") / "_chunked"


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: int
    message: str


def _safe_name(filename: str, fallback: str) -> str:
    """Return a safe basename."""
    safe_name = Path(filename or fallback).name
    return safe_name if safe_name else fallback


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
    max_frames: int,
    resize_width: int,
) -> Dict[str, str]:
    """Create and enqueue a processing job."""
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Video uploaded, waiting for processing",
        "video_path": str(video_path),
        "output_path": None,
        "results": None,
        "settings": {
            "model_path": chosen_model_path,
            "analysis_fps": max(0.1, float(analysis_fps)),
            "max_frames": max(1, int(max_frames)),
            "resize_width": max(256, int(resize_width)),
        },
    }

    background_tasks.add_task(process_video, job_id)
    return {"job_id": job_id, "status": "pending"}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_path: Optional[str] = Form(default=None),
    analysis_fps: float = Form(default=1.0),
    max_frames: int = Form(default=5400),
    resize_width: int = Form(default=960),
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

    chosen_model_path = model_path or DEFAULT_MODEL_PATH

    if model_file is not None and model_file.filename:
        model_dir = Path("uploaded_models")
        model_dir.mkdir(exist_ok=True)
        safe_name = _safe_name(model_file.filename, "model.pt")
        saved_model_path = model_dir / f"{uuid.uuid4()}_{safe_name}"
        await _save_upload_file(model_file, saved_model_path)
        chosen_model_path = str(saved_model_path)

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
    analysis_fps: float = Form(default=1.0),
    max_frames: int = Form(default=5400),
    resize_width: int = Form(default=960),
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

        chosen_model_path = model_path or DEFAULT_MODEL_PATH

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


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get processing status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
    )


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not complete")

    return job["results"]


@app.get("/video/{job_id}")
async def get_video(job_id: str):
    """Download annotated video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    video_path = job.get("output_path")

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(video_path, media_type="video/avi", filename=f"{job_id}_analyzed.avi")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "default_model_exists": os.path.exists(DEFAULT_MODEL_PATH)}


def process_video(job_id: str) -> None:
    """Process video in background."""
    job = jobs[job_id]

    try:
        video_path = job["video_path"]
        settings = job["settings"]

        model_path = settings["model_path"]
        analysis_fps = settings["analysis_fps"]
        max_frames = settings["max_frames"]
        resize_width = settings["resize_width"]

        job["status"] = "processing"
        job["message"] = "Reading video"
        job["progress"] = 5

        video_frames = read_video_sampled(
            video_path,
            target_fps=analysis_fps,
            max_frames=max_frames,
            resize_width=resize_width,
        )

        if not video_frames:
            raise ValueError("Could not read video")

        job["message"] = "Initializing tracker"
        job["progress"] = 12
        tracker = Tracker(model_path)

        job["message"] = "Detecting and tracking objects"
        job["progress"] = 25
        tracks = tracker.get_object_tracks(video_frames)
        tracker.add_position_to_tracks(tracks)

        job["message"] = "Estimating camera movement"
        job["progress"] = 40
        camera_estimator = CameraMovementEstimator(video_frames[0])
        camera_movements = camera_estimator.get_camera_movement(video_frames)
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movements)

        job["message"] = "Transforming coordinates"
        job["progress"] = 50
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        job["message"] = "Interpolating ball positions"
        job["progress"] = 58
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        job["message"] = "Calculating speed and distance"
        job["progress"] = 66
        speed_estimator = SpeedDistanceEstimator(frame_rate=max(1, int(round(analysis_fps))))
        speed_estimator.add_speed_and_distance_to_tracks(tracks)

        if not tracks["players"] or not tracks["players"][0]:
            raise ValueError("No players detected in first frame")

        job["message"] = "Assigning teams"
        job["progress"] = 74
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

        job["message"] = "Assigning ball possession"
        job["progress"] = 82
        ball_assigner = BallAssigner()
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks["players"]):
            ball_frame = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
            ball_bbox = ball_frame.get(1, {}).get("bbox")

            if ball_bbox:
                assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
                if assigned_player != -1:
                    tracks["players"][frame_num][assigned_player]["has_ball"] = True
                    team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
                else:
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        team_ball_control_np = np.array(team_ball_control)

        job["message"] = "Drawing annotations"
        job["progress"] = 92
        output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control_np)
        output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movements)
        output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{job_id}_output.avi"
        save_video(output_frames, str(output_path), fps=analysis_fps)

        total_possession = len(team_ball_control_np)
        possession_team1 = float((team_ball_control_np == 1).sum() / total_possession * 100) if total_possession > 0 else 50.0
        possession_team2 = float((team_ball_control_np == 2).sum() / total_possession * 100) if total_possession > 0 else 50.0

        player_stats = []
        last_frame_idx = len(tracks["players"]) - 1
        if tracks["players"] and tracks["players"][last_frame_idx]:
            for player_id, track in tracks["players"][last_frame_idx].items():
                player_stats.append({
                    "id": int(player_id),
                    "name": f"Player {player_id}",
                    "team": int(track.get("team", 1)),
                    "distance_km": float(track.get("distance", 0)) / 1000.0,
                    "max_speed_kmh": float(track.get("speed", 0)),
                })

        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Analysis complete"
        job["output_path"] = str(output_path)
        job["results"] = {
            "job_id": job_id,
            "annotated_video_url": f"/video/{job_id}",
            "stats": {
                "possession_team1": possession_team1,
                "possession_team2": possession_team2,
                "total_passes": 0,
                "total_shots": 0,
                "player_count": len(tracks["players"][0]) if tracks["players"] else 0,
                "processed_frames": len(video_frames),
                "analysis_fps": analysis_fps,
            },
            "tactical_analysis": {
                "formation_team1": "4-3-3",
                "formation_team2": "4-4-2",
                "pressing_intensity": "medium",
                "key_moments": [],
            },
            "player_stats": player_stats,
        }

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Error: {str(e)}"
        job["progress"] = 0


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("stubs", exist_ok=True)
    os.makedirs("uploaded_models", exist_ok=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)
