"""Memory-aware batch analysis for long football videos."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.ball_assigner.ball_assigner import BallAssigner
from src.team_assigner.team_assigner import TeamAssigner
from src.trackers.tracker import Tracker
from src.utils.video_utils import get_video_properties, iter_video_frames_sampled_with_indices
from src.view_transformer.view_transformer import ViewTransformer


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


def _write_report_files(job_id: str, report: Dict[str, Any], player_stats: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Path]:
    """Write JSON and CSV report artifacts."""
    report_dir = output_dir / "reports"
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


def _resolve_output_fps(props: Dict[str, Any], output_fps: Optional[float], analysis_fps: float) -> float:
    """Choose a smooth but bounded output FPS."""
    source_fps = float(props.get("fps") or 0.0)
    if output_fps and output_fps > 0:
        return min(float(output_fps), source_fps) if source_fps > 0 else float(output_fps)
    if source_fps > 0:
        return min(source_fps, 30.0)
    return max(1.0, float(analysis_fps))


def _render_smooth_review_video(
    *,
    tracker: Tracker,
    video_path: str,
    output_path: Path,
    annotation_states: List[Dict[str, Any]],
    output_fps: float,
    resize_width: int,
    max_source_frame_idx: Optional[int],
) -> int:
    """Render original video frames at output FPS using the latest sampled annotations."""
    if not annotation_states:
        return 0

    writer: Optional[cv2.VideoWriter] = None
    state_idx = 0
    current_state: Optional[Dict[str, Any]] = None
    rendered_frames = 0

    for source_frame_idx, _, frame in iter_video_frames_sampled_with_indices(
        video_path,
        target_fps=float(output_fps),
        resize_width=int(resize_width),
    ):
        if max_source_frame_idx is not None and source_frame_idx > max_source_frame_idx:
            break

        while (
            state_idx < len(annotation_states)
            and annotation_states[state_idx]["source_frame_idx"] <= source_frame_idx
        ):
            current_state = annotation_states[state_idx]
            state_idx += 1

        if current_state is None:
            annotated = frame
        else:
            annotated = _draw_review_frame(
                tracker=tracker,
                frame=frame,
                player_track=current_state["players"],
                referee_track=current_state["referees"],
                ball_track=current_state["ball"],
                team_ball_counts=current_state["team_ball_counts"],
                processed_frames=current_state["sample_number"],
            )

        if writer is None:
            writer = _create_video_writer(output_path, annotated, float(output_fps))
        writer.write(annotated)
        rendered_frames += 1

    if writer is not None:
        writer.release()

    return rendered_frames


def run_batch_analysis(
    *,
    job_id: str,
    video_path: str,
    model_path: str,
    output_dir: str | Path,
    analysis_fps: float = 3.0,
    resize_width: int = 1280,
    output_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Run a conservative, memory-aware analysis and write local artifacts."""
    warnings: List[str] = []
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{job_id}_output.mp4"

    tracker = Tracker(model_path)
    team_assigner = TeamAssigner()
    ball_assigner = BallAssigner()
    view_transformer = ViewTransformer()

    props = get_video_properties(video_path)
    resolved_output_fps = _resolve_output_fps(props, output_fps, float(analysis_fps))
    annotation_states: List[Dict[str, Any]] = []
    processed_frames = 0
    rendered_frames = 0
    player_frame_detections = 0
    referee_frame_detections = 0
    ball_detections = 0
    max_players_in_frame = 0
    team_ball_counts = {1: 0, 2: 0}
    last_team_control = 0
    player_summary: Dict[int, Dict[str, Any]] = {}
    calibration_checked = False
    field_calibration_confidence = 0.0

    def process_batch(entries: List[tuple[int, np.ndarray]]) -> None:
        nonlocal processed_frames
        nonlocal rendered_frames
        nonlocal player_frame_detections
        nonlocal referee_frame_detections
        nonlocal ball_detections
        nonlocal max_players_in_frame
        nonlocal last_team_control
        nonlocal calibration_checked
        nonlocal field_calibration_confidence

        if not entries:
            return

        frames = [entry[1] for entry in entries]
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

        for local_idx, (source_frame_idx, frame) in enumerate(entries):
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
            annotation_states.append(
                {
                    "source_frame_idx": int(source_frame_idx),
                    "sample_number": int(processed_frames),
                    "players": deepcopy(player_track),
                    "referees": deepcopy(referee_track),
                    "ball": deepcopy(ball_track),
                    "team_ball_counts": dict(team_ball_counts),
                }
            )

    batch: List[tuple[int, np.ndarray]] = []
    for source_frame_idx, _, frame in iter_video_frames_sampled_with_indices(
        video_path,
        target_fps=float(analysis_fps),
        max_frames=max_frames,
        resize_width=int(resize_width),
    ):
        batch.append((source_frame_idx, frame))
        if len(batch) >= max(1, batch_size):
            process_batch(batch)
            batch = []
    process_batch(batch)

    if processed_frames == 0:
        raise ValueError(f"Could not read video: {video_path}. Try MP4 (H.264) or re-upload the file.")

    max_source_frame_idx = None
    if max_frames is not None and annotation_states:
        source_fps = float(props.get("fps") or 0.0)
        sample_step = max(1, int(round(source_fps / float(analysis_fps)))) if source_fps > 0 else 1
        max_source_frame_idx = int(annotation_states[-1]["source_frame_idx"]) + sample_step - 1

    rendered_frames = _render_smooth_review_video(
        tracker=tracker,
        video_path=video_path,
        output_path=output_path,
        annotation_states=annotation_states,
        output_fps=resolved_output_fps,
        resize_width=int(resize_width),
        max_source_frame_idx=max_source_frame_idx,
    )

    if rendered_frames == 0:
        raise ValueError(f"Could not render output video: {output_path}")

    if team_assigner.kmeans is None:
        warnings.append("Team assignment was unavailable because not enough players were detected in any sampled frame.")

    counted_possession = team_ball_counts[1] + team_ball_counts[2]
    possession_team1 = (team_ball_counts[1] / counted_possession * 100.0) if counted_possession else None
    possession_team2 = (team_ball_counts[2] / counted_possession * 100.0) if counted_possession else None
    possession_confidence = counted_possession / processed_frames if processed_frames else 0.0
    player_stats = sorted(player_summary.values(), key=lambda item: item["id"])

    report = {
        "job_id": job_id,
        "status": "completed",
        "stats": {
            "possession_team1": possession_team1,
            "possession_team2": possession_team2,
            "possession_confidence": possession_confidence,
            "total_passes": None,
            "total_shots": None,
            "player_count": len(player_summary),
            "max_players_in_frame": max_players_in_frame,
            "processed_frames": processed_frames,
            "rendered_output_frames": rendered_frames,
            "analysis_fps": float(analysis_fps),
            "output_fps": float(resolved_output_fps),
            "resize_width": int(resize_width),
            "source_duration_seconds": props.get("duration_seconds"),
            "source_fps": props.get("fps"),
            "ball_detection_frames": ball_detections,
            "player_frame_detections": player_frame_detections,
            "referee_frame_detections": referee_frame_detections,
        },
        "tactical_analysis": {
            "formation_team1": None,
            "formation_team2": None,
            "pressing_intensity": None,
            "key_moments": [],
            "status": "unavailable_without_field_calibration",
        },
        "player_stats": player_stats,
        "warnings": warnings,
        "confidence": {
            "field_calibration": field_calibration_confidence,
            "possession": possession_confidence,
            "distance_speed": 0.0,
            "formation": 0.0,
        },
        "unavailable_metrics": [
            "passes",
            "shots",
            "formations",
            "distance_km",
            "max_speed_kmh",
        ],
    }

    report_paths = _write_report_files(job_id, report, player_stats, output_root)
    report["artifacts"] = {
        "annotated_video": {"local_path": str(output_path), "content_type": "video/mp4"},
        "report_json": {"local_path": str(report_paths["report_json"]), "content_type": "application/json"},
        "report_csv": {"local_path": str(report_paths["report_csv"]), "content_type": "text/csv"},
    }
    report_paths["annotated_video"] = output_path

    return {"report": report, "paths": report_paths}
