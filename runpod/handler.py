"""
RunPod Serverless Handler for Football Tactical AI.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests
import runpod

# Ensure local imports work inside worker image
sys.path.append(str(Path(__file__).parent.parent))

from src.trackers.tracker import Tracker
from src.team_assigner.team_assigner import TeamAssigner
from src.ball_assigner.ball_assigner import BallAssigner
from src.camera_movement.camera_movement_estimator import CameraMovementEstimator
from src.view_transformer.view_transformer import ViewTransformer
from src.speed_distance.speed_distance_estimator import SpeedDistanceEstimator
from src.tactical_analysis.tactical_analyzer import TacticalAnalyzer
from src.advanced_features.pass_detector import PassDetector
from src.advanced_features.shot_detector import ShotDetector
from src.utils.video_utils import read_video_sampled, save_video, get_video_properties

MODEL_PATH = "models/abdullah_yolov5.pt"
INPUT_VIDEO_PATH = "/tmp/input_video.mp4"
OUTPUT_VIDEO_PATH = "/tmp/output_video.avi"


def _download_video(video_url: str, target_path: str) -> None:
    """Download input video file from URL."""
    urllib.request.urlretrieve(video_url, target_path)


def _upload_file(upload_url: str, file_path: str) -> Dict[str, Any]:
    """Upload output file to a pre-signed URL via HTTP PUT."""
    with open(file_path, "rb") as f:
        response = requests.put(upload_url, data=f, timeout=300)

    return {
        "status_code": response.status_code,
        "ok": response.ok,
        "message": "uploaded" if response.ok else response.text[:300],
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless entrypoint."""
    input_data = event.get("input", {})

    video_url = input_data.get("video_url")
    if not video_url:
        return {"error": "No video_url provided"}

    # Memory-aware defaults for long videos.
    analysis_fps = float(input_data.get("analysis_fps", 1.0))
    max_frames = int(input_data.get("max_frames", 5400))
    resize_width = int(input_data.get("resize_width", 960))

    detect_passes = bool(input_data.get("detect_passes", True))
    detect_shots = bool(input_data.get("detect_shots", True))
    model_path = input_data.get("model_path", MODEL_PATH)
    output_upload_url: Optional[str] = input_data.get("output_upload_url")

    try:
        _download_video(video_url, INPUT_VIDEO_PATH)
        source_meta = get_video_properties(INPUT_VIDEO_PATH)

        video_frames = read_video_sampled(
            INPUT_VIDEO_PATH,
            target_fps=analysis_fps,
            max_frames=max_frames,
            resize_width=resize_width,
        )

        if not video_frames:
            return {"error": "Could not read video frames"}

        tracker = Tracker(model_path)
        tracks = tracker.get_object_tracks(video_frames)
        tracker.add_position_to_tracks(tracks)

        camera_estimator = CameraMovementEstimator(video_frames[0])
        camera_movements = camera_estimator.get_camera_movement(video_frames)
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movements)

        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        speed_estimator = SpeedDistanceEstimator(frame_rate=int(analysis_fps) if analysis_fps >= 1 else 1)
        speed_estimator.add_speed_and_distance_to_tracks(tracks)

        if not tracks["players"] or not tracks["players"][0]:
            return {"error": "No players detected in first frame"}

        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
                tracks["players"][frame_num][player_id]["team"] = team

        team_ball_control = []
        ball_assigner = BallAssigner()

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

        team_ball_control_array = np.array(team_ball_control)

        tactical_analyzer = TacticalAnalyzer()
        tactical_report = tactical_analyzer.analyze_match(tracks, team_ball_control)

        passes = []
        if detect_passes:
            pass_detector = PassDetector()
            passes = pass_detector.detect_passes(tracks, team_ball_control)

        shots = []
        if detect_shots:
            shot_detector = ShotDetector()
            shots = shot_detector.detect_shots(tracks, team_ball_control)

        output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control_array)
        output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movements)
        output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)
        save_video(output_frames, OUTPUT_VIDEO_PATH, fps=analysis_fps if analysis_fps > 0 else 1)

        total_possession = len(team_ball_control_array)
        team1_poss = float((team_ball_control_array == 1).sum() / total_possession * 100) if total_possession > 0 else 0.0
        team2_poss = float((team_ball_control_array == 2).sum() / total_possession * 100) if total_possession > 0 else 0.0

        upload_result = None
        if output_upload_url:
            upload_result = _upload_file(output_upload_url, OUTPUT_VIDEO_PATH)

        return {
            "status": "completed",
            "video_meta": {
                "source": source_meta,
                "processed_frames": len(video_frames),
                "analysis_fps": analysis_fps,
                "resize_width": resize_width,
                "max_frames": max_frames,
            },
            "stats": {
                "total_frames": len(video_frames),
                "duration_seconds_processed": int(len(video_frames) / analysis_fps) if analysis_fps > 0 else 0,
                "possession_team1": team1_poss,
                "possession_team2": team2_poss,
                "player_count": len(tracks["players"][0]) if tracks["players"] else 0,
                "passes_detected": len(passes),
                "shots_detected": len(shots),
            },
            "tactical_analysis": tactical_report,
            "output": {
                "local_path": OUTPUT_VIDEO_PATH,
                "upload": upload_result,
                "note": "Set output_upload_url input to persist the output video outside worker storage.",
            },
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
