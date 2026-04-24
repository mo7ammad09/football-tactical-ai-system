"""
Main pipeline - Football Tactical AI System.

Usage:
    python main.py --input input_videos/test.mp4 --output output_videos/output_video.avi
    python main.py --config config/config.yaml --stubs --target-fps 10
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np
import yaml

from src.utils.video_utils import read_video_sampled, save_video
from src.trackers.tracker import Tracker
from src.team_assigner.team_assigner import TeamAssigner
from src.ball_assigner.ball_assigner import BallAssigner
from src.camera_movement.camera_movement_estimator import CameraMovementEstimator
from src.view_transformer.view_transformer import ViewTransformer
from src.speed_distance.speed_distance_estimator import SpeedDistanceEstimator


DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file.

    Args:
        config_path: Path to YAML config.

    Returns:
        Parsed config dictionary.
    """
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Football Tactical AI local pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    parser.add_argument("--input", default=None, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--stubs", action="store_true", help="Use cached stubs when available")
    parser.add_argument("--target-fps", type=float, default=None, help="Sample input video to this FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to read from input video")
    return parser.parse_args()


def main() -> None:
    """Run full analysis pipeline."""
    args = parse_args()
    config = load_config(args.config)

    input_path = args.input or config.get("video", {}).get("input_file", "input_videos/test.mp4")
    output_path = args.output or config.get("video", {}).get("output_file", "output_videos/output_video.avi")
    model_path = args.model or config.get("detection", {}).get("model_path", "models/best.pt")
    target_fps = args.target_fps or config.get("video", {}).get("analysis_fps")
    max_frames = args.max_frames or config.get("video", {}).get("max_frames")

    os.makedirs("stubs", exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"📹 Reading video: {input_path}")
    video_frames = read_video_sampled(input_path, target_fps=target_fps, max_frames=max_frames)
    print(f"✅ Loaded {len(video_frames)} frames")
    if not video_frames:
        raise ValueError(f"No frames loaded from: {input_path}")

    print(f"🔍 Loading model: {model_path}")
    tracker = Tracker(model_path)

    print("🔍 Detecting and tracking objects...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=args.stubs,
        stub_path="stubs/track_stubs.pkl"
    )
    tracker.add_position_to_tracks(tracks)

    print("📹 Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=args.stubs,
        stub_path="stubs/camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    print("🔄 Transforming to field coordinates...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    print("⚽ Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print("🏃 Calculating speed and distance...")
    speed_distance_estimator = SpeedDistanceEstimator(
        frame_rate=int(target_fps) if target_fps else int(config.get("video", {}).get("fps", 24))
    )
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    print("👥 Assigning teams...")
    team_assigner = TeamAssigner()
    if not tracks["players"] or not tracks["players"][0]:
        raise ValueError("No players detected in first frame; cannot assign teams")

    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    print("⚽ Assigning ball possession...")
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

    team_ball_control = np.array(team_ball_control)

    print("🎨 Drawing annotations...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    print(f"💾 Saving output video: {output_path}")
    save_video(output_video_frames, output_path, fps=float(target_fps) if target_fps else 24.0)

    print("✅ Analysis complete")


if __name__ == "__main__":
    main()
