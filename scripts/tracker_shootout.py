"""Compare tracker backends on a short football video.

This is a diagnostic tool, not a final benchmark. Without manual ground truth,
it reports proxy signals that help identify obvious ID instability:

- Too many unique player IDs compared with max players in frame.
- Many very short tracks.
- Role flicker for the same displayed ID.
- Low ball detection coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trackers.tracker import Tracker
from src.utils.video_utils import iter_video_frames_sampled_with_indices


def _chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    """Yield fixed-size chunks."""
    for index in range(0, len(items), max(1, size)):
        yield items[index: index + size]


def _summarize_tracker(
    *,
    backend: str,
    video_path: str,
    model_path: str,
    analysis_fps: float,
    resize_width: int,
    max_frames: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Run one backend and return ID-stability proxy metrics."""
    tracker = Tracker(model_path, tracker_backend=backend)
    backend_reid_strategy = {
        "botsort": "ultralytics_botsort_reid",
        "strongsort": "boxmot_strongsort_osnet",
        "bytetrack": "none",
    }.get(backend, "unknown")
    entries = list(
        iter_video_frames_sampled_with_indices(
            video_path,
            target_fps=float(analysis_fps),
            max_frames=max_frames,
            resize_width=int(resize_width),
        )
    )
    if not entries:
        raise ValueError(f"Could not read frames from {video_path}")

    track_lengths: Counter[int] = Counter()
    role_counts_by_id: Dict[int, Counter[str]] = defaultdict(Counter)
    player_counts: List[int] = []
    referee_counts: List[int] = []
    ball_frames = 0
    raw_ids_by_display_id: Dict[int, set[int]] = defaultdict(set)

    for batch in _chunks(entries, batch_size):
        frames = [item[2] for item in batch]
        tracks = tracker.get_object_tracks_for_frames(frames)
        for frame_players, frame_referees, frame_ball in zip(
            tracks.get("players", []),
            tracks.get("referees", []),
            tracks.get("ball", []),
        ):
            player_counts.append(len(frame_players))
            referee_counts.append(len(frame_referees))
            if frame_ball:
                ball_frames += 1

            for display_id, track in frame_players.items():
                display_id = int(display_id)
                track_lengths[display_id] += 1
                role_counts_by_id[display_id][str(track.get("role", "player"))] += 1
                if track.get("raw_track_id") is not None:
                    raw_ids_by_display_id[display_id].add(int(track["raw_track_id"]))

            for display_id, track in frame_referees.items():
                display_id = int(display_id)
                track_lengths[int(display_id)] += 1
                role_counts_by_id[int(display_id)]["referee"] += 1
                if track.get("raw_track_id") is not None:
                    raw_ids_by_display_id[int(display_id)].add(int(track["raw_track_id"]))

    lengths = list(track_lengths.values())
    role_flicker_ids = [
        track_id
        for track_id, counts in role_counts_by_id.items()
        if len([role for role, count in counts.items() if count > 0]) > 1
    ]
    short_track_ids = [track_id for track_id, length in track_lengths.items() if length <= 2]
    max_players = max(player_counts) if player_counts else 0
    unique_ids = len(track_lengths)
    id_churn_ratio = float(unique_ids / max(1, max_players))

    return {
        "backend": backend,
        "reid_strategy": backend_reid_strategy,
        "reid_enabled": backend_reid_strategy != "none",
        "processed_frames": len(entries),
        "unique_display_ids": unique_ids,
        "max_players_in_frame": max_players,
        "avg_players_in_frame": float(sum(player_counts) / len(player_counts)) if player_counts else 0.0,
        "avg_referees_in_frame": float(sum(referee_counts) / len(referee_counts)) if referee_counts else 0.0,
        "ball_detection_frames": ball_frames,
        "ball_detection_rate": float(ball_frames / len(entries)),
        "median_track_length": float(median(lengths)) if lengths else 0.0,
        "short_track_count": len(short_track_ids),
        "role_flicker_count": len(role_flicker_ids),
        "display_ids_with_multiple_raw_ids": sum(1 for raw_ids in raw_ids_by_display_id.values() if len(raw_ids) > 1),
        "id_churn_ratio": id_churn_ratio,
        "notes": [
            "Lower id_churn_ratio, short_track_count, and role_flicker_count are better.",
            "This is a proxy report. Final ID quality still needs manual ground-truth review.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tracker backends on one video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", default="models/abdullah_yolov5.pt", help="Detector model path.")
    parser.add_argument("--backends", nargs="+", default=["botsort", "bytetrack", "strongsort"], help="Tracker backends to try.")
    parser.add_argument("--analysis-fps", type=float, default=3.0)
    parser.add_argument("--resize-width", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="output_videos/tracker_shootout")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for backend in args.backends:
        try:
            result = _summarize_tracker(
                backend=backend,
                video_path=args.video,
                model_path=args.model,
                analysis_fps=args.analysis_fps,
                resize_width=args.resize_width,
                max_frames=args.max_frames,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            result = {
                "backend": backend,
                "error": str(exc),
            }
        results.append(result)

    json_path = output_dir / "tracker_shootout.json"
    csv_path = output_dir / "tracker_shootout.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in results for key in row.keys() if key != "notes"})
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
