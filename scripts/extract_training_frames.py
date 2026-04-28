#!/usr/bin/env python3
"""Extract annotation frames for football detector training.

This script samples real match footage into still images that can be labeled
for a YOLO11 four-class detector: player, goalkeeper, referee, ball.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def _iter_video_paths(paths: Iterable[str]) -> list[Path]:
    """Return all supported video paths from files or directories."""
    videos: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            videos.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS
                )
            )
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return videos


def _safe_stem(path: Path) -> str:
    """Create a filesystem-safe video stem."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def _resize_if_needed(frame, max_width: int):
    """Resize frame while preserving aspect ratio."""
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def extract_frames(
    *,
    video_path: Path,
    output_dir: Path,
    every_seconds: float,
    max_width: int,
    start_seconds: float,
    end_seconds: float | None,
    max_frames_per_video: int | None,
) -> list[dict[str, str | int | float]]:
    """Extract frames from one video and return metadata rows."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 25.0

    start_frame = max(0, int(round(start_seconds * fps)))
    end_frame = total_frames - 1 if end_seconds is None else int(round(end_seconds * fps))
    end_frame = max(start_frame, min(end_frame, total_frames - 1 if total_frames else end_frame))
    step = max(1, int(round(every_seconds * fps)))

    video_stem = _safe_stem(video_path)
    video_output_dir = output_dir / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    emitted = 0
    frame_idx = start_frame
    try:
        while frame_idx <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            frame = _resize_if_needed(frame, max_width)
            timestamp = frame_idx / fps
            output_name = f"{video_stem}_f{frame_idx:07d}_t{timestamp:09.3f}.jpg"
            output_path = video_output_dir / output_name
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            rows.append(
                {
                    "image_path": str(output_path),
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "source_frame_idx": frame_idx,
                    "timestamp_seconds": round(timestamp, 3),
                    "fps": round(fps, 3),
                }
            )

            emitted += 1
            if max_frames_per_video is not None and emitted >= max_frames_per_video:
                break
            frame_idx += step
    finally:
        cap.release()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames for YOLO11 football annotation.")
    parser.add_argument("inputs", nargs="+", help="Video files or folders containing videos.")
    parser.add_argument("--output-dir", default="training_data/raw_frames", help="Where images will be written.")
    parser.add_argument("--every-seconds", type=float, default=2.0, help="Sample interval per video.")
    parser.add_argument("--max-width", type=int, default=1920, help="Resize images wider than this value.")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Skip the start of each video.")
    parser.add_argument("--end-seconds", type=float, default=None, help="Stop time per video.")
    parser.add_argument("--max-frames-per-video", type=int, default=None, help="Cap frames per video.")
    parser.add_argument("--metadata", default="training_data/raw_frames/metadata.csv", help="CSV metadata path.")
    args = parser.parse_args()

    videos = _iter_video_paths(args.inputs)
    if not videos:
        raise SystemExit("No supported videos found.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata).expanduser().resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str | int | float]] = []
    for video_path in videos:
        rows = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            every_seconds=args.every_seconds,
            max_width=args.max_width,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            max_frames_per_video=args.max_frames_per_video,
        )
        print(f"{video_path.name}: extracted {len(rows)} frames")
        all_rows.extend(rows)

    with open(metadata_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "video_path",
                "video_name",
                "source_frame_idx",
                "timestamp_seconds",
                "fps",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Total frames: {len(all_rows)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
