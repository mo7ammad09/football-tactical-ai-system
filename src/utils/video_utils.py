"""
دوال مساعدة لقراءة وكتابة الفيديو.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np


def read_video(video_path: str) -> List[np.ndarray]:
    """Read full video into a list of frames.

    Args:
        video_path: مسار ملف الفيديو.

    Returns:
        قائمة بالفريمات.
    """
    return read_video_sampled(video_path)


def _read_video_with_opencv(
    video_path: str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = None,
) -> List[np.ndarray]:
    """Read frames using OpenCV only."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if target_fps and source_fps > 0:
        step = max(1, int(round(source_fps / target_fps)))
    else:
        step = 1

    frames: List[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                if resize_width and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    resized_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (resize_width, resized_height), interpolation=cv2.INTER_AREA)
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()

    return frames


def _iter_video_frames_with_opencv(
    video_path: str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield sampled frames using OpenCV without retaining the full video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if target_fps and source_fps > 0:
        step = max(1, int(round(source_fps / target_fps)))
    else:
        step = 1

    emitted = 0
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                if resize_width and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    resized_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (resize_width, resized_height), interpolation=cv2.INTER_AREA)
                yield frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()


def _transcode_to_h264_mp4(video_path: str) -> Optional[str]:
    """Transcode to a broadly-compatible MP4 using ffmpeg when available."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return None

    fd, output_path = tempfile.mkstemp(prefix="football_transcoded_", suffix=".mp4")
    os.close(fd)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-an",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1800,
        )
        return output_path
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


def read_video_sampled(
    video_path: str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = None
) -> List[np.ndarray]:
    """Read video frames with optional FPS sampling and frame cap.

    Args:
        video_path: Path to input video.
        target_fps: Target FPS for sampling (None = full FPS).
        max_frames: Maximum number of frames to return.
        resize_width: Resize output frames to this width while preserving aspect ratio.

    Returns:
        List of sampled frames.
    """
    frames = _read_video_with_opencv(
        video_path=video_path,
        target_fps=target_fps,
        max_frames=max_frames,
        resize_width=resize_width,
    )
    if frames:
        return frames

    transcoded_path = _transcode_to_h264_mp4(video_path)
    if not transcoded_path:
        return []

    try:
        frames = _read_video_with_opencv(
            video_path=transcoded_path,
            target_fps=target_fps,
            max_frames=max_frames,
            resize_width=resize_width,
        )
    finally:
        if os.path.exists(transcoded_path):
            os.remove(transcoded_path)

    return frames


def iter_video_frames_sampled(
    video_path: str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield sampled video frames without loading the full file into memory."""
    cap = cv2.VideoCapture(video_path)
    can_open = cap.isOpened()
    cap.release()

    if can_open:
        yield from _iter_video_frames_with_opencv(
            video_path=video_path,
            target_fps=target_fps,
            max_frames=max_frames,
            resize_width=resize_width,
        )
        return

    transcoded_path = _transcode_to_h264_mp4(video_path)
    if not transcoded_path:
        return

    try:
        yield from _iter_video_frames_with_opencv(
            video_path=transcoded_path,
            target_fps=target_fps,
            max_frames=max_frames,
            resize_width=resize_width,
        )
    finally:
        if os.path.exists(transcoded_path):
            os.remove(transcoded_path)


def save_video(
    output_video_frames: List[np.ndarray],
    output_video_path: str,
    fps: float = 24.0
) -> None:
    """Save frames to video file.

    Args:
        output_video_frames: قائمة الفريمات.
        output_video_path: مسار حفظ الفيديو.
        fps: معدل الإطارات.
    """
    if not output_video_frames:
        return

    output_ext = Path(output_video_path).suffix.lower()
    if output_ext == ".mp4":
        codec_candidates = ["avc1", "mp4v", "H264"]
    else:
        codec_candidates = ["XVID", "MJPG", "mp4v"]

    height, width = output_video_frames[0].shape[:2]
    target_fps = max(1.0, float(fps))

    out = None
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))
        if writer.isOpened():
            out = writer
            break
        writer.release()

    if out is None:
        raise ValueError(f"Could not initialize video writer for: {output_video_path}")

    for frame in output_video_frames:
        out.write(frame)

    out.release()


def get_video_properties(video_path: str) -> Dict:
    """Get video properties.

    Args:
        video_path: مسار الفيديو.

    Returns:
        قاموس بالخصائص.
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    properties = {
        "fps": fps,
        "frame_count": frame_count,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": int(frame_count / fps) if fps else 0,
    }

    cap.release()
    return properties
