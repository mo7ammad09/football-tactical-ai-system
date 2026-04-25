from pathlib import Path

import cv2
import numpy as np

from src.utils.video_utils import read_video_sampled


def _write_test_video(path: Path, frame_count: int = 20, fps: float = 10.0):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(frame_count):
        frame = np.full((64, 64, 3), i % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_read_video_sampled_by_target_fps(tmp_path):
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, frame_count=20, fps=10.0)

    frames = read_video_sampled(str(video_path), target_fps=5.0)

    assert 9 <= len(frames) <= 11


def test_read_video_sampled_with_max_frames(tmp_path):
    video_path = tmp_path / "sample_max.avi"
    _write_test_video(video_path, frame_count=30, fps=10.0)

    frames = read_video_sampled(str(video_path), max_frames=7)

    assert len(frames) == 7


def test_read_video_sampled_with_resize(tmp_path):
    video_path = tmp_path / "sample_resize.avi"
    _write_test_video(video_path, frame_count=5, fps=10.0)

    frames = read_video_sampled(str(video_path), resize_width=32)

    assert len(frames) == 5
    assert frames[0].shape[1] == 32
