from pathlib import Path

import cv2
import numpy as np

from src.utils.video_utils import iter_video_frames_sampled


def _write_test_video(path: Path, frame_count: int = 20, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(frame_count):
        frame = np.full((64, 64, 3), i % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_iter_video_frames_sampled_streams_with_max_frames(tmp_path):
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, frame_count=30, fps=10.0)

    frames = list(iter_video_frames_sampled(str(video_path), target_fps=5.0, max_frames=4))

    assert len(frames) == 4
    assert frames[0].shape == (64, 64, 3)


def test_iter_video_frames_sampled_resizes(tmp_path):
    video_path = tmp_path / "sample_resize.avi"
    _write_test_video(video_path, frame_count=5, fps=10.0)

    frames = list(iter_video_frames_sampled(str(video_path), resize_width=32, max_frames=2))

    assert len(frames) == 2
    assert frames[0].shape[1] == 32
