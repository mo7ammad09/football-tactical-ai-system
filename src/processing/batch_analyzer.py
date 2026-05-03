"""Memory-aware batch analysis for long football videos."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile

import cv2
import numpy as np

from src.ball_assigner.ball_assigner import BallAssigner
from src.identity.pre_render_audit import (
    KNOWN_BAD_RUNPOD_IMAGES,
    RUNPOD_BASELINE_IMAGE,
    apply_safe_correction_plan_to_raw_records,
    build_correction_candidates,
    build_dry_run_correction_plan,
    build_final_render_identity_manifest,
    build_identity_events,
    build_player_crop_index_plan,
    build_render_identity_audit,
    build_vision_review_results,
    build_vision_review_queue,
    post_fix_audit_improved,
)
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
        display_role = str(player.get("display_role") or player.get("role") or "player")
        fallback_color = (
            (255, 0, 255)
            if display_role == "goalkeeper"
            else (0, 0, 255)
        )
        color = _normalize_color(
            player.get("display_color", player.get("team_color")),
            fallback_color,
        )
        out = tracker.draw_ellipse(
            out,
            player["bbox"],
            color,
            player.get("display_label", track_id),
        )
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

    fieldnames = [
        "id",
        "name",
        "role",
        "team",
        "frames_seen",
        "distance_km",
        "max_speed_kmh",
        "distance_speed_confidence",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in player_stats:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return {"report_json": json_path, "report_csv": csv_path}


def _json_safe(value: Any) -> Any:
    """Convert numpy/counter values into JSON-serializable primitives."""
    if isinstance(value, Counter):
        return {str(key): int(count) for key, count in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as jsonl_f:
        for record in records:
            jsonl_f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def _write_identity_artifacts(
    job_id: str,
    *,
    raw_tracklet_records: List[Dict[str, Any]],
    identity_debug: Dict[str, Any],
    identity_events: Dict[str, Any],
    render_audit_before: Dict[str, Any],
    render_audit_after: Dict[str, Any],
    correction_candidates: Dict[str, Any],
    correction_plan: Dict[str, Any],
    correction_applied: Dict[str, Any],
    vision_review_queue: Dict[str, Any],
    vision_review_results: Dict[str, Any],
    final_render_identity_manifest: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Write identity debugging artifacts for offline review."""
    identity_dir = output_dir / "identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    raw_tracklets_path = identity_dir / f"{job_id}_raw_tracklets.jsonl"
    identity_debug_path = identity_dir / f"{job_id}_identity_debug.json"
    identity_events_path = identity_dir / f"{job_id}_identity_events.json"
    render_audit_before_path = identity_dir / f"{job_id}_render_audit_before.json"
    render_audit_after_path = identity_dir / f"{job_id}_render_audit_after.json"
    correction_candidates_path = identity_dir / f"{job_id}_correction_candidates.json"
    correction_plan_path = identity_dir / f"{job_id}_correction_plan.json"
    correction_applied_path = identity_dir / f"{job_id}_correction_applied.json"
    vision_review_queue_path = identity_dir / f"{job_id}_vision_review_queue.json"
    vision_review_results_path = identity_dir / f"{job_id}_vision_review_results.json"
    final_render_identity_manifest_path = identity_dir / f"{job_id}_final_render_identity_manifest.json"

    _write_jsonl(raw_tracklets_path, raw_tracklet_records)
    identity_debug_path.write_text(
        json.dumps(_json_safe(identity_debug), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    identity_events_path.write_text(
        json.dumps(_json_safe(identity_events), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_audit_before_path.write_text(
        json.dumps(_json_safe(render_audit_before), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_audit_after_path.write_text(
        json.dumps(_json_safe(render_audit_after), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    correction_candidates_path.write_text(
        json.dumps(_json_safe(correction_candidates), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    correction_plan_path.write_text(
        json.dumps(_json_safe(correction_plan), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    correction_applied_path.write_text(
        json.dumps(_json_safe(correction_applied), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    vision_review_queue_path.write_text(
        json.dumps(_json_safe(vision_review_queue), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    vision_review_results_path.write_text(
        json.dumps(_json_safe(vision_review_results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_render_identity_manifest_path.write_text(
        json.dumps(_json_safe(final_render_identity_manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "raw_tracklets_jsonl": raw_tracklets_path,
        "identity_debug_json": identity_debug_path,
        "identity_events_json": identity_events_path,
        "render_audit_before_json": render_audit_before_path,
        "render_audit_after_json": render_audit_after_path,
        "correction_candidates_json": correction_candidates_path,
        "correction_plan_json": correction_plan_path,
        "correction_applied_json": correction_applied_path,
        "vision_review_queue_json": vision_review_queue_path,
        "vision_review_results_json": vision_review_results_path,
        "final_render_identity_manifest_json": final_render_identity_manifest_path,
    }


def _resize_frame_for_analysis(frame: np.ndarray, resize_width: int) -> np.ndarray:
    """Resize a source frame the same way the analysis loop does."""
    if resize_width and frame.shape[1] > int(resize_width):
        scale = float(resize_width) / float(frame.shape[1])
        resized_height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (int(resize_width), resized_height), interpolation=cv2.INTER_AREA)
    return frame


def _read_requested_frames(
    video_path: str,
    frame_indices: List[int],
    resize_width: int,
) -> Dict[int, np.ndarray]:
    """Read exact source frames needed for vision evidence crops."""
    requested = sorted({int(frame_idx) for frame_idx in frame_indices if int(frame_idx) >= 0})
    if not requested:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    frames: Dict[int, np.ndarray] = {}
    try:
        for frame_idx in requested:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames[frame_idx] = _resize_frame_for_analysis(frame, int(resize_width))
    finally:
        cap.release()
    return frames


def _crop_with_padding(frame: np.ndarray, bbox: Any, padding_ratio: float = 0.18) -> Optional[np.ndarray]:
    """Crop a bbox from a frame with modest context padding."""
    xyxy = _as_bbox_array(bbox)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * float(padding_ratio)
    pad_y = box_h * float(padding_ratio)
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right]
    return crop if crop.size else None


def _make_contact_sheet(crops: List[tuple[Path, str]], output_path: Path) -> bool:
    """Create a compact contact sheet for one vision-review case."""
    if not crops:
        return False

    cell_w = 180
    cell_h = 260
    label_h = 34
    cols = min(4, max(1, len(crops)))
    rows = int(np.ceil(len(crops) / cols))
    sheet = np.full((rows * (cell_h + label_h), cols * cell_w, 3), 245, dtype=np.uint8)

    for idx, (crop_path, label) in enumerate(crops):
        crop = cv2.imread(str(crop_path))
        if crop is None or crop.size == 0:
            continue
        row = idx // cols
        col = idx % cols
        scale = min(cell_w / crop.shape[1], cell_h / crop.shape[0])
        new_w = max(1, int(crop.shape[1] * scale))
        new_h = max(1, int(crop.shape[0] * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = col * cell_w + (cell_w - new_w) // 2
        y = row * (cell_h + label_h) + (cell_h - new_h) // 2
        sheet[y : y + new_h, x : x + new_w] = resized
        cv2.putText(
            sheet,
            label[:24],
            (col * cell_w + 6, row * (cell_h + label_h) + cell_h + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet))


def _zip_directory(source_dir: Path, output_path: Path) -> None:
    """Zip a directory, preserving relative paths."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_f:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zip_f.write(path, path.relative_to(source_dir))


def _write_vision_evidence_artifacts(
    job_id: str,
    *,
    video_path: str,
    raw_tracklet_records: List[Dict[str, Any]],
    vision_review_queue: Dict[str, Any],
    render_audit: Dict[str, Any],
    output_dir: Path,
    resize_width: int,
) -> Dict[str, Path]:
    """Write Phase 5 crop index and contact sheets for queued vision cases."""
    identity_dir = output_dir / "identity"
    crop_root = identity_dir / "vision_crops"
    sheet_root = identity_dir / "contact_sheets"
    crop_root.mkdir(parents=True, exist_ok=True)
    sheet_root.mkdir(parents=True, exist_ok=True)

    crop_index = build_player_crop_index_plan(
        raw_tracklet_rows=raw_tracklet_records,
        vision_review_queue=vision_review_queue,
        render_audit=render_audit,
    )
    requested_frames = [
        int(crop["source_frame_idx"])
        for case in crop_index.get("cases", [])
        for crop in case.get("crop_requests", [])
        if crop.get("source_frame_idx") is not None
    ]
    frames = _read_requested_frames(video_path, requested_frames, int(resize_width))

    written_crop_count = 0
    for case in crop_index.get("cases", []):
        case_id = str(case.get("case_id") or "unknown_case").replace("/", "_")
        case_crops: List[tuple[Path, str]] = []
        for crop in case.get("crop_requests", []):
            frame = frames.get(int(crop.get("source_frame_idx", -1)))
            if frame is None:
                crop["status"] = "frame_unavailable"
                continue
            crop_image = _crop_with_padding(frame, crop.get("bbox"))
            if crop_image is None:
                crop["status"] = "invalid_bbox"
                continue
            crop_path = crop_root / case_id / f"{crop['crop_id']}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(crop_path), crop_image):
                crop["status"] = "written"
                crop["crop_path"] = str(crop_path)
                written_crop_count += 1
                label = f"t{crop.get('track_id')} f{crop.get('source_frame_idx')}"
                case_crops.append((crop_path, label))
            else:
                crop["status"] = "write_failed"

        sheet_path = sheet_root / f"{case_id}.jpg"
        if _make_contact_sheet(case_crops, sheet_path):
            case["contact_sheet_path"] = str(sheet_path)
            case["contact_sheet_status"] = "written"
        else:
            case["contact_sheet_status"] = "not_written"

    crop_index["crop_root"] = str(crop_root)
    crop_index["contact_sheets_root"] = str(sheet_root)
    crop_index["written_crop_count"] = int(written_crop_count)
    crop_index["phase"] = "phase_5_crop_evidence"
    crop_index_path = identity_dir / f"{job_id}_player_crop_index.json"
    crop_index_path.write_text(
        json.dumps(_json_safe(crop_index), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contact_sheets_zip = identity_dir / f"{job_id}_contact_sheets.zip"
    _zip_directory(sheet_root, contact_sheets_zip)
    return {
        "player_crop_index_json": crop_index_path,
        "vision_contact_sheets_zip": contact_sheets_zip,
    }


def _append_raw_tracklet_records(
    records: List[Dict[str, Any]],
    *,
    sample_number: int,
    source_frame_idx: int,
    object_type: str,
    tracks: Dict[int, Dict[str, Any]],
) -> None:
    """Append frame-level track records for offline ID debugging."""
    for track_id, track in tracks.items():
        reid_vector = _normalize_vector(track.get("reid_embedding"))
        records.append(
            {
                "sample_number": int(sample_number),
                "source_frame_idx": int(source_frame_idx),
                "object_type": str(object_type),
                "track_id": int(track_id),
                "raw_track_id": (
                    int(track["raw_track_id"])
                    if track.get("raw_track_id") is not None
                    else None
                ),
                "merged_from_id": (
                    int(track["merged_from_id"])
                    if track.get("merged_from_id") is not None
                    else None
                ),
                "role": track.get("role"),
                "detected_role": track.get("detected_role"),
                "team": int(track.get("team", 0) or 0),
                "display_label": (
                    str(track["display_label"])
                    if track.get("display_label") is not None
                    else None
                ),
                "display_role": (
                    str(track["display_role"])
                    if track.get("display_role") is not None
                    else None
                ),
                "display_team": (
                    int(track["display_team"])
                    if track.get("display_team") is not None
                    else None
                ),
                "display_color": (
                    list(_normalize_color(track["display_color"]))
                    if track.get("display_color") is not None
                    else None
                ),
                "goalkeeper_display_locked": bool(track.get("goalkeeper_display_locked", False)),
                "role_display_suppressed": bool(track.get("role_display_suppressed", False)),
                "confidence": (
                    float(track["confidence"])
                    if track.get("confidence") is not None
                    else None
                ),
                "has_ball": bool(track.get("has_ball", False)),
                "reid_available": reid_vector is not None,
                "reid_dim": int(reid_vector.size) if reid_vector is not None else 0,
                "bbox": _as_bbox_array(track.get("bbox")).tolist(),
            }
        )


class _GoalkeeperDisplayLock:
    """Stabilize client-facing goalkeeper labels without merging raw identities."""

    def __init__(
        self,
        min_evidence_frames: int = 5,
        max_follow_gap_frames: int = 120,
        follow_distance_threshold: float = 2.5,
    ):
        self.min_evidence_frames = max(1, int(min_evidence_frames))
        self.max_follow_gap_frames = max(1, int(max_follow_gap_frames))
        self.follow_distance_threshold = max(0.5, float(follow_distance_threshold))
        self._goalkeeper_evidence: Counter = Counter()
        self._locked_keys: set[tuple[str, int]] = set()
        self._last_locked_bbox: Optional[List[float]] = None
        self._last_locked_source_frame_idx: Optional[int] = None
        self.locked = False
        self.suppressed_frames = 0
        self.locked_frames = 0

    @staticmethod
    def _candidate_key(track_id: int, track: Dict[str, Any]) -> tuple[str, int]:
        raw_track_id = track.get("raw_track_id")
        if raw_track_id is not None:
            return ("raw", int(raw_track_id))
        return ("display", int(track_id))

    @staticmethod
    def _is_goalkeeper_evidence(track: Dict[str, Any]) -> bool:
        """Return whether either raw or smoothed role says goalkeeper."""
        return (
            str(track.get("role", "player")) == "goalkeeper"
            or str(track.get("detected_role", "player")) == "goalkeeper"
        )

    @staticmethod
    def _is_referee_like(track: Dict[str, Any]) -> bool:
        """Avoid promoting referee detections into goalkeeper display locks."""
        return (
            str(track.get("role", "player")) == "referee"
            or str(track.get("detected_role", "player")) == "referee"
            or str(track.get("object_type", "player")) == "referee"
        )

    @staticmethod
    def _current_goalkeeper_score(track: Dict[str, Any]) -> float:
        """Score one frame's evidence for choosing the visible GK."""
        score = float(track.get("confidence") or 0.0)
        role = str(track.get("role", "player"))
        detected_role = str(track.get("detected_role", role))
        team = int(track.get("team", 0) or 0)

        if role == "goalkeeper":
            score += 3.0
        if detected_role == "goalkeeper":
            score += 2.0
        if team == 0:
            score += 1.0
        if role == "player" and team not in {0, None}:
            score -= 2.0
        if _GoalkeeperDisplayLock._is_referee_like(track):
            score -= 10.0
        return score

    @staticmethod
    def _bbox_distance(a: Any, b: Any) -> float:
        """Return normalized center distance between two boxes."""
        a_bbox = _as_bbox_array(a)
        b_bbox = _as_bbox_array(b)
        scale = max(_bbox_scale(a_bbox), _bbox_scale(b_bbox), 1.0)
        return float(np.linalg.norm(_bbox_center(a_bbox) - _bbox_center(b_bbox)) / scale)

    def _is_recent_spatial_follow(self, track: Dict[str, Any], source_frame_idx: Optional[int]) -> bool:
        """Return whether a player-classified row continues the active GK path."""
        if self._last_locked_bbox is None or self._last_locked_source_frame_idx is None:
            return False
        if source_frame_idx is None:
            return False
        if source_frame_idx - self._last_locked_source_frame_idx > self.max_follow_gap_frames:
            return False
        if self._is_referee_like(track):
            return False
        distance = self._bbox_distance(self._last_locked_bbox, track.get("bbox"))
        return distance <= self.follow_distance_threshold

    def _suppress_prelock(self, track: Dict[str, Any]) -> None:
        """Keep early noisy goalkeeper flashes from becoming client-facing IDs."""
        track["role_display_suppressed"] = True
        track["display_role"] = "player"
        track["display_color"] = (128, 128, 128)
        if str(track.get("role", "player")) == "goalkeeper":
            track["role"] = "player"
            track["team"] = 0
            track["team_color"] = (128, 128, 128)
        self.suppressed_frames += 1

    def _suppress_duplicate_display(self, track_id: int, track: Dict[str, Any]) -> None:
        """Remove client-facing GK display from duplicate/weak same-frame rows."""
        track["display_label"] = str(track_id)
        track["display_role"] = "player"
        track["display_team"] = int(track.get("team", 0) or 0)
        track["display_color"] = None
        track["goalkeeper_display_locked"] = False
        track["role_display_suppressed"] = True
        self.suppressed_frames += 1

    def _apply_locked_display(
        self,
        track_id: int,
        track: Dict[str, Any],
        key: tuple[str, int],
        source_frame_idx: Optional[int],
    ) -> None:
        """Force the visible goalkeeper identity while preserving internal evidence."""
        self._locked_keys.add(key)
        self.locked = True
        track["display_label"] = "GK"
        track["display_role"] = "goalkeeper"
        track["display_team"] = 0
        track["display_color"] = (255, 0, 255)
        track["goalkeeper_display_locked"] = True
        if str(track.get("role", "player")) == "goalkeeper":
            track["team"] = 0
            track["team_color"] = (255, 0, 255)
        self.locked_frames += 1
        self._last_locked_bbox = _as_bbox_array(track.get("bbox")).tolist()
        self._last_locked_source_frame_idx = source_frame_idx

    def apply(
        self,
        player_track: Dict[int, Dict[str, Any]],
        source_frame_idx: Optional[int] = None,
    ) -> None:
        """Mark stable goalkeeper detections with a fixed visible label."""
        if not player_track:
            return

        goalkeeper_items: List[tuple[int, Dict[str, Any], tuple[str, int]]] = []
        for track_id, track in player_track.items():
            if not self._is_goalkeeper_evidence(track):
                continue
            key = self._candidate_key(int(track_id), track)
            self._goalkeeper_evidence[key] += 1
            goalkeeper_items.append((int(track_id), track, key))
            if self._goalkeeper_evidence[key] >= self.min_evidence_frames:
                self._locked_keys.add(key)

        self.locked = bool(self._locked_keys)
        if not self.locked:
            for _, track, _ in goalkeeper_items:
                self._suppress_prelock(track)
            return

        locked_items: List[tuple[int, Dict[str, Any], tuple[str, int]]] = []
        for track_id, track in player_track.items():
            key = self._candidate_key(int(track_id), track)
            if key in self._locked_keys:
                if self._is_goalkeeper_evidence(track) or self._is_recent_spatial_follow(
                    track,
                    source_frame_idx,
                ):
                    locked_items.append((int(track_id), track, key))
                continue
            if (
                self._is_goalkeeper_evidence(track)
                and self._goalkeeper_evidence[key] >= self.min_evidence_frames
            ):
                locked_items.append((int(track_id), track, key))
                continue
            if (
                self._is_goalkeeper_evidence(track)
                and self._is_recent_spatial_follow(track, source_frame_idx)
            ):
                locked_items.append((int(track_id), track, key))

        locked_keys_this_frame = {key for _, _, key in locked_items}
        for _, track, key in goalkeeper_items:
            if key not in locked_keys_this_frame:
                self._suppress_prelock(track)

        if len(locked_items) > 1:
            same_frame_locked_items = locked_items
            selected = max(
                same_frame_locked_items,
                key=lambda item: self._current_goalkeeper_score(item[1]),
            )
            locked_items = [selected]
            for track_id, track, key in same_frame_locked_items:
                if track is not selected[1]:
                    self._suppress_duplicate_display(track_id, track)

        for track_id, track, key in locked_items:
            self._apply_locked_display(track_id, track, key, source_frame_idx)

    def summary(self) -> Dict[str, Any]:
        """Return report-friendly lock statistics."""
        return {
            "enabled": True,
            "locked": bool(self.locked),
            "min_evidence_frames": int(self.min_evidence_frames),
            "max_follow_gap_frames": int(self.max_follow_gap_frames),
            "follow_distance_threshold": float(self.follow_distance_threshold),
            "locked_goalkeeper_frames": int(self.locked_frames),
            "suppressed_prelock_frames": int(self.suppressed_frames),
            "locked_keys": [
                f"{kind}:{identifier}"
                for kind, identifier in sorted(self._locked_keys)
            ],
            "candidate_evidence": {
                f"{kind}:{identifier}": int(count)
                for (kind, identifier), count in self._goalkeeper_evidence.items()
            },
        }


def _normalize_identity_merge_map(identity_merge_map: Optional[Dict[Any, Any]]) -> Dict[int, int]:
    """Normalize user-provided source_id -> target_id corrections."""
    if not identity_merge_map:
        return {}

    normalized: Dict[int, int] = {}
    for source_id, target_id in identity_merge_map.items():
        try:
            source = int(source_id)
            target = int(target_id)
        except (TypeError, ValueError):
            continue
        if source > 0 and target > 0 and source != target:
            normalized[source] = target
    return normalized


def _bbox_area(track: Dict[str, Any]) -> float:
    """Return bbox area for conflict resolution."""
    bbox = track.get("bbox") or [0, 0, 0, 0]
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _merge_track_ids(track: Dict[int, Dict[str, Any]], merge_map: Dict[int, int]) -> tuple[Dict[int, Dict[str, Any]], int]:
    """Apply source_id -> target_id merges to one frame of tracks."""
    if not merge_map:
        return track, 0

    merged: Dict[int, Dict[str, Any]] = {}
    conflicts = 0
    for track_id, track_data in track.items():
        source_id = int(track_id)
        target_id = int(merge_map.get(source_id, source_id))
        if target_id != source_id:
            track_data["merged_from_id"] = source_id

        if target_id in merged:
            conflicts += 1
            if _bbox_area(track_data) > _bbox_area(merged[target_id]):
                merged[target_id] = track_data
            continue
        merged[target_id] = track_data

    return merged, conflicts


def _apply_identity_merge_map(batch_tracks: Dict[str, List[Dict]], merge_map: Dict[int, int]) -> int:
    """Apply manual identity merges to players and referees in a batch."""
    conflicts = 0
    if not merge_map:
        return conflicts

    for object_type in ("players", "referees"):
        for frame_idx, frame_tracks in enumerate(batch_tracks.get(object_type, [])):
            merged, frame_conflicts = _merge_track_ids(frame_tracks, merge_map)
            frame_tracks.clear()
            frame_tracks.update(merged)
            batch_tracks[object_type][frame_idx] = merged
            conflicts += frame_conflicts
    return conflicts


def _safe_plan_action_ids(correction_plan: Dict[str, Any]) -> set[str]:
    """Return validator-accepted action IDs for Phase 3 rendering."""
    validation = correction_plan.get("validation", {})
    if validation.get("verdict") != "PASS":
        return set()
    return {
        str(action_id)
        for action_id in validation.get("accepted_action_ids", [])
    }


def _apply_safe_correction_plan_to_annotation_states(
    annotation_states: List[Dict[str, Any]],
    correction_plan: Dict[str, Any],
) -> int:
    """Apply safe display-only corrections to rendered annotation states."""
    accepted_action_ids = _safe_plan_action_ids(correction_plan)
    if not accepted_action_ids:
        return 0

    updated_tracks = 0
    for action in correction_plan.get("actions", []):
        if str(action.get("action_id")) not in accepted_action_ids:
            continue
        if str(action.get("action_type")) != "display_override":
            continue
        if str(action.get("status")) != "safe_fix_dry_run":
            continue
        if str(action.get("set_display_role", "unknown")) == "goalkeeper":
            continue
        if str(action.get("set_display_label", "")).upper() == "GK":
            continue
        if str(action.get("set_display_color_policy", "")) != "team":
            continue

        target_track_id = int(action.get("track_id"))
        target_raw_track_id = (
            int(action["raw_track_id"])
            if action.get("raw_track_id") is not None
            else None
        )
        first_frame = int(action.get("first_source_frame_idx", -1))
        last_frame = int(action.get("last_source_frame_idx", -1))
        for state in annotation_states:
            source_frame_idx = int(state.get("source_frame_idx", -1))
            if source_frame_idx < first_frame or source_frame_idx > last_frame:
                continue
            track = state.get("players", {}).get(target_track_id)
            if track is None:
                continue
            if target_raw_track_id is not None and track.get("raw_track_id") is not None:
                if int(track["raw_track_id"]) != target_raw_track_id:
                    continue
            track["display_role"] = str(action.get("set_display_role", "player"))
            track["display_label"] = str(action.get("set_display_label", target_track_id))
            track["display_team"] = int(track.get("team", 0) or 0)
            track["display_color"] = track.get("team_color")
            track["goalkeeper_display_locked"] = False
            track["role_display_suppressed"] = False
            updated_tracks += 1
    return updated_tracks


def _as_bbox_array(bbox: Any) -> np.ndarray:
    """Convert a bbox-like value to an xyxy float array."""
    try:
        values = np.asarray(bbox, dtype=float).reshape(-1)
    except Exception:
        values = np.zeros(4, dtype=float)
    if len(values) < 4:
        return np.zeros(4, dtype=float)
    return values[:4]


def _bbox_center(bbox: Any) -> np.ndarray:
    """Return bbox center."""
    xyxy = _as_bbox_array(bbox)
    return np.array(
        [
            (float(xyxy[0]) + float(xyxy[2])) / 2.0,
            (float(xyxy[1]) + float(xyxy[3])) / 2.0,
        ],
        dtype=float,
    )


def _bbox_scale(bbox: Any) -> float:
    """Return a stable scale for normalized distance."""
    xyxy = _as_bbox_array(bbox)
    return max(1.0, float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1]))


def _normalize_vector(values: Any) -> Optional[np.ndarray]:
    """Return a finite L2-normalized vector."""
    if values is None:
        return None
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _cosine_distance(vector_a: Optional[np.ndarray], vector_b: Optional[np.ndarray]) -> float:
    """Return cosine distance for normalized identity embeddings."""
    a = _normalize_vector(vector_a)
    b = _normalize_vector(vector_b)
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return float(max(0.0, min(2.0, 1.0 - float(np.dot(a, b)))))


def _extract_color_signature(frame: np.ndarray, bbox: Any) -> Optional[np.ndarray]:
    """Extract a jersey/body color signature for fallback identity linking."""
    xyxy = _as_bbox_array(bbox)
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1, int(xyxy[0])))
    y1 = max(0, min(frame_h - 1, int(xyxy[1])))
    x2 = max(x1 + 1, min(frame_w, int(xyxy[2])))
    y2 = max(y1 + 1, min(frame_h, int(xyxy[3])))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    body_crop = crop[: max(1, int(crop.shape[0] * 0.68)), :]
    hsv_crop = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv_crop],
        [0, 1],
        None,
        [24, 12],
        [0, 180, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(float) if np.isfinite(hist).all() else None


def _merge_profile_vector(profile: Dict[str, Any], key: str, count_key: str, values: Any) -> None:
    """Update a running mean vector in a profile."""
    vector = _normalize_vector(values)
    if vector is None:
        return
    if profile.get(key) is None:
        profile[key] = vector
        profile[count_key] = 1
        return
    count = int(profile.get(count_key, 0))
    current = np.asarray(profile[key], dtype=float)
    profile[key] = ((current * count) + vector) / float(count + 1)
    profile[count_key] = count + 1


def _update_identity_profile(
    profiles: Dict[int, Dict[str, Any]],
    *,
    track_id: int,
    role: str,
    team: int,
    sample_number: int,
    source_frame_idx: int,
    bbox: Any,
    raw_track_id: Optional[int] = None,
    object_type: str = "player",
    reid_embedding: Any = None,
    color_signature: Any = None,
) -> None:
    """Collect a compact tracklet profile for offline identity reconciliation."""
    profile = profiles.setdefault(
        int(track_id),
        {
            "id": int(track_id),
            "role_counts": Counter(),
            "team_counts": Counter(),
            "frames_seen": 0,
            "first_sample": int(sample_number),
            "last_sample": int(sample_number),
            "first_source_frame_idx": int(source_frame_idx),
            "last_source_frame_idx": int(source_frame_idx),
            "first_bbox": _as_bbox_array(bbox).tolist(),
            "last_bbox": _as_bbox_array(bbox).tolist(),
            "raw_track_counts": Counter(),
            "object_type_counts": Counter(),
            "role_segments": [],
            "reid_embedding": None,
            "reid_count": 0,
            "color_signature": None,
            "color_count": 0,
        },
    )
    profile["role_counts"][str(role)] += 1
    profile["object_type_counts"][str(object_type)] += 1
    if raw_track_id is not None:
        profile["raw_track_counts"][int(raw_track_id)] += 1
    if team:
        profile["team_counts"][int(team)] += 1
    profile["frames_seen"] += 1
    profile["last_sample"] = int(sample_number)
    profile["last_source_frame_idx"] = int(source_frame_idx)
    profile["last_bbox"] = _as_bbox_array(bbox).tolist()
    segments = profile.setdefault("role_segments", [])
    segment_raw_track_id = int(raw_track_id) if raw_track_id is not None else None
    if (
        not segments
        or segments[-1]["role"] != str(role)
        or int(segments[-1]["team"]) != int(team)
        or segments[-1].get("raw_track_id") != segment_raw_track_id
        or str(segments[-1].get("object_type", object_type)) != str(object_type)
    ):
        segments.append(
            {
                "role": str(role),
                "team": int(team),
                "raw_track_id": segment_raw_track_id,
                "object_type": str(object_type),
                "first_sample": int(sample_number),
                "last_sample": int(sample_number),
                "first_source_frame_idx": int(source_frame_idx),
                "last_source_frame_idx": int(source_frame_idx),
                "frames_seen": 1,
            }
        )
    else:
        segments[-1]["last_sample"] = int(sample_number)
        segments[-1]["last_source_frame_idx"] = int(source_frame_idx)
        segments[-1]["frames_seen"] += 1
    _merge_profile_vector(profile, "reid_embedding", "reid_count", reid_embedding)
    _merge_profile_vector(profile, "color_signature", "color_count", color_signature)


def _dominant_count_value(counter: Counter, fallback: Any = None) -> tuple[Any, float]:
    """Return dominant counter value and confidence."""
    total = sum(counter.values())
    if total <= 0:
        return fallback, 0.0
    value, count = counter.most_common(1)[0]
    return value, float(count) / float(total)


def _roles_compatible(role_a: str, role_b: str) -> bool:
    """Allow identity linking only between roles that can reasonably flicker."""
    if role_a == role_b:
        return True
    return {role_a, role_b}.issubset({"player", "goalkeeper"})


def _teams_compatible(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> bool:
    """Reject high-confidence team contradictions while allowing noisy teams."""
    team_a, confidence_a = _dominant_count_value(profile_a.get("team_counts", Counter()), 0)
    team_b, confidence_b = _dominant_count_value(profile_b.get("team_counts", Counter()), 0)
    if not team_a or not team_b:
        return True
    if team_a == team_b:
        return True
    return not (confidence_a >= 0.85 and confidence_b >= 0.85)


def _profile_position_distance(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> float:
    """Return normalized distance between old tracklet end and new tracklet start."""
    center_a = _bbox_center(profile_a.get("last_bbox"))
    center_b = _bbox_center(profile_b.get("first_bbox"))
    scale = max(_bbox_scale(profile_a.get("last_bbox")), _bbox_scale(profile_b.get("first_bbox")))
    return float(np.linalg.norm(center_a - center_b) / scale)


def _reid_candidate_threshold(role_a: str, role_b: str, position_distance: float) -> float:
    """Return a broad ReID threshold for surfacing human-review candidates."""
    if role_a == "referee" and role_b == "referee":
        return 0.32 if position_distance <= 3.0 else 0.28
    if role_a != role_b:
        return 0.28
    return 0.36 if position_distance <= 2.5 else 0.28


def _auto_reid_threshold(role_a: str, role_b: str) -> float:
    """Return the stricter ReID threshold used for automatic merges."""
    if role_a == "referee" and role_b == "referee":
        return 0.16
    if role_a != role_b:
        return 0.08
    if role_a in {"player", "goalkeeper"}:
        return 0.12
    return 0.10


def _identity_candidate(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Score whether two non-overlapping tracklets are the same person."""
    if int(profile_a["last_source_frame_idx"]) >= int(profile_b["first_source_frame_idx"]):
        return None

    role_a, _ = _dominant_count_value(profile_a.get("role_counts", Counter()), "player")
    role_b, _ = _dominant_count_value(profile_b.get("role_counts", Counter()), "player")
    if not _roles_compatible(str(role_a), str(role_b)):
        return None
    if not _teams_compatible(profile_a, profile_b):
        return None

    gap = int(profile_b["first_source_frame_idx"]) - int(profile_a["last_source_frame_idx"])
    position_distance = _profile_position_distance(profile_a, profile_b)
    reid_count = min(int(profile_a.get("reid_count", 0)), int(profile_b.get("reid_count", 0)))
    color_count = min(int(profile_a.get("color_count", 0)), int(profile_b.get("color_count", 0)))

    if reid_count > 0:
        reid_distance = _cosine_distance(profile_a.get("reid_embedding"), profile_b.get("reid_embedding"))
        threshold = _reid_candidate_threshold(str(role_a), str(role_b), position_distance)
        if reid_distance > threshold:
            return None
        return {
            "source_id": int(profile_b["id"]),
            "target_id": int(profile_a["id"]),
            "score": float(reid_distance + min(position_distance, 6.0) * 0.015 + min(gap, 9000) / 9000.0 * 0.025),
            "evidence": "osnet_reid",
            "role_a": str(role_a),
            "role_b": str(role_b),
            "reid_count": int(reid_count),
            "reid_distance": float(reid_distance),
            "reid_threshold": float(threshold),
            "auto_reid_threshold": float(_auto_reid_threshold(str(role_a), str(role_b))),
            "position_distance": float(position_distance),
            "gap_source_frames": int(gap),
        }

    if color_count > 0:
        color_distance = _cosine_distance(profile_a.get("color_signature"), profile_b.get("color_signature"))
        if color_distance > 0.08 or position_distance > 1.25:
            return None
        return {
            "source_id": int(profile_b["id"]),
            "target_id": int(profile_a["id"]),
            "score": float(color_distance + position_distance * 0.05),
            "evidence": "color_position",
            "color_distance": float(color_distance),
            "position_distance": float(position_distance),
            "gap_source_frames": int(gap),
        }

    return None


def _identity_candidate_debug(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> Dict[str, Any]:
    """Explain whether two tracklets are identity-link candidates."""
    base = {
        "source_id": int(profile_b["id"]),
        "target_id": int(profile_a["id"]),
        "old_tracklet_id": int(profile_a["id"]),
        "new_tracklet_id": int(profile_b["id"]),
        "old_last_source_frame_idx": int(profile_a["last_source_frame_idx"]),
        "new_first_source_frame_idx": int(profile_b["first_source_frame_idx"]),
    }

    if int(profile_a["last_source_frame_idx"]) >= int(profile_b["first_source_frame_idx"]):
        return {**base, "status": "rejected", "reason": "overlap_or_reverse_time"}

    role_a, role_confidence_a = _dominant_count_value(profile_a.get("role_counts", Counter()), "player")
    role_b, role_confidence_b = _dominant_count_value(profile_b.get("role_counts", Counter()), "player")
    team_a, team_confidence_a = _dominant_count_value(profile_a.get("team_counts", Counter()), 0)
    team_b, team_confidence_b = _dominant_count_value(profile_b.get("team_counts", Counter()), 0)
    gap = int(profile_b["first_source_frame_idx"]) - int(profile_a["last_source_frame_idx"])
    position_distance = _profile_position_distance(profile_a, profile_b)
    reid_count = min(int(profile_a.get("reid_count", 0)), int(profile_b.get("reid_count", 0)))
    color_count = min(int(profile_a.get("color_count", 0)), int(profile_b.get("color_count", 0)))

    details = {
        **base,
        "role_a": str(role_a),
        "role_b": str(role_b),
        "role_confidence_a": float(role_confidence_a),
        "role_confidence_b": float(role_confidence_b),
        "team_a": int(team_a or 0),
        "team_b": int(team_b or 0),
        "team_confidence_a": float(team_confidence_a),
        "team_confidence_b": float(team_confidence_b),
        "gap_source_frames": int(gap),
        "position_distance": float(position_distance),
        "reid_count": int(reid_count),
        "color_count": int(color_count),
    }

    if not _roles_compatible(str(role_a), str(role_b)):
        return {**details, "status": "rejected", "reason": "role_incompatible"}
    if not _teams_compatible(profile_a, profile_b):
        return {**details, "status": "rejected", "reason": "team_conflict"}

    if reid_count > 0:
        reid_distance = _cosine_distance(profile_a.get("reid_embedding"), profile_b.get("reid_embedding"))
        threshold = _reid_candidate_threshold(str(role_a), str(role_b), position_distance)
        status = "candidate" if reid_distance <= threshold else "rejected"
        reason = "candidate_reid" if status == "candidate" else "reid_distance_above_threshold"
        score = float(reid_distance + min(position_distance, 6.0) * 0.015 + min(gap, 9000) / 9000.0 * 0.025)
        return {
            **details,
            "status": status,
            "reason": reason,
            "score": score,
            "evidence": "osnet_reid",
            "reid_distance": float(reid_distance),
            "reid_threshold": float(threshold),
            "auto_reid_threshold": float(_auto_reid_threshold(str(role_a), str(role_b))),
        }

    if color_count > 0:
        color_distance = _cosine_distance(profile_a.get("color_signature"), profile_b.get("color_signature"))
        is_candidate = color_distance <= 0.08 and position_distance <= 1.25
        return {
            **details,
            "status": "candidate" if is_candidate else "rejected",
            "reason": "candidate_color_position" if is_candidate else "color_or_position_above_threshold",
            "score": float(color_distance + position_distance * 0.05),
            "evidence": "color_position",
            "color_distance": float(color_distance),
            "color_threshold": 0.08,
            "position_threshold": 1.25,
        }

    return {**details, "status": "rejected", "reason": "no_reid_or_color_signal"}


def _intervals_overlap(intervals_a: List[tuple[int, int]], intervals_b: List[tuple[int, int]]) -> bool:
    """Return True if any source-frame intervals overlap."""
    for start_a, end_a in intervals_a:
        for start_b, end_b in intervals_b:
            if start_a <= end_b and start_b <= end_a:
                return True
    return False


def _build_auto_identity_merge_map(profiles: Dict[int, Dict[str, Any]]) -> tuple[Dict[int, int], List[Dict[str, Any]]]:
    """Build high-confidence automatic tracklet merges after the full pass."""
    if len(profiles) < 2:
        return {}, []

    candidates: List[Dict[str, Any]] = []
    profile_values = sorted(profiles.values(), key=lambda item: int(item["first_source_frame_idx"]))
    for old_profile in profile_values:
        for new_profile in profile_values:
            if int(old_profile["id"]) == int(new_profile["id"]):
                continue
            candidate = _identity_candidate(old_profile, new_profile)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        return {}, []

    best_previous_for_source: Dict[int, Dict[str, Any]] = {}
    best_next_for_target: Dict[int, Dict[str, Any]] = {}
    candidates_by_source: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    candidates_by_target: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        source_id = int(candidate["source_id"])
        target_id = int(candidate["target_id"])
        candidates_by_source[source_id].append(candidate)
        candidates_by_target[target_id].append(candidate)
        if (
            source_id not in best_previous_for_source
            or candidate["score"] < best_previous_for_source[source_id]["score"]
        ):
            best_previous_for_source[source_id] = candidate
        if (
            target_id not in best_next_for_target
            or candidate["score"] < best_next_for_target[target_id]["score"]
        ):
            best_next_for_target[target_id] = candidate

    for grouped_candidates in list(candidates_by_source.values()) + list(candidates_by_target.values()):
        grouped_candidates.sort(key=lambda item: float(item["score"]))

    def is_unambiguous_best(candidate: Dict[str, Any]) -> bool:
        """Only auto-merge when the best link clearly beats nearby alternatives."""
        if candidate.get("evidence") != "osnet_reid":
            return False
        if int(candidate.get("reid_count", 0)) < 8:
            return False

        role_a = str(candidate.get("role_a", "player"))
        role_b = str(candidate.get("role_b", "player"))
        reid_distance = float(candidate.get("reid_distance", 1.0))
        if reid_distance > _auto_reid_threshold(role_a, role_b):
            return False

        margin = 0.035
        source_candidates = candidates_by_source.get(int(candidate["source_id"]), [])
        target_candidates = candidates_by_target.get(int(candidate["target_id"]), [])
        if len(source_candidates) > 1 and float(source_candidates[1]["score"]) - float(candidate["score"]) < margin:
            return False
        if len(target_candidates) > 1 and float(target_candidates[1]["score"]) - float(candidate["score"]) < margin:
            return False
        return True

    parent = {int(profile_id): int(profile_id) for profile_id in profiles}
    intervals = {
        int(profile_id): [
            (
                int(profile["first_source_frame_idx"]),
                int(profile["last_source_frame_idx"]),
            )
        ]
        for profile_id, profile in profiles.items()
    }

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    accepted_links: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: float(item["score"])):
        source_id = int(candidate["source_id"])
        target_id = int(candidate["target_id"])
        if best_previous_for_source.get(source_id) is not candidate:
            continue
        if best_next_for_target.get(target_id) is not candidate:
            continue
        if not is_unambiguous_best(candidate):
            continue

        source_root = find(source_id)
        target_root = find(target_id)
        if source_root == target_root:
            continue
        if _intervals_overlap(intervals[source_root], intervals[target_root]):
            continue

        parent[source_root] = target_root
        intervals[target_root].extend(intervals[source_root])
        accepted_links.append(candidate)

    merge_map = {
        track_id: find(track_id)
        for track_id in sorted(parent)
        if find(track_id) != track_id
    }
    return merge_map, accepted_links


def _identity_profile_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact JSON-safe identity profile."""
    role, role_confidence = _dominant_count_value(profile.get("role_counts", Counter()), "player")
    team, team_confidence = _dominant_count_value(profile.get("team_counts", Counter()), 0)
    return {
        "id": int(profile["id"]),
        "role": str(role),
        "role_confidence": float(role_confidence),
        "team": int(team or 0),
        "team_confidence": float(team_confidence),
        "frames_seen": int(profile.get("frames_seen", 0)),
        "first_sample": int(profile.get("first_sample", 0)),
        "last_sample": int(profile.get("last_sample", 0)),
        "first_source_frame_idx": int(profile.get("first_source_frame_idx", 0)),
        "last_source_frame_idx": int(profile.get("last_source_frame_idx", 0)),
        "first_bbox": _as_bbox_array(profile.get("first_bbox")).tolist(),
        "last_bbox": _as_bbox_array(profile.get("last_bbox")).tolist(),
        "reid_count": int(profile.get("reid_count", 0)),
        "color_count": int(profile.get("color_count", 0)),
    }


def _compact_role_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Return the fields needed to audit role/color flicker."""
    return {
        "role": str(segment.get("role", "player")),
        "team": int(segment.get("team", 0) or 0),
        "raw_track_id": (
            int(segment["raw_track_id"])
            if segment.get("raw_track_id") is not None
            else None
        ),
        "object_type": str(segment.get("object_type", "player")),
        "first_sample": int(segment.get("first_sample", 0)),
        "last_sample": int(segment.get("last_sample", 0)),
        "first_source_frame_idx": int(segment.get("first_source_frame_idx", 0)),
        "last_source_frame_idx": int(segment.get("last_source_frame_idx", 0)),
        "frames_seen": int(segment.get("frames_seen", 0)),
    }


def _build_role_stability_diagnostics(profile_values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find role flicker and goalkeeper ID fragmentation visible in tracklets."""
    role_flicker_tracklets: List[Dict[str, Any]] = []
    goalkeeper_segments: List[Dict[str, Any]] = []

    for profile in profile_values:
        track_id = int(profile["id"])
        frames_seen = int(profile.get("frames_seen", 0))
        role_counts = Counter(profile.get("role_counts", Counter()))
        team_counts = Counter(profile.get("team_counts", Counter()))
        raw_track_counts = Counter(profile.get("raw_track_counts", Counter()))
        object_type_counts = Counter(profile.get("object_type_counts", Counter()))
        dominant_role, role_confidence = _dominant_count_value(role_counts, "player")
        segments = [
            _compact_role_segment(segment)
            for segment in profile.get("role_segments", [])
        ]
        role_transitions = sum(
            1
            for previous, current in zip(segments, segments[1:])
            if previous["role"] != current["role"]
        )
        goalkeeper_like_segments = [
            segment for segment in segments if segment["role"] == "goalkeeper" or segment["team"] == 0
        ]
        goalkeeper_frame_count = sum(int(segment["frames_seen"]) for segment in goalkeeper_like_segments)

        for segment in goalkeeper_like_segments:
            goalkeeper_segments.append({"track_id": track_id, **segment})

        if (
            role_transitions > 0
            or (str(dominant_role) != "goalkeeper" and goalkeeper_frame_count >= 3)
            or (frames_seen >= 20 and float(role_confidence) < 0.85)
        ):
            role_flicker_tracklets.append(
                {
                    "id": track_id,
                    "dominant_role": str(dominant_role),
                    "role_confidence": float(role_confidence),
                    "frames_seen": frames_seen,
                    "first_source_frame_idx": int(profile.get("first_source_frame_idx", 0)),
                    "last_source_frame_idx": int(profile.get("last_source_frame_idx", 0)),
                    "role_transition_count": int(role_transitions),
                    "goalkeeper_frame_count": int(goalkeeper_frame_count),
                    "role_counts": dict(role_counts),
                    "team_counts": dict(team_counts),
                    "object_type_counts": dict(object_type_counts),
                    "raw_track_counts": dict(raw_track_counts.most_common(8)),
                    "goalkeeper_segments": goalkeeper_like_segments[:12],
                    "segments_truncated": len(goalkeeper_like_segments) > 12,
                }
            )

    goalkeeper_segments = sorted(
        goalkeeper_segments,
        key=lambda item: (
            int(item.get("first_source_frame_idx", 0)),
            int(item.get("track_id", 0)),
        ),
    )
    substantial_goalkeeper_ids = {
        int(segment["track_id"])
        for segment in goalkeeper_segments
        if int(segment.get("frames_seen", 0)) >= 3
    }
    goalkeeper_id_switch_count = sum(
        1
        for previous, current in zip(goalkeeper_segments, goalkeeper_segments[1:])
        if int(previous["track_id"]) != int(current["track_id"])
    )

    return {
        "role_flicker_tracklet_count": len(role_flicker_tracklets),
        "goalkeeper_segment_count": len(goalkeeper_segments),
        "goalkeeper_display_id_count": len({int(segment["track_id"]) for segment in goalkeeper_segments}),
        "substantial_goalkeeper_display_id_count": len(substantial_goalkeeper_ids),
        "goalkeeper_id_switch_count": int(goalkeeper_id_switch_count),
        "role_flicker_tracklets": role_flicker_tracklets,
        "goalkeeper_timeline": goalkeeper_segments[:80],
        "goalkeeper_timeline_truncated": len(goalkeeper_segments) > 80,
    }


def _build_identity_debug_report(
    *,
    profiles: Dict[int, Dict[str, Any]],
    manual_merge_map: Dict[int, int],
    auto_merge_map: Dict[int, int],
    auto_links: List[Dict[str, Any]],
    max_rejected_examples: int = 500,
) -> Dict[str, Any]:
    """Build a post-match identity report for human inspection."""
    profile_values = sorted(profiles.values(), key=lambda item: int(item["first_source_frame_idx"]))
    candidate_links: List[Dict[str, Any]] = []
    actionable_rejected_examples: List[Dict[str, Any]] = []
    overlap_rejected_examples: List[Dict[str, Any]] = []
    reject_reason_counts: Counter = Counter()
    profiles_with_reid = sum(1 for profile in profile_values if int(profile.get("reid_count", 0)) > 0)
    profiles_with_color = sum(1 for profile in profile_values if int(profile.get("color_count", 0)) > 0)
    role_stability = _build_role_stability_diagnostics(profile_values)

    for old_profile in profile_values:
        for new_profile in profile_values:
            if int(old_profile["id"]) == int(new_profile["id"]):
                continue
            audit = _identity_candidate_debug(old_profile, new_profile)
            if audit["status"] == "candidate":
                candidate_links.append(audit)
            else:
                reject_reason_counts[str(audit.get("reason", "unknown"))] += 1
                if audit.get("reason") == "overlap_or_reverse_time":
                    if len(overlap_rejected_examples) < 20:
                        overlap_rejected_examples.append(audit)
                elif len(actionable_rejected_examples) < max_rejected_examples:
                    actionable_rejected_examples.append(audit)

    def _rejection_sort_key(item: Dict[str, Any]) -> tuple:
        return (
            float(item.get("score", 999.0)),
            int(item.get("gap_source_frames", 999999)),
            float(item.get("position_distance", 999.0)),
        )

    rejected_examples = sorted(actionable_rejected_examples, key=_rejection_sort_key)
    if len(rejected_examples) < max_rejected_examples:
        rejected_examples.extend(
            overlap_rejected_examples[: max_rejected_examples - len(rejected_examples)]
        )

    warnings = []
    if profile_values and profiles_with_reid == 0:
        warnings.append(
            {
                "code": "missing_reid_embeddings",
                "message": (
                    "No OSNet/ReID embeddings were captured. Automatic identity "
                    "linking can only use jersey color and position, so conservative "
                    "auto-merges are expected to be rare."
                ),
            }
        )
    if role_stability["role_flicker_tracklet_count"] > 0:
        warnings.append(
            {
                "code": "role_flicker_detected",
                "message": (
                    "Some display IDs switch roles over time. Review "
                    "role_stability.role_flicker_tracklets before trusting "
                    "the annotated colors."
                ),
                "tracklet_count": role_stability["role_flicker_tracklet_count"],
            }
        )
    if role_stability["substantial_goalkeeper_display_id_count"] > 1:
        warnings.append(
            {
                "code": "goalkeeper_identity_fragmentation",
                "message": (
                    "Goalkeeper-like segments appear under multiple display IDs. "
                    "Review role_stability.goalkeeper_timeline before accepting "
                    "goalkeeper ID consistency."
                ),
                "display_id_count": role_stability["substantial_goalkeeper_display_id_count"],
            }
        )

    return {
        "summary": {
            "tracklet_count": len(profile_values),
            "profiles_with_reid": profiles_with_reid,
            "profiles_with_color": profiles_with_color,
            "candidate_link_count": len(candidate_links),
            "accepted_auto_link_count": len(auto_links),
            "manual_merge_count": len(manual_merge_map),
            "auto_merge_count": len(auto_merge_map),
            "rejected_pair_count": sum(reject_reason_counts.values()),
            "rejected_examples_truncated": sum(reject_reason_counts.values()) > len(rejected_examples),
        },
        "manual_merge_map": manual_merge_map,
        "auto_merge_map": auto_merge_map,
        "accepted_auto_links": auto_links,
        "candidate_links": sorted(candidate_links, key=lambda item: float(item.get("score", 999.0))),
        "reject_reason_counts": dict(reject_reason_counts),
        "warnings": warnings,
        "role_stability": role_stability,
        "rejected_examples": rejected_examples,
        "tracklet_profiles": [_identity_profile_summary(profile) for profile in profile_values],
    }


def _rebuild_player_outputs(annotation_states: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuild player stats and final identity spans after all identity merges."""
    summaries: Dict[int, Dict[str, Any]] = {}
    spans: Dict[int, Dict[str, Any]] = {}

    for state in annotation_states:
        sample_number = int(state["sample_number"])
        source_frame_idx = int(state["source_frame_idx"])
        for player_id, track in state.get("players", {}).items():
            player_id = int(player_id)
            role = str(track.get("role", "player"))
            team = int(track.get("team", 0) or 0)
            summary = summaries.setdefault(
                player_id,
                {
                    "id": player_id,
                    "name": f"Player {player_id}",
                    "role": role,
                    "team": team,
                    "frames_seen": 0,
                    "distance_km": None,
                    "max_speed_kmh": None,
                    "distance_speed_confidence": 0.0,
                },
            )
            summary["frames_seen"] += 1
            summary["role"] = role
            if team:
                summary["team"] = team

            span = spans.setdefault(
                player_id,
                {
                    "id": player_id,
                    "role": role,
                    "frames_seen": 0,
                    "first_sample": sample_number,
                    "last_sample": sample_number,
                    "first_source_frame_idx": source_frame_idx,
                    "last_source_frame_idx": source_frame_idx,
                    "merged_from_ids": set(),
                },
            )
            span["frames_seen"] += 1
            span["role"] = role
            span["last_sample"] = sample_number
            span["last_source_frame_idx"] = source_frame_idx
            if track.get("merged_from_id"):
                span["merged_from_ids"].add(int(track["merged_from_id"]))

    player_stats = sorted(summaries.values(), key=lambda item: item["id"])
    identity_track_spans = []
    for span in sorted(spans.values(), key=lambda item: item["id"]):
        output_span = dict(span)
        output_span["merged_from_ids"] = sorted(output_span["merged_from_ids"])
        identity_track_spans.append(output_span)
    return player_stats, identity_track_spans


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
    identity_merge_map: Optional[Dict[Any, Any]] = None,
    tracker_backend: str = "botsort",
) -> Dict[str, Any]:
    """Run a conservative, memory-aware analysis and write local artifacts."""
    warnings: List[str] = []
    normalized_identity_merge_map = _normalize_identity_merge_map(identity_merge_map)
    if normalized_identity_merge_map:
        warnings.append(
            f"Manual identity merge map applied: {normalized_identity_merge_map}"
        )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{job_id}_output.mp4"

    tracker = Tracker(model_path, tracker_backend=tracker_backend)
    team_assigner = TeamAssigner()
    ball_assigner = BallAssigner()
    view_transformer = ViewTransformer()
    goalkeeper_display_lock = _GoalkeeperDisplayLock()

    props = get_video_properties(video_path)
    resolved_output_fps = _resolve_output_fps(props, output_fps, float(analysis_fps))
    annotation_states: List[Dict[str, Any]] = []
    processed_frames = 0
    rendered_frames = 0
    player_frame_detections = 0
    goalkeeper_frame_detections = 0
    referee_frame_detections = 0
    ball_detections = 0
    max_players_in_frame = 0
    team_ball_counts = {1: 0, 2: 0}
    last_team_control = 0
    identity_profiles: Dict[int, Dict[str, Any]] = {}
    raw_tracklet_records: List[Dict[str, Any]] = []
    identity_merge_conflicts = 0
    auto_identity_merge_map: Dict[int, int] = {}
    auto_identity_links: List[Dict[str, Any]] = []
    calibration_checked = False
    field_calibration_confidence = 0.0

    def process_batch(entries: List[tuple[int, np.ndarray]]) -> None:
        nonlocal processed_frames
        nonlocal rendered_frames
        nonlocal player_frame_detections
        nonlocal goalkeeper_frame_detections
        nonlocal referee_frame_detections
        nonlocal ball_detections
        nonlocal max_players_in_frame
        nonlocal last_team_control
        nonlocal identity_merge_conflicts
        nonlocal calibration_checked
        nonlocal field_calibration_confidence

        if not entries:
            return

        frames = [entry[1] for entry in entries]
        batch_tracks = tracker.get_object_tracks_for_frames(frames)
        identity_merge_conflicts += _apply_identity_merge_map(
            batch_tracks,
            normalized_identity_merge_map,
        )
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
                field_players = {
                    player_id: track
                    for player_id, track in player_track.items()
                    if track.get("role", "player") == "player"
                }
                if len(field_players) >= 2:
                    try:
                        team_assigner.assign_team_color(frame, field_players)
                    except Exception as exc:
                        warnings.append(f"Team color assignment failed on an early frame: {exc}")
                    break

        for local_idx, (source_frame_idx, frame) in enumerate(entries):
            player_track = batch_tracks["players"][local_idx]
            referee_track = batch_tracks["referees"][local_idx]
            ball_track = batch_tracks["ball"][local_idx]
            goalkeeper_display_lock.apply(
                player_track,
                source_frame_idx=int(source_frame_idx),
            )

            max_players_in_frame = max(max_players_in_frame, len(player_track))
            player_frame_detections += len(player_track)
            referee_frame_detections += len(referee_track)
            if ball_track:
                ball_detections += 1

            for player_id, track in player_track.items():
                role = track.get("role", "player")
                if role == "goalkeeper":
                    goalkeeper_frame_detections += 1

                team = 0
                if team_assigner.kmeans is not None and role == "player":
                    try:
                        team = team_assigner.get_player_team(frame, track["bbox"], player_id)
                    except Exception:
                        team = 0
                track["team"] = int(team)
                if role == "goalkeeper":
                    track["team_color"] = (255, 0, 255)
                else:
                    track["team_color"] = team_assigner.team_colors.get(team, (128, 128, 128))

                reid_embedding = track.get("reid_embedding")
                profile_role = str(track.get("display_role") or role)
                profile_team = (
                    int(track["display_team"])
                    if track.get("display_team") is not None
                    else int(team)
                )
                _update_identity_profile(
                    identity_profiles,
                    track_id=int(player_id),
                    role=profile_role,
                    team=profile_team,
                    sample_number=int(processed_frames + 1),
                    source_frame_idx=int(source_frame_idx),
                    bbox=track["bbox"],
                    raw_track_id=(
                        int(track["raw_track_id"])
                        if track.get("raw_track_id") is not None
                        else None
                    ),
                    object_type="player",
                    reid_embedding=reid_embedding,
                    color_signature=_extract_color_signature(frame, track["bbox"]),
                )

            for referee_id, referee in referee_track.items():
                role = str(referee.get("role", "referee"))
                reid_embedding = referee.get("reid_embedding")
                _update_identity_profile(
                    identity_profiles,
                    track_id=int(referee_id),
                    role=role,
                    team=0,
                    sample_number=int(processed_frames + 1),
                    source_frame_idx=int(source_frame_idx),
                    bbox=referee["bbox"],
                    raw_track_id=(
                        int(referee["raw_track_id"])
                        if referee.get("raw_track_id") is not None
                        else None
                    ),
                    object_type="referee",
                    reid_embedding=reid_embedding,
                    color_signature=_extract_color_signature(frame, referee["bbox"]),
                )

            ball_bbox = ball_track.get(1, {}).get("bbox") if ball_track else None
            assigned_player = -1
            if ball_bbox:
                assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1 and assigned_player in player_track:
                player_track[assigned_player]["has_ball"] = True
                last_team_control = int(player_track[assigned_player].get("team", 0))

            if last_team_control in team_ball_counts:
                team_ball_counts[last_team_control] += 1

            sample_number = int(processed_frames + 1)
            _append_raw_tracklet_records(
                raw_tracklet_records,
                sample_number=sample_number,
                source_frame_idx=int(source_frame_idx),
                object_type="player",
                tracks=player_track,
            )
            _append_raw_tracklet_records(
                raw_tracklet_records,
                sample_number=sample_number,
                source_frame_idx=int(source_frame_idx),
                object_type="referee",
                tracks=referee_track,
            )
            _append_raw_tracklet_records(
                raw_tracklet_records,
                sample_number=sample_number,
                source_frame_idx=int(source_frame_idx),
                object_type="ball",
                tracks=ball_track,
            )

            for player in player_track.values():
                player.pop("reid_embedding", None)
            for referee in referee_track.values():
                referee.pop("reid_embedding", None)

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

    auto_identity_merge_map, auto_identity_links = _build_auto_identity_merge_map(identity_profiles)
    if auto_identity_merge_map:
        annotation_tracks = {
            "players": [state["players"] for state in annotation_states],
            "referees": [state["referees"] for state in annotation_states],
        }
        identity_merge_conflicts += _apply_identity_merge_map(annotation_tracks, auto_identity_merge_map)
        warnings.append(
            f"Automatic identity reconciliation linked {len(auto_identity_merge_map)} tracklets into stable player IDs."
        )

    identity_debug = _build_identity_debug_report(
        profiles=identity_profiles,
        manual_merge_map=normalized_identity_merge_map,
        auto_merge_map=auto_identity_merge_map,
        auto_links=auto_identity_links,
    )
    render_audit_before = build_render_identity_audit(
        raw_tracklet_records,
        identity_debug,
        baseline_image=RUNPOD_BASELINE_IMAGE,
    )
    identity_events = build_identity_events(raw_tracklet_records, render_audit_before)
    correction_candidates = build_correction_candidates(render_audit_before, identity_debug)
    correction_plan = build_dry_run_correction_plan(
        correction_candidates=correction_candidates,
        render_audit=render_audit_before,
        identity_events=identity_events,
    )
    corrected_raw_tracklet_records, correction_applied = apply_safe_correction_plan_to_raw_records(
        raw_tracklet_records,
        correction_plan,
    )
    candidate_render_audit_after = build_render_identity_audit(
        corrected_raw_tracklet_records,
        identity_debug,
        baseline_image=RUNPOD_BASELINE_IMAGE,
    )
    if correction_applied.get("correction_applied") and post_fix_audit_improved(
        render_audit_before,
        candidate_render_audit_after,
    ):
        updated_annotation_tracks = _apply_safe_correction_plan_to_annotation_states(
            annotation_states,
            correction_plan,
        )
        correction_applied["kept"] = True
        correction_applied["candidate_correction_applied"] = True
        correction_applied["updated_annotation_track_count"] = updated_annotation_tracks
        raw_tracklet_records = corrected_raw_tracklet_records
        render_audit_after = candidate_render_audit_after
        warnings.append(
            f"Pre-render identity correction applied {correction_applied['applied_action_count']} safe display fixes."
        )
    else:
        if correction_applied.get("correction_applied"):
            correction_applied["candidate_correction_applied"] = True
            correction_applied["correction_applied"] = False
            correction_applied["kept"] = False
            correction_applied["rollback_reason"] = "post_fix_audit_not_improved"
        else:
            correction_applied["candidate_correction_applied"] = False
            correction_applied["kept"] = False
        correction_applied["updated_annotation_track_count"] = 0
        render_audit_after = render_audit_before
    vision_review_queue = build_vision_review_queue(
        correction_plan=correction_plan,
        render_audit_before=render_audit_before,
        render_audit_after=render_audit_after,
        correction_applied=correction_applied,
    )
    vision_evidence_paths = _write_vision_evidence_artifacts(
        job_id,
        video_path=video_path,
        raw_tracklet_records=raw_tracklet_records,
        vision_review_queue=vision_review_queue,
        render_audit=render_audit_after,
        output_dir=output_root,
        resize_width=int(resize_width),
    )
    player_crop_index_path = vision_evidence_paths.get("player_crop_index_json")
    if player_crop_index_path is not None:
        player_crop_index = json.loads(Path(player_crop_index_path).read_text(encoding="utf-8"))
    else:
        player_crop_index = {"schema_version": "1.0", "phase": "phase_5_crop_evidence", "cases": []}
    vision_review_results = build_vision_review_results(
        vision_review_queue=vision_review_queue,
        player_crop_index=player_crop_index,
    )

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
    final_render_identity_manifest = build_final_render_identity_manifest(
        render_audit_after=render_audit_after,
        correction_applied=correction_applied,
        vision_review_queue=vision_review_queue,
        vision_review_results=vision_review_results,
        rendered_output_frames=rendered_frames,
        baseline_image=RUNPOD_BASELINE_IMAGE,
    )

    if team_assigner.kmeans is None:
        warnings.append("Team assignment was unavailable because not enough players were detected in any sampled frame.")

    counted_possession = team_ball_counts[1] + team_ball_counts[2]
    possession_team1 = (team_ball_counts[1] / counted_possession * 100.0) if counted_possession else None
    possession_team2 = (team_ball_counts[2] / counted_possession * 100.0) if counted_possession else None
    possession_confidence = counted_possession / processed_frames if processed_frames else 0.0
    player_stats, identity_track_spans = _rebuild_player_outputs(annotation_states)

    report = {
        "job_id": job_id,
        "status": "completed",
        "stats": {
            "possession_team1": possession_team1,
            "possession_team2": possession_team2,
            "possession_confidence": possession_confidence,
            "total_passes": None,
            "total_shots": None,
            "player_count": len(player_stats),
            "max_players_in_frame": max_players_in_frame,
            "processed_frames": processed_frames,
            "rendered_output_frames": rendered_frames,
            "analysis_fps": float(analysis_fps),
            "output_fps": float(resolved_output_fps),
            "resize_width": int(resize_width),
            "tracker_backend": str(tracker.tracker_backend),
            "source_duration_seconds": props.get("duration_seconds"),
            "source_fps": props.get("fps"),
            "ball_detection_frames": ball_detections,
            "player_frame_detections": player_frame_detections,
            "goalkeeper_frame_detections": goalkeeper_frame_detections,
            "referee_frame_detections": referee_frame_detections,
            "identity_merge_conflicts": identity_merge_conflicts,
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
        "identity_quality": {
            "manual_merge_map": normalized_identity_merge_map,
            "auto_merge_map": auto_identity_merge_map,
            "auto_links": auto_identity_links,
            "track_spans": identity_track_spans,
            "merge_conflicts": identity_merge_conflicts,
            "goalkeeper_display_lock": goalkeeper_display_lock.summary(),
        },
        "pre_render_identity_correction": {
            "phase": "phase_1_audit_only",
            "baseline_image": RUNPOD_BASELINE_IMAGE,
            "known_bad_images": KNOWN_BAD_RUNPOD_IMAGES,
            "correction_applied": False,
        },
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
    report_paths.update(vision_evidence_paths)
    report["pre_render_identity_correction"].update(
        {
            "phase": "phase_7_final_render_integration",
            "vision_review_phase": "phase_6_vision_review_results",
            "correction_phase": "phase_3_safe_apply",
            "release_status": final_render_identity_manifest.get("release_status"),
            "output_identity_mode": final_render_identity_manifest.get("output_identity_mode"),
            "render_policy": final_render_identity_manifest.get("render_policy"),
            "final_manifest_validation": final_render_identity_manifest.get("validation", {}),
            "render_audit_verdict": render_audit_before.get("verdict"),
            "render_audit_score": render_audit_before.get("score"),
            "render_audit_summary": render_audit_before.get("summary", {}),
            "render_audit_after_verdict": render_audit_after.get("verdict"),
            "render_audit_after_score": render_audit_after.get("score"),
            "render_audit_after_summary": render_audit_after.get("summary", {}),
            "correction_candidate_count": correction_candidates.get("candidate_count", 0),
            "correction_plan_summary": correction_plan.get("summary", {}),
            "correction_plan_validation": correction_plan.get("validation", {}),
            "correction_applied": correction_applied.get("correction_applied", False),
            "correction_application": correction_applied,
            "vision_review_queue_count": vision_review_queue.get("case_count", 0),
            "vision_review_results_count": vision_review_results.get("case_count", 0),
            "vision_review_unresolved_count": vision_review_results.get("unresolved_count", 0),
            "vision_review_validation": vision_review_results.get("validation", {}),
            "vision_model_invoked": vision_review_results.get("vision_model_invoked", False),
            "crop_evidence_prepared": bool(vision_evidence_paths.get("player_crop_index_json")),
        }
    )
    identity_paths = _write_identity_artifacts(
        job_id,
        raw_tracklet_records=raw_tracklet_records,
        identity_debug=identity_debug,
        identity_events=identity_events,
        render_audit_before=render_audit_before,
        render_audit_after=render_audit_after,
        correction_candidates=correction_candidates,
        correction_plan=correction_plan,
        correction_applied=correction_applied,
        vision_review_queue=vision_review_queue,
        vision_review_results=vision_review_results,
        final_render_identity_manifest=final_render_identity_manifest,
        output_dir=output_root,
    )
    report_paths.update(identity_paths)
    report["artifacts"] = {
        "annotated_video": {"local_path": str(output_path), "content_type": "video/mp4"},
        "report_json": {"local_path": str(report_paths["report_json"]), "content_type": "application/json"},
        "report_csv": {"local_path": str(report_paths["report_csv"]), "content_type": "text/csv"},
        "raw_tracklets_jsonl": {"local_path": str(report_paths["raw_tracklets_jsonl"]), "content_type": "application/x-ndjson"},
        "identity_debug_json": {"local_path": str(report_paths["identity_debug_json"]), "content_type": "application/json"},
        "identity_events_json": {"local_path": str(report_paths["identity_events_json"]), "content_type": "application/json"},
        "render_audit_before_json": {"local_path": str(report_paths["render_audit_before_json"]), "content_type": "application/json"},
        "render_audit_after_json": {"local_path": str(report_paths["render_audit_after_json"]), "content_type": "application/json"},
        "correction_candidates_json": {"local_path": str(report_paths["correction_candidates_json"]), "content_type": "application/json"},
        "correction_plan_json": {"local_path": str(report_paths["correction_plan_json"]), "content_type": "application/json"},
        "correction_applied_json": {"local_path": str(report_paths["correction_applied_json"]), "content_type": "application/json"},
        "vision_review_queue_json": {"local_path": str(report_paths["vision_review_queue_json"]), "content_type": "application/json"},
        "vision_review_results_json": {"local_path": str(report_paths["vision_review_results_json"]), "content_type": "application/json"},
        "final_render_identity_manifest_json": {"local_path": str(report_paths["final_render_identity_manifest_json"]), "content_type": "application/json"},
        "player_crop_index_json": {"local_path": str(report_paths["player_crop_index_json"]), "content_type": "application/json"},
        "vision_contact_sheets_zip": {"local_path": str(report_paths["vision_contact_sheets_zip"]), "content_type": "application/zip"},
    }
    report_paths["report_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_paths["annotated_video"] = output_path

    return {"report": report, "paths": report_paths}
