"""
Object detection and tracking using YOLOv8 + ByteTrack.
Based on: football_analysis_yolo by TrishamBP
"""

import os
import pickle
import inspect
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, get_bbox_width


class Tracker:
    """Detect and track players, referees, and ball in football video."""

    PERSON_ROLES = {"player", "goalkeeper", "referee"}
    CLASS_ALIASES = {
        "players": "player",
        "football-player": "player",
        "football_player": "player",
        "football player": "player",
        "goalkeeper": "goalkeeper",
        "goal keeper": "goalkeeper",
        "keeper": "goalkeeper",
        "ref": "referee",
        "referee": "referee",
        "official": "referee",
        "match official": "referee",
        "assistant referee": "referee",
        "assistant_referee": "referee",
        "linesman": "referee",
        "line referee": "referee",
        "ball": "ball",
        "football": "ball",
        "sports ball": "ball",
        "sports_ball": "ball",
    }

    def __init__(
        self,
        model_path: str,
        detection_confidence: float = 0.12,
        person_confidence: float = 0.25,
        ball_confidence: float = 0.08,
        role_stability_window: int = 12,
    ):
        """Initialize tracker with YOLO model.

        Args:
            model_path: Path to trained YOLO model.
            detection_confidence: Base YOLO confidence threshold.
            person_confidence: Minimum confidence for player/referee/goalkeeper detections.
            ball_confidence: Minimum confidence for ball detections.
            role_stability_window: Number of recent role votes kept for each track ID.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(**self._bytetrack_kwargs())
        self.detection_confidence = detection_confidence
        self.person_confidence = person_confidence
        self.ball_confidence = ball_confidence
        self.role_stability_window = max(1, int(role_stability_window))
        self._role_votes: Dict[int, List[str]] = defaultdict(list)
        self._frame_index = 0
        self._next_display_id = 1
        self._display_id_by_tracker_id: Dict[int, int] = {}
        self._last_seen_by_display_id: Dict[int, Dict] = {}
        self.identity_lost_buffer = 90

    @staticmethod
    def _bytetrack_kwargs_for_signature(parameter_names: Set[str]) -> Dict:
        """Build ByteTrack kwargs for old and new supervision versions."""
        desired = {
            "track_activation_threshold": 0.2,
            "lost_track_buffer": 90,
            "minimum_matching_threshold": 0.7,
            "minimum_consecutive_frames": 1,
            "track_thresh": 0.2,
            "track_buffer": 90,
            "match_thresh": 0.7,
            "frame_rate": 30,
        }
        return {
            name: value
            for name, value in desired.items()
            if name in parameter_names
        }

    def _bytetrack_kwargs(self) -> Dict:
        """Return kwargs accepted by the installed supervision ByteTrack."""
        try:
            signature = inspect.signature(sv.ByteTrack)
        except (TypeError, ValueError):
            return {}
        return self._bytetrack_kwargs_for_signature(set(signature.parameters))

    def add_position_to_tracks(self, tracks: Dict) -> None:
        """Add foot/center position to all tracked objects.

        Args:
            tracks: Tracking dictionary.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions: List[Dict]) -> List[Dict]:
        """Interpolate missing ball positions using pandas.

        Args:
            ball_positions: List of ball position dicts per frame.

        Returns:
            Interpolated ball positions.
        """
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames: List[np.ndarray], batch_size: int = 20) -> List:
        """Detect objects in frames using YOLO.

        Args:
            frames: List of video frames.
            batch_size: Batch size for inference.

        Returns:
            List of detection results.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=self.detection_confidence,
                verbose=False,
            )
            detections += detections_batch
        return detections

    def _normalize_class_name(self, name: str) -> str:
        """Normalize model class labels into stable project roles."""
        normalized = str(name).strip().lower().replace("-", " ").replace("_", " ")
        return self.CLASS_ALIASES.get(normalized, normalized)

    def _class_id_for_role(self, cls_names_inv: Dict[str, int], role: str) -> Optional[int]:
        """Find the model class id for a normalized role."""
        for class_name, class_id in cls_names_inv.items():
            if self._normalize_class_name(class_name) == role:
                return int(class_id)
        return None

    def _bbox_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute intersection-over-union for two xyxy boxes."""
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
            0.0,
            float(box_a[3] - box_a[1]),
        )
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
            0.0,
            float(box_b[3] - box_b[1]),
        )
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _role_for_tracked_bbox(
        self,
        tracked_bbox: np.ndarray,
        original_boxes: np.ndarray,
        original_roles: List[str],
    ) -> str:
        """Recover the original model role for a tracked bbox."""
        if len(original_boxes) == 0:
            return "player"
        best_idx = 0
        best_iou = -1.0
        for idx, original_box in enumerate(original_boxes):
            iou = self._bbox_iou(np.asarray(tracked_bbox), np.asarray(original_box))
            if iou > best_iou:
                best_idx = idx
                best_iou = iou
        return original_roles[best_idx] if best_iou >= 0.5 else "player"

    def _stable_role_for_track(self, track_id: int, detected_role: str) -> str:
        """Return a smoothed role for a tracked person."""
        votes = self._role_votes[int(track_id)]
        votes.append(detected_role)
        if len(votes) > self.role_stability_window:
            del votes[0: len(votes) - self.role_stability_window]
        return Counter(votes).most_common(1)[0][0]

    def _bbox_center(self, bbox: np.ndarray) -> np.ndarray:
        """Return center point for an xyxy bbox."""
        return np.array(
            [
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                (float(bbox[1]) + float(bbox[3])) / 2.0,
            ],
            dtype=float,
        )

    def _role_compatible_for_identity(self, previous_role: str, current_role: str) -> bool:
        """Allow relinking only between roles that can reasonably flicker."""
        if previous_role == current_role:
            return True
        return {previous_role, current_role}.issubset({"player", "goalkeeper"})

    def _match_previous_display_id(
        self,
        bbox: np.ndarray,
        role: str,
        used_display_ids: Set[int],
    ) -> Optional[int]:
        """Relink a new ByteTrack ID to a recent display ID when likely same person."""
        best_display_id = None
        best_score = -999.0
        current_center = self._bbox_center(bbox)
        current_width = max(1.0, float(bbox[2] - bbox[0]))
        current_height = max(1.0, float(bbox[3] - bbox[1]))
        current_scale = max(current_width, current_height)

        for display_id, state in self._last_seen_by_display_id.items():
            if display_id in used_display_ids:
                continue
            age = self._frame_index - int(state["frame_index"])
            if age <= 0 or age > self.identity_lost_buffer:
                continue
            previous_role = str(state.get("role", "player"))
            if not self._role_compatible_for_identity(previous_role, role):
                continue

            previous_bbox = np.asarray(state["bbox"], dtype=float)
            previous_center = self._bbox_center(previous_bbox)
            previous_width = max(1.0, float(previous_bbox[2] - previous_bbox[0]))
            previous_height = max(1.0, float(previous_bbox[3] - previous_bbox[1]))
            previous_scale = max(previous_width, previous_height)
            scale = max(current_scale, previous_scale)
            center_distance = float(np.linalg.norm(current_center - previous_center))
            normalized_distance = center_distance / scale
            iou = self._bbox_iou(bbox, previous_bbox)

            if iou < 0.08 and normalized_distance > 0.95:
                continue

            score = (2.0 * iou) - normalized_distance - (0.02 * age)
            if score > best_score:
                best_score = score
                best_display_id = int(display_id)

        return best_display_id

    def _display_id_for_track(
        self,
        raw_track_id: int,
        bbox: np.ndarray,
        role: str,
        used_display_ids: Set[int],
    ) -> int:
        """Map noisy ByteTrack IDs to stable user-facing IDs."""
        if raw_track_id in self._display_id_by_tracker_id:
            display_id = self._display_id_by_tracker_id[raw_track_id]
            if display_id not in used_display_ids:
                return display_id

        display_id = self._match_previous_display_id(bbox, role, used_display_ids)
        if display_id is None:
            display_id = self._next_display_id
            self._next_display_id += 1

        self._display_id_by_tracker_id[raw_track_id] = display_id
        return display_id

    def _remember_display_track(self, display_id: int, bbox: np.ndarray, role: str) -> None:
        """Store last position for lightweight ID relinking."""
        self._last_seen_by_display_id[int(display_id)] = {
            "bbox": np.asarray(bbox, dtype=float).tolist(),
            "role": role,
            "frame_index": int(self._frame_index),
        }

    def _prune_identity_cache(self) -> None:
        """Remove old identity candidates."""
        expired = [
            display_id
            for display_id, state in self._last_seen_by_display_id.items()
            if self._frame_index - int(state["frame_index"]) > self.identity_lost_buffer
        ]
        for display_id in expired:
            del self._last_seen_by_display_id[display_id]

    def _filter_detections(self, detections: sv.Detections, cls_names: Dict[int, str]) -> sv.Detections:
        """Remove low-confidence detections using class-specific thresholds."""
        if detections.class_id is None or len(detections) == 0:
            return detections

        confidence = detections.confidence
        if confidence is None:
            confidence = np.ones(len(detections), dtype=float)

        keep_mask = []
        for class_id, conf in zip(detections.class_id, confidence):
            role = self._normalize_class_name(cls_names.get(int(class_id), ""))
            if role in self.PERSON_ROLES:
                keep_mask.append(float(conf) >= self.person_confidence)
            elif role == "ball":
                keep_mask.append(float(conf) >= self.ball_confidence)
            else:
                keep_mask.append(True)

        return detections[np.array(keep_mask, dtype=bool)]

    def get_object_tracks_for_frames(self, frames: List[np.ndarray]) -> Dict:
        """Track a batch of frames using the current tracker state.

        This is intended for long videos where the caller processes sampled
        frames in batches instead of keeping the full match in memory.
        """
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_supervision = self._filter_detections(detection_supervision, cls_names)
            class_ids = (
                detection_supervision.class_id
                if detection_supervision.class_id is not None
                else np.array([], dtype=int)
            )
            normalized_roles = [
                self._normalize_class_name(cls_names.get(int(class_id), ""))
                for class_id in class_ids
            ]
            person_mask = np.array(
                [role in self.PERSON_ROLES for role in normalized_roles],
                dtype=bool,
            )
            ball_mask = np.array(
                [role == "ball" for role in normalized_roles],
                dtype=bool,
            )

            person_detections = detection_supervision[person_mask]
            ball_detections = detection_supervision[ball_mask]
            person_class_ids = (
                person_detections.class_id
                if person_detections.class_id is not None
                else np.array([], dtype=int)
            )
            original_boxes = np.asarray(person_detections.xyxy).copy()
            original_roles = [
                self._normalize_class_name(cls_names.get(int(class_id), ""))
                for class_id in person_class_ids
            ]

            common_person_class_id = (
                self._class_id_for_role(cls_names_inv, "person")
                or self._class_id_for_role(cls_names_inv, "player")
            )
            if common_person_class_id is None and len(person_class_ids) > 0:
                common_person_class_id = int(person_class_ids[0])
            if common_person_class_id is not None and person_detections.class_id is not None:
                person_detections.class_id[:] = int(common_person_class_id)

            detection_with_tracks = self.tracker.update_with_detections(person_detections)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            tracked_xyxy = getattr(detection_with_tracks, "xyxy", [])
            tracked_class_ids = getattr(detection_with_tracks, "class_id", None)
            tracked_ids = getattr(detection_with_tracks, "tracker_id", None)

            if tracked_class_ids is None or tracked_ids is None:
                tracked_xyxy = []
                tracked_class_ids = []
                tracked_ids = []

            used_display_ids: Set[int] = set()
            for bbox, _cls_id, track_id in zip(tracked_xyxy, tracked_class_ids, tracked_ids):
                if track_id is None:
                    continue
                raw_role = self._role_for_tracked_bbox(np.asarray(bbox), original_boxes, original_roles)
                if raw_role not in self.PERSON_ROLES:
                    raw_role = "player"
                raw_track_id = int(track_id)
                stable_role = self._stable_role_for_track(raw_track_id, raw_role)
                display_id = self._display_id_for_track(
                    raw_track_id=raw_track_id,
                    bbox=np.asarray(bbox, dtype=float),
                    role=stable_role,
                    used_display_ids=used_display_ids,
                )
                used_display_ids.add(display_id)
                self._remember_display_track(display_id, np.asarray(bbox), stable_role)
                track_data = {
                    "bbox": bbox.tolist(),
                    "role": stable_role,
                    "detected_role": raw_role,
                    "raw_track_id": raw_track_id,
                }

                if stable_role == "referee":
                    tracks["referees"][frame_num][display_id] = track_data
                else:
                    tracks["players"][frame_num][display_id] = track_data

            if len(ball_detections) > 0:
                ball_confidences = ball_detections.confidence
                if ball_confidences is None:
                    best_ball_idx = 0
                else:
                    best_ball_idx = int(np.argmax(ball_confidences))
                tracks["ball"][frame_num][1] = {
                    "bbox": ball_detections.xyxy[best_ball_idx].tolist()
                }

            self._prune_identity_cache()
            self._frame_index += 1

        return tracks

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None
    ) -> Dict:
        """Get tracked objects across frames.

        Args:
            frames: List of video frames.
            read_from_stub: Read from cached file.
            stub_path: Path to stub file.

        Returns:
            Dictionary with tracks for players, referees, ball.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracks = self.get_object_tracks_for_frames(frames)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame: np.ndarray, bbox: list, color: tuple, track_id: Optional[int] = None) -> np.ndarray:
        """Draw ellipse at player foot position.

        Args:
            frame: Video frame.
            bbox: Bounding box [x1, y1, x2, y2].
            color: RGB color tuple.
            track_id: Player track ID.

        Returns:
            Annotated frame.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw track ID label
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame: np.ndarray, bbox: list, color: tuple) -> np.ndarray:
        """Draw triangle above object (for ball or ball possessor).

        Args:
            frame: Video frame.
            bbox: Bounding box.
            color: RGB color.

        Returns:
            Annotated frame.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame: np.ndarray, frame_num: int, team_ball_control: np.ndarray) -> np.ndarray:
        """Draw team ball control percentage overlay.

        Args:
            frame: Video frame.
            frame_num: Current frame number.
            team_ball_control: Array of team ball control per frame.

        Returns:
            Annotated frame.
        """
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get number of times each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        if team_1_num_frames + team_2_num_frames > 0:
            team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
            team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)
        else:
            team_1 = team_2 = 0.5

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(
        self,
        video_frames: List[np.ndarray],
        tracks: Dict,
        team_ball_control: np.ndarray
    ) -> List[np.ndarray]:
        """Draw all annotations on video frames.

        Args:
            video_frames: List of original frames.
            tracks: Tracking dictionary.
            team_ball_control: Ball control array.

        Returns:
            List of annotated frames.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                fallback_color = (
                    (255, 0, 255)
                    if player.get("role") == "goalkeeper"
                    else (0, 0, 255)
                )
                color = player.get("team_color", fallback_color)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
