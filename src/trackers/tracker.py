"""
Object detection and tracking using YOLOv8 + ByteTrack.
Based on: football_analysis_yolo by TrishamBP
"""

import os
import pickle
import inspect
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
import yaml

from src.field_filter import FieldFilter
from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, get_bbox_width


class Tracker:
    """Detect and track players, referees, and ball in football video."""

    PERSON_ROLES = {"player", "goalkeeper", "referee"}
    DEFAULT_BOTSORT_SETTINGS = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.30,
        "track_buffer": 1800,
        "match_thresh": 0.90,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.90,
        "appearance_thresh": 0.30,
        "with_reid": True,
        "model": "auto",
    }
    DEFAULT_FIELD_FILTER_SETTINGS = {
        "hsv_lower": [35, 27, 40],
        "hsv_upper": [95, 200, 200],
        "alpha_smooth": 0.7,
        "kernel_erode_size": 18,
        "kernel_dilate_size": 13,
        "kernel_close_size": 30,
        "morph_iterations": 2,
        "binary_threshold": 100,
        "min_area_ratio": 0.01,
        "poly_epsilon": 0.01,
        "bottom_zone_ratio": 0.80,
        "clipping_margin": 5,
        "thresh_clipping": -5.0,
        "thresh_bottom": 5.0,
        "thresh_standard": -2.0,
    }
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
        tracker_backend: Optional[str] = None,
        enable_field_filter: Optional[bool] = None,
    ):
        """Initialize tracker with YOLO model.

        Args:
            model_path: Path to trained YOLO model.
            detection_confidence: Base YOLO confidence threshold.
            person_confidence: Minimum confidence for player/referee/goalkeeper detections.
            ball_confidence: Minimum confidence for ball detections.
            role_stability_window: Number of recent role votes kept for each track ID.
            tracker_backend: "strongsort", "botsort", or "bytetrack".
            enable_field_filter: Remove non-pitch detections when True.
        """
        self.model = YOLO(model_path)
        self.tracker_backend = (tracker_backend or os.environ.get("TRACKER_BACKEND", "botsort")).lower()
        self.tracker = sv.ByteTrack(**self._bytetrack_kwargs())
        self.strongsort_tracker = None
        self.botsort_tracker_path = self._create_botsort_yaml()
        if enable_field_filter is None:
            enable_field_filter = os.environ.get("ENABLE_FIELD_FILTER", "1").strip() != "0"
        self.field_filter = (
            FieldFilter(self.DEFAULT_FIELD_FILTER_SETTINGS)
            if enable_field_filter
            else None
        )
        self.detection_confidence = detection_confidence
        self.person_confidence = person_confidence
        self.ball_confidence = ball_confidence
        self.role_stability_window = max(1, int(role_stability_window))
        self._role_votes: Dict[int, List[str]] = defaultdict(list)
        self._frame_index = 0
        self._next_display_id = 1
        self._display_id_by_tracker_id: Dict[int, int] = {}
        self._last_seen_by_display_id: Dict[int, Dict] = {}
        self.identity_lost_buffer = 900
        self.identity_long_buffer = 5400
        self.appearance_match_threshold = 0.38

    def _create_strongsort_tracker(self):
        """Create the optional StrongSORT/OSNet tracker."""
        try:
            import torch
            from boxmot.trackers.strongsort.strongsort import StrongSort
        except Exception as exc:
            raise RuntimeError(
                "StrongSORT tracking requires BoxMOT. Install boxmot and its "
                "runtime dependencies, or use tracker_backend='botsort'."
            ) from exc

        default_weights = Path("/app/tracking/weights/osnet_ain_x1_0_msmt17.pt")
        if not default_weights.exists():
            default_weights = Path("models/reid/osnet_ain_x1_0_msmt17.pt")

        weights_path = Path(
            os.environ.get(
                "STRONGSORT_REID_WEIGHTS",
                str(default_weights),
            )
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        device_name = os.environ.get(
            "STRONGSORT_DEVICE",
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )
        device = torch.device(device_name)
        half = (
            device.type != "cpu"
            and os.environ.get("STRONGSORT_HALF", "1").strip() != "0"
        )

        return StrongSort(
            reid_weights=weights_path,
            device=device,
            half=half,
            min_conf=float(os.environ.get("STRONGSORT_MIN_CONF", str(self.person_confidence))),
            max_cos_dist=float(os.environ.get("STRONGSORT_MAX_COS_DIST", "0.35")),
            max_iou_dist=float(os.environ.get("STRONGSORT_MAX_IOU_DIST", "0.95")),
            max_age=int(os.environ.get("STRONGSORT_MAX_AGE", "1800")),
            n_init=int(os.environ.get("STRONGSORT_N_INIT", "1")),
            nn_budget=int(os.environ.get("STRONGSORT_NN_BUDGET", "500")),
            mc_lambda=float(os.environ.get("STRONGSORT_MC_LAMBDA", "0.995")),
            ema_alpha=float(os.environ.get("STRONGSORT_EMA_ALPHA", "0.90")),
        )

    def _create_botsort_yaml(self) -> str:
        """Create a temporary BoT-SORT config for Ultralytics tracking."""
        config_dir = Path(tempfile.gettempdir()) / "football_tactical_ai"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "botsort_reid.yaml"
        config_path.write_text(
            yaml.safe_dump(self.DEFAULT_BOTSORT_SETTINGS, sort_keys=False),
            encoding="utf-8",
        )
        return str(config_path)

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

    def _extract_appearance(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract a lightweight HSV color fingerprint for long-term relinking."""
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(frame_w - 1, int(bbox[0])))
        y1 = max(0, min(frame_h - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(frame_w, int(bbox[2])))
        y2 = max(y1 + 1, min(frame_h, int(bbox[3])))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        jersey_crop = crop[: max(1, int(crop.shape[0] * 0.60)), :]
        hsv_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv_crop],
            [0, 1],
            None,
            [16, 8],
            [0, 180, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten()
        if not np.isfinite(hist).all() or float(np.sum(hist)) == 0.0:
            return None
        return hist.astype(float)

    def _appearance_distance(
        self,
        appearance_a: Optional[np.ndarray],
        appearance_b: Optional[np.ndarray],
    ) -> float:
        """Compare two normalized HSV histograms."""
        if appearance_a is None or appearance_b is None:
            return 1.0
        return float(cv2.compareHist(
            appearance_a.astype(np.float32),
            appearance_b.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        ))

    def _match_previous_display_id(
        self,
        bbox: np.ndarray,
        role: str,
        appearance: Optional[np.ndarray],
        used_display_ids: Set[int],
    ) -> Optional[int]:
        """Relink a new tracker ID to a previous display ID when likely same person."""
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
            if age <= 0 or age > self.identity_long_buffer:
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
            appearance_distance = self._appearance_distance(
                appearance,
                np.asarray(state["appearance"], dtype=float)
                if state.get("appearance") is not None
                else None,
            )

            appearance_match = appearance_distance <= self.appearance_match_threshold
            loose_position_match = normalized_distance <= 2.25 or iou >= 0.02

            if age > self.identity_lost_buffer:
                if not appearance_match and not loose_position_match:
                    continue
                score = (
                    (0.65 * (1.0 - appearance_distance))
                    + (0.35 * max(0.0, 1.0 - (normalized_distance / 2.25)))
                    + (0.20 * iou)
                    - (0.0002 * age)
                )
                if score > best_score:
                    best_score = score
                    best_display_id = int(display_id)
                continue

            if iou < 0.08 and normalized_distance > 1.50 and not appearance_match:
                continue

            score = (
                (2.0 * iou)
                - normalized_distance
                - (0.002 * age)
                + (0.2 * (1.0 - appearance_distance))
            )
            if score > best_score:
                best_score = score
                best_display_id = int(display_id)

        return best_display_id

    def _display_id_for_track(
        self,
        raw_track_id: int,
        bbox: np.ndarray,
        role: str,
        appearance: Optional[np.ndarray],
        used_display_ids: Set[int],
    ) -> int:
        """Map noisy ByteTrack IDs to stable user-facing IDs."""
        if raw_track_id in self._display_id_by_tracker_id:
            display_id = self._display_id_by_tracker_id[raw_track_id]
            if display_id not in used_display_ids:
                return display_id

        display_id = self._match_previous_display_id(
            bbox,
            role,
            appearance,
            used_display_ids,
        )
        if display_id is None:
            display_id = self._next_display_id
            self._next_display_id += 1

        self._display_id_by_tracker_id[raw_track_id] = display_id
        return display_id

    def _remember_display_track(
        self,
        display_id: int,
        bbox: np.ndarray,
        role: str,
        appearance: Optional[np.ndarray],
    ) -> None:
        """Store last position for lightweight ID relinking."""
        previous = self._last_seen_by_display_id.get(int(display_id), {})
        previous_appearance = previous.get("appearance")
        if appearance is None and previous_appearance is not None:
            stored_appearance = previous_appearance
        elif appearance is not None and previous_appearance is not None:
            stored_appearance = (
                0.70 * np.asarray(previous_appearance, dtype=float)
                + 0.30 * appearance
            ).tolist()
        elif appearance is not None:
            stored_appearance = appearance.tolist()
        else:
            stored_appearance = None

        self._last_seen_by_display_id[int(display_id)] = {
            "bbox": np.asarray(bbox, dtype=float).tolist(),
            "role": role,
            "appearance": stored_appearance,
            "frame_index": int(self._frame_index),
        }

    def _prune_identity_cache(self) -> None:
        """Remove old identity candidates."""
        expired = [
            display_id
            for display_id, state in self._last_seen_by_display_id.items()
            if self._frame_index - int(state["frame_index"]) > self.identity_long_buffer
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

    def _get_object_tracks_with_botsort(self, frames: List[np.ndarray]) -> Dict:
        """Track frames with Ultralytics BoT-SORT + ReID settings."""
        results = self.model.track(
            source=frames,
            conf=self.detection_confidence,
            iou=0.7,
            tracker=self.botsort_tracker_path,
            persist=True,
            verbose=False,
            stream=False,
            agnostic_nms=True,
        )

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for result_index, result in enumerate(results):
            frame = frames[result_index]
            cls_names = getattr(result, "names", {}) or {}
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None:
                self._prune_identity_cache()
                self._frame_index += 1
                continue

            xyxy_values = boxes.xyxy.cpu().numpy()
            class_values = boxes.cls.int().cpu().numpy()
            confidence_values = boxes.conf.cpu().numpy()
            track_id_values = boxes.id.int().cpu().numpy()

            people_items = []
            best_ball = None
            best_ball_confidence = -1.0

            for bbox, class_id, confidence, raw_track_id in zip(
                xyxy_values,
                class_values,
                confidence_values,
                track_id_values,
            ):
                role = self._normalize_class_name(cls_names.get(int(class_id), ""))
                confidence = float(confidence)

                if role in self.PERSON_ROLES:
                    if confidence < self.person_confidence:
                        continue
                    people_items.append(
                        {
                            "xyxy": np.asarray(bbox, dtype=float),
                            "raw_track_id": int(raw_track_id),
                            "role": role,
                            "confidence": confidence,
                        }
                    )
                elif role == "ball" and confidence >= self.ball_confidence:
                    if confidence > best_ball_confidence:
                        best_ball_confidence = confidence
                        best_ball = np.asarray(bbox, dtype=float)

            if self.field_filter is not None:
                people_items = self.field_filter.filter_track_items(frame, people_items)

            used_display_ids: Set[int] = set()
            for item in people_items:
                bbox = np.asarray(item["xyxy"], dtype=float)
                raw_track_id = int(item["raw_track_id"])
                raw_role = str(item["role"])
                stable_role = self._stable_role_for_track(raw_track_id, raw_role)
                appearance = self._extract_appearance(frame, bbox)
                display_id = self._display_id_for_track(
                    raw_track_id=raw_track_id,
                    bbox=bbox,
                    role=stable_role,
                    appearance=appearance,
                    used_display_ids=used_display_ids,
                )
                used_display_ids.add(display_id)
                self._remember_display_track(display_id, bbox, stable_role, appearance)

                track_data = {
                    "bbox": bbox.tolist(),
                    "role": stable_role,
                    "detected_role": raw_role,
                    "raw_track_id": raw_track_id,
                    "confidence": float(item["confidence"]),
                }
                if stable_role == "referee":
                    tracks["referees"][result_index][display_id] = track_data
                else:
                    tracks["players"][result_index][display_id] = track_data

            if best_ball is not None:
                tracks["ball"][result_index][1] = {"bbox": best_ball.tolist()}

            self._prune_identity_cache()
            self._frame_index += 1

        return tracks

    def _get_object_tracks_with_strongsort(self, frames: List[np.ndarray]) -> Dict:
        """Track frames with StrongSORT + OSNet ReID for maximum ID stability."""
        if self.strongsort_tracker is None:
            self.strongsort_tracker = self._create_strongsort_tracker()

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }
        role_to_class = {"player": 0, "goalkeeper": 1, "referee": 2}
        class_to_role = {value: key for key, value in role_to_class.items()}

        for frame_num, detection in enumerate(detections):
            frame = frames[frame_num]
            cls_names = detection.names
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_supervision = self._filter_detections(detection_supervision, cls_names)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            class_ids = (
                detection_supervision.class_id
                if detection_supervision.class_id is not None
                else np.array([], dtype=int)
            )
            confidences = (
                detection_supervision.confidence
                if detection_supervision.confidence is not None
                else np.ones(len(detection_supervision), dtype=float)
            )

            people_items = []
            best_ball = None
            best_ball_confidence = -1.0
            for bbox, class_id, confidence in zip(
                np.asarray(detection_supervision.xyxy),
                class_ids,
                confidences,
            ):
                role = self._normalize_class_name(cls_names.get(int(class_id), ""))
                confidence = float(confidence)
                if role in self.PERSON_ROLES:
                    if confidence < self.person_confidence:
                        continue
                    people_items.append(
                        {
                            "xyxy": np.asarray(bbox, dtype=float),
                            "role": role,
                            "confidence": confidence,
                        }
                    )
                elif role == "ball" and confidence >= self.ball_confidence:
                    if confidence > best_ball_confidence:
                        best_ball_confidence = confidence
                        best_ball = np.asarray(bbox, dtype=float)

            if self.field_filter is not None:
                people_items = self.field_filter.filter_track_items(frame, people_items)

            if people_items:
                dets = np.asarray(
                    [
                        [
                            *item["xyxy"].tolist(),
                            float(item["confidence"]),
                            float(role_to_class.get(str(item["role"]), 0)),
                        ]
                        for item in people_items
                    ],
                    dtype=float,
                )
            else:
                dets = np.empty((0, 6), dtype=float)

            if len(dets) > 0:
                strongsort_embeddings = self.strongsort_tracker.model.get_features(dets[:, 0:4], frame)
            else:
                strongsort_embeddings = np.empty((0, 0), dtype=float)

            strongsort_outputs = self.strongsort_tracker.update(dets, frame, embs=strongsort_embeddings)
            if strongsort_outputs is None:
                strongsort_outputs = np.empty((0, 8), dtype=float)

            used_display_ids: Set[int] = set()
            for output in np.asarray(strongsort_outputs):
                if len(output) < 8:
                    continue
                bbox = np.asarray(output[0:4], dtype=float)
                raw_track_id = int(output[4])
                confidence = float(output[5])
                class_id = int(output[6])
                det_ind = int(output[7])
                if 0 <= det_ind < len(people_items):
                    raw_role = str(people_items[det_ind]["role"])
                else:
                    raw_role = class_to_role.get(class_id, "player")
                reid_embedding = None
                if 0 <= det_ind < len(strongsort_embeddings):
                    reid_embedding = np.asarray(strongsort_embeddings[det_ind], dtype=float)

                stable_role = self._stable_role_for_track(raw_track_id, raw_role)
                appearance = self._extract_appearance(frame, bbox)
                display_id = self._display_id_for_track(
                    raw_track_id=raw_track_id,
                    bbox=bbox,
                    role=stable_role,
                    appearance=appearance,
                    used_display_ids=used_display_ids,
                )
                used_display_ids.add(display_id)
                self._remember_display_track(display_id, bbox, stable_role, appearance)

                track_data = {
                    "bbox": bbox.tolist(),
                    "role": stable_role,
                    "detected_role": raw_role,
                    "raw_track_id": raw_track_id,
                    "confidence": confidence,
                    "tracker_backend": "strongsort",
                }
                if reid_embedding is not None and np.isfinite(reid_embedding).all():
                    track_data["reid_embedding"] = reid_embedding.tolist()
                if stable_role == "referee":
                    tracks["referees"][frame_num][display_id] = track_data
                else:
                    tracks["players"][frame_num][display_id] = track_data

            if best_ball is not None:
                tracks["ball"][frame_num][1] = {"bbox": best_ball.tolist()}

            self._prune_identity_cache()
            self._frame_index += 1

        return tracks

    def get_object_tracks_for_frames(self, frames: List[np.ndarray]) -> Dict:
        """Track a batch of frames using the current tracker state.

        This is intended for long videos where the caller processes sampled
        frames in batches instead of keeping the full match in memory.
        """
        if self.tracker_backend == "strongsort":
            return self._get_object_tracks_with_strongsort(frames)

        if self.tracker_backend == "botsort":
            try:
                return self._get_object_tracks_with_botsort(frames)
            except Exception:
                self.tracker_backend = "bytetrack"

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
                frame = frames[frame_num]
                appearance = self._extract_appearance(frame, np.asarray(bbox, dtype=float))
                display_id = self._display_id_for_track(
                    raw_track_id=raw_track_id,
                    bbox=np.asarray(bbox, dtype=float),
                    role=stable_role,
                    appearance=appearance,
                    used_display_ids=used_display_ids,
                )
                used_display_ids.add(display_id)
                self._remember_display_track(
                    display_id,
                    np.asarray(bbox),
                    stable_role,
                    appearance,
                )
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
