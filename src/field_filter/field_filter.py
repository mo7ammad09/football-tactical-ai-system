"""
HSV pitch filtering for football detections.

Adapted from the MIT-licensed project:
https://github.com/SimoneFaraulo/Soccer-Player-Tracking-and-Region-Based-Behavior-Analysis
Original copyright (c) 2026 Simone Faraulo.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


class FieldFilter:
    """Filter detections whose foot point is outside the visible pitch."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize HSV and morphology settings.

        Args:
            settings: Optional threshold/morphology overrides.
        """
        if settings is None:
            settings = {}

        hsv_lower = settings.get("hsv_lower", [35, 27, 40])
        hsv_upper = settings.get("hsv_upper", [95, 200, 200])
        self.lower_green = np.array(hsv_lower, dtype=np.uint8)
        self.upper_green = np.array(hsv_upper, dtype=np.uint8)

        self.kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self._kernel_size(settings.get("kernel_erode_size", 15)),
        )
        self.kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            self._kernel_size(settings.get("kernel_dilate_size", 13)),
        )
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self._kernel_size(settings.get("kernel_close_size", 30)),
        )
        self.morph_iterations = int(settings.get("morph_iterations", 2))
        self.alpha = float(settings.get("alpha_smooth", 0.7))
        self.binary_threshold = int(settings.get("binary_threshold", 100))
        self.min_area_ratio = float(settings.get("min_area_ratio", 0.01))
        self.poly_epsilon = float(settings.get("poly_epsilon", 0.01))
        self.bottom_zone_ratio = float(settings.get("bottom_zone_ratio", 0.80))
        self.clipping_margin = int(settings.get("clipping_margin", 5))
        self.thresh_clipping = float(settings.get("thresh_clipping", -5.0))
        self.thresh_bottom = float(settings.get("thresh_bottom", 5.0))
        self.thresh_standard = float(settings.get("thresh_standard", -2.0))

        self.prev_mask: Optional[np.ndarray] = None
        self.last_contour: Optional[np.ndarray] = None

    def _kernel_size(self, value: int) -> Tuple[int, int]:
        """Return a valid OpenCV kernel size."""
        size = max(1, int(value))
        return size, size

    def get_field_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract a smoothed pitch contour from a frame."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        current_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        if self.prev_mask is None:
            self.prev_mask = current_mask.astype(np.float32)
        else:
            cv2.accumulateWeighted(current_mask, self.prev_mask, 1.0 - self.alpha)

        _, processed_mask = cv2.threshold(
            self.prev_mask,
            self.binary_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        processed_mask = processed_mask.astype(np.uint8)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, self.kernel_close)
        processed_mask = cv2.erode(
            processed_mask,
            self.kernel_erode,
            iterations=self.morph_iterations,
        )
        processed_mask = cv2.dilate(
            processed_mask,
            self.kernel_dilate,
            iterations=self.morph_iterations,
        )

        contours, _ = cv2.findContours(
            processed_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        min_area = frame.shape[0] * frame.shape[1] * self.min_area_ratio
        significant = [contour for contour in contours if cv2.contourArea(contour) > min_area]
        if not significant:
            self.last_contour = None
            return None

        hull = cv2.convexHull(np.vstack(significant))
        epsilon = self.poly_epsilon * cv2.arcLength(hull, True)
        self.last_contour = cv2.approxPolyDP(hull, epsilon, True)
        return self.last_contour

    def is_bbox_on_field(self, frame: np.ndarray, bbox: Iterable[float]) -> bool:
        """Return True when the detection foot point is inside/near the pitch."""
        contour = self.get_field_contour(frame)
        if contour is None:
            return True

        x1, y1, x2, y2 = [float(value) for value in bbox]
        feet_x = (x1 + x2) / 2.0
        feet_y = y2
        frame_h = frame.shape[0]
        bottom_limit_y = frame_h * self.bottom_zone_ratio
        distance = cv2.pointPolygonTest(contour, (feet_x, feet_y), True)

        if feet_y >= frame_h - self.clipping_margin:
            threshold = self.thresh_clipping
        elif feet_y > bottom_limit_y:
            threshold = self.thresh_bottom
        else:
            threshold = self.thresh_standard

        return bool(distance >= threshold)

    def filter_track_items(self, frame: np.ndarray, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detection dictionaries that contain an xyxy bbox."""
        contour = self.get_field_contour(frame)
        if contour is None:
            return items

        filtered = []
        frame_h = frame.shape[0]
        bottom_limit_y = frame_h * self.bottom_zone_ratio

        for item in items:
            x1, _y1, x2, y2 = [float(value) for value in item["xyxy"]]
            feet_x = (x1 + x2) / 2.0
            feet_y = y2
            distance = cv2.pointPolygonTest(contour, (feet_x, feet_y), True)

            if feet_y >= frame_h - self.clipping_margin:
                threshold = self.thresh_clipping
            elif feet_y > bottom_limit_y:
                threshold = self.thresh_bottom
            else:
                threshold = self.thresh_standard

            if distance >= threshold:
                filtered.append(item)

        return filtered
