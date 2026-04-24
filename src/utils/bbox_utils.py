"""
Utility functions for bounding box operations.
Taken from: football_analysis_yolo (TrishamBP)
"""

from typing import Tuple


def get_center_of_bbox(bbox: list) -> Tuple[int, int]:
    """Get center point of bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: list) -> float:
    """Get width of bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        Width value.
    """
    return bbox[2] - bbox[0]


def measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        p1: (x, y) point 1
        p2: (x, y) point 2

    Returns:
        Euclidean distance.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate x and y distance separately.

    Args:
        p1: (x, y) point 1
        p2: (x, y) point 2

    Returns:
        (dx, dy)
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox: list) -> Tuple[int, int]:
    """Get foot position (bottom center) of bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (x, y) foot position.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
