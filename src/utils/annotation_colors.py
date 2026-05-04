"""Helpers for stable annotation colors."""

from __future__ import annotations

from typing import Any


PLAYER_FALLBACK_COLOR: tuple[int, int, int] = (128, 128, 128)
GOALKEEPER_COLOR: tuple[int, int, int] = (255, 0, 255)


def normalize_color(
    color: Any,
    fallback: tuple[int, int, int] = PLAYER_FALLBACK_COLOR,
) -> tuple[int, int, int]:
    """Convert color-like values into OpenCV BGR tuples.

    Args:
        color: Any list/tuple-like BGR value.
        fallback: Color returned when color is missing or malformed.

    Returns:
        A 3-channel BGR tuple.
    """
    if color is None:
        return fallback
    try:
        values = list(color)
        if len(values) < 3:
            return fallback
        return tuple(int(max(0, min(255, value))) for value in values[:3])
    except Exception:
        return fallback


def resolve_player_annotation_color(player: dict[str, Any]) -> tuple[int, int, int]:
    """Return the client-facing player annotation color.

    The key detail is that an explicit ``display_color=None`` must not mask a
    valid ``team_color``. This prevents corrected non-goalkeepers from falling
    back to the legacy red player color after the GK display is removed.
    """
    display_role = str(player.get("display_role") or player.get("role") or "player")
    fallback = GOALKEEPER_COLOR if display_role == "goalkeeper" else PLAYER_FALLBACK_COLOR
    color = player.get("display_color")
    if color is None:
        color = player.get("team_color")
    return normalize_color(color, fallback)
