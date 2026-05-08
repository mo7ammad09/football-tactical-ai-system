"""Global identity display stabilizer for pre-render output.

This phase does not merge identities and does not claim uncertain identities.
It only writes stable client-facing display metadata when the raw tracklet
history contains enough evidence for a player role and team.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable

from src.utils.annotation_colors import PLAYER_FALLBACK_COLOR, is_goalkeeper_color, normalize_color


PERSON_OBJECT_TYPES = {"player", "referee", "goalkeeper"}
READY_STATUS = "ready_for_safe_apply"
REVIEW_STATUS = "needs_review"
APPLIED_STATUS = "applied"
NOOP_STATUS = "no_ready_actions"
REJECTED_STATUS = "rejected"
MIN_PLAYER_ROLE_CONFIDENCE = 0.80
MIN_PLAYER_TEAM_CONFIDENCE = 0.85
MIN_STRONG_PLAYER_TEAM_CONFIDENCE = 0.94
MAX_MINOR_ROLE_SEGMENT_FRAMES = 2
MAX_MINOR_TEAM_SEGMENT_FRAMES = 12
MAX_MINOR_TEAM_SEGMENT_RATIO = 0.08
MAX_STRONG_MINOR_TEAM_SEGMENT_FRAMES = 30
MAX_STRONG_MINOR_TEAM_SEGMENT_RATIO = 0.06
VALID_ACTION_TYPES = {
    "stable_player_display_override",
    "stable_player_role_display_override",
    "stable_player_team_display_override",
}


def _as_int(value: Any, fallback: int = 0) -> int:
    """Convert values to int while keeping dirty rows safe."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: Any, fallback: float = 0.0) -> float:
    """Convert values to float while keeping dirty rows safe."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _counter_dict(counter: Counter) -> dict[str, int]:
    """Return a JSON-safe counter dict."""
    return {str(key): int(value) for key, value in counter.items()}


def _dominant(counter: Counter, fallback: Any = None) -> tuple[Any, float, int, int]:
    """Return dominant value, confidence, top count, and runner-up count."""
    total = sum(counter.values())
    if total <= 0:
        return fallback, 0.0, 0, 0
    values = counter.most_common(2)
    top_value, top_count = values[0]
    runner_up = values[1][1] if len(values) > 1 else 0
    return top_value, float(top_count) / float(total), int(top_count), int(runner_up)


def _iter_person_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield person-like records."""
    for row in rows:
        if str(row.get("object_type", "")) in PERSON_OBJECT_TYPES:
            yield row


def _source_track_id(row: dict[str, Any]) -> int:
    """Return the source track id."""
    return _as_int(row.get("track_id"), -1)


def _frame_idx(row: dict[str, Any]) -> int:
    """Return the source frame index."""
    return _as_int(row.get("source_frame_idx"))


def _sample_number(row: dict[str, Any]) -> int:
    """Return the sampled frame number."""
    return _as_int(row.get("sample_number"), _frame_idx(row))


def _raw_role(row: dict[str, Any]) -> str:
    """Return raw role evidence."""
    object_type = str(row.get("object_type") or "")
    if object_type == "referee":
        return "referee"
    return str(row.get("role") or row.get("detected_role") or object_type or "unknown")


def _display_role(row: dict[str, Any]) -> str:
    """Return client-facing role evidence."""
    return str(row.get("display_role") or row.get("role") or row.get("detected_role") or "unknown")


def _display_team(row: dict[str, Any]) -> int:
    """Return client-facing team evidence."""
    if row.get("display_team") is not None:
        return _as_int(row.get("display_team"))
    return _as_int(row.get("team"))


def _display_label(row: dict[str, Any]) -> str:
    """Return client-facing label."""
    if row.get("display_label") is not None:
        return str(row.get("display_label"))
    return str(row.get("track_id"))


def _is_display_goalkeeper(row: dict[str, Any]) -> bool:
    """Return whether the row is rendered as goalkeeper-like."""
    return (
        _display_role(row) == "goalkeeper"
        or _display_label(row).upper() == "GK"
        or is_goalkeeper_color(row.get("display_color"))
    )


def _safe_color(color: Any) -> tuple[int, int, int] | None:
    """Return a non-goalkeeper color tuple."""
    if color is None or is_goalkeeper_color(color):
        return None
    normalized = normalize_color(color)
    if is_goalkeeper_color(normalized):
        return None
    return normalized


def _transition_count(values: list[Any]) -> int:
    """Count adjacent transitions in an ordered value stream."""
    return sum(1 for before, after in zip(values, values[1:]) if before != after)


def _segments(values: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    """Return contiguous value segments in frame order."""
    if not values:
        return []
    ordered = sorted(values, key=lambda item: item[0])
    current_value = ordered[0][1]
    start = end = ordered[0][0]
    count = 1
    segments: list[dict[str, Any]] = []
    for frame, value in ordered[1:]:
        if value == current_value:
            end = frame
            count += 1
            continue
        segments.append(
            {
                "value": current_value,
                "first_source_frame_idx": int(start),
                "last_source_frame_idx": int(end),
                "frames_seen": int(count),
            }
        )
        current_value = value
        start = end = frame
        count = 1
    segments.append(
        {
            "value": current_value,
            "first_source_frame_idx": int(start),
            "last_source_frame_idx": int(end),
            "frames_seen": int(count),
        }
    )
    return segments


def _max_non_value_segment(segments: list[dict[str, Any]], dominant_value: Any) -> tuple[int, float]:
    """Return largest non-dominant segment count and ratio."""
    total = sum(_as_int(segment.get("frames_seen")) for segment in segments)
    max_count = max(
        (
            _as_int(segment.get("frames_seen"))
            for segment in segments
            if segment.get("value") != dominant_value
        ),
        default=0,
    )
    ratio = (max_count / float(total)) if total else 0.0
    return int(max_count), float(ratio)


def _safe_player_role_decision(
    *,
    role_counts: Counter,
    role_segments: list[dict[str, Any]],
    frames_seen: int,
    min_role_confidence: float,
    min_role_margin_ratio: float,
    max_minor_role_segment_frames: int,
) -> tuple[bool, str, float, int]:
    """Return whether the track has enough evidence to render as a player."""
    role, confidence, top_count, runner_up = _dominant(role_counts, "unknown")
    margin = top_count - runner_up
    margin_ratio = (margin / float(frames_seen)) if frames_seen else 0.0
    max_minor_role_segment, _ = _max_non_value_segment(role_segments, role)
    is_safe = (
        str(role) == "player"
        and confidence >= min_role_confidence
        and margin_ratio >= min_role_margin_ratio
        and max_minor_role_segment <= max_minor_role_segment_frames
    )
    return is_safe, str(role), float(confidence), int(margin)


def _safe_player_team_decision(
    *,
    team_counts: Counter,
    team_segments: list[dict[str, Any]],
    frames_seen: int,
    min_team_confidence: float,
    min_team_margin_ratio: float,
    min_team_frames: int,
    max_minor_team_segment_frames: int,
    max_minor_team_segment_ratio: float,
) -> tuple[bool, int, float, int]:
    """Return whether the track has enough evidence for a player team."""
    team, confidence, top_count, runner_up = _dominant(team_counts, 0)
    margin = top_count - runner_up
    margin_ratio = (margin / float(frames_seen)) if frames_seen else 0.0
    team_int = _as_int(team, 0)
    max_minor_team_segment, max_minor_team_ratio = _max_non_value_segment(team_segments, team_int)
    is_safe = (
        team_int in {1, 2}
        and top_count >= min_team_frames
        and confidence >= min_team_confidence
        and margin_ratio >= min_team_margin_ratio
        and max_minor_team_segment <= max_minor_team_segment_frames
        and max_minor_team_ratio <= max_minor_team_segment_ratio
    )
    return is_safe, int(team_int), float(confidence), int(margin)


def _strong_player_team_safe(profile: dict[str, Any]) -> bool:
    """Return whether team evidence is strong enough for display-only locking."""
    return (
        bool(profile.get("player_role_safe"))
        and _as_int(profile.get("dominant_player_team"), 0) in {1, 2}
        and _as_float(profile.get("player_team_confidence"), 0.0)
        >= MIN_STRONG_PLAYER_TEAM_CONFIDENCE
        and _as_int(profile.get("max_minor_team_segment_frames"), 0)
        <= MAX_STRONG_MINOR_TEAM_SEGMENT_FRAMES
        and _as_float(profile.get("max_minor_team_segment_ratio"), 0.0)
        <= MAX_STRONG_MINOR_TEAM_SEGMENT_RATIO
    )


def _action_sets_role(action: dict[str, Any]) -> bool:
    """Return whether an action writes display role/label."""
    return action.get("set_display_role") is not None


def _action_sets_team(action: dict[str, Any]) -> bool:
    """Return whether an action writes display team/color."""
    return action.get("set_display_team") is not None


def _build_player_display_action(
    profile: dict[str, Any],
    *,
    action_type: str,
    set_role: bool,
    set_team: bool,
    reason: str,
) -> dict[str, Any]:
    """Build a JSON-safe display stabilization action."""
    track_id = _as_int(profile.get("track_id"), -1)
    action: dict[str, Any] = {
        "action_id": f"{action_type}_{track_id}",
        "action_type": action_type,
        "status": READY_STATUS,
        "track_id": int(track_id),
        "role_confidence": float(profile.get("raw_role_confidence", 0.0)),
        "team_confidence": float(profile.get("player_team_confidence", 0.0)),
        "role_margin": int(profile.get("raw_role_margin", 0)),
        "team_margin": int(profile.get("player_team_margin", 0)),
        "max_minor_role_segment_frames": int(
            profile.get("max_minor_role_segment_frames", 0)
        ),
        "max_minor_role_segment_ratio": float(
            profile.get("max_minor_role_segment_ratio", 0.0)
        ),
        "max_minor_team_segment_frames": int(
            profile.get("max_minor_team_segment_frames", 0)
        ),
        "max_minor_team_segment_ratio": float(
            profile.get("max_minor_team_segment_ratio", 0.0)
        ),
        "reason": reason,
        "evidence": {
            "raw_role_counts": profile.get("raw_role_counts", {}),
            "player_team_counts": profile.get("player_team_counts", {}),
            "display_role_counts": profile.get("display_role_counts", {}),
            "display_team_counts": profile.get("display_team_counts", {}),
            "raw_role_segments": profile.get("raw_role_segments", []),
            "player_team_segments": profile.get("player_team_segments", []),
        },
    }
    if set_role:
        action["set_display_role"] = "player"
        action["set_display_label"] = str(track_id)
    if set_team:
        action["set_display_team"] = int(profile.get("dominant_player_team", 0))
        action["set_display_color"] = profile.get("canonical_display_color")
    return action


def _track_profiles(raw_tracklet_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Build display-stability profiles by track id."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_person_rows(raw_tracklet_rows):
        track_id = _source_track_id(row)
        if track_id >= 0:
            grouped[track_id].append(row)

    profiles: dict[int, dict[str, Any]] = {}
    for track_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (_sample_number(item), _frame_idx(item)))
        frames_seen = len(ordered)
        raw_roles = Counter(_raw_role(row) for row in ordered)
        display_roles = Counter(_display_role(row) for row in ordered)
        display_teams = Counter(_display_team(row) for row in ordered)
        object_types = Counter(str(row.get("object_type") or "unknown") for row in ordered)
        player_team_counts: Counter = Counter()
        colors_by_team: dict[int, Counter] = defaultdict(Counter)

        for row in ordered:
            raw_role = _raw_role(row)
            team = _as_int(row.get("team"), 0)
            if raw_role == "player" and team in {1, 2}:
                player_team_counts[team] += 1
                color = _safe_color(row.get("team_color"))
                if color is not None:
                    colors_by_team[team][color] += 1

        display_role_sequence = [_display_role(row) for row in ordered]
        display_team_sequence = [_display_team(row) for row in ordered]
        raw_role_segments = _segments(
            [(_frame_idx(row), _raw_role(row)) for row in ordered]
        )
        player_team_segments = _segments(
            [
                (_frame_idx(row), _as_int(row.get("team"), 0))
                for row in ordered
                if _raw_role(row) == "player" and _as_int(row.get("team"), 0) in {1, 2}
            ]
        )
        player_role_safe, dominant_role, role_confidence, role_margin = _safe_player_role_decision(
            role_counts=raw_roles,
            role_segments=raw_role_segments,
            frames_seen=frames_seen,
            min_role_confidence=MIN_PLAYER_ROLE_CONFIDENCE,
            min_role_margin_ratio=0.10,
            max_minor_role_segment_frames=MAX_MINOR_ROLE_SEGMENT_FRAMES,
        )
        player_team_safe, dominant_team, team_confidence, team_margin = _safe_player_team_decision(
            team_counts=player_team_counts,
            team_segments=player_team_segments,
            frames_seen=max(1, sum(player_team_counts.values())),
            min_team_confidence=MIN_PLAYER_TEAM_CONFIDENCE,
            min_team_margin_ratio=0.10,
            min_team_frames=3,
            max_minor_team_segment_frames=MAX_MINOR_TEAM_SEGMENT_FRAMES,
            max_minor_team_segment_ratio=MAX_MINOR_TEAM_SEGMENT_RATIO,
        )
        display_role, display_role_confidence, _, _ = _dominant(display_roles, "unknown")
        display_team, display_team_confidence, _, _ = _dominant(display_teams, 0)
        color, _, _, _ = _dominant(colors_by_team.get(dominant_team, Counter()), None)
        max_minor_role_segment, max_minor_role_segment_ratio = _max_non_value_segment(
            raw_role_segments,
            dominant_role,
        )
        max_minor_team_segment, max_minor_team_segment_ratio = _max_non_value_segment(
            player_team_segments,
            dominant_team,
        )

        profiles[track_id] = {
            "track_id": int(track_id),
            "frames_seen": int(frames_seen),
            "first_source_frame_idx": _frame_idx(ordered[0]) if ordered else 0,
            "last_source_frame_idx": _frame_idx(ordered[-1]) if ordered else 0,
            "raw_role_counts": _counter_dict(raw_roles),
            "display_role_counts": _counter_dict(display_roles),
            "display_team_counts": _counter_dict(display_teams),
            "player_team_counts": _counter_dict(player_team_counts),
            "object_type_counts": _counter_dict(object_types),
            "dominant_raw_role": dominant_role,
            "raw_role_confidence": float(role_confidence),
            "raw_role_margin": int(role_margin),
            "dominant_player_team": int(dominant_team),
            "player_team_confidence": float(team_confidence),
            "player_team_margin": int(team_margin),
            "dominant_display_role": str(display_role),
            "display_role_confidence": float(display_role_confidence),
            "dominant_display_team": int(_as_int(display_team, 0)),
            "display_team_confidence": float(display_team_confidence),
            "display_role_transition_count": int(_transition_count(display_role_sequence)),
            "display_team_transition_count": int(_transition_count(display_team_sequence)),
            "display_goalkeeper_frame_count": sum(1 for row in ordered if _is_display_goalkeeper(row)),
            "player_role_safe": bool(player_role_safe),
            "player_team_safe": bool(player_team_safe),
            "raw_role_segments": raw_role_segments[:20],
            "raw_role_segments_truncated": len(raw_role_segments) > 20,
            "player_team_segments": player_team_segments[:20],
            "player_team_segments_truncated": len(player_team_segments) > 20,
            "max_minor_role_segment_frames": int(max_minor_role_segment),
            "max_minor_role_segment_ratio": float(max_minor_role_segment_ratio),
            "max_minor_team_segment_frames": int(max_minor_team_segment),
            "max_minor_team_segment_ratio": float(max_minor_team_segment_ratio),
            "canonical_display_color": list(color) if color is not None else list(PLAYER_FALLBACK_COLOR),
        }
    return profiles


def validate_global_identity_stability_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate that the stabilizer plan can be safely applied."""
    rejected: list[dict[str, Any]] = []
    for action in plan.get("actions", []) or []:
        reasons: list[str] = []
        if action.get("status") != READY_STATUS:
            continue
        action_type = str(action.get("action_type"))
        sets_role = _action_sets_role(action)
        sets_team = _action_sets_team(action)
        if action_type not in VALID_ACTION_TYPES:
            reasons.append("unsupported_action_type")
        if not sets_role and not sets_team:
            reasons.append("display_action_must_set_role_or_team")
        if action_type == "stable_player_display_override" and (not sets_role or not sets_team):
            reasons.append("combined_display_override_requires_role_and_team")
        if action_type == "stable_player_role_display_override" and not sets_role:
            reasons.append("role_display_override_requires_role")
        if action_type == "stable_player_team_display_override" and not sets_team:
            reasons.append("team_display_override_requires_team")
        if sets_role:
            if str(action.get("set_display_role")) != "player":
                reasons.append("only_player_display_role_overrides_are_supported")
            if _as_float(action.get("role_confidence"), 0.0) < MIN_PLAYER_ROLE_CONFIDENCE:
                reasons.append("role_confidence_below_threshold")
            if _as_int(action.get("max_minor_role_segment_frames"), 0) > MAX_MINOR_ROLE_SEGMENT_FRAMES:
                reasons.append("minor_role_segment_too_long")
        if sets_team:
            if _as_int(action.get("set_display_team"), 0) not in {1, 2}:
                reasons.append("player_display_team_must_be_team_1_or_2")
            if _as_float(action.get("role_confidence"), 0.0) < MIN_PLAYER_ROLE_CONFIDENCE:
                reasons.append("team_lock_requires_stable_player_role")
            if _as_int(action.get("max_minor_role_segment_frames"), 0) > MAX_MINOR_ROLE_SEGMENT_FRAMES:
                reasons.append("team_lock_requires_stable_player_role_segments")
            if _as_float(action.get("team_confidence"), 0.0) < MIN_PLAYER_TEAM_CONFIDENCE:
                reasons.append("team_confidence_below_threshold")
            if _as_int(action.get("max_minor_team_segment_frames"), 0) > MAX_STRONG_MINOR_TEAM_SEGMENT_FRAMES:
                reasons.append("minor_team_segment_too_long")
            if _as_float(action.get("max_minor_team_segment_ratio"), 0.0) > MAX_STRONG_MINOR_TEAM_SEGMENT_RATIO:
                reasons.append("minor_team_segment_ratio_too_high")
        if action.get("track_id") is None:
            reasons.append("missing_track_id")
        if reasons:
            rejected.append(
                {
                    "action_id": action.get("action_id"),
                    "track_id": action.get("track_id"),
                    "reasons": reasons,
                }
            )

    return {
        "schema_version": "1.0",
        "validator": "global_identity_stability_validator",
        "verdict": "PASS" if not rejected else "FAIL",
        "accepted_action_count": len(
            [
                action
                for action in plan.get("actions", []) or []
                if action.get("status") == READY_STATUS
            ]
        )
        - len(rejected),
        "rejected_action_count": len(rejected),
        "rejected_actions": rejected,
    }


def build_global_identity_stability_plan(
    raw_tracklet_rows: list[dict[str, Any]],
    *,
    min_frames: int = 4,
) -> dict[str, Any]:
    """Build a deterministic plan to stabilize player display role/team."""
    profiles = _track_profiles(raw_tracklet_rows)
    actions: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []

    for track_id in sorted(profiles):
        profile = profiles[track_id]
        frames_seen = _as_int(profile.get("frames_seen"), 0)
        display_role_transitions = _as_int(profile.get("display_role_transition_count"), 0)
        display_team_transitions = _as_int(profile.get("display_team_transition_count"), 0)
        has_visible_flicker = display_role_transitions > 0 or display_team_transitions > 0
        has_gk_leak = _as_int(profile.get("display_goalkeeper_frame_count"), 0) > 0
        has_hidden_role_conflict = (
            str(profile.get("dominant_raw_role")) == "player"
            and _as_int(profile.get("max_minor_role_segment_frames"), 0)
            > MAX_MINOR_ROLE_SEGMENT_FRAMES
        )
        player_team_counts = profile.get("player_team_counts") or {}
        has_multiple_player_teams = (
            sum(1 for count in player_team_counts.values() if _as_int(count) > 0) > 1
        )
        has_hidden_team_conflict = has_multiple_player_teams and (
            _as_float(profile.get("player_team_confidence"), 0.0) < MIN_PLAYER_TEAM_CONFIDENCE
            or _as_int(profile.get("max_minor_team_segment_frames"), 0)
            > MAX_MINOR_TEAM_SEGMENT_FRAMES
            or _as_float(profile.get("max_minor_team_segment_ratio"), 0.0)
            > MAX_MINOR_TEAM_SEGMENT_RATIO
        )

        if frames_seen < min_frames:
            if has_visible_flicker or has_gk_leak or has_hidden_role_conflict or has_hidden_team_conflict:
                review.append(
                    {
                        "track_id": int(track_id),
                        "status": REVIEW_STATUS,
                        "reason": "track_too_short_for_safe_stabilization",
                        "profile": profile,
                    }
                )
            continue

        action_added = False
        if profile.get("player_role_safe") and profile.get("player_team_safe"):
            actions.append(
                _build_player_display_action(
                    profile,
                    action_type="stable_player_display_override",
                    set_role=True,
                    set_team=True,
                    reason=(
                        "Track has enough raw player-role and team evidence to keep "
                        "client-facing role/team stable across rendered frames."
                    ),
                )
            )
            action_added = True
        else:
            if profile.get("player_role_safe") and (display_role_transitions > 0 or has_gk_leak):
                actions.append(
                    _build_player_display_action(
                        profile,
                        action_type="stable_player_role_display_override",
                        set_role=True,
                        set_team=False,
                        reason=(
                            "Track has isolated role flicker only; lock the "
                            "client-facing role to player without forcing team."
                        ),
                    )
                )
                action_added = True
            if (
                display_team_transitions > 0
                and _strong_player_team_safe(profile)
            ):
                actions.append(
                    _build_player_display_action(
                        profile,
                        action_type="stable_player_team_display_override",
                        set_role=False,
                        set_team=True,
                        reason=(
                            "Track has strong player-team evidence with a small "
                            "minor team segment; lock display team/color only."
                        ),
                    )
                )
                action_added = True

        if has_visible_flicker or has_gk_leak or has_hidden_role_conflict or has_hidden_team_conflict:
            reasons: list[str] = []
            if not profile.get("player_role_safe"):
                reasons.append("insufficient_role_evidence")
            if not profile.get("player_team_safe"):
                reasons.append("insufficient_team_evidence")
            if _as_int(profile.get("max_minor_role_segment_frames"), 0) > MAX_MINOR_ROLE_SEGMENT_FRAMES:
                reasons.append("minor_role_segment_too_long")
            if _as_int(profile.get("max_minor_team_segment_frames"), 0) > MAX_MINOR_TEAM_SEGMENT_FRAMES:
                reasons.append("minor_team_segment_too_long")
            if has_hidden_role_conflict:
                reasons.append("hidden_role_conflict")
            if has_hidden_team_conflict:
                reasons.append("hidden_team_conflict")
            review.append(
                {
                    "track_id": int(track_id),
                    "status": REVIEW_STATUS,
                    "reason": ",".join(reasons) if reasons else "display_flicker_needs_review",
                    "recommended_action": (
                        "segment_split_required"
                        if has_hidden_role_conflict or has_hidden_team_conflict
                        else "review_remaining_display_evidence"
                    ),
                    "partial_safe_action_applied": bool(action_added),
                    "profile": profile,
                }
            )

    plan = {
        "schema_version": "1.0",
        "phase": "phase_12_global_identity_stabilizer",
        "summary": {
            "profile_count": len(profiles),
            "ready_action_count": len(actions),
            "review_track_count": len(review),
        },
        "actions": actions,
        "needs_review": review[:120],
        "needs_review_truncated": len(review) > 120,
        "source_profiles": [profiles[track_id] for track_id in sorted(profiles)],
    }
    plan["validation"] = validate_global_identity_stability_plan(plan)
    return plan


def _ready_actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validator-approved stabilizer actions."""
    if plan.get("validation", {}).get("verdict") != "PASS":
        return []
    return [
        action
        for action in plan.get("actions", []) or []
        if isinstance(action, dict) and action.get("status") == READY_STATUS
    ]


def apply_global_identity_stability_plan_to_raw_records(
    raw_tracklet_rows: list[dict[str, Any]],
    stability_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply safe display stabilization to raw-tracklet rows."""
    rows = deepcopy(raw_tracklet_rows)
    validation = stability_plan.get("validation") or {}
    if validation.get("verdict") != "PASS":
        return rows, {
            "schema_version": "1.0",
            "phase": "phase_12_global_identity_stabilizer_apply",
            "status": REJECTED_STATUS,
            "applied_action_count": 0,
            "updated_record_count": 0,
            "validation": validation,
        }

    updated_record_count = 0
    applied_actions: list[dict[str, Any]] = []
    for action in _ready_actions(stability_plan):
        track_id = _as_int(action.get("track_id"), -1)
        action_updates = 0
        sets_role = _action_sets_role(action)
        sets_team = _action_sets_team(action)
        color = normalize_color(action.get("set_display_color"), PLAYER_FALLBACK_COLOR)
        for row in rows:
            if str(row.get("object_type")) not in PERSON_OBJECT_TYPES:
                continue
            if _source_track_id(row) != track_id:
                continue
            if sets_team and not sets_role and _raw_role(row) != "player":
                continue
            if sets_role:
                row["display_role"] = "player"
                row["display_label"] = str(action.get("set_display_label", track_id))
                row["goalkeeper_display_locked"] = False
                row["role_display_suppressed"] = False
            if sets_team:
                row["display_team"] = int(action.get("set_display_team"))
                row["display_color"] = list(color)
            row["identity_stability_status"] = "phase12_safe_applied"
            row["identity_stability_action_id"] = action.get("action_id")
            row["identity_stability_role_confidence"] = float(action.get("role_confidence", 0.0))
            row["identity_stability_team_confidence"] = float(action.get("team_confidence", 0.0))
            action_updates += 1
        if action_updates:
            updated_record_count += action_updates
            applied_actions.append(
                {
                    "action_id": action.get("action_id"),
                    "track_id": track_id,
                    "updated_record_count": action_updates,
                    "set_display_role": action.get("set_display_role"),
                    "set_display_team": (
                        int(action.get("set_display_team"))
                        if action.get("set_display_team") is not None
                        else None
                    ),
                }
            )

    return rows, {
        "schema_version": "1.0",
        "phase": "phase_12_global_identity_stabilizer_apply",
        "status": APPLIED_STATUS if applied_actions else NOOP_STATUS,
        "applied_action_count": len(applied_actions),
        "updated_record_count": updated_record_count,
        "applied_actions": applied_actions,
        "validation": validation,
    }


def apply_global_identity_stability_plan_to_annotation_states(
    annotation_states: list[dict[str, Any]],
    stability_plan: dict[str, Any],
) -> int:
    """Apply safe display stabilization to render annotation states."""
    ready = _ready_actions(stability_plan)
    if not ready:
        return 0

    updated = 0
    for action in ready:
        track_id = _as_int(action.get("track_id"), -1)
        sets_role = _action_sets_role(action)
        sets_team = _action_sets_team(action)
        color = normalize_color(action.get("set_display_color"), PLAYER_FALLBACK_COLOR)
        for state in annotation_states:
            for bucket_name in ("players", "referees"):
                track = (state.get(bucket_name) or {}).get(track_id)
                if track is None:
                    continue
                if sets_team and not sets_role and str(track.get("role") or bucket_name) != "player":
                    continue
                if sets_role:
                    track["display_role"] = "player"
                    track["display_label"] = str(action.get("set_display_label", track_id))
                    track["goalkeeper_display_locked"] = False
                    track["role_display_suppressed"] = False
                if sets_team:
                    track["display_team"] = int(action.get("set_display_team"))
                    track["display_color"] = tuple(color)
                track["identity_stability_status"] = "phase12_safe_applied"
                track["identity_stability_action_id"] = action.get("action_id")
                track["identity_stability_role_confidence"] = float(action.get("role_confidence", 0.0))
                track["identity_stability_team_confidence"] = float(action.get("team_confidence", 0.0))
                updated += 1
    return updated
