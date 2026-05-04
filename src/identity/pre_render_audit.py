"""Pre-render identity audit artifacts for safe video export."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable


RUNPOD_BASELINE_IMAGE = "ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-bbe8dec"
KNOWN_BAD_RUNPOD_IMAGES = [
    {
        "image": "ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6",
        "reason": (
            "Regressed goalkeeper display locking: GK stayed visible but spread "
            "to defenders/field players with the goalkeeper color."
        ),
    }
]

GOALKEEPER_ROLE = "goalkeeper"
PERSON_OBJECT_TYPES = {"player", "referee", "goalkeeper"}
VISION_REVIEW_VERDICTS = {
    "same_player",
    "different_player",
    "goalkeeper",
    "not_goalkeeper",
    "unresolved",
}
VISION_REVIEW_STATUSES = {
    "not_invoked",
    "reviewed",
    "missing_model_output",
    "invalid_model_output",
}


def _as_int(value: Any, fallback: int = 0) -> int:
    """Convert values to int while keeping audits resilient to dirty rows."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: Any, fallback: float = 0.0) -> float:
    """Convert values to float while keeping audits resilient to dirty rows."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _counter_dict(counter: Counter) -> dict[str, int]:
    """Return a JSON-safe counter dict."""
    return {str(key): int(count) for key, count in counter.items()}


def _dominant(counter: Counter, fallback: Any = None) -> tuple[Any, float]:
    """Return the dominant counter value and confidence."""
    total = sum(counter.values())
    if total <= 0:
        return fallback, 0.0
    value, count = counter.most_common(1)[0]
    return value, float(count) / float(total)


def _iter_person_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield person-like rows."""
    for row in rows:
        if str(row.get("object_type", "")) in PERSON_OBJECT_TYPES:
            yield row


def _source_track_id(row: dict[str, Any]) -> int:
    """Return the underlying tracker/display track id."""
    return _as_int(row.get("track_id"), -1)


def _raw_track_id(row: dict[str, Any]) -> int | None:
    """Return raw tracker id when available."""
    if row.get("raw_track_id") is None:
        return None
    return _as_int(row.get("raw_track_id"))


def _raw_role(row: dict[str, Any]) -> str:
    """Return the non-client-facing role evidence."""
    return str(row.get("role") or row.get("detected_role") or "unknown")


def _detected_role(row: dict[str, Any]) -> str:
    """Return detector/model role evidence."""
    return str(row.get("detected_role") or row.get("role") or "unknown")


def _visible_role(row: dict[str, Any]) -> str:
    """Return the role shown in the rendered video."""
    return str(row.get("display_role") or row.get("role") or "unknown")


def _visible_team(row: dict[str, Any]) -> int:
    """Return the team/color bucket shown in the rendered video."""
    if row.get("display_team") is not None:
        return _as_int(row.get("display_team"))
    return _as_int(row.get("team"))


def _visible_label(row: dict[str, Any]) -> str:
    """Return the label shown in the rendered video."""
    if row.get("display_label") is not None:
        return str(row.get("display_label"))
    return str(row.get("track_id"))


def _is_magenta_display(row: dict[str, Any]) -> bool:
    """Return whether display_color looks like the current goalkeeper color."""
    color = row.get("display_color")
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return False
    blue = _as_int(color[0])
    green = _as_int(color[1])
    red = _as_int(color[2])
    return blue >= 220 and red >= 220 and green <= 80


def _is_display_goalkeeper(row: dict[str, Any]) -> bool:
    """Return whether the row would be rendered as goalkeeper-like."""
    return (
        _visible_role(row) == GOALKEEPER_ROLE
        or _visible_label(row).upper() == "GK"
        or _is_magenta_display(row)
    )


def _frame_idx(row: dict[str, Any]) -> int:
    """Return source frame index."""
    return _as_int(row.get("source_frame_idx"))


def _sample_number(row: dict[str, Any]) -> int:
    """Return sampled frame number."""
    return _as_int(row.get("sample_number"), _frame_idx(row))


def _build_source_profiles(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Build profiles by underlying track id, not by visible label."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_person_rows(rows):
        source_id = _source_track_id(row)
        if source_id >= 0:
            grouped[source_id].append(row)

    profiles: dict[int, dict[str, Any]] = {}
    for source_id, track_rows in grouped.items():
        ordered = sorted(track_rows, key=lambda item: (_frame_idx(item), _sample_number(item)))
        raw_roles = Counter(_raw_role(row) for row in ordered)
        detected_roles = Counter(_detected_role(row) for row in ordered)
        visible_roles = Counter(_visible_role(row) for row in ordered)
        visible_labels = Counter(_visible_label(row) for row in ordered)
        teams = Counter(_as_int(row.get("team")) for row in ordered)
        visible_teams = Counter(_visible_team(row) for row in ordered)
        raw_ids = Counter(
            _raw_track_id(row)
            for row in ordered
            if _raw_track_id(row) is not None
        )
        frames = [_frame_idx(row) for row in ordered]
        dominant_raw_role, raw_role_confidence = _dominant(raw_roles, "unknown")
        dominant_detected_role, detected_role_confidence = _dominant(detected_roles, "unknown")
        display_goalkeeper_frames = sum(1 for row in ordered if _is_display_goalkeeper(row))
        raw_goalkeeper_frames = sum(
            1
            for row in ordered
            if _raw_role(row) == GOALKEEPER_ROLE or _detected_role(row) == GOALKEEPER_ROLE
        )
        profiles[source_id] = {
            "track_id": int(source_id),
            "frames_seen": len(ordered),
            "first_source_frame_idx": min(frames) if frames else 0,
            "last_source_frame_idx": max(frames) if frames else 0,
            "dominant_raw_role": str(dominant_raw_role),
            "raw_role_confidence": float(raw_role_confidence),
            "dominant_detected_role": str(dominant_detected_role),
            "detected_role_confidence": float(detected_role_confidence),
            "raw_role_counts": _counter_dict(raw_roles),
            "detected_role_counts": _counter_dict(detected_roles),
            "visible_role_counts": _counter_dict(visible_roles),
            "visible_label_counts": _counter_dict(visible_labels),
            "team_counts": _counter_dict(teams),
            "visible_team_counts": _counter_dict(visible_teams),
            "raw_track_counts": _counter_dict(raw_ids),
            "display_goalkeeper_frame_count": int(display_goalkeeper_frames),
            "raw_goalkeeper_frame_count": int(raw_goalkeeper_frames),
        }
    return profiles


def _goalkeeper_display_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build contiguous visible-goalkeeper segments by source track/raw id."""
    grouped: dict[tuple[int, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_person_rows(rows):
        if not _is_display_goalkeeper(row):
            continue
        key = (_source_track_id(row), _raw_track_id(row), _visible_label(row))
        grouped[key].append(row)

    segments: list[dict[str, Any]] = []
    for (track_id, raw_track_id, visible_label), track_rows in grouped.items():
        ordered = sorted(track_rows, key=lambda item: (_sample_number(item), _frame_idx(item)))
        current: dict[str, Any] | None = None
        last_sample = None
        for row in ordered:
            sample = _sample_number(row)
            frame = _frame_idx(row)
            if current is None or (last_sample is not None and sample - last_sample > 1):
                if current is not None:
                    segments.append(current)
                current = {
                    "track_id": int(track_id),
                    "raw_track_id": raw_track_id,
                    "visible_label": str(visible_label),
                    "visible_role": _visible_role(row),
                    "visible_team": _visible_team(row),
                    "first_sample": int(sample),
                    "last_sample": int(sample),
                    "first_source_frame_idx": int(frame),
                    "last_source_frame_idx": int(frame),
                    "frames_seen": 1,
                }
            else:
                current["last_sample"] = int(sample)
                current["last_source_frame_idx"] = int(frame)
                current["frames_seen"] += 1
            last_sample = sample
        if current is not None:
            segments.append(current)

    return sorted(
        segments,
        key=lambda item: (
            int(item.get("first_source_frame_idx", 0)),
            int(item.get("track_id", 0)),
        ),
    )


def _issue(
    *,
    issue_id: str,
    issue_type: str,
    severity: str,
    title: str,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    """Create a stable issue shape."""
    return {
        "issue_id": issue_id,
        "issue_type": issue_type,
        "severity": severity,
        "title": title,
        "reason": reason,
        **extra,
    }


def _score_issues(issues: list[dict[str, Any]]) -> tuple[int, str]:
    """Return score and verdict from render issues."""
    score = 100
    for issue in issues:
        severity = issue.get("severity")
        if severity == "critical":
            score -= 35
        elif severity == "high":
            score -= 18
        elif severity == "medium":
            score -= 8
        elif severity == "low":
            score -= 3
    score = max(0, min(100, score))
    if any(issue.get("severity") == "critical" for issue in issues):
        return score, "FAIL"
    if any(issue.get("severity") == "high" for issue in issues):
        return score, "REVIEW"
    return score, "PASS"


def build_render_identity_audit(
    raw_tracklet_rows: list[dict[str, Any]],
    identity_debug: dict[str, Any] | None = None,
    *,
    baseline_image: str = RUNPOD_BASELINE_IMAGE,
) -> dict[str, Any]:
    """Audit the client-facing identity state before final render.

    This audit intentionally looks at display fields, not just raw tracker fields.
    It catches regressions where a fixed GK label/color spreads to non-goalkeepers.
    """
    identity_debug = identity_debug or {}
    person_rows = list(_iter_person_rows(raw_tracklet_rows))
    source_profiles = _build_source_profiles(raw_tracklet_rows)
    gk_segments = _goalkeeper_display_segments(raw_tracklet_rows)
    issues: list[dict[str, Any]] = []

    for segment in gk_segments:
        profile = source_profiles.get(int(segment["track_id"]), {})
        dominant_role = str(profile.get("dominant_raw_role", "unknown"))
        raw_role_confidence = _as_float(profile.get("raw_role_confidence"))
        frames_seen = _as_int(profile.get("frames_seen"))
        display_gk_frames = _as_int(profile.get("display_goalkeeper_frame_count"))
        raw_gk_frames = _as_int(profile.get("raw_goalkeeper_frame_count"))
        if (
            dominant_role != GOALKEEPER_ROLE
            and frames_seen >= 4
            and display_gk_frames > 0
            and raw_role_confidence >= 0.65
        ):
            issues.append(
                _issue(
                    issue_id=(
                        "gk_false_positive_"
                        f"{segment['track_id']}_{segment['first_source_frame_idx']}_"
                        f"{segment['last_source_frame_idx']}"
                    ),
                    issue_type="gk_false_positive_segment",
                    severity="critical" if raw_gk_frames == 0 else "high",
                    title="Goalkeeper display is applied to a non-goalkeeper-dominant track",
                    reason=(
                        "The rendered GK label/color appears on an underlying "
                        f"{dominant_role}-dominant track. This is the class of "
                        "regression seen in the bad sha-5f969e6 image."
                    ),
                    track_id=int(segment["track_id"]),
                    raw_track_id=segment.get("raw_track_id"),
                    first_source_frame_idx=int(segment["first_source_frame_idx"]),
                    last_source_frame_idx=int(segment["last_source_frame_idx"]),
                    frames_seen=int(segment["frames_seen"]),
                    visible_label=str(segment["visible_label"]),
                    visible_role=str(segment["visible_role"]),
                    visible_team=int(segment["visible_team"]),
                    dominant_raw_role=dominant_role,
                    raw_role_confidence=float(raw_role_confidence),
                    raw_goalkeeper_frame_count=int(raw_gk_frames),
                    profile_frames_seen=int(frames_seen),
                )
            )

    gk_display_rows = [row for row in person_rows if _is_display_goalkeeper(row)]
    gk_source_ids = Counter(_source_track_id(row) for row in gk_display_rows)
    player_dominant_gk_source_ids = [
        track_id
        for track_id in gk_source_ids
        if source_profiles.get(track_id, {}).get("dominant_raw_role") not in {GOALKEEPER_ROLE, "unknown"}
        and _as_float(source_profiles.get(track_id, {}).get("raw_role_confidence")) >= 0.65
    ]
    if len(gk_source_ids) > 1 and player_dominant_gk_source_ids:
        issues.append(
            _issue(
                issue_id="unsafe_gk_display_spread",
                issue_type="unsafe_gk_display_spread",
                severity="critical",
                title="GK display spreads across player-dominant source tracks",
                reason=(
                    "A stable GK label is useful only if it follows the goalkeeper. "
                    "Here it spans underlying tracks that are player/referee dominant."
                ),
                source_track_ids=_counter_dict(gk_source_ids),
                player_dominant_source_track_ids=[
                    int(track_id) for track_id in player_dominant_gk_source_ids
                ],
            )
        )

    by_frame_team: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in gk_display_rows:
        by_frame_team[(_frame_idx(row), _visible_team(row))].add(_source_track_id(row))
    conflicts = [
        {
            "source_frame_idx": int(frame),
            "visible_team": int(team),
            "track_ids": sorted(int(track_id) for track_id in track_ids),
        }
        for (frame, team), track_ids in by_frame_team.items()
        if len(track_ids) > 1
    ]
    if conflicts:
        issues.append(
            _issue(
                issue_id="simultaneous_goalkeeper_display",
                issue_type="simultaneous_goalkeeper_display",
                severity="critical",
                title="Multiple visible goalkeepers appear for the same team/frame",
                reason="The final render would show more than one GK for the same visible team.",
                conflict_count=len(conflicts),
                examples=conflicts[:20],
                examples_truncated=len(conflicts) > 20,
            )
        )

    visible_risk_issue_types = {
        "gk_false_positive_segment",
        "unsafe_gk_display_spread",
        "simultaneous_goalkeeper_display",
    }
    has_visible_identity_risk = any(
        str(issue.get("issue_type")) in visible_risk_issue_types
        for issue in issues
    )

    role_stability = identity_debug.get("role_stability") or {}
    if int(role_stability.get("role_flicker_tracklet_count", 0) or 0) > 0:
        severity = "high" if has_visible_identity_risk else "medium"
        reason = (
            "Role flicker must be resolved or downgraded before trusting rendered colors."
            if has_visible_identity_risk
            else (
                "Raw identity debug still reports role flicker, but the client-facing "
                "goalkeeper display has no remaining high-risk spread or false-positive issue."
            )
        )
        issues.append(
            _issue(
                issue_id="identity_debug_role_flicker",
                issue_type="identity_debug_role_flicker",
                severity=severity,
                title="Identity debug reports role flicker",
                reason=reason,
                tracklet_count=int(role_stability.get("role_flicker_tracklet_count", 0) or 0),
                advisory=not has_visible_identity_risk,
            )
        )

    score, verdict = _score_issues(issues)
    issue_counts = Counter(str(issue.get("issue_type", "unknown")) for issue in issues)
    return {
        "schema_version": "1.0",
        "phase": "phase_1_audit_only",
        "baseline_image": baseline_image,
        "known_bad_images": KNOWN_BAD_RUNPOD_IMAGES,
        "verdict": verdict,
        "score": score,
        "summary": {
            "raw_record_count": len(raw_tracklet_rows),
            "person_record_count": len(person_rows),
            "visible_goalkeeper_record_count": len(gk_display_rows),
            "visible_goalkeeper_segment_count": len(gk_segments),
            "underlying_gk_source_track_count": len(gk_source_ids),
            "gk_false_positive_segment_count": int(issue_counts.get("gk_false_positive_segment", 0)),
            "simultaneous_goalkeeper_conflict_count": int(
                issue_counts.get("simultaneous_goalkeeper_display", 0)
            ),
            "unsafe_gk_display_spread_count": int(issue_counts.get("unsafe_gk_display_spread", 0)),
            "issue_counts": _counter_dict(issue_counts),
        },
        "issues": issues,
        "goalkeeper_display_segments": gk_segments[:120],
        "goalkeeper_display_segments_truncated": len(gk_segments) > 120,
        "source_profiles": [
            source_profiles[track_id]
            for track_id in sorted(source_profiles)
        ],
    }


def build_identity_events(
    raw_tracklet_rows: list[dict[str, Any]],
    render_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact identity events for LLM review."""
    render_audit = render_audit or {}
    events: list[dict[str, Any]] = []
    rows_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_person_rows(raw_tracklet_rows):
        rows_by_track[_source_track_id(row)].append(row)

    for track_id, rows in rows_by_track.items():
        ordered = sorted(rows, key=lambda item: (_sample_number(item), _frame_idx(item)))
        previous_role = None
        previous_team = None
        for row in ordered:
            role = _visible_role(row)
            team = _visible_team(row)
            frame = _frame_idx(row)
            if previous_role is not None and role != previous_role:
                events.append(
                    {
                        "event_id": f"role_change_{track_id}_{frame}",
                        "event_type": "role_change",
                        "severity": "high" if GOALKEEPER_ROLE in {role, previous_role} else "medium",
                        "track_id": int(track_id),
                        "raw_track_id": _raw_track_id(row),
                        "source_frame_idx": int(frame),
                        "before": {"role": previous_role},
                        "after": {"role": role},
                    }
                )
            if previous_team is not None and team != previous_team:
                events.append(
                    {
                        "event_id": f"team_change_{track_id}_{frame}",
                        "event_type": "team_change",
                        "severity": "medium",
                        "track_id": int(track_id),
                        "raw_track_id": _raw_track_id(row),
                        "source_frame_idx": int(frame),
                        "before": {"team": previous_team},
                        "after": {"team": team},
                    }
                )
            previous_role = role
            previous_team = team

    for issue in render_audit.get("issues", []):
        if issue.get("issue_type") not in {
            "gk_false_positive_segment",
            "unsafe_gk_display_spread",
            "simultaneous_goalkeeper_display",
        }:
            continue
        events.append(
            {
                "event_id": str(issue.get("issue_id")),
                "event_type": str(issue.get("issue_type")),
                "severity": str(issue.get("severity", "medium")),
                "track_id": issue.get("track_id"),
                "raw_track_id": issue.get("raw_track_id"),
                "first_source_frame_idx": issue.get("first_source_frame_idx"),
                "last_source_frame_idx": issue.get("last_source_frame_idx"),
                "evidence_issue_id": str(issue.get("issue_id")),
                "reason": str(issue.get("reason", "")),
            }
        )

    return {
        "schema_version": "1.0",
        "event_count": len(events),
        "events": events[:500],
        "events_truncated": len(events) > 500,
    }


def build_correction_candidates(
    render_audit: dict[str, Any],
    identity_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build dry-run correction candidates from deterministic audit issues."""
    identity_debug = identity_debug or {}
    candidates: list[dict[str, Any]] = []
    for issue in render_audit.get("issues", []):
        issue_type = str(issue.get("issue_type"))
        if issue_type == "gk_false_positive_segment":
            track_id = issue.get("track_id")
            candidates.append(
                {
                    "candidate_id": f"candidate_{issue.get('issue_id')}",
                    "source_issue_id": issue.get("issue_id"),
                    "candidate_type": "display_override",
                    "safety": "safe_fix_review_required",
                    "track_id": track_id,
                    "raw_track_id": issue.get("raw_track_id"),
                    "first_source_frame_idx": issue.get("first_source_frame_idx"),
                    "last_source_frame_idx": issue.get("last_source_frame_idx"),
                    "proposed_action": {
                        "action_type": "display_override",
                        "set_display_role": issue.get("dominant_raw_role", "player"),
                        "set_display_label": str(track_id),
                        "set_display_color_policy": "team",
                    },
                    "reason": (
                        "Remove client-facing GK label/color from a "
                        "non-goalkeeper-dominant segment."
                    ),
                }
            )
        elif issue_type in {"unsafe_gk_display_spread", "simultaneous_goalkeeper_display"}:
            candidates.append(
                {
                    "candidate_id": f"candidate_{issue.get('issue_id')}",
                    "source_issue_id": issue.get("issue_id"),
                    "candidate_type": "needs_vision_or_validator",
                    "safety": "do_not_auto_apply",
                    "proposed_action": {"action_type": "needs_vision"},
                    "reason": issue.get("reason"),
                }
            )

    warning_codes = {
        str(warning.get("code"))
        for warning in identity_debug.get("warnings", [])
        if warning.get("code")
    }
    if "goalkeeper_identity_fragmentation" in warning_codes:
        candidates.append(
            {
                "candidate_id": "candidate_goalkeeper_identity_fragmentation",
                "candidate_type": "identity_review",
                "safety": "do_not_auto_apply",
                "proposed_action": {"action_type": "needs_reid_topk_or_vision"},
                "reason": "Goalkeeper appears under multiple IDs; needs evidence before merge.",
            }
        )

    return {
        "schema_version": "1.0",
        "phase": "phase_1_audit_only",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _evidence_ids(render_audit: dict[str, Any], identity_events: dict[str, Any]) -> set[str]:
    """Return evidence ids that a plan is allowed to cite."""
    ids = {
        str(issue.get("issue_id"))
        for issue in render_audit.get("issues", [])
        if issue.get("issue_id")
    }
    ids.update(
        str(event.get("event_id"))
        for event in identity_events.get("events", [])
        if event.get("event_id")
    )
    return ids


def validate_correction_plan(
    correction_plan: dict[str, Any],
    *,
    render_audit: dict[str, Any],
    identity_events: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a dry-run correction plan before any future application step."""
    identity_events = identity_events or {}
    valid_evidence_ids = _evidence_ids(render_audit, identity_events)
    accepted_action_ids: list[str] = []
    rejected_actions: list[dict[str, Any]] = []

    for action in correction_plan.get("actions", []):
        action_id = str(action.get("action_id", "unknown_action"))
        action_type = str(action.get("action_type", "unknown"))
        reasons: list[str] = []

        cited_evidence = [
            str(evidence_id)
            for evidence_id in action.get("evidence_ids", [])
            if evidence_id is not None
        ]
        if not cited_evidence:
            reasons.append("missing_evidence_ids")
        elif not set(cited_evidence).issubset(valid_evidence_ids):
            reasons.append("unknown_evidence_id")

        if action_type in {"display_override", "role_override", "team_override"}:
            first_frame = action.get("first_source_frame_idx")
            last_frame = action.get("last_source_frame_idx")
            if first_frame is None or last_frame is None:
                reasons.append("missing_frame_range")
            elif _as_int(first_frame) > _as_int(last_frame):
                reasons.append("invalid_frame_range")
            if action.get("track_id") is None:
                reasons.append("missing_track_id")

        if action_type == "display_override":
            target_role = str(action.get("set_display_role", "unknown"))
            target_label = str(action.get("set_display_label", ""))
            color_policy = str(action.get("set_display_color_policy", ""))
            if target_role != GOALKEEPER_ROLE and target_label.upper() == "GK":
                reasons.append("non_goalkeeper_keeps_gk_label")
            if target_role != GOALKEEPER_ROLE and color_policy == "goalkeeper":
                reasons.append("non_goalkeeper_uses_goalkeeper_color")
            if target_role == GOALKEEPER_ROLE and float(action.get("confidence", 0.0)) < 0.95:
                reasons.append("goalkeeper_override_requires_high_confidence")

        if action_type == "merge_tracklets":
            if action.get("source_id") is None or action.get("target_id") is None:
                reasons.append("missing_merge_ids")
            if not action.get("non_overlapping"):
                reasons.append("merge_requires_non_overlapping_evidence")

        if reasons:
            rejected_actions.append(
                {
                    "action_id": action_id,
                    "action_type": action_type,
                    "reasons": reasons,
                }
            )
        else:
            accepted_action_ids.append(action_id)

    verdict = "PASS" if not rejected_actions else "FAIL"
    return {
        "schema_version": "1.0",
        "validator": "deterministic_pre_render_validator",
        "verdict": verdict,
        "accepted_action_count": len(accepted_action_ids),
        "rejected_action_count": len(rejected_actions),
        "accepted_action_ids": accepted_action_ids,
        "rejected_actions": rejected_actions,
    }


def build_dry_run_correction_plan(
    *,
    correction_candidates: dict[str, Any],
    render_audit: dict[str, Any],
    identity_events: dict[str, Any],
    planner: str = "deterministic_phase2",
) -> dict[str, Any]:
    """Build a Phase 2 dry-run correction plan without mutating tracklets."""
    actions: list[dict[str, Any]] = []
    needs_vision: list[dict[str, Any]] = []
    do_not_touch: list[dict[str, Any]] = []

    for candidate in correction_candidates.get("candidates", []):
        candidate_type = str(candidate.get("candidate_type", "unknown"))
        source_issue_id = str(candidate.get("source_issue_id") or "")
        proposed_action = candidate.get("proposed_action") or {}

        if candidate_type == "display_override":
            track_id = candidate.get("track_id")
            action_type = str(proposed_action.get("action_type", "display_override"))
            actions.append(
                {
                    "action_id": f"dry_run_{candidate.get('candidate_id')}",
                    "candidate_id": candidate.get("candidate_id"),
                    "action_type": action_type,
                    "status": "safe_fix_dry_run",
                    "confidence": 0.9,
                    "track_id": track_id,
                    "raw_track_id": candidate.get("raw_track_id"),
                    "first_source_frame_idx": candidate.get("first_source_frame_idx"),
                    "last_source_frame_idx": candidate.get("last_source_frame_idx"),
                    "set_display_role": proposed_action.get("set_display_role", "player"),
                    "set_display_label": proposed_action.get("set_display_label", str(track_id)),
                    "set_display_color_policy": proposed_action.get(
                        "set_display_color_policy",
                        "team",
                    ),
                    "evidence_ids": [source_issue_id] if source_issue_id else [],
                    "reason": candidate.get("reason"),
                    "dry_run_only": True,
                }
            )
        elif str(proposed_action.get("action_type")) == "needs_vision":
            needs_vision.append(
                {
                    "case_id": candidate.get("candidate_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "source_issue_id": candidate.get("source_issue_id"),
                    "question": "resolve_identity_or_goalkeeper_display_conflict",
                    "reason": candidate.get("reason"),
                    "dry_run_only": True,
                }
            )
        elif str(proposed_action.get("action_type")) == "needs_reid_topk_or_vision":
            needs_vision.append(
                {
                    "case_id": candidate.get("candidate_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "question": "goalkeeper_identity_fragmentation",
                    "reason": candidate.get("reason"),
                    "required_artifacts": ["reid_topk.json", "player_crop_index.json"],
                    "dry_run_only": True,
                }
            )
        else:
            do_not_touch.append(
                {
                    "case_id": candidate.get("candidate_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": candidate.get("reason") or "Unsupported candidate type.",
                    "dry_run_only": True,
                }
            )

    plan: dict[str, Any] = {
        "plan_version": "1.0",
        "phase": "phase_2_dry_run",
        "planner": planner,
        "baseline_image": render_audit.get("baseline_image", RUNPOD_BASELINE_IMAGE),
        "correction_applied": False,
        "summary": {
            "safe_fix_count": len(actions),
            "needs_vision_count": len(needs_vision),
            "do_not_touch_count": len(do_not_touch),
        },
        "actions": actions,
        "needs_vision": needs_vision,
        "do_not_touch": do_not_touch,
    }
    plan["validation"] = validate_correction_plan(
        plan,
        render_audit=render_audit,
        identity_events=identity_events,
    )
    return plan


def _action_matches_row(action: dict[str, Any], row: dict[str, Any]) -> bool:
    """Return whether a display action applies to a raw-tracklet row."""
    if str(row.get("object_type", "")) not in PERSON_OBJECT_TYPES:
        return False
    if _source_track_id(row) != _as_int(action.get("track_id"), -999999):
        return False
    action_raw_track_id = action.get("raw_track_id")
    if action_raw_track_id is not None and _raw_track_id(row) != _as_int(action_raw_track_id):
        return False
    frame = _frame_idx(row)
    return (
        _as_int(action.get("first_source_frame_idx"), -1) <= frame
        <= _as_int(action.get("last_source_frame_idx"), -1)
    )


def _safe_display_override_action(action: dict[str, Any], accepted_action_ids: set[str]) -> bool:
    """Return whether an action is safe enough for Phase 3 auto-application."""
    if str(action.get("action_id")) not in accepted_action_ids:
        return False
    if str(action.get("action_type")) != "display_override":
        return False
    if str(action.get("status")) != "safe_fix_dry_run":
        return False
    if str(action.get("set_display_role", "unknown")) == GOALKEEPER_ROLE:
        return False
    if str(action.get("set_display_label", "")).upper() == "GK":
        return False
    if str(action.get("set_display_color_policy", "")) != "team":
        return False
    return True


def apply_safe_correction_plan_to_raw_records(
    raw_tracklet_rows: list[dict[str, Any]],
    correction_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply validated Phase 3 display-only fixes to copied raw records."""
    corrected_rows = [deepcopy(row) for row in raw_tracklet_rows]
    validation = correction_plan.get("validation", {})
    if validation.get("verdict") != "PASS":
        return corrected_rows, {
            "schema_version": "1.0",
            "phase": "phase_3_safe_apply",
            "correction_applied": False,
            "applied_action_count": 0,
            "updated_record_count": 0,
            "skipped_reason": "plan_validation_not_pass",
            "applied_actions": [],
        }

    accepted_action_ids = {
        str(action_id)
        for action_id in validation.get("accepted_action_ids", [])
    }
    applied_actions: list[dict[str, Any]] = []
    updated_record_count = 0
    for action in correction_plan.get("actions", []):
        if not _safe_display_override_action(action, accepted_action_ids):
            continue

        action_update_count = 0
        for row in corrected_rows:
            if not _action_matches_row(action, row):
                continue
            row["display_role"] = str(action.get("set_display_role", "player"))
            row["display_label"] = str(action.get("set_display_label", row.get("track_id")))
            row["display_team"] = row.get("team")
            row["display_color"] = None
            row["goalkeeper_display_locked"] = False
            row["role_display_suppressed"] = False
            action_update_count += 1

        if action_update_count > 0:
            updated_record_count += action_update_count
            applied_actions.append(
                {
                    "action_id": action.get("action_id"),
                    "action_type": action.get("action_type"),
                    "track_id": action.get("track_id"),
                    "raw_track_id": action.get("raw_track_id"),
                    "first_source_frame_idx": action.get("first_source_frame_idx"),
                    "last_source_frame_idx": action.get("last_source_frame_idx"),
                    "updated_record_count": action_update_count,
                }
            )

    return corrected_rows, {
        "schema_version": "1.0",
        "phase": "phase_3_safe_apply",
        "correction_applied": bool(applied_actions),
        "applied_action_count": len(applied_actions),
        "updated_record_count": updated_record_count,
        "applied_actions": applied_actions,
    }


def post_fix_audit_improved(
    before_audit: dict[str, Any],
    after_audit: dict[str, Any],
) -> bool:
    """Return whether post-fix audit is safe to keep."""
    before_summary = before_audit.get("summary", {})
    after_summary = after_audit.get("summary", {})
    critical_issue_types = [
        "gk_false_positive_segment_count",
        "unsafe_gk_display_spread_count",
        "simultaneous_goalkeeper_conflict_count",
    ]
    for key in critical_issue_types:
        if _as_int(after_summary.get(key)) > _as_int(before_summary.get(key)):
            return False
    if after_audit.get("verdict") == "FAIL" and before_audit.get("verdict") != "FAIL":
        return False
    return _as_int(after_audit.get("score")) >= _as_int(before_audit.get("score"))


def build_vision_review_queue(
    *,
    correction_plan: dict[str, Any],
    render_audit_before: dict[str, Any],
    render_audit_after: dict[str, Any] | None = None,
    correction_applied: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Phase 4 vision escalation queue for unresolved identity cases."""
    render_audit_after = render_audit_after or render_audit_before
    correction_applied = correction_applied or {}
    cases: list[dict[str, Any]] = []

    unresolved_issue_ids = {
        str(issue.get("issue_id"))
        for issue in render_audit_after.get("issues", [])
        if issue.get("issue_id")
    }

    for item in correction_plan.get("needs_vision", []):
        source_issue_id = str(item.get("source_issue_id") or "")
        if source_issue_id and source_issue_id not in unresolved_issue_ids:
            continue
        priority = "high" if source_issue_id in unresolved_issue_ids else "medium"
        question = str(item.get("question") or "identity_review")
        required_artifacts = list(item.get("required_artifacts") or [])
        if "player_crop_index.json" not in required_artifacts:
            required_artifacts.append("player_crop_index.json")
        if "contact_sheets" not in required_artifacts:
            required_artifacts.append("contact_sheets")
        cases.append(
            {
                "case_id": item.get("case_id") or item.get("candidate_id"),
                "source_issue_id": source_issue_id or None,
                "question": question,
                "priority": priority,
                "status": "pending_vision",
                "reason": item.get("reason"),
                "required_artifacts": required_artifacts,
                "instructions": (
                    "Use paired crops/contact sheets only to answer the specific "
                    "question. Return structured JSON; do not invent jersey numbers."
                ),
            }
        )

    if (
        correction_applied.get("candidate_correction_applied")
        and not correction_applied.get("correction_applied")
    ):
        cases.append(
            {
                "case_id": "rollback_post_fix_audit_not_improved",
                "source_issue_id": None,
                "question": "review_rollback_reason",
                "priority": "high",
                "status": "pending_vision_or_human_review",
                "reason": correction_applied.get("rollback_reason"),
                "required_artifacts": [
                    "render_audit_before.json",
                    "render_audit_after.json",
                    "correction_plan.json",
                    "player_crop_index.json",
                    "contact_sheets",
                ],
                "instructions": (
                    "Review why the deterministic safe fix did not improve the "
                    "post-fix audit before allowing any final identity correction."
                ),
            }
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    cases = sorted(
        cases,
        key=lambda item: (
            severity_rank.get(str(item.get("priority")), 9),
            str(item.get("case_id")),
        ),
    )
    return {
        "schema_version": "1.0",
        "phase": "phase_4_vision_on_demand_queue",
        "vision_model_invoked": False,
        "case_count": len(cases),
        "cases": cases,
        "notes": [
            "This queue prepares Vision-on-demand; it does not run a vision model.",
            "Only unresolved or unsafe cases should enter this queue.",
        ],
    }


def _bbox_area_from_row(row: dict[str, Any]) -> float:
    """Return bbox area for crop ranking."""
    bbox = row.get("bbox") or [0, 0, 0, 0]
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _issue_by_id(render_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return render-audit issues keyed by issue id."""
    return {
        str(issue.get("issue_id")): issue
        for issue in render_audit.get("issues", [])
        if issue.get("issue_id")
    }


def _target_tracks_for_case(
    case: dict[str, Any],
    render_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return track targets that should get visual evidence for one case."""
    issues = _issue_by_id(render_audit)
    source_issue_id = str(case.get("source_issue_id") or "")
    issue = issues.get(source_issue_id, {})
    targets: list[dict[str, Any]] = []

    if issue.get("track_id") is not None:
        targets.append(
            {
                "track_id": _as_int(issue.get("track_id")),
                "raw_track_id": issue.get("raw_track_id"),
                "first_source_frame_idx": issue.get("first_source_frame_idx"),
                "last_source_frame_idx": issue.get("last_source_frame_idx"),
                "source_issue_id": source_issue_id,
            }
        )

    for track_id in issue.get("player_dominant_source_track_ids", []) or []:
        targets.append(
            {
                "track_id": _as_int(track_id),
                "raw_track_id": None,
                "first_source_frame_idx": None,
                "last_source_frame_idx": None,
                "source_issue_id": source_issue_id,
            }
        )

    source_track_ids = issue.get("source_track_ids") or {}
    if isinstance(source_track_ids, dict):
        for track_id in source_track_ids:
            targets.append(
                {
                    "track_id": _as_int(track_id),
                    "raw_track_id": None,
                    "first_source_frame_idx": None,
                    "last_source_frame_idx": None,
                    "source_issue_id": source_issue_id,
                }
            )

    if str(case.get("question")) == "goalkeeper_identity_fragmentation":
        seen_track_ids = {target["track_id"] for target in targets}
        for segment in render_audit.get("goalkeeper_display_segments", [])[:12]:
            track_id = _as_int(segment.get("track_id"))
            if track_id in seen_track_ids:
                continue
            seen_track_ids.add(track_id)
            targets.append(
                {
                    "track_id": track_id,
                    "raw_track_id": segment.get("raw_track_id"),
                    "first_source_frame_idx": segment.get("first_source_frame_idx"),
                    "last_source_frame_idx": segment.get("last_source_frame_idx"),
                    "source_issue_id": source_issue_id or None,
                }
            )

    unique: dict[tuple[int, Any, Any, Any], dict[str, Any]] = {}
    for target in targets:
        key = (
            _as_int(target.get("track_id")),
            target.get("raw_track_id"),
            target.get("first_source_frame_idx"),
            target.get("last_source_frame_idx"),
        )
        unique[key] = target
    return list(unique.values())


def _select_crop_rows(
    raw_tracklet_rows: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    """Select high-value crop rows for one target."""
    track_id = _as_int(target.get("track_id"), -999999)
    raw_track_id = target.get("raw_track_id")
    first_frame = target.get("first_source_frame_idx")
    last_frame = target.get("last_source_frame_idx")
    rows = [
        row
        for row in _iter_person_rows(raw_tracklet_rows)
        if _source_track_id(row) == track_id
    ]
    if raw_track_id is not None:
        rows = [row for row in rows if _raw_track_id(row) == _as_int(raw_track_id)]
    if first_frame is not None and last_frame is not None:
        start = _as_int(first_frame)
        end = _as_int(last_frame)
        ranged_rows = [row for row in rows if start <= _frame_idx(row) <= end]
        if ranged_rows:
            rows = ranged_rows

    def row_score(row: dict[str, Any]) -> tuple[float, float, int]:
        confidence = _as_float(row.get("confidence"), 0.0)
        display_bonus = 1 if _is_display_goalkeeper(row) else 0
        return (display_bonus, confidence, _bbox_area_from_row(row))

    rows = sorted(rows, key=row_score, reverse=True)
    selected: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    for row in rows:
        frame = _frame_idx(row)
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        selected.append(row)
        if len(selected) >= max_rows:
            break
    return sorted(selected, key=_frame_idx)


def build_player_crop_index_plan(
    *,
    raw_tracklet_rows: list[dict[str, Any]],
    vision_review_queue: dict[str, Any],
    render_audit: dict[str, Any],
    max_crops_per_track: int = 3,
    max_crops_per_case: int = 12,
) -> dict[str, Any]:
    """Build deterministic crop requests for Phase 5 visual evidence."""
    cases: list[dict[str, Any]] = []
    total_crop_requests = 0

    for case in vision_review_queue.get("cases", []):
        crop_requests: list[dict[str, Any]] = []
        targets = _target_tracks_for_case(case, render_audit)
        for target in targets:
            for row in _select_crop_rows(
                raw_tracklet_rows,
                target,
                max_rows=max(1, int(max_crops_per_track)),
            ):
                crop_id = (
                    f"{case.get('case_id')}_track{_source_track_id(row)}_"
                    f"frame{_frame_idx(row)}"
                )
                crop_requests.append(
                    {
                        "crop_id": str(crop_id).replace("/", "_"),
                        "case_id": case.get("case_id"),
                        "track_id": _source_track_id(row),
                        "raw_track_id": _raw_track_id(row),
                        "source_frame_idx": _frame_idx(row),
                        "sample_number": _sample_number(row),
                        "bbox": row.get("bbox"),
                        "role": _raw_role(row),
                        "detected_role": _detected_role(row),
                        "team": _as_int(row.get("team")),
                        "display_label": _visible_label(row),
                        "display_role": _visible_role(row),
                        "display_team": _visible_team(row),
                        "confidence": row.get("confidence"),
                        "crop_path": None,
                    }
                )
                if len(crop_requests) >= max(1, int(max_crops_per_case)):
                    break
            if len(crop_requests) >= max(1, int(max_crops_per_case)):
                break

        total_crop_requests += len(crop_requests)
        cases.append(
            {
                "case_id": case.get("case_id"),
                "question": case.get("question"),
                "priority": case.get("priority"),
                "status": "crop_requests_ready" if crop_requests else "no_matching_track_rows",
                "source_issue_id": case.get("source_issue_id"),
                "reason": case.get("reason"),
                "target_count": len(targets),
                "crop_request_count": len(crop_requests),
                "crop_requests": crop_requests,
                "contact_sheet_path": None,
            }
        )

    return {
        "schema_version": "1.0",
        "phase": "phase_5_crop_evidence_plan",
        "vision_model_invoked": False,
        "case_count": len(cases),
        "total_crop_request_count": total_crop_requests,
        "cases": cases,
    }


def _crop_cases_by_id(player_crop_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return crop evidence cases keyed by case id."""
    return {
        str(case.get("case_id")): case
        for case in player_crop_index.get("cases", [])
        if case.get("case_id") is not None
    }


def _vision_evidence_for_case(crop_case: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact crop evidence metadata for a vision-review result."""
    evidence: list[dict[str, Any]] = []
    for crop in crop_case.get("crop_requests", []):
        evidence.append(
            {
                "crop_id": crop.get("crop_id"),
                "crop_path": crop.get("crop_path"),
                "status": crop.get("status"),
                "track_id": crop.get("track_id"),
                "raw_track_id": crop.get("raw_track_id"),
                "source_frame_idx": crop.get("source_frame_idx"),
                "bbox": crop.get("bbox"),
                "role": crop.get("role"),
                "detected_role": crop.get("detected_role"),
                "display_label": crop.get("display_label"),
                "display_role": crop.get("display_role"),
            }
        )
    return evidence


def _sanitize_model_output(output: dict[str, Any]) -> dict[str, Any]:
    """Normalize externally supplied vision output without trusting it blindly."""
    verdict = str(output.get("verdict") or "unresolved")
    status = str(output.get("status") or "reviewed")
    try:
        confidence = float(output.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "status": status,
        "verdict": verdict,
        "confidence": confidence,
        "reason": str(output.get("reason") or ""),
        "model_evidence": output.get("evidence") if isinstance(output.get("evidence"), list) else [],
        "raw_output": output,
    }


def build_vision_review_results(
    *,
    vision_review_queue: dict[str, Any],
    player_crop_index: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    provider_enabled: bool = False,
    model_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build Phase 6 structured vision review results.

    No visual judgement is made unless an external provider explicitly supplies
    model outputs. The default path is intentionally conservative: every queued
    case remains unresolved instead of being guessed from text-only evidence.
    """
    crop_cases = _crop_cases_by_id(player_crop_index)
    outputs_by_case = {
        str(output.get("case_id")): output
        for output in (model_outputs or [])
        if isinstance(output, dict) and output.get("case_id") is not None
    }
    vision_model_invoked = bool(provider_enabled and provider and model and model_outputs is not None)
    results: list[dict[str, Any]] = []

    for case in vision_review_queue.get("cases", []):
        case_id = str(case.get("case_id") or "unknown_case")
        crop_case = crop_cases.get(case_id, {})
        evidence = _vision_evidence_for_case(crop_case)
        contact_sheet_path = crop_case.get("contact_sheet_path")

        if not vision_model_invoked:
            status = "not_invoked"
            verdict = "unresolved"
            confidence = 0.0
            reason = (
                "Vision provider is not configured/enabled; no visual decision was made."
            )
            limits = ["no_vision_provider_configured"]
            model_evidence: list[Any] = []
        elif case_id not in outputs_by_case:
            status = "missing_model_output"
            verdict = "unresolved"
            confidence = 0.0
            reason = "Vision provider ran, but no structured result was supplied for this case."
            limits = ["missing_model_output"]
            model_evidence = []
        else:
            sanitized = _sanitize_model_output(outputs_by_case[case_id])
            status = sanitized["status"]
            verdict = sanitized["verdict"]
            confidence = sanitized["confidence"]
            reason = sanitized["reason"]
            limits = []
            model_evidence = sanitized["model_evidence"]

        results.append(
            {
                "case_id": case_id,
                "source_issue_id": case.get("source_issue_id"),
                "question": case.get("question"),
                "priority": case.get("priority"),
                "status": status,
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
                "recommended_action": (
                    "keep_unresolved_no_identity_mutation"
                    if verdict == "unresolved"
                    else "requires_validator_before_identity_mutation"
                ),
                "contact_sheet_path": contact_sheet_path,
                "crop_count": len(evidence),
                "evidence": evidence,
                "model_evidence": model_evidence,
                "limits": limits,
            }
        )

    payload = {
        "schema_version": "1.0",
        "phase": "phase_6_vision_review_results",
        "vision_model_invoked": vision_model_invoked,
        "provider": provider if vision_model_invoked else None,
        "model": model if vision_model_invoked else None,
        "case_count": len(results),
        "unresolved_count": sum(1 for result in results if result.get("verdict") == "unresolved"),
        "results": results,
        "notes": [
            "Phase 6 records structured vision-review outcomes before final correction.",
            "Without an enabled provider and explicit model outputs, all cases remain unresolved.",
        ],
    }
    payload["validation"] = validate_vision_review_results(
        payload,
        vision_review_queue=vision_review_queue,
    )
    return payload


def validate_vision_review_results(
    vision_review_results: dict[str, Any],
    *,
    vision_review_queue: dict[str, Any],
) -> dict[str, Any]:
    """Validate Phase 6 review results before any future correction step."""
    queued_case_ids = {
        str(case.get("case_id"))
        for case in vision_review_queue.get("cases", [])
        if case.get("case_id") is not None
    }
    result_case_ids: set[str] = set()
    rejected_results: list[dict[str, Any]] = []
    vision_model_invoked = bool(vision_review_results.get("vision_model_invoked"))

    for result in vision_review_results.get("results", []):
        case_id = str(result.get("case_id") or "")
        result_case_ids.add(case_id)
        reasons: list[str] = []
        status = str(result.get("status") or "")
        verdict = str(result.get("verdict") or "")
        confidence = _as_float(result.get("confidence"), -1.0)

        if not case_id:
            reasons.append("missing_case_id")
        elif case_id not in queued_case_ids:
            reasons.append("unknown_case_id")
        if status not in VISION_REVIEW_STATUSES:
            reasons.append("invalid_status")
        if verdict not in VISION_REVIEW_VERDICTS:
            reasons.append("invalid_verdict")
        if confidence < 0.0 or confidence > 1.0:
            reasons.append("invalid_confidence")
        if not isinstance(result.get("evidence"), list):
            reasons.append("evidence_must_be_list")

        if not vision_model_invoked:
            if status != "not_invoked":
                reasons.append("not_invoked_requires_not_invoked_status")
            if verdict != "unresolved":
                reasons.append("not_invoked_must_stay_unresolved")
            if confidence != 0.0:
                reasons.append("not_invoked_confidence_must_be_zero")
        elif verdict != "unresolved":
            if status != "reviewed":
                reasons.append("resolved_verdict_requires_reviewed_status")
            if confidence < 0.70:
                reasons.append("resolved_verdict_requires_confidence")
            if not result.get("model_evidence"):
                reasons.append("resolved_verdict_requires_model_evidence")

        if reasons:
            rejected_results.append(
                {
                    "case_id": case_id or None,
                    "status": status,
                    "verdict": verdict,
                    "reasons": reasons,
                }
            )

    missing_case_ids = sorted(queued_case_ids - result_case_ids)
    for case_id in missing_case_ids:
        rejected_results.append(
            {
                "case_id": case_id,
                "status": None,
                "verdict": None,
                "reasons": ["missing_result_for_queued_case"],
            }
        )

    verdict = "PASS" if not rejected_results else "FAIL"
    return {
        "schema_version": "1.0",
        "validator": "deterministic_phase6_vision_result_validator",
        "verdict": verdict,
        "queued_case_count": len(queued_case_ids),
        "result_count": len(result_case_ids),
        "rejected_result_count": len(rejected_results),
        "missing_case_ids": missing_case_ids,
        "rejected_results": rejected_results,
    }


def _high_risk_issue_count(render_audit: dict[str, Any]) -> int:
    """Return count of issues that make identity output review-required."""
    return sum(
        1
        for issue in render_audit.get("issues", [])
        if str(issue.get("severity")) in {"critical", "high"}
    )


def build_final_render_identity_manifest(
    *,
    render_audit_after: dict[str, Any],
    correction_applied: dict[str, Any],
    vision_review_queue: dict[str, Any],
    vision_review_results: dict[str, Any],
    rendered_output_frames: int,
    baseline_image: str = RUNPOD_BASELINE_IMAGE,
) -> dict[str, Any]:
    """Build Phase 7 final render/identity integration manifest.

    The manifest does not block video creation. It classifies whether the
    rendered output can be treated as identity-trusted or must remain a review
    output with supporting artifacts.
    """
    audit_verdict = str(render_audit_after.get("verdict") or "UNKNOWN")
    audit_score = _as_int(render_audit_after.get("score"), 0)
    high_risk_issue_count = _high_risk_issue_count(render_audit_after)
    unresolved_results = [
        result
        for result in vision_review_results.get("results", [])
        if str(result.get("verdict")) == "unresolved"
    ]
    unresolved_case_count = len(unresolved_results)
    vision_validation = vision_review_results.get("validation") or {}
    correction_validation = correction_applied.get("validation") or {}
    final_video_produced = int(rendered_output_frames) > 0

    blockers: list[str] = []
    review_reasons: list[str] = []
    if not final_video_produced:
        blockers.append("rendered_output_has_no_frames")
    if vision_validation.get("verdict") not in {None, "PASS"}:
        blockers.append("vision_review_results_failed_validation")
    if audit_verdict == "FAIL" or high_risk_issue_count > 0:
        review_reasons.append("post_correction_audit_has_high_risk_identity_issues")
    if unresolved_case_count > 0:
        review_reasons.append("vision_review_cases_unresolved")
    if correction_applied.get("candidate_correction_applied") and not correction_applied.get("kept"):
        review_reasons.append("candidate_correction_was_rolled_back")

    if blockers:
        release_status = "invalid_render"
        output_identity_mode = "not_released"
        render_policy = "do_not_claim_final_identity"
    elif review_reasons:
        release_status = "review_required"
        output_identity_mode = "review_output_with_identity_artifacts"
        render_policy = "produce_video_with_identity_review_artifacts"
    else:
        release_status = "identity_trusted"
        output_identity_mode = "final_identity_output"
        render_policy = "produce_standard_final_video"

    manifest = {
        "schema_version": "1.0",
        "phase": "phase_7_final_render_integration",
        "baseline_image": baseline_image,
        "known_bad_images": KNOWN_BAD_RUNPOD_IMAGES,
        "final_video_produced": final_video_produced,
        "rendered_output_frames": int(rendered_output_frames),
        "release_status": release_status,
        "output_identity_mode": output_identity_mode,
        "render_policy": render_policy,
        "deterministic_correction_applied": bool(
            correction_applied.get("correction_applied")
        ),
        "vision_model_invoked": bool(vision_review_results.get("vision_model_invoked")),
        "audit": {
            "verdict": audit_verdict,
            "score": audit_score,
            "summary": render_audit_after.get("summary", {}),
            "high_risk_issue_count": high_risk_issue_count,
        },
        "vision_review": {
            "queued_case_count": _as_int(vision_review_queue.get("case_count"), 0),
            "result_count": _as_int(vision_review_results.get("case_count"), 0),
            "unresolved_case_count": unresolved_case_count,
            "validation": vision_validation,
        },
        "correction": {
            "correction_applied": bool(correction_applied.get("correction_applied")),
            "candidate_correction_applied": bool(
                correction_applied.get("candidate_correction_applied")
            ),
            "kept": bool(correction_applied.get("kept")),
            "updated_annotation_track_count": _as_int(
                correction_applied.get("updated_annotation_track_count"),
                0,
            ),
            "validation": correction_validation,
            "rollback_reason": correction_applied.get("rollback_reason"),
        },
        "blockers": blockers,
        "review_reasons": review_reasons,
        "unresolved_cases": [
            {
                "case_id": result.get("case_id"),
                "question": result.get("question"),
                "priority": result.get("priority"),
                "reason": result.get("reason"),
                "contact_sheet_path": result.get("contact_sheet_path"),
            }
            for result in unresolved_results[:50]
        ],
        "unresolved_cases_truncated": len(unresolved_results) > 50,
        "required_review_artifacts": [
            "render_audit_after.json",
            "correction_plan.json",
            "correction_applied.json",
            "vision_review_queue.json",
            "player_crop_index.json",
            "vision_review_results.json",
        ],
    }
    manifest["validation"] = validate_final_render_identity_manifest(manifest)
    return manifest


def validate_final_render_identity_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate Phase 7 manifest consistency."""
    reasons: list[str] = []
    release_status = str(manifest.get("release_status") or "")
    output_identity_mode = str(manifest.get("output_identity_mode") or "")
    render_policy = str(manifest.get("render_policy") or "")
    final_video_produced = bool(manifest.get("final_video_produced"))
    blockers = manifest.get("blockers")
    review_reasons = manifest.get("review_reasons")

    if release_status not in {"identity_trusted", "review_required", "invalid_render"}:
        reasons.append("invalid_release_status")
    if output_identity_mode not in {
        "final_identity_output",
        "review_output_with_identity_artifacts",
        "not_released",
    }:
        reasons.append("invalid_output_identity_mode")
    if render_policy not in {
        "produce_standard_final_video",
        "produce_video_with_identity_review_artifacts",
        "do_not_claim_final_identity",
    }:
        reasons.append("invalid_render_policy")
    if not isinstance(blockers, list):
        reasons.append("blockers_must_be_list")
        blockers = []
    if not isinstance(review_reasons, list):
        reasons.append("review_reasons_must_be_list")
        review_reasons = []

    if not final_video_produced and release_status != "invalid_render":
        reasons.append("missing_video_requires_invalid_render")
    if blockers and release_status != "invalid_render":
        reasons.append("blockers_require_invalid_render")
    if release_status == "identity_trusted":
        if blockers:
            reasons.append("trusted_output_cannot_have_blockers")
        if review_reasons:
            reasons.append("trusted_output_cannot_have_review_reasons")
        if _as_int(manifest.get("vision_review", {}).get("unresolved_case_count"), 0):
            reasons.append("trusted_output_cannot_have_unresolved_cases")
        if _as_int(manifest.get("audit", {}).get("high_risk_issue_count"), 0):
            reasons.append("trusted_output_cannot_have_high_risk_issues")
    if release_status == "review_required" and not review_reasons:
        reasons.append("review_required_needs_review_reason")

    verdict = "PASS" if not reasons else "FAIL"
    return {
        "schema_version": "1.0",
        "validator": "deterministic_phase7_final_render_validator",
        "verdict": verdict,
        "reasons": reasons,
    }
