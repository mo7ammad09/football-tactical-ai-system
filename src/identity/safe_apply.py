"""Safe application layer for Phase 9 identity resolution proposals.

Phase 10 never trusts a model output directly. It only applies proposals that
already passed the Phase 9 resolver validator, then records metadata that later
stages can use to promote identity-trusted output. Client-facing colors and
roles are kept unchanged unless a later validator explicitly allows them.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from src.identity.resolver import (
    LOCK_DISPLAY_ROLE_ACTION,
    LOCK_DISPLAY_TEAM_ACTION,
    MARK_IDENTITY_CLUSTER_REQUIRED_ACTION,
    MARK_SEGMENT_SPLIT_REQUIRED_ACTION,
    MERGE_CANDIDATE_ACTION,
    MERGE_GOALKEEPER_ACTION,
    READY_STATUS,
)
from src.utils.annotation_colors import (
    GOALKEEPER_COLOR,
    PLAYER_FALLBACK_COLOR,
    is_goalkeeper_color,
    normalize_color,
)


APPLIED_STATUS = "applied"
NOOP_STATUS = "no_ready_proposals"
REJECTED_STATUS = "rejected"
GOALKEEPER_DISPLAY_ID = 900_000

SUPPORTED_ACTIONS = {
    MERGE_GOALKEEPER_ACTION,
    MERGE_CANDIDATE_ACTION,
    LOCK_DISPLAY_TEAM_ACTION,
    LOCK_DISPLAY_ROLE_ACTION,
}


def _as_int(value: Any, fallback: int = 0) -> int:
    """Convert value to int with a stable fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: Any, fallback: float = 0.0) -> float:
    """Convert value to float with a stable fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _target_track_ids(proposal: dict[str, Any]) -> list[int]:
    """Return unique positive target track ids for a proposal."""
    return sorted(
        {
            _as_int(track_id, -1)
            for track_id in proposal.get("target_track_ids", []) or []
            if _as_int(track_id, -1) >= 0
        }
    )


def _source_frame_idx(row: dict[str, Any]) -> int:
    """Return a row source-frame index."""
    return _as_int(row.get("source_frame_idx"), -1)


def _row_raw_track_id(row: dict[str, Any]) -> int | None:
    """Return row raw_track_id when available."""
    if row.get("raw_track_id") is None:
        return None
    return _as_int(row.get("raw_track_id"), -1)


def _ready_proposals(identity_resolution_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return proposals that are eligible for Phase 10 safe apply."""
    if identity_resolution_plan.get("validation", {}).get("verdict") != "PASS":
        return []
    return [
        proposal
        for proposal in identity_resolution_plan.get("resolution_proposals", []) or []
        if isinstance(proposal, dict)
        and proposal.get("status") == READY_STATUS
        and proposal.get("proposed_action") in SUPPORTED_ACTIONS
    ]


def _topology_guard_proposals(identity_resolution_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return non-merge topology proposals that should protect the render."""
    if identity_resolution_plan.get("validation", {}).get("verdict") != "PASS":
        return []
    proposals: list[dict[str, Any]] = []
    for proposal in identity_resolution_plan.get("resolution_proposals", []) or []:
        if not isinstance(proposal, dict):
            continue
        action = str(proposal.get("proposed_action") or "")
        proposal_type = str(proposal.get("proposal_type") or "")
        if action in {
            MARK_SEGMENT_SPLIT_REQUIRED_ACTION,
            MARK_IDENTITY_CLUSTER_REQUIRED_ACTION,
        } or proposal_type in {"segment_split_required", "identity_cluster_required"}:
            proposals.append(proposal)
    return proposals


def _target_display_team(proposal: dict[str, Any]) -> int:
    """Return target display team from proposal evidence."""
    return _as_int((proposal.get("evidence") or {}).get("target_display_team"), 0)


def _target_display_role(proposal: dict[str, Any]) -> str:
    """Return target display role from proposal evidence."""
    return str((proposal.get("evidence") or {}).get("target_display_role") or "")


def _safe_color_value(value: Any) -> tuple[int, int, int] | None:
    """Return a non-goalkeeper display color or None."""
    if value is None or is_goalkeeper_color(value):
        return None
    color = normalize_color(value, PLAYER_FALLBACK_COLOR)
    return None if is_goalkeeper_color(color) else color


def _canonical_team_color(
    rows: list[dict[str, Any]],
    *,
    target_track_ids: set[int],
    target_team: int,
) -> tuple[int, int, int]:
    """Choose a stable color for a display-team lock from matching rows."""
    colors: Counter[tuple[int, int, int]] = Counter()
    for row in rows:
        if _as_int(row.get("track_id"), -1) not in target_track_ids:
            continue
        row_team = _as_int(row.get("team"), _as_int(row.get("display_team"), 0))
        if row_team != target_team:
            continue
        color = _safe_color_value(row.get("team_color")) or _safe_color_value(
            row.get("display_color")
        )
        if color is not None:
            colors[color] += 1
    if colors:
        return colors.most_common(1)[0][0]
    return PLAYER_FALLBACK_COLOR


def _row_player_color(row: dict[str, Any]) -> tuple[int, int, int]:
    """Return a non-goalkeeper color suitable for player display locks."""
    return (
        _safe_color_value(row.get("team_color"))
        or _safe_color_value(row.get("display_color"))
        or PLAYER_FALLBACK_COLOR
    )


def validate_identity_safe_apply_plan(
    identity_resolution_plan: dict[str, Any],
) -> dict[str, Any]:
    """Validate that Phase 10 may consume a Phase 9 resolution plan."""
    rejected: list[dict[str, Any]] = []
    plan_validation = identity_resolution_plan.get("validation") or {}
    if plan_validation.get("verdict") != "PASS":
        rejected.append(
            {
                "proposal_id": None,
                "reasons": ["phase9_resolution_plan_failed_validation"],
            }
        )

    for proposal in identity_resolution_plan.get("resolution_proposals", []) or []:
        if not isinstance(proposal, dict):
            rejected.append(
                {
                    "proposal_id": None,
                    "reasons": ["proposal_must_be_dict"],
                }
            )
            continue
        if proposal.get("status") != READY_STATUS:
            continue

        reasons: list[str] = []
        action = str(proposal.get("proposed_action") or "")
        confidence = _as_float(proposal.get("confidence"), 0.0)
        targets = _target_track_ids(proposal)
        apply_policy = proposal.get("apply_policy") or {}

        if action not in SUPPORTED_ACTIONS:
            reasons.append("unsupported_ready_action")
        if confidence < 0.60:
            reasons.append("ready_proposal_confidence_below_threshold")
        if action in {MERGE_GOALKEEPER_ACTION, MERGE_CANDIDATE_ACTION} and len(targets) < 2:
            reasons.append("ready_proposal_needs_multiple_targets")
        if action in {LOCK_DISPLAY_TEAM_ACTION, LOCK_DISPLAY_ROLE_ACTION} and len(targets) != 1:
            reasons.append("display_lock_needs_one_target")
        if not apply_policy.get("requires_safe_apply_validator"):
            reasons.append("proposal_did_not_require_safe_apply_validator")
        if action == MERGE_GOALKEEPER_ACTION:
            evidence = proposal.get("evidence") or {}
            if evidence.get("vision_verdict") != "same_player":
                reasons.append("goalkeeper_apply_requires_same_player_verdict")
            if evidence.get("vision_status") != "reviewed":
                reasons.append("goalkeeper_apply_requires_reviewed_status")
        if action == LOCK_DISPLAY_TEAM_ACTION:
            evidence = proposal.get("evidence") or {}
            target_team = _target_display_team(proposal)
            if target_team not in {1, 2}:
                reasons.append("team_lock_apply_requires_target_team")
            if evidence.get("vision_verdict") != f"team_{target_team}":
                reasons.append("team_lock_apply_requires_matching_verdict")
            if not apply_policy.get("display_only"):
                reasons.append("team_lock_apply_must_be_display_only")
        if action == LOCK_DISPLAY_ROLE_ACTION:
            evidence = proposal.get("evidence") or {}
            target_role = _target_display_role(proposal)
            if target_role not in {"player", "referee"}:
                reasons.append("role_lock_apply_requires_target_role")
            if evidence.get("vision_verdict") != target_role:
                reasons.append("role_lock_apply_requires_matching_verdict")
            if not apply_policy.get("display_only"):
                reasons.append("role_lock_apply_must_be_display_only")

        if reasons:
            rejected.append(
                {
                    "proposal_id": proposal.get("proposal_id"),
                    "proposed_action": action,
                    "reasons": reasons,
                }
            )

    return {
        "schema_version": "1.0",
        "validator": "deterministic_phase10_safe_apply_validator",
        "verdict": "PASS" if not rejected else "FAIL",
        "ready_proposal_count": len(_ready_proposals(identity_resolution_plan)),
        "rejected_proposal_count": len(rejected),
        "rejected_proposals": rejected,
    }


def _row_visible_goalkeeper(row: dict[str, Any]) -> bool:
    """Return whether a raw row is currently rendered as goalkeeper."""
    return (
        str(row.get("display_role") or row.get("role") or "").lower() == "goalkeeper"
        or str(row.get("display_label") or "").upper() == "GK"
    )


def _row_goalkeeper_evidence(row: dict[str, Any]) -> bool:
    """Return whether a row has goalkeeper evidence before display mutation."""
    role = str(row.get("role") or "").lower()
    detected_role = str(row.get("detected_role") or "").lower()
    display_role = str(row.get("display_role") or "").lower()
    label = str(row.get("display_label") or "").upper()
    return (
        role == "goalkeeper"
        or detected_role == "goalkeeper"
        or display_role == "goalkeeper"
        or label == "GK"
    )


def _track_visible_goalkeeper(track: dict[str, Any]) -> bool:
    """Return whether an annotation track is currently rendered as goalkeeper."""
    return (
        str(track.get("display_role") or track.get("role") or "").lower()
        == "goalkeeper"
        or str(track.get("display_label") or "").upper() == "GK"
    )


def _track_goalkeeper_evidence(track: dict[str, Any]) -> bool:
    """Return whether an annotation track has goalkeeper evidence."""
    role = str(track.get("role") or "").lower()
    detected_role = str(track.get("detected_role") or "").lower()
    display_role = str(track.get("display_role") or "").lower()
    label = str(track.get("display_label") or "").upper()
    return (
        role == "goalkeeper"
        or detected_role == "goalkeeper"
        or display_role == "goalkeeper"
        or label == "GK"
    )


def _proposal_resolution_payload(proposal: dict[str, Any]) -> dict[str, Any]:
    """Build compact metadata written by Phase 10."""
    action = str(proposal.get("proposed_action") or "")
    if action == MERGE_GOALKEEPER_ACTION:
        resolved_identity_id: str | int = "gk"
        resolved_identity_label = "GK"
    else:
        targets = _target_track_ids(proposal)
        resolved_identity_id = targets[0] if targets else "unresolved"
        resolved_identity_label = str(resolved_identity_id)

    return {
        "resolved_identity_id": resolved_identity_id,
        "resolved_identity_label": resolved_identity_label,
        "identity_resolution_case_id": proposal.get("case_id"),
        "identity_resolution_proposal_id": proposal.get("proposal_id"),
        "identity_resolution_action": action,
        "identity_resolution_confidence": _as_float(proposal.get("confidence"), 0.0),
        "identity_resolution_status": "phase10_safe_applied",
    }


def _proposal_raw_track_ids(proposal: dict[str, Any]) -> set[int]:
    """Return raw track ids explicitly referenced by proposal evidence."""
    evidence = proposal.get("evidence") or {}
    raw_ids = evidence.get("raw_track_ids") or []
    parsed = {_as_int(raw_id, -1) for raw_id in raw_ids}
    return {raw_id for raw_id in parsed if raw_id >= 0}


def _proposal_frame_span(proposal: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return proposal frame span when available."""
    span = (proposal.get("evidence") or {}).get("frame_span") or {}
    first = _as_int(span.get("first_source_frame_idx"), -1)
    last = _as_int(span.get("last_source_frame_idx"), -1)
    if first < 0 or last < 0:
        return None, None
    return first, last


def _in_proposal_scope(row: dict[str, Any], proposal: dict[str, Any]) -> bool:
    """Return whether a row belongs to a proposal's explicit evidence scope."""
    frame = _source_frame_idx(row)
    first, last = _proposal_frame_span(proposal)
    if first is not None and last is not None and not (first <= frame <= last):
        return False
    raw_ids = _proposal_raw_track_ids(proposal)
    raw_id = _row_raw_track_id(row)
    if raw_ids and raw_id is not None:
        return raw_id in raw_ids
    return True


def _is_goalkeeper_cluster_proposal(proposal: dict[str, Any]) -> bool:
    """Return whether a cluster proposal is specifically about the goalkeeper."""
    case_id = str(proposal.get("case_id") or "").lower()
    proposal_type = str(proposal.get("proposal_type") or "").lower()
    evidence = proposal.get("evidence") or {}
    question = str(evidence.get("question") or "").lower()
    reason = str(evidence.get("identity_cluster_reason") or proposal.get("reason") or "").lower()
    return (
        "goalkeeper" in case_id
        or "goalkeeper" in proposal_type
        or question == "goalkeeper_identity_fragmentation"
        or "goalkeeper" in reason
    )


def _segment_key(row: dict[str, Any]) -> tuple[str, int, int | None]:
    """Return a conservative segment key for a mixed display track."""
    role = str(row.get("role") or row.get("detected_role") or row.get("display_role") or "player")
    if role not in {"player", "goalkeeper", "referee"}:
        role = "player"
    if role == "goalkeeper":
        # Segment-split cases are meant to stop a mixed track from sharing one
        # player ID. They must not promote detector GK flicker to a rendered GK;
        # only an explicit goalkeeper identity-cluster proposal may do that.
        role = "player"
    if role == "referee":
        team = 0
    else:
        team = _as_int(row.get("team"), _as_int(row.get("display_team"), 0))
    return role, team, _row_raw_track_id(row)


def _segment_label(base_track_id: int, segment_index: int, role: str) -> str:
    """Build a short display label for a protected segment."""
    if role == "referee":
        return "REF"
    if role == "goalkeeper":
        return "GK"
    suffix = chr(ord("A") + min(max(segment_index - 1, 0), 25))
    return f"{base_track_id}{suffix}"


def _build_segment_specs(
    rows: list[dict[str, Any]],
    proposal: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build contiguous display segments for one mixed track proposal."""
    target_ids = set(_target_track_ids(proposal))
    if len(target_ids) != 1:
        return []
    target_id = next(iter(target_ids))
    target_rows = sorted(
        [
            row
            for row in rows
            if _as_int(row.get("track_id"), -1) == target_id
        ],
        key=lambda item: (_source_frame_idx(item), _as_int(item.get("sample_number"), 0)),
    )
    if not target_rows:
        return []

    specs: list[dict[str, Any]] = []
    segment_index = 0
    current_key: tuple[str, int, int | None] | None = None
    current: dict[str, Any] | None = None

    for row in target_rows:
        key = _segment_key(row)
        frame = _source_frame_idx(row)
        if key != current_key:
            segment_index += 1
            role, team, raw_id = key
            current = {
                "track_id": target_id,
                "segment_index": segment_index,
                "display_identity_id": (target_id * 1000) + segment_index,
                "display_label": _segment_label(target_id, segment_index, role),
                "display_role": role,
                "display_team": team,
                "raw_track_id": raw_id,
                "first_source_frame_idx": frame,
                "last_source_frame_idx": frame,
                "row_count": 0,
            }
            specs.append(current)
            current_key = key
        if current is not None:
            current["last_source_frame_idx"] = frame
            current["row_count"] += 1

    return specs


def _match_segment_spec(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Return whether a raw/annotation row belongs to a segment spec."""
    if _as_int(row.get("track_id"), -1) != _as_int(spec.get("track_id"), -1):
        return False
    frame = _source_frame_idx(row)
    if frame < _as_int(spec.get("first_source_frame_idx"), -1):
        return False
    if frame > _as_int(spec.get("last_source_frame_idx"), -1):
        return False
    raw_id = _row_raw_track_id(row)
    spec_raw_id = spec.get("raw_track_id")
    if spec_raw_id is not None and raw_id is not None:
        return raw_id == _as_int(spec_raw_id, -1)
    return True


def _apply_segment_spec_to_row(row: dict[str, Any], spec: dict[str, Any]) -> None:
    """Apply one segment split display spec to a row-like dict."""
    role = str(spec.get("display_role") or "player")
    row["display_identity_id"] = int(spec["display_identity_id"])
    row["display_label"] = str(spec.get("display_label") or row.get("display_label") or row.get("track_id"))
    row["display_role"] = role
    row["identity_topology_status"] = "phase10_segment_split_applied"
    row["identity_topology_action"] = MARK_SEGMENT_SPLIT_REQUIRED_ACTION
    row["identity_segment_index"] = int(spec.get("segment_index", 0))
    if role == "goalkeeper":
        row["display_team"] = 0
        row["display_color"] = list(GOALKEEPER_COLOR)
    elif role == "referee":
        row["display_team"] = 0
        row["display_color"] = None
    else:
        team = _as_int(spec.get("display_team"), _as_int(row.get("team"), 0))
        if team in {1, 2}:
            row["display_team"] = team
        row["display_color"] = list(_row_player_color(row))


def _apply_goalkeeper_cluster_to_row(row: dict[str, Any]) -> None:
    """Apply a display-only goalkeeper identity cluster to one row-like dict."""
    row["display_identity_id"] = GOALKEEPER_DISPLAY_ID
    row["display_label"] = "GK"
    row["display_role"] = "goalkeeper"
    row["display_team"] = 0
    row["display_color"] = list(GOALKEEPER_COLOR)
    row["model_confirmed_role"] = "goalkeeper"
    row["model_confirmed_role_source"] = MARK_IDENTITY_CLUSTER_REQUIRED_ACTION
    row["resolved_identity_id"] = "gk"
    row["resolved_identity_label"] = "GK"
    row["identity_topology_status"] = "phase10_identity_cluster_applied"
    row["identity_topology_action"] = MARK_IDENTITY_CLUSTER_REQUIRED_ACTION


def apply_identity_resolution_plan_to_raw_records(
    raw_tracklet_records: list[dict[str, Any]],
    identity_resolution_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply ready Phase 9 proposals to raw tracklet metadata.

    This function does not re-key track IDs. It annotates rows with resolved
    identity metadata so future render/report stages can trust the resolution
    only after the post-apply audit stays safe.
    """
    validation = validate_identity_safe_apply_plan(identity_resolution_plan)
    rows = deepcopy(raw_tracklet_records)
    if validation.get("verdict") != "PASS":
        return rows, {
            "schema_version": "1.0",
            "phase": "phase_10_identity_safe_apply",
            "safe_apply_status": REJECTED_STATUS,
            "safe_apply_candidate": False,
            "applied_proposal_count": 0,
            "updated_record_count": 0,
            "actions": [],
            "validation": validation,
        }

    ready = _ready_proposals(identity_resolution_plan)
    topology = _topology_guard_proposals(identity_resolution_plan)
    actions: list[dict[str, Any]] = []
    updated_record_count = 0

    for proposal in ready:
        target_ids = set(_target_track_ids(proposal))
        payload = _proposal_resolution_payload(proposal)
        proposal_updated = 0
        action = str(proposal.get("proposed_action") or "")
        team_color = _canonical_team_color(
            rows,
            target_track_ids=target_ids,
            target_team=_target_display_team(proposal),
        )
        for row in rows:
            if _as_int(row.get("track_id"), -1) not in target_ids:
                continue
            if action == MERGE_GOALKEEPER_ACTION and not _row_visible_goalkeeper(row):
                continue
            if action == LOCK_DISPLAY_TEAM_ACTION:
                if _row_visible_goalkeeper(row):
                    continue
                row["display_role"] = str(row.get("display_role") or row.get("role") or "player")
                row["display_team"] = _target_display_team(proposal)
                row["display_color"] = list(team_color)
                row["model_confirmed_team"] = _target_display_team(proposal)
                row["model_confirmed_team_source"] = LOCK_DISPLAY_TEAM_ACTION
            elif action == LOCK_DISPLAY_ROLE_ACTION:
                target_role = _target_display_role(proposal)
                row["display_role"] = target_role
                row["model_confirmed_role"] = target_role
                row["model_confirmed_role_source"] = LOCK_DISPLAY_ROLE_ACTION
                if target_role == "player":
                    row["display_label"] = str(row.get("display_label") or row.get("track_id"))
                    if _as_int(row.get("display_team"), 0) == 0 and _as_int(row.get("team"), 0) in {1, 2}:
                        row["display_team"] = _as_int(row.get("team"))
                    row["display_color"] = list(_row_player_color(row))
                elif target_role == "referee":
                    row["display_team"] = 0
            row.update(payload)
            proposal_updated += 1
        updated_record_count += proposal_updated
        actions.append(
            {
                "proposal_id": proposal.get("proposal_id"),
                "case_id": proposal.get("case_id"),
                "action": action,
                "target_track_ids": sorted(target_ids),
                "updated_record_count": proposal_updated,
                "confidence": proposal.get("confidence"),
            }
        )

    for proposal in topology:
        target_ids = set(_target_track_ids(proposal))
        action = str(proposal.get("proposed_action") or proposal.get("proposal_type") or "")
        proposal_updated = 0
        segment_specs: list[dict[str, Any]] = []
        if action == MARK_SEGMENT_SPLIT_REQUIRED_ACTION or proposal.get("proposal_type") == "segment_split_required":
            segment_specs = _build_segment_specs(rows, proposal)
            for row in rows:
                for spec in segment_specs:
                    if not _match_segment_spec(row, spec):
                        continue
                    _apply_segment_spec_to_row(row, spec)
                    proposal_updated += 1
                    break
        elif (
            action == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION
            or proposal.get("proposal_type") == "identity_cluster_required"
        ) and _is_goalkeeper_cluster_proposal(proposal):
            for row in rows:
                if _as_int(row.get("track_id"), -1) not in target_ids:
                    continue
                if not _row_visible_goalkeeper(row) and not (
                    _in_proposal_scope(row, proposal) and _row_goalkeeper_evidence(row)
                ):
                    continue
                _apply_goalkeeper_cluster_to_row(row)
                proposal_updated += 1

        if proposal_updated:
            updated_record_count += proposal_updated
            actions.append(
                {
                    "proposal_id": proposal.get("proposal_id"),
                    "case_id": proposal.get("case_id"),
                    "action": action,
                    "target_track_ids": sorted(target_ids),
                    "updated_record_count": proposal_updated,
                    "segment_count": len(segment_specs),
                    "confidence": proposal.get("confidence"),
                }
            )

    applied_count = len([action for action in actions if action.get("updated_record_count", 0) > 0])
    return rows, {
        "schema_version": "1.0",
        "phase": "phase_10_identity_safe_apply",
        "safe_apply_status": APPLIED_STATUS if actions else NOOP_STATUS,
        "safe_apply_candidate": bool(actions),
        "applied_proposal_count": applied_count,
        "updated_record_count": updated_record_count,
        "actions": actions,
        "validation": validation,
        "notes": [
            "Phase 10 writes safe display metadata and topology guards only.",
            "Track IDs are not re-keyed; segment and cluster display IDs are render-facing metadata.",
        ],
    }


def apply_identity_resolution_plan_to_annotation_states(
    annotation_states: list[dict[str, Any]],
    identity_resolution_plan: dict[str, Any],
) -> int:
    """Apply ready Phase 9 proposals to annotation-state metadata."""
    ready = _ready_proposals(identity_resolution_plan)
    topology = _topology_guard_proposals(identity_resolution_plan)
    if not ready and not topology:
        return 0

    updated = 0
    for proposal in ready:
        target_ids = set(_target_track_ids(proposal))
        payload = _proposal_resolution_payload(proposal)
        action = str(proposal.get("proposed_action") or "")
        for state in annotation_states:
            for track_id, track in state.get("players", {}).items():
                if _as_int(track_id, -1) not in target_ids:
                    continue
                if action == MERGE_GOALKEEPER_ACTION and not _track_visible_goalkeeper(track):
                    continue
                if action == LOCK_DISPLAY_TEAM_ACTION:
                    if _track_visible_goalkeeper(track):
                        continue
                    target_team = _target_display_team(proposal)
                    track["display_team"] = target_team
                    track["model_confirmed_team"] = target_team
                    track["model_confirmed_team_source"] = LOCK_DISPLAY_TEAM_ACTION
                    color = _safe_color_value(track.get("team_color")) or _safe_color_value(
                        track.get("display_color")
                    )
                    track["display_color"] = tuple(color or PLAYER_FALLBACK_COLOR)
                elif action == LOCK_DISPLAY_ROLE_ACTION:
                    target_role = _target_display_role(proposal)
                    track["display_role"] = target_role
                    track["model_confirmed_role"] = target_role
                    track["model_confirmed_role_source"] = LOCK_DISPLAY_ROLE_ACTION
                    if target_role == "player":
                        track["display_label"] = str(track.get("display_label") or track_id)
                        if _as_int(track.get("display_team"), 0) == 0 and _as_int(track.get("team"), 0) in {1, 2}:
                            track["display_team"] = _as_int(track.get("team"))
                        color = _safe_color_value(track.get("team_color")) or _safe_color_value(
                            track.get("display_color")
                        )
                        track["display_color"] = tuple(color or PLAYER_FALLBACK_COLOR)
                    elif target_role == "referee":
                        track["display_team"] = 0
                track.update(payload)
                updated += 1

    if topology:
        pseudo_rows: list[dict[str, Any]] = []
        for state in annotation_states:
            source_frame_idx = _as_int(state.get("source_frame_idx"), -1)
            sample_number = _as_int(state.get("sample_number"), 0)
            for track_id, track in state.get("players", {}).items():
                row = dict(track)
                row["track_id"] = _as_int(track_id, -1)
                row["source_frame_idx"] = source_frame_idx
                row["sample_number"] = sample_number
                pseudo_rows.append(row)
        segment_specs_by_proposal: dict[str, list[dict[str, Any]]] = {
            str(proposal.get("proposal_id")): _build_segment_specs(pseudo_rows, proposal)
            for proposal in topology
        }

        for proposal in topology:
            target_ids = set(_target_track_ids(proposal))
            action = str(proposal.get("proposed_action") or proposal.get("proposal_type") or "")
            specs = segment_specs_by_proposal.get(str(proposal.get("proposal_id"))) or []
            for state in annotation_states:
                source_frame_idx = _as_int(state.get("source_frame_idx"), -1)
                for track_id, track in state.get("players", {}).items():
                    row_view = dict(track)
                    row_view["track_id"] = _as_int(track_id, -1)
                    row_view["source_frame_idx"] = source_frame_idx
                    if action == MARK_SEGMENT_SPLIT_REQUIRED_ACTION or proposal.get("proposal_type") == "segment_split_required":
                        for spec in specs:
                            if not _match_segment_spec(row_view, spec):
                                continue
                            _apply_segment_spec_to_row(track, spec)
                            updated += 1
                            break
                    elif (
                        action == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION
                        or proposal.get("proposal_type") == "identity_cluster_required"
                    ) and _is_goalkeeper_cluster_proposal(proposal):
                        if _as_int(track_id, -1) not in target_ids:
                            continue
                        if not _track_visible_goalkeeper(track) and not (
                            _in_proposal_scope(row_view, proposal)
                            and _track_goalkeeper_evidence(track)
                        ):
                            continue
                        _apply_goalkeeper_cluster_to_row(track)
                        updated += 1
    return updated
