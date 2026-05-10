"""Dry-run resolver for pre-render identity review cases.

This phase converts review artifacts into structured resolution proposals. It
does not mutate tracklets or rendered annotations; every proposal must pass a
future safe-apply validator before any identity change is made.
"""

from __future__ import annotations

from typing import Any


READY_STATUS = "ready_for_safe_apply"
DEFERRED_STATUS = "deferred_needs_evidence"
REJECTED_STATUS = "rejected_identity_mutation"
BLOCKED_STATUS = "blocked_render_not_safe"

MERGE_GOALKEEPER_ACTION = "merge_goalkeeper_display_ids"
MERGE_CANDIDATE_ACTION = "merge_candidate_identity_link"
LOCK_DISPLAY_TEAM_ACTION = "lock_display_team"
LOCK_DISPLAY_ROLE_ACTION = "lock_display_role"
MARK_SEGMENT_SPLIT_REQUIRED_ACTION = "mark_segment_split_required"
MARK_IDENTITY_CLUSTER_REQUIRED_ACTION = "mark_identity_cluster_required"
KEEP_UNRESOLVED_ACTION = "keep_unresolved_no_identity_mutation"
REJECT_MERGE_ACTION = "reject_identity_merge"

DISPLAY_LOCK_ACTIONS = {LOCK_DISPLAY_TEAM_ACTION, LOCK_DISPLAY_ROLE_ACTION}
READY_ACTIONS = {
    MERGE_GOALKEEPER_ACTION,
    MERGE_CANDIDATE_ACTION,
    LOCK_DISPLAY_TEAM_ACTION,
    LOCK_DISPLAY_ROLE_ACTION,
}
VALID_STATUSES = {
    READY_STATUS,
    DEFERRED_STATUS,
    REJECTED_STATUS,
    BLOCKED_STATUS,
}
VALID_ACTIONS = {
    MERGE_GOALKEEPER_ACTION,
    MERGE_CANDIDATE_ACTION,
    LOCK_DISPLAY_TEAM_ACTION,
    LOCK_DISPLAY_ROLE_ACTION,
    MARK_SEGMENT_SPLIT_REQUIRED_ACTION,
    MARK_IDENTITY_CLUSTER_REQUIRED_ACTION,
    KEEP_UNRESOLVED_ACTION,
    REJECT_MERGE_ACTION,
}

MIN_DISPLAY_LOCK_MODEL_CONFIDENCE = 0.90
MIN_SAFE_TEAM_EVIDENCE_CONFIDENCE = 0.88
MIN_SAFE_ROLE_EVIDENCE_CONFIDENCE = 0.98
MAX_SAFE_MINOR_TEAM_SEGMENT_RATIO = 0.12
MAX_SAFE_MINOR_TEAM_SEGMENT_FRAMES = 30
MAX_SAFE_MINOR_ROLE_SEGMENT_FRAMES = 2
SEGMENT_SPLIT_REASON_KEYWORDS = (
    "segment_split_required",
    "different player",
    "different people",
    "different identities",
    "multiple different identities",
    "not following a single individual",
    "track switch",
    "track has merged",
    "track merge",
    "merged two different",
)
IDENTITY_CLUSTER_REASON_KEYWORDS = (
    "identity_cluster_required",
    "same real player is split",
    "same player is split",
    "same individual is split",
    "fragmented across track ids",
    "fragmented across display track",
)


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


def _target_team_from_verdict(verdict: str) -> int | None:
    """Return team id for a model team verdict."""
    if verdict == "team_1":
        return 1
    if verdict == "team_2":
        return 2
    return None


def _looks_like_segment_split(vision_result: dict[str, Any]) -> bool:
    """Return whether model text indicates one display track contains identities."""
    reason = str(vision_result.get("reason") or "").lower()
    return any(keyword in reason for keyword in SEGMENT_SPLIT_REASON_KEYWORDS)


def _looks_like_identity_cluster(vision_result: dict[str, Any]) -> bool:
    """Return whether model text indicates one identity spans display tracks."""
    reason = str(vision_result.get("reason") or "").lower()
    return any(keyword in reason for keyword in IDENTITY_CLUSTER_REASON_KEYWORDS)


def _audit_evidence(queue_case: dict[str, Any]) -> dict[str, Any]:
    """Return audit evidence attached to a review queue case."""
    evidence = queue_case.get("audit_evidence")
    return evidence if isinstance(evidence, dict) else {}


def _target_track_id(
    *,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    crop_case: dict[str, Any],
) -> int | None:
    """Return the single target track for display-only proposals."""
    audit_track_id = _as_int(_audit_evidence(queue_case).get("track_id"), -1)
    if audit_track_id >= 0:
        return audit_track_id
    target_ids = _track_ids_from_evidence(vision_result, crop_case)
    if len(target_ids) == 1:
        return target_ids[0]
    return None


def _dominant_team_from_counts(counts: dict[str, Any]) -> int | None:
    """Return dominant team id from a counts dictionary."""
    parsed: list[tuple[int, int]] = []
    for key, value in (counts or {}).items():
        team = _as_int(key, 0)
        count = _as_int(value, 0)
        if team in {1, 2} and count > 0:
            parsed.append((team, count))
    if not parsed:
        return None
    parsed.sort(key=lambda item: item[1], reverse=True)
    return parsed[0][0]


def _model_evidence_count(vision_result: dict[str, Any]) -> int:
    """Return model evidence count for a vision result."""
    return len(vision_result.get("model_evidence") or [])


def _display_lock_evidence_pack(
    *,
    question: str,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    crop_case: dict[str, Any],
    target_track_id: int,
) -> dict[str, Any]:
    """Build compact evidence for a display-only safe-apply proposal."""
    audit = _audit_evidence(queue_case)
    return {
        "question": question,
        "issue_type": audit.get("issue_type"),
        "issue_id": audit.get("issue_id") or queue_case.get("source_issue_id"),
        "priority": queue_case.get("priority") or vision_result.get("priority"),
        "vision_status": vision_result.get("status"),
        "vision_verdict": vision_result.get("verdict"),
        "vision_confidence": _as_float(vision_result.get("confidence"), 0.0),
        "contact_sheet_path": vision_result.get("contact_sheet_path")
        or crop_case.get("contact_sheet_path"),
        "crop_count": _as_int(
            vision_result.get("crop_count"),
            _as_int(crop_case.get("crop_request_count"), 0),
        ),
        "target_track_ids": [target_track_id],
        "raw_track_ids": _raw_track_ids_from_evidence(vision_result, crop_case),
        "frame_span": _frame_span_from_evidence(vision_result, crop_case),
        "model_evidence_count": _model_evidence_count(vision_result),
        "audit_evidence": audit,
    }


def _safe_team_lock_reason(
    *,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    target_team: int,
) -> str | None:
    """Return None when a team-lock proposal is safe enough to apply."""
    audit = _audit_evidence(queue_case)
    if audit.get("issue_type") not in {
        "display_team_flicker",
        "hidden_team_switch_after_stabilizer",
    }:
        return "team_lock_only_handles_display_team_flicker"
    if str(vision_result.get("status") or "") != "reviewed":
        return "team_lock_requires_reviewed_model_result"
    if _model_evidence_count(vision_result) <= 0:
        return "team_lock_requires_model_evidence"
    if _looks_like_segment_split(vision_result):
        return "team_lock_blocked_by_segment_split_evidence"
    if _looks_like_identity_cluster(vision_result):
        return "team_lock_blocked_by_identity_cluster_evidence"
    if _as_float(vision_result.get("confidence"), 0.0) < MIN_DISPLAY_LOCK_MODEL_CONFIDENCE:
        return "team_lock_requires_high_model_confidence"
    if _dominant_team_from_counts(audit.get("player_team_counts") or {}) != target_team:
        return "team_lock_requires_model_verdict_to_match_numeric_dominant_team"
    if (
        _as_float(vision_result.get("confidence"), 0.0) >= MIN_DISPLAY_LOCK_MODEL_CONFIDENCE
        and _as_float(audit.get("raw_role_confidence"), 0.0) >= 0.90
    ):
        return None
    if _as_float(audit.get("player_team_confidence"), 0.0) < MIN_SAFE_TEAM_EVIDENCE_CONFIDENCE:
        return "team_lock_requires_strong_numeric_team_evidence"
    if _as_float(audit.get("max_minor_player_team_segment_ratio"), 1.0) > MAX_SAFE_MINOR_TEAM_SEGMENT_RATIO:
        return "team_lock_minor_segment_ratio_too_large"
    if _as_int(audit.get("max_minor_player_team_segment_frames"), 999999) > MAX_SAFE_MINOR_TEAM_SEGMENT_FRAMES:
        return "team_lock_minor_segment_too_long"
    if _as_float(audit.get("raw_role_confidence"), 0.0) < 0.95:
        return "team_lock_requires_stable_player_role"
    return None


def _safe_role_lock_reason(
    *,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    target_role: str,
) -> str | None:
    """Return None when a role-lock proposal is safe enough to apply."""
    audit = _audit_evidence(queue_case)
    if audit.get("issue_type") not in {
        "display_role_flicker",
        "hidden_role_switch_after_stabilizer",
    }:
        return "role_lock_only_handles_visible_display_role_flicker"
    if target_role not in {"player", "referee"}:
        return "role_lock_rejects_goalkeeper_or_unknown_role"
    if str(vision_result.get("status") or "") != "reviewed":
        return "role_lock_requires_reviewed_model_result"
    if _model_evidence_count(vision_result) <= 0:
        return "role_lock_requires_model_evidence"
    if _looks_like_segment_split(vision_result):
        return "role_lock_blocked_by_segment_split_evidence"
    if _looks_like_identity_cluster(vision_result):
        return "role_lock_blocked_by_identity_cluster_evidence"
    if _as_float(vision_result.get("confidence"), 0.0) < MIN_DISPLAY_LOCK_MODEL_CONFIDENCE:
        return "role_lock_requires_high_model_confidence"
    if str(audit.get("dominant_raw_role") or "") != target_role:
        return "role_lock_requires_model_verdict_to_match_numeric_dominant_role"
    if (
        _as_float(vision_result.get("confidence"), 0.0) >= MIN_DISPLAY_LOCK_MODEL_CONFIDENCE
        and _as_float(audit.get("raw_role_confidence"), 0.0) >= 0.85
    ):
        return None
    if _as_float(audit.get("raw_role_confidence"), 0.0) < MIN_SAFE_ROLE_EVIDENCE_CONFIDENCE:
        return "role_lock_requires_stable_numeric_role_evidence"
    if _as_int(audit.get("max_minor_raw_role_segment_frames"), 999999) > MAX_SAFE_MINOR_ROLE_SEGMENT_FRAMES:
        return "role_lock_minor_segment_too_long"
    return None


def _case_id(value: dict[str, Any]) -> str:
    """Return a stable case id from a review payload."""
    return str(value.get("case_id") or value.get("decision_id") or "unknown_case")


def _cases_by_id(payload: dict[str, Any], key: str = "cases") -> dict[str, dict[str, Any]]:
    """Return nested case dictionaries keyed by case id."""
    return {
        _case_id(case): case
        for case in payload.get(key, []) or []
        if isinstance(case, dict)
    }


def _results_by_case_id(vision_review_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return vision-review results keyed by case id."""
    return _cases_by_id(vision_review_results, key="results")


def _decision_items(identity_review_decisions: dict[str, Any]) -> list[dict[str, Any]]:
    """Return review engine decision items."""
    return [
        decision
        for decision in identity_review_decisions.get("decisions", []) or []
        if isinstance(decision, dict)
    ]


def _unresolved_case_ids(identity_review_decisions: dict[str, Any]) -> list[str]:
    """Return case ids that the review engine kept unresolved."""
    case_ids: list[str] = []
    for decision in _decision_items(identity_review_decisions):
        if decision.get("status") != "needs_vision_or_llm":
            continue
        evidence = decision.get("evidence") or {}
        case_id = str(evidence.get("case_id") or decision.get("decision_id") or "")
        if case_id:
            case_ids.append(case_id)
    return sorted(set(case_ids))


def _review_case_ids(
    identity_review_decisions: dict[str, Any],
    vision_review_results: dict[str, Any],
) -> list[str]:
    """Return unresolved and model-resolved case ids for Phase 9 proposals."""
    case_ids = set(_unresolved_case_ids(identity_review_decisions))
    for result in vision_review_results.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        case_id = str(result.get("case_id") or "")
        if not case_id:
            continue
        verdict = str(result.get("verdict") or "")
        if (
            verdict != "unresolved"
            or _looks_like_segment_split(result)
            or _looks_like_identity_cluster(result)
        ):
            case_ids.add(case_id)
    return sorted(case_ids)


def _candidate_link_decisions(
    identity_review_decisions: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return candidate-link review decisions."""
    return [
        decision
        for decision in _decision_items(identity_review_decisions)
        if decision.get("decision_type") == "candidate_identity_link"
    ]


def _track_ids_from_evidence(*items: dict[str, Any]) -> list[int]:
    """Collect unique track ids from crop/model evidence payloads."""
    track_ids: set[int] = set()
    for item in items:
        for key in ("target_track_ids", "track_ids"):
            for track_id in item.get(key, []) or []:
                parsed = _as_int(track_id, -1)
                if parsed >= 0:
                    track_ids.add(parsed)
        for evidence_key in ("evidence", "model_evidence", "crop_requests"):
            for row in item.get(evidence_key, []) or []:
                if not isinstance(row, dict):
                    continue
                parsed = _as_int(row.get("track_id"), -1)
                if parsed >= 0:
                    track_ids.add(parsed)
    return sorted(track_ids)


def _raw_track_ids_from_evidence(*items: dict[str, Any]) -> list[int]:
    """Collect unique raw track ids from crop/model evidence payloads."""
    raw_ids: set[int] = set()
    for item in items:
        for evidence_key in ("evidence", "model_evidence", "crop_requests"):
            for row in item.get(evidence_key, []) or []:
                if not isinstance(row, dict):
                    continue
                parsed = _as_int(row.get("raw_track_id"), -1)
                if parsed >= 0:
                    raw_ids.add(parsed)
    return sorted(raw_ids)


def _frame_span_from_evidence(*items: dict[str, Any]) -> dict[str, int | None]:
    """Return min/max source frame from crop evidence."""
    frames: list[int] = []
    for item in items:
        for evidence_key in ("evidence", "model_evidence", "crop_requests"):
            for row in item.get(evidence_key, []) or []:
                if not isinstance(row, dict):
                    continue
                parsed = _as_int(row.get("source_frame_idx"), -1)
                if parsed >= 0:
                    frames.append(parsed)
    if not frames:
        return {"first_source_frame_idx": None, "last_source_frame_idx": None}
    return {
        "first_source_frame_idx": min(frames),
        "last_source_frame_idx": max(frames),
    }


def _goalkeeper_targets(
    *,
    vision_result: dict[str, Any],
    crop_case: dict[str, Any],
    render_audit_after: dict[str, Any],
    identity_review_decisions: dict[str, Any],
) -> list[int]:
    """Return goalkeeper target display ids for a fragmentation case."""
    target_ids = _track_ids_from_evidence(vision_result, crop_case)
    if target_ids:
        return target_ids
    summary_ids = (
        identity_review_decisions.get("summary", {}).get("goalkeeper_display_track_ids")
        or []
    )
    target_ids = sorted({_as_int(track_id, -1) for track_id in summary_ids})
    target_ids = [track_id for track_id in target_ids if track_id >= 0]
    if target_ids:
        return target_ids
    return sorted(
        {
            _as_int(segment.get("track_id"), -1)
            for segment in render_audit_after.get("goalkeeper_display_segments", []) or []
            if isinstance(segment, dict)
        }
        - {-1}
    )


def _build_goalkeeper_fragmentation_proposal(
    *,
    case_id: str,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    crop_case: dict[str, Any],
    identity_review_decisions: dict[str, Any],
    render_audit_after: dict[str, Any],
) -> dict[str, Any]:
    """Build one proposal for goalkeeper identity fragmentation."""
    render_safe = bool(identity_review_decisions.get("render_safe"))
    target_track_ids = _goalkeeper_targets(
        vision_result=vision_result,
        crop_case=crop_case,
        render_audit_after=render_audit_after,
        identity_review_decisions=identity_review_decisions,
    )
    verdict = str(vision_result.get("verdict") or "unresolved")
    status = str(vision_result.get("status") or "not_invoked")
    confidence = _as_float(vision_result.get("confidence"), 0.0)
    vision_invoked = bool(identity_review_decisions.get("vision_model_invoked")) or bool(
        vision_result.get("model_evidence")
    )

    if _looks_like_identity_cluster(vision_result):
        proposal_status = DEFERRED_STATUS
        action = MARK_IDENTITY_CLUSTER_REQUIRED_ACTION
        reason = (
            "Review evidence says goalkeeper identity is fragmented across display "
            "tracks; apply a display-only cluster guard before final render."
        )
    elif not render_safe:
        proposal_status = BLOCKED_STATUS
        action = KEEP_UNRESOLVED_ACTION
        reason = "Render identity is not safe; resolver cannot propose identity merges."
    elif verdict == "same_player" and status == "reviewed" and confidence >= 0.80:
        proposal_status = READY_STATUS
        action = MERGE_GOALKEEPER_ACTION
        reason = "Vision evidence supports one goalkeeper identity across display IDs."
    elif verdict in {"different_player", "not_goalkeeper"} and status == "reviewed":
        proposal_status = REJECTED_STATUS
        action = REJECT_MERGE_ACTION
        reason = "Review evidence rejects a goalkeeper identity merge."
    else:
        proposal_status = DEFERRED_STATUS
        action = KEEP_UNRESOLVED_ACTION
        reason = (
            "No trusted Vision/LLM decision is available; keep the case unresolved."
            if not vision_invoked
            else "Review evidence is insufficient for a safe identity merge."
        )

    evidence_pack = {
        "question": queue_case.get("question") or vision_result.get("question"),
        "priority": queue_case.get("priority") or vision_result.get("priority"),
        "vision_status": status,
        "vision_verdict": verdict,
        "vision_confidence": confidence,
        "contact_sheet_path": vision_result.get("contact_sheet_path")
        or crop_case.get("contact_sheet_path"),
        "crop_count": _as_int(
            vision_result.get("crop_count"),
            _as_int(crop_case.get("crop_request_count"), 0),
        ),
        "target_track_ids": target_track_ids,
        "raw_track_ids": _raw_track_ids_from_evidence(vision_result, crop_case),
        "frame_span": _frame_span_from_evidence(vision_result, crop_case),
        "model_evidence_count": len(vision_result.get("model_evidence") or []),
    }
    if action == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION:
        evidence_pack["identity_cluster_reason"] = vision_result.get("reason")

    return {
        "proposal_id": f"resolve_{case_id}",
        "case_id": case_id,
        "proposal_type": (
            "identity_cluster_required"
            if action == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION
            else "goalkeeper_identity_fragmentation"
        ),
        "status": proposal_status,
        "proposed_action": action,
        "confidence": confidence if proposal_status == READY_STATUS else 0.0,
        "target_track_ids": target_track_ids,
        "reason": reason,
        "evidence": evidence_pack,
        "apply_policy": {
            "dry_run_only": True,
            "requires_safe_apply_validator": True,
            "mutates_raw_tracklets": False,
            "mutates_render_annotations": action == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION,
            "display_only": action == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION,
            "requires_cross_track_identity_resolution": action
            == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION,
        },
    }


def _build_candidate_link_proposal(decision: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative proposal for one candidate identity link."""
    evidence = decision.get("evidence") or {}
    source_id = evidence.get("source_id")
    target_id = evidence.get("target_id")
    return {
        "proposal_id": f"resolve_candidate_link_{source_id}_{target_id}",
        "case_id": f"candidate_link_{source_id}_{target_id}",
        "proposal_type": "candidate_identity_link",
        "status": DEFERRED_STATUS,
        "proposed_action": KEEP_UNRESOLVED_ACTION,
        "confidence": 0.0,
        "target_track_ids": [
            track_id
            for track_id in (_as_int(target_id, -1), _as_int(source_id, -1))
            if track_id >= 0
        ],
        "reason": (
            "Candidate link is outside strict auto-merge policy; keep it for "
            "text/visual review before any identity mutation."
        ),
        "evidence": evidence,
        "apply_policy": {
            "dry_run_only": True,
            "requires_safe_apply_validator": True,
            "mutates_raw_tracklets": False,
            "mutates_render_annotations": False,
        },
    }


def _build_segment_split_required_proposal(
    *,
    case_id: str,
    question: str,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    crop_case: dict[str, Any],
) -> dict[str, Any]:
    """Build a deferred proposal for tracks that appear internally mixed."""
    target_track_id = _target_track_id(
        queue_case=queue_case,
        vision_result=vision_result,
        crop_case=crop_case,
    )
    target_track_ids = [target_track_id] if target_track_id is not None else _track_ids_from_evidence(
        vision_result,
        crop_case,
    )
    evidence = _display_lock_evidence_pack(
        question=question,
        queue_case=queue_case,
        vision_result=vision_result,
        crop_case=crop_case,
        target_track_id=target_track_ids[0] if target_track_ids else -1,
    )
    evidence["segment_split_reason"] = vision_result.get("reason")
    return {
        "proposal_id": f"resolve_{case_id}",
        "case_id": case_id,
        "proposal_type": "segment_split_required",
        "status": DEFERRED_STATUS,
        "proposed_action": MARK_SEGMENT_SPLIT_REQUIRED_ACTION,
        "confidence": 0.0,
        "target_track_ids": target_track_ids,
        "reason": (
            "Review evidence indicates this display track may contain multiple "
            "identities; do not force one team or role across the whole track."
        ),
        "evidence": evidence,
        "apply_policy": {
            "dry_run_only": True,
            "requires_safe_apply_validator": True,
            "mutates_raw_tracklets": False,
            "mutates_render_annotations": False,
            "requires_segment_level_resolution": True,
        },
    }


def _build_identity_cluster_required_proposal(
    *,
    case_id: str,
    question: str,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    crop_case: dict[str, Any],
) -> dict[str, Any]:
    """Build a deferred proposal for identities that span display tracks."""
    target_track_ids = _track_ids_from_evidence(vision_result, crop_case)
    if not target_track_ids:
        target_track_id = _target_track_id(
            queue_case=queue_case,
            vision_result=vision_result,
            crop_case=crop_case,
        )
        target_track_ids = [target_track_id] if target_track_id is not None else []
    evidence = _display_lock_evidence_pack(
        question=question,
        queue_case=queue_case,
        vision_result=vision_result,
        crop_case=crop_case,
        target_track_id=target_track_ids[0] if target_track_ids else -1,
    )
    evidence["identity_cluster_reason"] = vision_result.get("reason")
    return {
        "proposal_id": f"resolve_{case_id}",
        "case_id": case_id,
        "proposal_type": "identity_cluster_required",
        "status": DEFERRED_STATUS,
        "proposed_action": MARK_IDENTITY_CLUSTER_REQUIRED_ACTION,
        "confidence": 0.0,
        "target_track_ids": target_track_ids,
        "reason": (
            "Review evidence indicates one real identity may be fragmented across "
            "multiple display tracks; cluster/merge must be solved before final trust."
        ),
        "evidence": evidence,
        "apply_policy": {
            "dry_run_only": True,
            "requires_safe_apply_validator": True,
            "mutates_raw_tracklets": False,
            "mutates_render_annotations": False,
            "requires_cross_track_identity_resolution": True,
        },
    }


def _build_display_review_proposal(
    *,
    case_id: str,
    question: str,
    queue_case: dict[str, Any],
    vision_result: dict[str, Any],
    crop_case: dict[str, Any],
) -> dict[str, Any]:
    """Build a proposal for team/role review cases."""
    if _looks_like_segment_split(vision_result):
        return _build_segment_split_required_proposal(
            case_id=case_id,
            question=question,
            queue_case=queue_case,
            vision_result=vision_result,
            crop_case=crop_case,
        )
    if _looks_like_identity_cluster(vision_result):
        return _build_identity_cluster_required_proposal(
            case_id=case_id,
            question=question,
            queue_case=queue_case,
            vision_result=vision_result,
            crop_case=crop_case,
        )

    verdict = str(vision_result.get("verdict") or "unresolved")
    target_track_id = _target_track_id(
        queue_case=queue_case,
        vision_result=vision_result,
        crop_case=crop_case,
    )
    if target_track_id is None:
        return {
            "proposal_id": f"resolve_{case_id}",
            "case_id": case_id,
            "proposal_type": question or "identity_review",
            "status": DEFERRED_STATUS,
            "proposed_action": KEEP_UNRESOLVED_ACTION,
            "confidence": 0.0,
            "target_track_ids": _track_ids_from_evidence(vision_result, crop_case),
            "reason": "Display review case does not resolve to exactly one target track.",
            "evidence": {
                "question": question,
                "vision_status": vision_result.get("status"),
                "vision_verdict": vision_result.get("verdict"),
            },
            "apply_policy": {
                "dry_run_only": True,
                "requires_safe_apply_validator": True,
                "mutates_raw_tracklets": False,
                "mutates_render_annotations": False,
            },
        }

    if question == "team_assignment_uncertain":
        target_team = _target_team_from_verdict(verdict)
        if target_team is not None:
            block_reason = _safe_team_lock_reason(
                queue_case=queue_case,
                vision_result=vision_result,
                target_team=target_team,
            )
            evidence = _display_lock_evidence_pack(
                question=question,
                queue_case=queue_case,
                vision_result=vision_result,
                crop_case=crop_case,
                target_track_id=target_track_id,
            )
            evidence["target_display_team"] = target_team
            return {
                "proposal_id": f"resolve_{case_id}",
                "case_id": case_id,
                "proposal_type": "display_team_lock",
                "status": READY_STATUS if block_reason is None else DEFERRED_STATUS,
                "proposed_action": (
                    LOCK_DISPLAY_TEAM_ACTION
                    if block_reason is None
                    else KEEP_UNRESOLVED_ACTION
                ),
                "confidence": (
                    _as_float(vision_result.get("confidence"), 0.0)
                    if block_reason is None
                    else 0.0
                ),
                "target_track_ids": [target_track_id],
                "reason": (
                    "Model and numeric evidence support a safe display-team lock."
                    if block_reason is None
                    else block_reason
                ),
                "evidence": evidence,
                "apply_policy": {
                    "dry_run_only": True,
                    "requires_safe_apply_validator": True,
                    "mutates_raw_tracklets": False,
                    "mutates_render_annotations": block_reason is None,
                    "display_only": True,
                },
            }

    if question == "role_stability_flicker" and verdict in {"player", "referee"}:
        block_reason = _safe_role_lock_reason(
            queue_case=queue_case,
            vision_result=vision_result,
            target_role=verdict,
        )
        evidence = _display_lock_evidence_pack(
            question=question,
            queue_case=queue_case,
            vision_result=vision_result,
            crop_case=crop_case,
            target_track_id=target_track_id,
        )
        evidence["target_display_role"] = verdict
        return {
            "proposal_id": f"resolve_{case_id}",
            "case_id": case_id,
            "proposal_type": "display_role_lock",
            "status": READY_STATUS if block_reason is None else DEFERRED_STATUS,
            "proposed_action": (
                LOCK_DISPLAY_ROLE_ACTION
                if block_reason is None
                else KEEP_UNRESOLVED_ACTION
            ),
            "confidence": (
                _as_float(vision_result.get("confidence"), 0.0)
                if block_reason is None
                else 0.0
            ),
            "target_track_ids": [target_track_id],
            "reason": (
                "Model and numeric evidence support a safe display-role lock."
                if block_reason is None
                else block_reason
            ),
            "evidence": evidence,
            "apply_policy": {
                "dry_run_only": True,
                "requires_safe_apply_validator": True,
                "mutates_raw_tracklets": False,
                "mutates_render_annotations": block_reason is None,
                "display_only": True,
            },
        }

    return {
        "proposal_id": f"resolve_{case_id}",
        "case_id": case_id,
        "proposal_type": question or "identity_review",
        "status": DEFERRED_STATUS,
        "proposed_action": KEEP_UNRESOLVED_ACTION,
        "confidence": 0.0,
        "target_track_ids": [target_track_id],
        "reason": "No safe display-lock resolver rule matched this review case.",
        "evidence": {
            "question": question,
            "vision_status": vision_result.get("status"),
            "vision_verdict": vision_result.get("verdict"),
            "contact_sheet_path": vision_result.get("contact_sheet_path")
            or crop_case.get("contact_sheet_path"),
            "audit_evidence": _audit_evidence(queue_case),
        },
        "apply_policy": {
            "dry_run_only": True,
            "requires_safe_apply_validator": True,
            "mutates_raw_tracklets": False,
            "mutates_render_annotations": False,
        },
    }


def validate_identity_resolution_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a Phase 9 dry-run identity resolution plan."""
    rejected: list[dict[str, Any]] = []
    proposals = plan.get("resolution_proposals", [])
    render_safe = bool(plan.get("render_safe"))

    if not isinstance(proposals, list):
        return {
            "schema_version": "1.0",
            "validator": "deterministic_phase9_identity_resolver_validator",
            "verdict": "FAIL",
            "rejected_proposal_count": 1,
            "ready_for_safe_apply_count": 0,
            "rejected_proposals": [
                {
                    "proposal_id": None,
                    "reasons": ["resolution_proposals_must_be_list"],
                }
            ],
        }

    for proposal in proposals:
        reasons: list[str] = []
        proposal_id = proposal.get("proposal_id")
        status = str(proposal.get("status") or "")
        action = str(proposal.get("proposed_action") or "")
        confidence = _as_float(proposal.get("confidence"), -1.0)
        target_track_ids = proposal.get("target_track_ids")
        evidence = proposal.get("evidence")
        apply_policy = proposal.get("apply_policy") or {}

        if not proposal_id:
            reasons.append("missing_proposal_id")
        if status not in VALID_STATUSES:
            reasons.append("invalid_status")
        if action not in VALID_ACTIONS:
            reasons.append("invalid_action")
        if confidence < 0.0 or confidence > 1.0:
            reasons.append("invalid_confidence")
        if not isinstance(target_track_ids, list):
            reasons.append("target_track_ids_must_be_list")
            target_track_ids = []
        if not isinstance(evidence, dict):
            reasons.append("evidence_must_be_dict")
            evidence = {}
        if not apply_policy.get("dry_run_only"):
            reasons.append("phase9_must_remain_dry_run_only")
        if not apply_policy.get("requires_safe_apply_validator"):
            reasons.append("safe_apply_validator_required")

        if status == READY_STATUS:
            if not render_safe and action not in DISPLAY_LOCK_ACTIONS:
                reasons.append("ready_proposal_requires_render_safe")
            if action not in READY_ACTIONS:
                reasons.append("ready_proposal_requires_ready_action")
            if confidence < 0.80:
                reasons.append("ready_proposal_requires_high_confidence")
            if action in {MERGE_GOALKEEPER_ACTION, MERGE_CANDIDATE_ACTION} and len(set(target_track_ids)) < 2:
                reasons.append("ready_proposal_requires_multiple_target_tracks")
            if action in DISPLAY_LOCK_ACTIONS and len(set(target_track_ids)) != 1:
                reasons.append("display_lock_requires_one_target_track")
            if action == MERGE_GOALKEEPER_ACTION:
                if evidence.get("vision_verdict") != "same_player":
                    reasons.append("goalkeeper_merge_requires_same_player_verdict")
                if evidence.get("vision_status") != "reviewed":
                    reasons.append("goalkeeper_merge_requires_reviewed_status")
                if _as_int(evidence.get("model_evidence_count"), 0) <= 0:
                    reasons.append("goalkeeper_merge_requires_model_evidence")
            if action == LOCK_DISPLAY_TEAM_ACTION:
                target_team = _as_int(evidence.get("target_display_team"), 0)
                if target_team not in {1, 2}:
                    reasons.append("team_lock_requires_target_display_team")
                if evidence.get("vision_verdict") != f"team_{target_team}":
                    reasons.append("team_lock_requires_matching_team_verdict")
                if evidence.get("vision_status") != "reviewed":
                    reasons.append("team_lock_requires_reviewed_status")
                if _as_int(evidence.get("model_evidence_count"), 0) <= 0:
                    reasons.append("team_lock_requires_model_evidence")
                if not apply_policy.get("display_only"):
                    reasons.append("team_lock_must_be_display_only")
                if not apply_policy.get("mutates_render_annotations"):
                    reasons.append("team_lock_must_mutate_render_annotations")
            if action == LOCK_DISPLAY_ROLE_ACTION:
                target_role = str(evidence.get("target_display_role") or "")
                if target_role not in {"player", "referee"}:
                    reasons.append("role_lock_requires_target_display_role")
                if evidence.get("vision_verdict") != target_role:
                    reasons.append("role_lock_requires_matching_role_verdict")
                if evidence.get("vision_status") != "reviewed":
                    reasons.append("role_lock_requires_reviewed_status")
                if _as_int(evidence.get("model_evidence_count"), 0) <= 0:
                    reasons.append("role_lock_requires_model_evidence")
                if not apply_policy.get("display_only"):
                    reasons.append("role_lock_must_be_display_only")
                if not apply_policy.get("mutates_render_annotations"):
                    reasons.append("role_lock_must_mutate_render_annotations")

        if reasons:
            rejected.append(
                {
                    "proposal_id": proposal_id,
                    "status": status,
                    "proposed_action": action,
                    "reasons": reasons,
                }
            )

    return {
        "schema_version": "1.0",
        "validator": "deterministic_phase9_identity_resolver_validator",
        "verdict": "PASS" if not rejected else "FAIL",
        "proposal_count": len(proposals),
        "ready_for_safe_apply_count": sum(
            1 for proposal in proposals if proposal.get("status") == READY_STATUS
        ),
        "rejected_proposal_count": len(rejected),
        "rejected_proposals": rejected,
    }


def build_identity_resolution_plan(
    *,
    identity_review_decisions: dict[str, Any],
    vision_review_queue: dict[str, Any],
    vision_review_results: dict[str, Any],
    player_crop_index: dict[str, Any] | None = None,
    render_audit_after: dict[str, Any] | None = None,
    final_render_identity_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Phase 9 dry-run identity resolution proposals.

    Args:
        identity_review_decisions: Phase 8 review decision artifact.
        vision_review_queue: Phase 4 queued review cases.
        vision_review_results: Phase 6 review results.
        player_crop_index: Optional crop/contact-sheet evidence.
        render_audit_after: Optional post-correction render audit.
        final_render_identity_manifest: Optional final manifest.

    Returns:
        JSON-safe dry-run resolution plan.
    """
    player_crop_index = player_crop_index or {}
    render_audit_after = render_audit_after or {}
    final_render_identity_manifest = final_render_identity_manifest or {}
    queue_cases = _cases_by_id(vision_review_queue)
    crop_cases = _cases_by_id(player_crop_index)
    vision_results = _results_by_case_id(vision_review_results)
    proposals: list[dict[str, Any]] = []

    for case_id in _review_case_ids(identity_review_decisions, vision_review_results):
        queue_case = queue_cases.get(case_id, {})
        vision_result = vision_results.get(case_id, {})
        question = str(
            queue_case.get("question") or vision_result.get("question") or ""
        )
        if question == "goalkeeper_identity_fragmentation":
            proposals.append(
                _build_goalkeeper_fragmentation_proposal(
                    case_id=case_id,
                    queue_case=queue_case,
                    vision_result=vision_result,
                    crop_case=crop_cases.get(case_id, {}),
                    identity_review_decisions=identity_review_decisions,
                    render_audit_after=render_audit_after,
                )
            )
        elif question in {"team_assignment_uncertain", "role_stability_flicker"}:
            proposals.append(
                _build_display_review_proposal(
                    case_id=case_id,
                    question=question,
                    queue_case=queue_case,
                    vision_result=vision_result,
                    crop_case=crop_cases.get(case_id, {}),
                )
            )
        else:
            proposals.append(
                {
                    "proposal_id": f"resolve_{case_id}",
                    "case_id": case_id,
                    "proposal_type": question or "identity_review",
                    "status": DEFERRED_STATUS,
                    "proposed_action": KEEP_UNRESOLVED_ACTION,
                    "confidence": 0.0,
                    "target_track_ids": _track_ids_from_evidence(
                        vision_result,
                        crop_cases.get(case_id, {}),
                    ),
                    "reason": "No resolver rule exists for this review case yet.",
                    "evidence": {
                        "question": question,
                        "vision_status": vision_result.get("status"),
                        "vision_verdict": vision_result.get("verdict"),
                        "contact_sheet_path": vision_result.get("contact_sheet_path")
                        or crop_cases.get(case_id, {}).get("contact_sheet_path"),
                    },
                    "apply_policy": {
                        "dry_run_only": True,
                        "requires_safe_apply_validator": True,
                        "mutates_raw_tracklets": False,
                        "mutates_render_annotations": False,
                    },
                }
            )

    for decision in _candidate_link_decisions(identity_review_decisions):
        proposals.append(_build_candidate_link_proposal(decision))

    ready_count = sum(1 for proposal in proposals if proposal.get("status") == READY_STATUS)
    deferred_count = sum(
        1 for proposal in proposals if proposal.get("status") == DEFERRED_STATUS
    )
    rejected_count = sum(
        1 for proposal in proposals if proposal.get("status") == REJECTED_STATUS
    )
    blocked_count = sum(
        1 for proposal in proposals if proposal.get("status") == BLOCKED_STATUS
    )

    if blocked_count:
        recommendation = "fix_render_identity_risks_before_resolution"
    elif ready_count and not deferred_count:
        recommendation = "safe_apply_available"
    elif ready_count:
        recommendation = "partial_safe_apply_available"
    elif rejected_count and not deferred_count:
        recommendation = "no_identity_merge_apply"
    elif proposals:
        recommendation = "keep_review_required_until_evidence"
    else:
        recommendation = "no_resolution_needed"

    plan = {
        "schema_version": "1.0",
        "phase": "phase_9_identity_review_resolver",
        "resolver": "deterministic_dry_run_identity_resolver_v1",
        "llm_model_invoked": False,
        "vision_model_invoked": bool(vision_review_results.get("vision_model_invoked")),
        "render_safe": bool(identity_review_decisions.get("render_safe")),
        "manifest_release_status": final_render_identity_manifest.get("release_status"),
        "recommendation": recommendation,
        "summary": {
            "proposal_count": len(proposals),
            "ready_for_safe_apply_count": ready_count,
            "deferred_count": deferred_count,
            "rejected_count": rejected_count,
            "blocked_count": blocked_count,
            "goalkeeper_merge_proposal_count": sum(
                1
                for proposal in proposals
                if proposal.get("proposed_action") == MERGE_GOALKEEPER_ACTION
            ),
            "candidate_link_proposal_count": sum(
                1
                for proposal in proposals
                if proposal.get("proposal_type") == "candidate_identity_link"
            ),
            "display_lock_proposal_count": sum(
                1
                for proposal in proposals
                if proposal.get("proposed_action") in DISPLAY_LOCK_ACTIONS
            ),
            "segment_split_required_count": sum(
                1
                for proposal in proposals
                if proposal.get("proposed_action") == MARK_SEGMENT_SPLIT_REQUIRED_ACTION
            ),
            "identity_cluster_required_count": sum(
                1
                for proposal in proposals
                if proposal.get("proposed_action") == MARK_IDENTITY_CLUSTER_REQUIRED_ACTION
            ),
        },
        "resolution_proposals": proposals,
        "safety_policy": {
            "auto_apply_resolver_output": False,
            "phase9_is_dry_run_only": True,
            "require_phase10_safe_apply_validator": True,
            "allow_merge_without_visual_or_llm_evidence": False,
        },
    }
    plan["validation"] = validate_identity_resolution_plan(plan)
    return plan
