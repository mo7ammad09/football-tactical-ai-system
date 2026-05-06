"""Safe application layer for Phase 9 identity resolution proposals.

Phase 10 never trusts a model output directly. It only applies proposals that
already passed the Phase 9 resolver validator, then records metadata that later
stages can use to promote identity-trusted output. Client-facing colors and
roles are kept unchanged unless a later validator explicitly allows them.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.identity.resolver import (
    MERGE_CANDIDATE_ACTION,
    MERGE_GOALKEEPER_ACTION,
    READY_STATUS,
)


APPLIED_STATUS = "applied"
NOOP_STATUS = "no_ready_proposals"
REJECTED_STATUS = "rejected"

SUPPORTED_ACTIONS = {
    MERGE_GOALKEEPER_ACTION,
    MERGE_CANDIDATE_ACTION,
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
        if confidence < 0.80:
            reasons.append("ready_proposal_confidence_below_threshold")
        if len(targets) < 2:
            reasons.append("ready_proposal_needs_multiple_targets")
        if not apply_policy.get("requires_safe_apply_validator"):
            reasons.append("proposal_did_not_require_safe_apply_validator")
        if action == MERGE_GOALKEEPER_ACTION:
            evidence = proposal.get("evidence") or {}
            if evidence.get("vision_verdict") != "same_player":
                reasons.append("goalkeeper_apply_requires_same_player_verdict")
            if evidence.get("vision_status") != "reviewed":
                reasons.append("goalkeeper_apply_requires_reviewed_status")

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


def _track_visible_goalkeeper(track: dict[str, Any]) -> bool:
    """Return whether an annotation track is currently rendered as goalkeeper."""
    return (
        str(track.get("display_role") or track.get("role") or "").lower()
        == "goalkeeper"
        or str(track.get("display_label") or "").upper() == "GK"
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
    actions: list[dict[str, Any]] = []
    updated_record_count = 0

    for proposal in ready:
        target_ids = set(_target_track_ids(proposal))
        payload = _proposal_resolution_payload(proposal)
        proposal_updated = 0
        action = str(proposal.get("proposed_action") or "")
        for row in rows:
            if _as_int(row.get("track_id"), -1) not in target_ids:
                continue
            if action == MERGE_GOALKEEPER_ACTION and not _row_visible_goalkeeper(row):
                continue
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

    return rows, {
        "schema_version": "1.0",
        "phase": "phase_10_identity_safe_apply",
        "safe_apply_status": APPLIED_STATUS if ready else NOOP_STATUS,
        "safe_apply_candidate": bool(ready),
        "applied_proposal_count": len(ready),
        "updated_record_count": updated_record_count,
        "actions": actions,
        "validation": validation,
        "notes": [
            "Phase 10 writes resolved identity metadata only.",
            "Track IDs and client-facing colors are not re-keyed by this phase.",
        ],
    }


def apply_identity_resolution_plan_to_annotation_states(
    annotation_states: list[dict[str, Any]],
    identity_resolution_plan: dict[str, Any],
) -> int:
    """Apply ready Phase 9 proposals to annotation-state metadata."""
    ready = _ready_proposals(identity_resolution_plan)
    if not ready:
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
                track.update(payload)
                updated += 1
    return updated
