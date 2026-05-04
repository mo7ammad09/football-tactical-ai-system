"""Deterministic pre-render identity review engine.

This is the text-first layer that reads QA artifacts and decides what can be
trusted immediately, what should stay unresolved for Vision/LLM, and what must
never be applied automatically.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


HIGH_RISK_SEVERITIES = {"critical", "high"}


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


def _high_risk_issues(render_audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Return audit issues that make rendered identity unsafe."""
    return [
        issue
        for issue in render_audit.get("issues", [])
        if str(issue.get("severity")) in HIGH_RISK_SEVERITIES
    ]


def _unresolved_vision_results(vision_review_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Return cases that still need visual or human review."""
    return [
        result
        for result in vision_review_results.get("results", [])
        if str(result.get("verdict")) == "unresolved"
    ]


def _goalkeeper_display_tracks(render_audit: dict[str, Any]) -> list[int]:
    """Return track ids that currently carry the client-facing GK display."""
    track_ids = {
        _as_int(segment.get("track_id"), -1)
        for segment in render_audit.get("goalkeeper_display_segments", [])
    }
    return sorted(track_id for track_id in track_ids if track_id >= 0)


def _candidate_link_summary(identity_debug: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact candidate links for text-first review."""
    links: list[dict[str, Any]] = []
    for link in identity_debug.get("candidate_links", []) or []:
        role_a = str(link.get("role_a") or "")
        role_b = str(link.get("role_b") or "")
        team_a = link.get("team_a")
        team_b = link.get("team_b")
        reid_distance = _as_float(link.get("reid_distance"), 999.0)
        auto_threshold = _as_float(link.get("auto_reid_threshold"), 0.0)
        same_role = role_a == role_b
        same_team = team_a == team_b
        auto_safe = (
            same_role
            and same_team
            and reid_distance <= auto_threshold
            and _as_int(link.get("gap_source_frames"), 999999) >= 0
        )
        links.append(
            {
                "source_id": link.get("source_id"),
                "target_id": link.get("target_id"),
                "roles": [role_a, role_b],
                "teams": [team_a, team_b],
                "gap_source_frames": link.get("gap_source_frames"),
                "reid_distance": link.get("reid_distance"),
                "auto_reid_threshold": link.get("auto_reid_threshold"),
                "decision": "auto_safe_candidate" if auto_safe else "needs_review",
                "reason": (
                    "Text evidence is inside the strict auto threshold."
                    if auto_safe
                    else "Candidate link is not strict enough for automatic identity merge."
                ),
            }
        )
    return links


def _identity_warning_counts(identity_debug: dict[str, Any]) -> dict[str, int]:
    """Return warning counts by warning code."""
    counts: Counter[str] = Counter()
    for warning in identity_debug.get("warnings", []) or []:
        code = str(warning.get("code") or "unknown")
        counts[code] += _as_int(warning.get("tracklet_count") or warning.get("display_id_count") or 1)
    return {code: int(count) for code, count in counts.items()}


def build_identity_review_decisions(
    *,
    identity_debug: dict[str, Any],
    render_audit_after: dict[str, Any],
    correction_applied: dict[str, Any],
    vision_review_queue: dict[str, Any],
    vision_review_results: dict[str, Any],
    final_render_identity_manifest: dict[str, Any],
    player_crop_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic identity review decisions from QA artifacts.

    Args:
        identity_debug: Identity profile/debug artifact.
        render_audit_after: Post-correction render audit.
        correction_applied: Applied deterministic correction artifact.
        vision_review_queue: Vision queue artifact.
        vision_review_results: Vision result artifact.
        final_render_identity_manifest: Final manifest artifact.
        player_crop_index: Optional crop/contact-sheet metadata.

    Returns:
        JSON-safe identity review decision artifact.
    """
    player_crop_index = player_crop_index or {}
    high_risk_issues = _high_risk_issues(render_audit_after)
    unresolved_results = _unresolved_vision_results(vision_review_results)
    candidate_links = _candidate_link_summary(identity_debug)
    review_candidate_links = [
        link for link in candidate_links if link.get("decision") != "auto_safe_candidate"
    ]
    audit_verdict = str(render_audit_after.get("verdict") or "UNKNOWN")
    render_safe = audit_verdict != "FAIL" and not high_risk_issues
    unresolved_case_count = len(unresolved_results)
    final_identity_needs_review = bool(unresolved_case_count or review_candidate_links)

    if not render_safe:
        recommendation = "render_not_safe"
    elif final_identity_needs_review:
        recommendation = "review_required"
    else:
        recommendation = "identity_trusted"

    decisions: list[dict[str, Any]] = [
        {
            "decision_id": "render_display_safety",
            "decision_type": "render_safety",
            "status": "accepted" if render_safe else "blocked",
            "confidence": 1.0 if render_safe else 0.0,
            "reason": (
                "Post-correction audit has no high-risk client-facing identity issues."
                if render_safe
                else "Post-correction audit still has high-risk identity issues."
            ),
            "evidence": {
                "audit_verdict": audit_verdict,
                "audit_score": render_audit_after.get("score"),
                "high_risk_issue_count": len(high_risk_issues),
                "summary": render_audit_after.get("summary", {}),
            },
        }
    ]

    if correction_applied.get("correction_applied"):
        decisions.append(
            {
                "decision_id": "deterministic_safe_corrections",
                "decision_type": "correction_summary",
                "status": "applied",
                "confidence": 1.0,
                "reason": "Deterministic display-only corrections were applied and kept.",
                "evidence": {
                    "applied_action_count": correction_applied.get("applied_action_count", 0),
                    "updated_record_count": correction_applied.get("updated_record_count", 0),
                    "updated_annotation_track_count": correction_applied.get(
                        "updated_annotation_track_count",
                        0,
                    ),
                },
            }
        )

    for result in unresolved_results:
        decisions.append(
            {
                "decision_id": f"unresolved_{result.get('case_id')}",
                "decision_type": str(result.get("question") or "identity_review"),
                "status": "needs_vision_or_llm",
                "confidence": 0.0,
                "reason": result.get("reason") or "No visual/LLM decision was made.",
                "evidence": {
                    "case_id": result.get("case_id"),
                    "priority": result.get("priority"),
                    "contact_sheet_path": result.get("contact_sheet_path"),
                    "crop_case_count": player_crop_index.get("case_count", 0),
                },
            }
        )

    for link in review_candidate_links:
        decisions.append(
            {
                "decision_id": f"candidate_link_{link.get('source_id')}_{link.get('target_id')}",
                "decision_type": "candidate_identity_link",
                "status": "needs_text_or_visual_review",
                "confidence": 0.0,
                "reason": link.get("reason"),
                "evidence": link,
            }
        )

    return {
        "schema_version": "1.0",
        "phase": "phase_8_identity_review_engine",
        "engine": "deterministic_text_first_v1",
        "llm_model_invoked": False,
        "vision_model_invoked": bool(vision_review_results.get("vision_model_invoked")),
        "recommendation": recommendation,
        "render_safe": render_safe,
        "final_identity_needs_review": final_identity_needs_review,
        "summary": {
            "decision_count": len(decisions),
            "high_risk_issue_count": len(high_risk_issues),
            "unresolved_case_count": unresolved_case_count,
            "candidate_link_count": len(candidate_links),
            "candidate_link_review_count": len(review_candidate_links),
            "goalkeeper_display_track_ids": _goalkeeper_display_tracks(render_audit_after),
            "identity_warning_counts": _identity_warning_counts(identity_debug),
            "manifest_release_status": final_render_identity_manifest.get("release_status"),
            "queued_vision_case_count": vision_review_queue.get("case_count", 0),
        },
        "decisions": decisions,
        "next_step": (
            "fix_render_identity_risks"
            if not render_safe
            else "invoke_vision_or_llm_for_unresolved_identity_cases"
            if final_identity_needs_review
            else "final_identity_ready"
        ),
        "safety_policy": {
            "auto_apply_llm_output": False,
            "require_validator_before_merge": True,
            "allow_display_only_safe_fixes": True,
            "allow_identity_merge_without_non_overlap_evidence": False,
        },
    }
