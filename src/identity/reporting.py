"""Presentation helpers for identity-correction reports."""

from __future__ import annotations

from typing import Any


IDENTITY_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("raw_tracklets_jsonl_url", "Raw tracklets"),
    ("identity_debug_json_url", "Identity debug"),
    ("identity_events_json_url", "Identity events"),
    ("render_audit_before_json_url", "Render audit before"),
    ("render_audit_after_json_url", "Render audit after"),
    ("correction_candidates_json_url", "Correction candidates"),
    ("correction_plan_json_url", "Correction plan"),
    ("correction_applied_json_url", "Correction applied"),
    ("vision_review_queue_json_url", "Vision review queue"),
    ("player_crop_index_json_url", "Player crop index"),
    ("vision_review_results_json_url", "Vision review results"),
    ("final_render_identity_manifest_json_url", "Final identity manifest"),
    ("identity_review_decisions_json_url", "Identity review decisions"),
    ("vision_contact_sheets_zip_url", "Vision contact sheets"),
)


def _artifact_public_url(results: dict[str, Any], artifact_name: str) -> str | None:
    """Return artifact URL from either flattened RunPod fields or artifact metadata."""
    url = results.get(f"{artifact_name}_url")
    if isinstance(url, str) and url:
        return url

    artifact = (results.get("artifacts") or {}).get(artifact_name)
    if isinstance(artifact, dict):
        public_url = artifact.get("public_url") or artifact.get("url")
        if isinstance(public_url, str) and public_url:
            return public_url
    return None


def build_identity_artifact_links(results: dict[str, Any]) -> list[dict[str, str]]:
    """Build stable UI links for identity review artifacts."""
    links: list[dict[str, str]] = []
    for url_key, label in IDENTITY_ARTIFACTS:
        artifact_name = url_key.removesuffix("_url")
        url = _artifact_public_url(results, artifact_name)
        if url:
            links.append({"key": url_key, "label": label, "url": url})
    return links


def summarize_pre_render_identity(results: dict[str, Any]) -> dict[str, Any]:
    """Return a compact identity status summary for UI and API consumers."""
    correction = results.get("pre_render_identity_correction") or {}
    release_status = str(correction.get("release_status") or "unknown")
    audit_verdict = str(
        correction.get("render_audit_after_verdict")
        or correction.get("render_audit_verdict")
        or "UNKNOWN"
    )
    validation = correction.get("final_manifest_validation") or {}
    validation_verdict = str(validation.get("verdict") or "UNKNOWN")
    unresolved_count = int(correction.get("vision_review_unresolved_count") or 0)
    queue_count = int(correction.get("vision_review_queue_count") or 0)
    correction_applied = bool(correction.get("correction_applied"))
    vision_invoked = bool(correction.get("vision_model_invoked"))

    if release_status == "identity_trusted":
        severity = "success"
        headline = "Identity trusted"
    elif release_status == "review_required":
        severity = "warning"
        headline = "Identity review required"
    elif release_status == "invalid_render":
        severity = "error"
        headline = "Identity render invalid"
    else:
        severity = "info"
        headline = "Identity status unavailable"

    return {
        "phase": correction.get("phase"),
        "release_status": release_status,
        "output_identity_mode": correction.get("output_identity_mode"),
        "render_policy": correction.get("render_policy"),
        "severity": severity,
        "headline": headline,
        "audit_verdict": audit_verdict,
        "audit_score": correction.get("render_audit_after_score")
        or correction.get("render_audit_score"),
        "validation_verdict": validation_verdict,
        "unresolved_case_count": unresolved_count,
        "queued_case_count": queue_count,
        "correction_applied": correction_applied,
        "vision_model_invoked": vision_invoked,
        "artifact_links": build_identity_artifact_links(results),
    }
